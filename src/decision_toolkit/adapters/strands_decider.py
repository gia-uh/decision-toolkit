"""`decision_toolkit.adapters.strands_decider` — Strands Decider ONNX layout.

Reference: ``onnx-community/strands-decider-2B-hobson-v19-ONNX`` and the
upstream ``strands-decider`` prompt renderer.  Each question is rendered as a
``<state>…</state>`` prefix plus a ``<question>`` block ending in the
``<answer>`` marker; the graph returns one pointer logit per option.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from ..engine import OnnxEngine
from ..errors import AdapterError
from .base import (
    build_choice_answer,
    build_noul_answer,
    build_score_answer,
    question_options,
    softmax,
)

logger = logging.getLogger(__name__)

NOUL_DEFAULT = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}
HEADERS = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}


def _render(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False)


class StrandsDeciderAdapter:
    """System One over the Strands Decider 2B ONNX export."""

    name = "strands-decider"
    required_roles = frozenset({"model", "tokenizer", "config"})

    @staticmethod
    def _encode_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            ids = getattr(encoded, "ids", None)
            offsets = getattr(encoded, "offsets", None)
            if ids is not None and offsets is not None:
                return [int(item) for item in ids], [(int(a), int(b)) for a, b in offsets]
        except TypeError:
            pass
        try:
            encoded = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )
            ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
                offsets = offsets[0]
            return [int(item) for item in ids], [(int(a), int(b)) for a, b in offsets]
        except Exception as error:
            raise AdapterError("the tokenizer cannot return offset mappings") from error

    @staticmethod
    def _option_lines(question: Mapping[str, Any]) -> tuple[str, list[str], list[str]]:
        kind, labels, descriptions = question_options(question)
        if kind == "noul":
            criteria = question.get("criteria") or {}
            false_desc = (criteria.get("false") if isinstance(criteria, Mapping) else None) or NOUL_DEFAULT["false"]
            true_desc = (criteria.get("true") if isinstance(criteria, Mapping) else None) or NOUL_DEFAULT["true"]
            return kind, ["false", "true"], [str(false_desc), str(true_desc)]
        if kind == "choice":
            return kind, labels, [description or "" for description in descriptions]
        return kind, [str(index) for index in range(len(labels))], labels

    def _prompt(
        self, state: Any, question: Mapping[str, Any]
    ) -> tuple[str, str, list[str], list[tuple[int, int]]]:
        kind, labels, descriptions = self._option_lines(question)
        instructions = _render(question.get("instructions"))
        header = HEADERS[kind]
        lines = [
            f"{index + 1}. {label}" + (f" \u2014 {_render(description)}" if description else "")
            for index, (label, description) in enumerate(zip(labels, descriptions))
        ]
        prefix = f'<question type="{kind}">\n{header}\n{instructions}\n<options>\n'
        cursor = len(prefix)
        spans: list[tuple[int, int]] = []
        for line in lines:
            spans.append((cursor, cursor + len(line)))
            cursor += len(line) + 1
        block = "\n".join(lines)
        question_text = prefix + block + "\n</options>\n</question>\n<answer>"
        state_text = _render(state)
        return state_text, question_text, labels, spans

    @staticmethod
    def _option_positions(
        offsets: Sequence[tuple[int, int]],
        spans: Sequence[tuple[int, int]],
        state_chars: int,
    ) -> list[int]:
        positions: list[int] = []
        for start, end in spans:
            start += state_chars
            end += state_chars
            candidates = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start and token_start < end and token_end <= end
            ]
            if not candidates:
                raise AdapterError("could not locate an option's last token")
            positions.append(candidates[-1])
        return positions

    def system_one(
        self, engine: OnnxEngine, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        import numpy as np

        config = engine.read_json("config")
        section = config.get("decider") if isinstance(config, Mapping) else None
        if not isinstance(section, Mapping):
            section = {}
        max_len = int(section.get("max_length", 4096))
        temperatures = section.get("temperature_by_kind") or {}
        default_temperature = float(section.get("temperature", 1.0))
        state_text, _, _, _ = self._prompt(state, next(iter(questions.values())))
        state_prefix = f"<state>\n{state_text}\n</state>\n"
        state_chars = len(state_prefix)

        answers: dict[str, dict[str, Any]] = {}
        for name, question in questions.items():
            _, question_text, labels, spans = self._prompt(state, question)
            prompt = state_prefix + question_text
            ids, offsets = self._encode_offsets(engine.tokenizer, prompt)
            if len(ids) > max_len:
                # The state is the only part that can be shortened without losing
                # the option binding. Keep the question suffix intact.
                excess = len(ids) - max_len
                state_ids, state_offsets = self._encode_offsets(engine.tokenizer, state_prefix)
                keep = max(1, len(state_ids) - excess)
                ids = state_ids[:keep] + ids[len(state_ids):]
                offsets = state_offsets[:keep] + offsets[len(state_ids):]
            positions = self._option_positions(offsets, spans, state_chars)
            input_ids = np.asarray([ids], dtype=np.int64)
            answer_pos = np.asarray([len(ids) - 1], dtype=np.int64)
            option_pos = np.asarray([positions], dtype=np.int64)
            logits = engine.session("model").run(None, {
                "input_ids": input_ids,
                "attention_mask": np.ones_like(input_ids),
                "answer_pos": answer_pos,
                "option_pos": option_pos,
            })[0][0]
            temperature = float(temperatures.get(question.get("type"), default_temperature))
            probabilities = softmax([float(value) / temperature for value in logits[: len(labels)]])
            kind = question.get("type")
            if kind == "choice":
                answers[name] = build_choice_answer(labels, probabilities)
            elif kind == "noul":
                answers[name] = build_noul_answer(probabilities[1], probabilities[0])
            else:
                levels = [str(level) for level in question.get("criteria") or []]
                answers[name] = build_score_answer(levels, probabilities)
        return answers


__all__ = ["StrandsDeciderAdapter"]
