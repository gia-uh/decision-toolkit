"""`decision_toolkit.adapters.julia` — Julia-1's single-graph layout.

Contract (ported from the official `julia/data.py` sequence builder and the
ONNX repository's browser runtime):

    [CLS] <type> question: <instructions> [SEP] ([MASK] opt0) ([MASK] opt1) … [SEP] <state> [SEP]
    one ONNX graph: input_ids, attention_mask, marker_pos, marker_mask, qtype -> logits
    probabilities = softmax(logits); noul options are [false, true], so p(true) = p[1]

Unlike Laya there is no separate head and no temperature: the exported graph
scores the options directly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ..engine import OnnxEngine, encode
from ..errors import AdapterError
from .base import (
    build_choice_answer,
    build_noul_answer,
    build_score_answer,
    question_options,
    softmax,
    state_to_text,
)

logger = logging.getLogger(__name__)

QTYPES = {"choice": 0, "score": 1, "noul": 2}


class JuliaAdapter:
    """System One over a Julia-1 ONNX export (one graph, marker/qtype inputs)."""

    name = "julia"

    def __init__(self, *, max_length: int = 1024, head_length: int = 256) -> None:
        self.max_length = int(max_length)
        self.head_length = int(head_length)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _special_ids(tokenizer: Any) -> tuple[int, int, int]:
        for mask_name, bos_name, eos_name in (
            ("<mask>", "<bos>", "<eos>"),
            ("[MASK]", "[CLS]", "[SEP]"),
        ):
            ids = (
                tokenizer.token_to_id(mask_name),
                tokenizer.token_to_id(bos_name),
                tokenizer.token_to_id(eos_name),
            )
            if all(token is not None for token in ids):
                return ids  # type: ignore[return-value]
        raise AdapterError(
            "the tokenizer exposes neither <mask>/<bos>/<eos> nor [MASK]/[CLS]/[SEP]; "
            "this is not a Julia-1 tokenizer"
        )

    @staticmethod
    def _render_options(question: Mapping[str, Any]) -> list[str]:
        criteria = question.get("criteria")
        kind = question.get("type")
        if kind == "choice":
            return [f"{key}: {value}" if value else key for key, value in criteria.items()]
        if kind == "score":
            return [str(level) for level in criteria]
        criteria = criteria or {}
        return [
            "false: " + (criteria.get("false") or "false"),
            "true: " + (criteria.get("true") or "true"),
        ]

    def _sequence(
        self,
        tokenizer: Any,
        state_text: str,
        question: Mapping[str, Any],
        options: list[str],
    ) -> tuple[list[int], list[int], bool]:
        mask, bos, eos = self._special_ids(tokenizer)
        truncated = False
        instructions = str(question.get("instructions") or "")
        head_ids = encode(tokenizer, f"{question['type']} question: {instructions}")
        option_ids: list[list[int]] = []
        for option in options:
            encoded = encode(tokenizer, " " + option)
            if len(encoded) > 48:
                truncated = True
            option_ids.append([mask, *encoded[:48]])

        budget = self.head_length - sum(len(option) for option in option_ids)
        if budget < 16:
            truncated = True
            per = max(4, (self.head_length - 16) // max(1, len(option_ids)))
            option_ids = [option[:per] for option in option_ids]
            budget = self.head_length - sum(len(option) for option in option_ids)
        if len(head_ids) > max(8, budget):
            truncated = True
        head_ids = head_ids[: max(8, budget)]

        ids = [bos, *head_ids, eos]
        markers: list[int] = []
        for option in option_ids:
            markers.append(len(ids))
            ids.extend(option)
        ids.append(eos)

        room = max(1, self.max_length - len(ids) - 1)
        state_ids = encode(tokenizer, state_text)
        if len(state_ids) > room:
            truncated = True
        state_ids = state_ids[:room]
        all_ids = (ids + state_ids + [eos])[: self.max_length]
        if len(markers) > sum(1 for marker in markers if marker < self.max_length):
            truncated = True
        return all_ids, [marker for marker in markers if marker < self.max_length], truncated

    # -------------------------------------------------------------- inference
    def system_one(
        self, engine: OnnxEngine, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        import numpy as np

        tokenizer = engine.tokenizer
        state_text = state_to_text(state)
        answers: dict[str, dict[str, Any]] = {}

        for name, question in questions.items():
            kind, labels, _ = question_options(question)
            options = self._render_options(question)
            ids, markers, truncated = self._sequence(tokenizer, state_text, question, options)
            if len(markers) != len(options):
                raise AdapterError(
                    f"{name!r}: the options do not fit in {self.head_length} tokens"
                )

            input_ids = np.asarray([ids], dtype=np.int64)
            attention = np.ones_like(input_ids)
            logits = engine.session("model").run(None, {
                "input_ids": input_ids,
                "attention_mask": attention,
                "marker_pos": np.asarray([markers], dtype=np.int64),
                "marker_mask": np.ones((1, len(markers)), dtype=bool),
                "qtype": np.asarray([QTYPES[kind]], dtype=np.int64),
            })[0]

            probabilities = softmax(np.asarray(logits)[0][: len(markers)].tolist())
            if kind == "choice":
                answer = build_choice_answer(labels, probabilities)
            elif kind == "noul":
                answer = build_noul_answer(probabilities[1], probabilities[0])
            else:
                answer = build_score_answer(labels, probabilities)
            if truncated:
                answer["truncated"] = True
                answer["warnings"] = [
                    f"input was truncated to fit max_length={self.max_length}"
                ]
            answers[name] = answer
        return answers


__all__ = ["JuliaAdapter"]
