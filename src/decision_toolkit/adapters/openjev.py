"""`decision_toolkit.adapters.openjev` — open-jev DeBERTa typed-decision layout.

Reference: ``onnx-community/open-jev-deberta-v3-large-ONNX``.  One DeBERTa
sequence contains the state and every question/option pair.  A span pooling
head scores each option; logits are grouped by question and softmaxed with the
checkpoint's calibrated temperature.
"""

from __future__ import annotations

import json
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

MARKERS = ("[CLS]", "[SEP]", "[STATE]", "[Q]", "[OPT]")


class OpenJevAdapter:
    """System One over the open-jev DeBERTa-v3-large ONNX export."""

    name = "open-jev"
    required_roles = frozenset({"model", "tokenizer", "config"})

    @staticmethod
    def _token_id(tokenizer: Any, token: str) -> int:
        converter = getattr(tokenizer, "convert_tokens_to_ids", None)
        if converter is not None:
            value = converter(token)
            if isinstance(value, int) and value >= 0:
                return int(value)
        low_level = getattr(tokenizer, "token_to_id", None)
        if low_level is not None:
            value = low_level(token)
            if isinstance(value, int):
                return int(value)
        ids = encode(tokenizer, token)
        if len(ids) == 1:
            return ids[0]
        raise AdapterError(f"the tokenizer does not expose the single token {token!r}")

    @staticmethod
    def _markers(tokenizer: Any) -> dict[str, int]:
        try:
            return {name: OpenJevAdapter._token_id(tokenizer, name) for name in MARKERS}
        except AdapterError as error:
            raise AdapterError(
                "the tokenizer must contain [CLS], [SEP], [STATE], [Q] and [OPT]"
            ) from error

    @staticmethod
    def _options(question: Mapping[str, Any]) -> tuple[str, list[str]]:
        kind, labels, _ = question_options(question)
        if kind == "noul":
            return kind, ["no", "yes"]
        return kind, labels

    def _tokens(
        self,
        tokenizer: Any,
        markers: Mapping[str, int],
        state_text: str,
        questions: Mapping[str, Mapping[str, Any]],
        max_state_tokens: int,
        max_len: int,
    ) -> tuple[list[int], list[int], list[int], list[int], bool]:
        cls_id = markers["[CLS]"]
        sep_id = markers["[SEP]"]
        state_id = markers["[STATE]"]
        q_id = markers["[Q]"]
        opt_id = markers["[OPT]"]

        state_tokens = encode(tokenizer, state_text)[:max_state_tokens]
        total_pairs = sum(len(self._options(question)[1]) for question in questions.values())

        tokens = [cls_id, state_id, *state_tokens]
        seg = [-1] * len(tokens)
        pair_q: list[int] = []
        pair_opt: list[int] = []

        for qi, question in enumerate(questions.values()):
            _, options = self._options(question)
            question_tokens = encode(tokenizer, str(question.get("instructions") or ""))
            tokens.append(q_id)
            seg.append(-1)
            tokens.extend(question_tokens)
            seg.extend([total_pairs + qi] * len(question_tokens))
            for option in options:
                option_tokens = encode(tokenizer, option)
                pair_index = len(pair_opt)
                tokens.append(opt_id)
                seg.append(-1)
                tokens.extend(option_tokens)
                seg.extend([pair_index] * len(option_tokens))
                pair_q.append(total_pairs + qi)
                pair_opt.append(pair_index)

        tokens.append(sep_id)
        seg.append(-1)

        truncated = False
        if len(tokens) > max_len:
            # Keep the question/option suffix; trim the state prefix.
            excess = len(tokens) - max_len
            state_start = 2
            removable = max(0, min(len(state_tokens), excess))
            if removable:
                del tokens[state_start : state_start + removable]
                del seg[state_start : state_start + removable]
                truncated = True
            if len(tokens) > max_len:
                raise AdapterError(
                    f"question and options exceed the {max_len}-token context window"
                )
        return tokens, seg, pair_q, pair_opt, truncated

    def system_one(
        self, engine: OnnxEngine, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        import numpy as np

        config = engine.read_json("config")
        section = config.get("open_jev") if isinstance(config, Mapping) else None
        if not isinstance(section, Mapping):
            section = {}
        max_len = int(section.get("max_len", 512))
        max_state_tokens = int(section.get("max_state_tokens", 256))
        temperature = float(section.get("temperature", 1.05))
        tokenizer = engine.tokenizer
        markers = self._markers(tokenizer)
        state_text = state_to_text(state)
        if isinstance(state, (Mapping, list, tuple)):
            state_text = json.dumps(state, ensure_ascii=False, sort_keys=True)

        tokens, seg, pair_q, pair_opt, truncated = self._tokens(
            tokenizer, markers, state_text, questions, max_state_tokens, max_len
        )
        ids = np.asarray([tokens], dtype=np.int64)
        seg_array = np.asarray([seg], dtype=np.int64)
        pair_q_array = np.asarray([pair_q], dtype=np.int64)
        pair_opt_array = np.asarray([pair_opt], dtype=np.int64)
        logits = engine.session("model").run(None, {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "seg": seg_array,
            "pair_q": pair_q_array,
            "pair_opt": pair_opt_array,
        })[0][0]

        answers: dict[str, dict[str, Any]] = {}
        cursor = 0
        for name, question in questions.items():
            kind, labels = self._options(question)
            count = len(labels)
            raw = [float(value) / temperature for value in np.asarray(logits)[cursor : cursor + count]]
            cursor += count
            probabilities = softmax(raw)
            if kind == "choice":
                answers[name] = build_choice_answer(labels, probabilities)
            elif kind == "noul":
                answers[name] = build_noul_answer(probabilities[1], probabilities[0])
            else:
                answers[name] = build_score_answer(labels, probabilities)
            if truncated:
                answers[name]["truncated"] = True
                answers[name]["warnings"] = [f"state was truncated to fit max_len={max_len}"]
        return answers


__all__ = ["OpenJevAdapter"]
