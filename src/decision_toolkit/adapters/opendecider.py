"""`decision_toolkit.adapters.opendecider` — OpenDecider nano's single-graph layout.

Contract (from the project's `opendecider/prompt.py::nano_ids` and the export's
`opendecider.json`):

    [CLS] question: <instructions> [SEP] ([MASK] opt) … [SEP] input: <state> [SEP]
    one ONNX graph: (input_ids, attention_mask) -> logits per token
    the answer is softmax over the logits at the [MASK] positions
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..engine import OnnxEngine, encode
from ..errors import AdapterError
from .base import (
    build_choice_answer,
    build_noul_answer,
    build_score_answer,
    softmax,
    state_to_text,
)

logger = logging.getLogger(__name__)


class OpendeciderAdapter:
    """System One over the official OpenDecider nano ONNX build."""

    name = "opendecider"

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _render_options(question: Mapping[str, Any]) -> dict[str, str | None]:
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            return {str(key): (None if value is None else str(value)) for key, value in criteria.items()}
        if kind == "score":
            return {str(level): None for level in criteria}
        criteria = criteria or {}
        return {
            "false": None if criteria.get("false") is None else str(criteria.get("false")),
            "true": None if criteria.get("true") is None else str(criteria.get("true")),
        }

    @staticmethod
    def _special_ids(config: Mapping[str, Any]) -> tuple[int, int, int]:
        ids = config.get("special_ids") or {}
        try:
            return int(ids["cls"]), int(ids["sep"]), int(ids["mask"])
        except (KeyError, TypeError, ValueError) as error:
            raise AdapterError(
                "opendecider.json must declare special_ids.cls/sep/mask"
            ) from error

    def _sequence(
        self,
        tokenizer: Any,
        state_text: str,
        instructions: str,
        options: Mapping[str, str | None],
        special_ids: tuple[int, int, int],
        max_len: int,
    ) -> tuple[list[int], list[int]]:
        cls_id, sep_id, mask_id = special_ids
        ids = [cls_id, *encode(tokenizer, f"question: {instructions}"), sep_id]
        markers: list[int] = []
        for key, value in options.items():
            markers.append(len(ids))
            ids.append(mask_id)
            text = f" {key}: {value}" if value and value != key else f" {key}"
            ids.extend(encode(tokenizer, text))
        ids.append(sep_id)

        state_ids = encode(tokenizer, f"input: {state_text}")
        room = max(max_len - len(ids) - 1, 0)
        ids = ids + state_ids[:room] + [sep_id]
        return ids[:max_len], [marker for marker in markers if marker < max_len]

    def _scores(
        self,
        engine: OnnxEngine,
        state_text: str,
        instructions: str,
        options: Mapping[str, str | None],
        special_ids: tuple[int, int, int],
        max_len: int,
    ) -> list[float]:
        import numpy as np

        ids, markers = self._sequence(
            engine.tokenizer, state_text, instructions, options, special_ids, max_len
        )
        input_ids = np.asarray([ids], dtype=np.int64)
        logits = engine.session("model").run(None, {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
        })[0][0]
        return [float(logits[position]) for position in markers]

    # -------------------------------------------------------------- inference
    def system_one(
        self, engine: OnnxEngine, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        config = engine.read_json("config")
        special_ids = self._special_ids(config)
        max_len = int(config.get("max_len", 2048))
        state_text = state_to_text(state)
        answers: dict[str, dict[str, Any]] = {}

        for name, question in questions.items():
            options = self._render_options(question)
            scores = self._scores(
                engine, state_text, str(question.get("instructions") or ""),
                options, special_ids, max_len,
            )
            probabilities = softmax(scores)
            kind = question.get("type")
            labels = list(options)
            if kind == "choice":
                answers[name] = build_choice_answer(labels, probabilities)
            elif kind == "noul":
                answers[name] = build_noul_answer(probabilities[1], probabilities[0])
            else:
                answers[name] = build_score_answer(labels, probabilities)
        return answers


__all__ = ["OpendeciderAdapter"]
