"""`decision_toolkit.adapters.gliner2_decide` — the fused GLiNER2.5-Decide export.

The fused ONNX export (`onnx-community/GLiNER2.5-Decide-ONNX`) differs from the
split multi-v1 layout: it expects the GLiNER2 classification sequence with
``[L]`` label markers and a ``[SEP_STRUCT]`` structural separator, and returns
one logit per marker in a single graph:

    ( [P] prompt [DESCRIPTION] … ( [L] label … ) ) [SEP_STRUCT] [SEP_TEXT] words
    inputs: input_ids, attention_mask, marker_positions
    output: logits [batch, markers] → softmax per question
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
from .gliner2 import WORD_PATTERN

logger = logging.getLogger(__name__)


class Gliner2DecideAdapter:
    """System One over the fused GLiNER2.5-Decide ONNX export."""

    name = "gliner2-decide"

    def __init__(self, *, use_descriptions: bool = True) -> None:
        self._use_descriptions = bool(use_descriptions)
        self._cache: dict[str, list[int]] = {}

    def _tokenize(self, tokenizer: Any, token: str) -> list[int]:
        ids = self._cache.get(token)
        if ids is None:
            ids = encode(tokenizer, token)
            self._cache[token] = ids
        return ids

    def _scores(
        self,
        engine: OnnxEngine,
        text: str,
        prompt: str,
        options: list[tuple[str, str | None]],
    ) -> list[float]:
        import numpy as np

        tokenizer = engine.tokenizer
        text = text if text.endswith((".", "!", "?")) else text + "."
        words = [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]

        parent = prompt
        if self._use_descriptions:
            for label, description in options:
                if description:
                    parent += f" [DESCRIPTION] {label}: {description}"

        schema = ["(", "[P]", parent, "("]
        for label, _ in options:
            schema += ["[L]", label]
        schema += [")", ")", "[SEP_STRUCT]", "[SEP_TEXT]"]

        input_ids: list[int] = []
        positions: list[int] = []
        for token in schema:
            if token == "[L]":
                positions.append(len(input_ids))
            input_ids.extend(self._tokenize(tokenizer, token))
        for word in words:
            input_ids.extend(self._tokenize(tokenizer, word))

        ids = np.asarray([input_ids], dtype=np.int64)
        logits = engine.session("model").run(None, {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "marker_positions": np.asarray([positions], dtype=np.int64),
        })[0][0]
        return [float(value) for value in np.asarray(logits)[: len(positions)]]

    def system_one(
        self, engine: OnnxEngine, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        state_text = state_to_text(state)
        answers: dict[str, dict[str, Any]] = {}
        for name, question in questions.items():
            kind, labels, descriptions = question_options(question)
            prompt = str(question.get("instructions") or "classify")
            scores = self._scores(
                engine, state_text, prompt, list(zip(labels, descriptions))
            )
            if len(scores) != len(labels):
                raise AdapterError(
                    f"{name!r}: the graph returned {len(scores)} logits for {len(labels)} labels"
                )
            probabilities = softmax(scores)
            if kind == "choice":
                answers[name] = build_choice_answer(labels, probabilities)
            elif kind == "noul":
                # question_options maps noul to labels ["yes", "no"]
                answers[name] = build_noul_answer(probabilities[0], probabilities[1])
            else:
                answers[name] = build_score_answer(labels, probabilities)
        return answers


__all__ = ["Gliner2DecideAdapter"]
