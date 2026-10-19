"""`decision_toolkit.adapters.gliner2` — the GLiNER2.5 classifier layout.

Contract implemented by the GLiNER2.5 exports (split encoder + classifier):

    schema = ( [P] <task> [DESCRIPTION] label: … ( [C] label … ) ) [SEP_TEXT] <words>
    encoder(input_ids, attention_mask) -> hidden states
    classifier(hidden_state at each [C]) -> one logit per option

Validated against the local mininvox export (15 routing cases, 9/15, 215 ms
on CPU). This adapter only runs models whose files follow that layout.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

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

#: Reference word pattern of the GLiNER2 schema (urls, emails, handles, words, punctuation).
WORD_PATTERN = re.compile(
    r"(?:https?://[^\s]+|www\.[^\s]+)"
    r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
    r"|@[a-z0-9_]+"
    r"|[\w]+(?:[-_][\w]+)*"
    r"|\S",
    re.I | re.U,
)


class Gliner2Adapter:
    """System One over a GLiNER2.5 classifier export (encoder + classifier)."""

    name = "gliner2"

    def __init__(self, *, task: str = "classify", use_descriptions: bool = True) -> None:
        self._task = task
        self._use_descriptions = bool(use_descriptions)
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------- tokenizing
    def _tokenize(self, tokenizer: Any, token: str) -> list[int]:
        ids = self._cache.get(token)
        if ids is None:
            ids = encode(tokenizer, token)
            self._cache[token] = ids
        return ids

    # --------------------------------------------------------------- inference
    def _scores(
        self,
        engine: OnnxEngine,
        text: str,
        options: list[tuple[str, str | None]],
        task: str,
    ) -> list[float]:
        import numpy as np  # only the real ONNX path needs arrays

        text = text if text.endswith((".", "!", "?")) else text + "."
        words = [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]
        tokenizer = engine.tokenizer

        parent = task
        if self._use_descriptions:
            for label, description in options:
                if description:
                    parent += f" [DESCRIPTION] {label}: {description}"

        schema = ["(", "[P]", parent, "("]
        for label, _ in options:
            schema += ["[C]", label]
        schema += [")", ")"]

        input_ids: list[int] = []
        positions: list[int] = []
        for token in schema:
            if token == "[C]":
                positions.append(len(input_ids))
            input_ids.extend(self._tokenize(tokenizer, token))
        input_ids.extend(self._tokenize(tokenizer, "[SEP_TEXT]"))
        for word in words:
            input_ids.extend(self._tokenize(tokenizer, word))

        ids = np.asarray([input_ids], dtype=np.int64)
        attention = np.ones_like(ids)

        # Fused export (e.g. onnx-community/GLiNER2.5-Decide-ONNX): one graph with
        # marker_positions and one logit per marker. Split export (e.g. the multi-v1
        # repo): encoder + classifier, with the classifier input name read from the graph.
        if "model" in engine.files:
            session = engine.session("model")
            input_names = {item.name for item in session.get_inputs()}
            feeds = {"input_ids": ids, "attention_mask": attention}
            if "marker_positions" in input_names:
                feeds["marker_positions"] = np.asarray([positions], dtype=np.int64)
            logits = session.run(None, feeds)[0]
            return [float(value) for value in np.asarray(logits)[0][: len(options)]]

        hidden = engine.session("encoder").run(
            None, {"input_ids": ids, "attention_mask": attention}
        )[0]
        queries = hidden[0][positions].astype(np.float32)
        classifier = engine.session("classifier")
        input_name = classifier.get_inputs()[0].name
        logits = classifier.run(None, {input_name: queries})[0].reshape(-1)
        return [float(value) for value in logits]

    def system_one(
        self, engine: OnnxEngine, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        state_text = state_to_text(state)
        answers: dict[str, dict[str, Any]] = {}
        for name, question in questions.items():
            kind, labels, descriptions = question_options(question)
            task = str(question.get("task") or self._task)
            scores = self._scores(engine, state_text, list(zip(labels, descriptions)), task)
            if len(scores) != len(labels):
                raise AdapterError(
                    f"the classifier returned {len(scores)} scores for {len(labels)} options"
                )
            probabilities = softmax(scores)
            if kind == "choice":
                answers[name] = build_choice_answer(labels, probabilities)
            elif kind == "noul":
                answers[name] = build_noul_answer(probabilities[0], probabilities[1])
            else:
                answers[name] = build_score_answer(labels, probabilities)
        return answers


__all__ = ["Gliner2Adapter"]
