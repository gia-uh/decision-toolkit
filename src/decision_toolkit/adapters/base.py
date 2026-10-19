"""`decision_toolkit.adapters.base` — the adapter contract and shared helpers.

An adapter is the only place that knows a model's **usage contract**: how to
build its input from a System One state/questions and how to read its output
back into probabilities. Several models can share an adapter when they share
that contract (e.g. the GLiNER2.5 family); a model with a different layout
gets its own adapter. No adapter is written without a real `.onnx` behind it.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..engine import OnnxEngine
from ..errors import AdapterError, InvalidQuestionError


@runtime_checkable
class Adapter(Protocol):
    """What :class:`~decision_toolkit.decision_model.DecisionModel` needs."""

    name: str

    def system_one(
        self, engine: OnnxEngine, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]: ...


def state_to_text(state: Any) -> str:
    """Render a System One ``state`` as text for text-based model families."""
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping) or isinstance(state, (list, tuple)):
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    if state is None:
        raise AdapterError("state cannot be None")
    return str(state)


def softmax(scores: Sequence[float]) -> list[float]:
    """Numerically stable softmax over raw scores."""
    if not scores:
        raise AdapterError("softmax needs at least one score")
    top = max(float(score) for score in scores)
    exponentials = [math.exp(float(score) - top) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def question_options(question: Mapping[str, Any]) -> tuple[str, list[str], list[str | None]]:
    """Extract ``(kind, labels, descriptions)`` from a System One question.

    * ``choice``: labels and descriptions come from the question's ``criteria``.
    * ``noul``: labels are ``["yes", "no"]``; ``criteria.true``/``criteria.false``
      become their descriptions.
    * ``score``: labels are the ordered levels; no descriptions.
    """
    kind = str(question.get("type"))
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or not criteria:
            raise InvalidQuestionError("a choice needs criteria (option labels) for this adapter")
        labels = [str(label) for label in criteria]
        descriptions = [None if value is None else str(value) for value in criteria.values()]
        return kind, labels, descriptions
    if kind == "noul":
        criteria = question.get("criteria") or {}
        if not isinstance(criteria, Mapping):
            criteria = {}
        true_description = criteria.get("true")
        false_description = criteria.get("false")
        return kind, ["yes", "no"], [
            None if true_description is None else str(true_description),
            None if false_description is None else str(false_description),
        ]
    if kind == "score":
        levels = question.get("criteria")
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or len(levels) < 2:
            raise InvalidQuestionError("a score needs its ordered levels in 'criteria'")
        labels = [str(level) for level in levels]
        return kind, labels, [None] * len(labels)
    raise InvalidQuestionError(f"unsupported question type {kind!r}")


def build_choice_answer(labels: Sequence[str], probabilities: Sequence[float]) -> dict[str, Any]:
    best = max(range(len(labels)), key=lambda index: probabilities[index])
    return {
        "type": "choice",
        "choice": labels[best],
        "probabilities": {label: float(p) for label, p in zip(labels, probabilities)},
        "confidence": float(probabilities[best]),
    }


def build_noul_answer(probability_yes: float, probability_no: float) -> dict[str, Any]:
    return {
        "type": "noul",
        "noul": float(probability_yes),
        "probabilities": {"true": float(probability_yes), "false": float(probability_no)},
        "confidence": float(max(probability_yes, probability_no)),
    }


def build_score_answer(labels: Sequence[str], probabilities: Sequence[float]) -> dict[str, Any]:
    expected = sum(index * probability for index, probability in enumerate(probabilities))
    best = max(range(len(labels)), key=lambda index: probabilities[index])
    return {
        "type": "score",
        "score": float(expected),
        "level": int(best),
        "legend": {str(index): label for index, label in enumerate(labels)},
        "probabilities": {str(index): float(p) for index, p in enumerate(probabilities)},
        "confidence": float(probabilities[best]),
    }


__all__ = [
    "Adapter",
    "state_to_text",
    "softmax",
    "question_options",
    "build_choice_answer",
    "build_noul_answer",
    "build_score_answer",
]
