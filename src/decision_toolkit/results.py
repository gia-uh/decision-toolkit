"""`decision_toolkit.results` — typed views over System One answers.

Adapters return raw System One answer dictionaries; these dataclasses give
Python callers (and the server) a stable, typed way to read them without
touching vendor-specific key spellings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Answer:
    """One named answer with the probabilities the model placed on its options."""

    key: str
    type: str
    data: Mapping[str, Any] = field(repr=False)

    @property
    def choice(self) -> Any:
        """The chosen option (``choice`` answers)."""
        return self.data.get("choice")

    @property
    def noul(self) -> float | None:
        """Probability that the yes/no answer is true (``noul`` answers)."""
        value = self.data.get("noul")
        if value is None:
            probabilities = self.data.get("probabilities")
            if isinstance(probabilities, Mapping):
                for candidate in ("true", "True", True):
                    if candidate in probabilities:
                        value = probabilities[candidate]
                        break
        return None if value is None else float(value)

    @property
    def score(self) -> float | None:
        """Expected score on the level scale (``score`` answers)."""
        value = self.data.get("score")
        return None if value is None else float(value)

    @property
    def level(self) -> Any:
        """Most likely level (``score`` answers); `score` is the expected value."""
        if "level" in self.data:
            return self.data["level"]
        probabilities = self.data.get("probabilities")
        if isinstance(probabilities, Mapping) and probabilities:
            return max(probabilities.items(), key=lambda item: float(item[1]))[0]
        return None

    @property
    def probabilities(self) -> Mapping[Any, float]:
        """The full distribution: ``{option: probability}``."""
        probabilities = self.data.get("probabilities") or {}
        return {key: float(value) for key, value in dict(probabilities).items()}

    @property
    def confidence(self) -> float | None:
        """The model's confidence in the chosen answer, when it reports one."""
        value = self.data.get("confidence")
        if value is not None:
            return float(value)
        probabilities = self.probabilities
        return max(probabilities.values()) if probabilities else None

    @property
    def legend(self) -> Mapping[str, str]:
        """Level names for a ``score`` answer, when the model reports them."""
        return dict(self.data.get("legend") or {})


@dataclass(frozen=True)
class Decision:
    """The answers to one System One request, keyed by question name."""

    answers: Mapping[str, Answer]
    model: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def __contains__(self, key: str) -> bool:
        return key in self.answers

    def keys(self) -> Sequence[str]:
        return tuple(self.answers)

    def to_dict(self) -> dict[str, Any]:
        """The raw System One answer payload (JSON-serializable)."""
        return dict(self.raw) if self.raw else {k: dict(a.data) for k, a in self.answers.items()}


@dataclass(frozen=True)
class Classification:
    """One-label classification: best label + calibrated confidence.

    Informative, not prescriptive (same contract as ``laya-router``): the best
    label always comes back, plus ``below_threshold`` so the caller decides
    what to do with a weak answer (ask again, show a menu, fall back).
    """

    label: str
    confidence: float
    below_threshold: bool
    scores: Mapping[str, float]

    def ranked(self, top_k: int | None = None) -> list[tuple[str, float]]:
        """Labels sorted by probability (descending)."""
        ordered = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        return ordered[:top_k] if top_k else ordered


def wrap_answers(answers: Mapping[str, Mapping[str, Any]], questions: Mapping[str, Mapping[str, Any]]
                 ) -> dict[str, Answer]:
    """Build :class:`Answer` objects for every question in the request."""
    wrapped: dict[str, Answer] = {}
    for key, data in answers.items():
        question = questions.get(key) or {}
        qtype = str(data.get("type") or question.get("type") or "unknown")
        wrapped[key] = Answer(key=key, type=qtype, data=dict(data))
    return wrapped


__all__ = ["Answer", "Decision", "Classification", "wrap_answers"]
