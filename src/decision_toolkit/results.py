"""`decision_toolkit.results` — typed views over System One answers.

Adapters return raw System One answer dictionaries; these dataclasses give
Python callers (and the CLI/REST layers) a stable, typed way to read them
without touching vendor-specific key spellings.

The core keeps a deliberately small, semantic vocabulary:

* :class:`Usage` — token accounting for one decision.
* :class:`Answer` — one named answer plus its confidence, warnings and
  truncation flag.
* :class:`Decision` — all answers for one state, including the model identity,
  latency, usage and request id.
* :class:`Classification` — the one-label convenience view.

Raw answers remain available (``to_dict`` / ``Decision.raw``) so existing
callers and adapters are not forced through the typed layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Usage:
    """Token accounting for one decision.

    ``estimated`` is ``True`` when the core counted tokens with a tokenizer but
    could not reproduce the exact model prompt, or when no tokenizer was
    available and a whitespace estimate was used.  It is deliberately explicit:
    the toolkit never presents an estimate as exact.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.total_tokens == 0:
            object.__setattr__(self, "total_tokens", self.input_tokens + self.output_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
        }


def _max_probability(probabilities: Mapping[Any, Any] | None) -> float | None:
    if not probabilities:
        return None
    try:
        return max(float(value) for value in probabilities.values())
    except (TypeError, ValueError):
        return None


def _confidence_from_data(data: Mapping[str, Any], qtype: str) -> float | None:
    value = data.get("confidence")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    probabilities = data.get("probabilities")
    if isinstance(probabilities, Mapping) and probabilities:
        return _max_probability(probabilities)

    if qtype == "noul":
        raw = data.get("noul", data.get("p_true"))
        if raw is not None:
            try:
                probability_true = float(raw)
                return max(probability_true, 1.0 - probability_true)
            except (TypeError, ValueError):
                return None
    return None


@dataclass(frozen=True)
class Answer:
    """One named answer with the probabilities the model placed on its options.

    ``type`` and ``data`` are the raw System One view kept for compatibility.
    ``kind``, ``value``, ``confidence_kind``, ``truncated`` and ``warnings`` are
    the typed, cross-model view.
    """

    key: str
    type: str
    data: Mapping[str, Any] = field(repr=False)

    @property
    def kind(self) -> str:
        """The answer type (``choice``, ``noul``, ``score``, ``box``, ``refusal``)."""
        return self.type

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
    def value(self) -> Any:
        """The canonical value for this answer kind (choice / noul / level / box)."""
        if self.type == "choice":
            return self.choice
        if self.type == "noul":
            return self.noul
        if self.type == "score":
            level = self.level
            return level if level is not None else self.score
        if self.type == "box":
            return self.data.get("box", self.data.get("value"))
        return None

    @property
    def probabilities(self) -> Mapping[Any, float]:
        """The full distribution: ``{option: probability}``."""
        probabilities = self.data.get("probabilities") or {}
        return {key: float(value) for key, value in dict(probabilities).items()}

    @property
    def confidence(self) -> float | None:
        """Probability of the chosen answer (normalized across adapters)."""
        return _confidence_from_data(self.data, self.type)

    @property
    def confidence_kind(self) -> str | None:
        """How ``confidence`` should be interpreted; currently ``probability``."""
        kind = self.data.get("confidence_kind")
        if kind is not None:
            return str(kind)
        return "probability" if self.confidence is not None else None

    @property
    def truncated(self) -> bool:
        """Whether the adapter had to truncate input for this answer."""
        return bool(self.data.get("truncated", False))

    @property
    def warnings(self) -> tuple[str, ...]:
        """Human-readable warnings attached by the adapter."""
        values = self.data.get("warnings") or ()
        if isinstance(values, str):
            return (values,)
        return tuple(str(value) for value in values)

    @property
    def action(self) -> Mapping[str, Any] | None:
        """Optional adapter action metadata (e.g. Laya's act probability)."""
        value = self.data.get("action")
        return dict(value) if isinstance(value, Mapping) else None

    @property
    def refusal(self) -> bool:
        """Whether this answer is a refusal (no usable decision was produced)."""
        return self.type == "refusal" or bool(self.data.get("refusal", False))

    @property
    def legend(self) -> Mapping[str, str]:
        """Level names for a ``score`` answer, when the model reports them."""
        return dict(self.data.get("legend") or {})

    def to_dict(self) -> dict[str, Any]:
        """The raw answer as a JSON-friendly dictionary."""
        payload = dict(self.data)
        if "warnings" in payload:
            payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True)
class Decision:
    """The answers to one System One request, keyed by question name."""

    answers: Mapping[str, Answer]
    model: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)
    variant: str | None = None
    adapter: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float | None = None
    warnings: tuple[str, ...] = ()
    request_id: str | None = None
    id: str | None = None

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def __contains__(self, key: str) -> bool:
        return key in self.answers

    def keys(self) -> Sequence[str]:
        return tuple(self.answers)

    @property
    def truncated(self) -> bool:
        """Whether any answer had to truncate its input."""
        return any(answer.truncated for answer in self.answers.values())

    @property
    def refused(self) -> bool:
        """Whether any answer is a refusal."""
        return any(answer.refusal for answer in self.answers.values())

    def answer_payload(self) -> dict[str, Any]:
        """Answers in their raw System One form (or typed form if no raw exists)."""
        if self.raw:
            return dict(self.raw)
        return {key: answer.to_dict() for key, answer in self.answers.items()}

    def to_dict(self) -> dict[str, Any]:
        """The raw System One answer payload (JSON-serializable).

        Kept as the compatibility method used by the CLI and REST layers.  Use
        :meth:`to_payload` when the caller also needs usage/latency metadata.
        """
        return self.answer_payload()

    def to_payload(self) -> dict[str, Any]:
        """The full decision payload: answers + identity, usage and latency."""
        payload: dict[str, Any] = {
            "id": self.id or self.request_id,
            "model": self.model,
            "variant": self.variant,
            "adapter": self.adapter,
            "answers": self.answer_payload(),
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "truncated": self.truncated,
            "warnings": list(self.warnings),
        }
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        return payload


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
    model: str | None = None
    variant: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float | None = None
    warnings: tuple[str, ...] = ()
    request_id: str | None = None

    def ranked(self, top_k: int | None = None) -> list[tuple[str, float]]:
        """Labels sorted by probability (descending)."""
        ordered = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        return ordered[:top_k] if top_k else ordered

    def to_dict(self, top_k: int | None = None) -> dict[str, Any]:
        """JSON-friendly view, optionally including the top-k ranked labels."""
        payload: dict[str, Any] = {
            "label": self.label,
            "confidence": self.confidence,
            "below_threshold": self.below_threshold,
            "scores": dict(self.scores),
            "model": self.model,
            "variant": self.variant,
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "warnings": list(self.warnings),
            "request_id": self.request_id,
        }
        if top_k is not None:
            payload["ranked"] = self.ranked(top_k)
        return payload


def wrap_answers(
    answers: Mapping[str, Mapping[str, Any]],
    questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Answer]:
    """Build :class:`Answer` objects for every raw answer in the response.

    Confidence is normalized here, once, rather than in every adapter: choice
    and score answers use the probability of the chosen option; noul answers use
    ``max(p_true, p_false)``.  The raw dictionaries are normalized too, so the
    CLI/REST payloads expose the same semantics as the typed views.
    """
    wrapped: dict[str, Answer] = {}
    for key, raw in answers.items():
        data = dict(raw)
        question = questions.get(key) or {}
        qtype = str(data.get("type") or question.get("type") or "unknown")
        if data.get("confidence") is None:
            confidence = _confidence_from_data(data, qtype)
            if confidence is not None:
                data["confidence"] = confidence
        if data.get("confidence_kind") is None and data.get("confidence") is not None:
            data["confidence_kind"] = "probability"
        wrapped[key] = Answer(key=key, type=qtype, data=data)
    return wrapped


__all__ = [
    "Answer",
    "Classification",
    "Decision",
    "Usage",
    "wrap_answers",
]
