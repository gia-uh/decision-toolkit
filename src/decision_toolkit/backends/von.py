"""`decision_toolkit.backends.von` — wfzyx's Von, in-process on CPU.

Von (395M, ModernBERT-large) is a non-autoregressive System One model that
summary-ships as ``von-sdk``: ``von.system_one(state=..., questions={...})``
returns rich answer objects. Its repository was archived on 2026-10-17, so the
weights and SDK are frozen — a stable, if unsupported, dependency.

The constructor accepts an injected ``sdk`` so tests never import the runtime:

>>> class FakeSdk:
...     def system_one(self, state, questions): return {"answers": {"q": {"type": "noul", "noul": 0.9}}}
>>> backend = VonBackend(resolved, sdk=FakeSdk())
"""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import BackendError, BackendNotAvailableError
from ..models import ResolvedModel
from .base import BaseBackend


def _read(source: Any, name: str) -> Any:
    """Read ``name`` from an object attribute or a mapping key."""
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


class VonBackend(BaseBackend):
    """System One over the ``von-sdk`` runtime (pip install "decision-toolkit[von]")."""

    name = "von"

    def __init__(
        self,
        resolved: ResolvedModel,
        *,
        device: str | None = None,
        sdk: Any | None = None,
        **options: Any,
    ) -> None:
        super().__init__(resolved, device=device, **options)
        self._sdk = sdk

    # ------------------------------------------------------------------ load
    def _load(self) -> Any:
        if self._sdk is None:
            try:
                import von  # type: ignore[import-not-found]
            except ImportError as error:  # pragma: no cover - depends on environment
                raise BackendNotAvailableError(
                    "the Von SDK is not installed: pip install \"decision-toolkit[von]\""
                ) from error
            if not hasattr(von, "system_one"):
                raise BackendNotAvailableError(
                    "`import von` resolved to an unexpected module; install von-sdk "
                    "(pip install \"decision-toolkit[von]\")"
                )
            self._sdk = von
        return self._sdk

    def warmup(self) -> None:
        self._load()

    # ------------------------------------------------------------- inference
    def system_one(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        sdk = self._load()
        response = sdk.system_one(state=state, questions=dict(questions))
        answers_source = _read(response, "answers")
        if answers_source is None:
            raise BackendError(f"unexpected Von response shape: {type(response).__name__}")

        answers: dict[str, dict[str, Any]] = {}
        for name, question in questions.items():
            source = answers_source if isinstance(answers_source, Mapping) else answers_source
            entry = _read(source, name)
            if entry is None:
                continue
            qtype = str(question.get("type"))
            answer: dict[str, Any] = {"type": qtype}
            if qtype == "choice":
                answer["choice"] = _read(entry, "choice")
                probabilities = _read(entry, "probabilities")
                if probabilities is not None:
                    answer["probabilities"] = {k: float(v) for k, v in dict(probabilities).items()}
                confidence = _read(entry, "confidence")
                if confidence is not None:
                    answer["confidence"] = float(confidence)
            elif qtype == "noul":
                value = _read(entry, "noul")
                if value is None:
                    value = _read(entry, "p_true")
                answer["noul"] = float(value) if value is not None else None
            elif qtype == "score":
                value = _read(entry, "score")
                if value is not None:
                    answer["score"] = float(value)
                level = _read(entry, "level")
                if level is not None:
                    answer["level"] = level
                legend = _read(entry, "legend")
                if legend is not None:
                    answer["legend"] = dict(legend)
                probabilities = _read(entry, "probabilities")
                if probabilities is not None:
                    answer["probabilities"] = {k: float(v) for k, v in dict(probabilities).items()}
            answers[name] = answer
        return answers


__all__ = ["VonBackend"]
