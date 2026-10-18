"""`decision_toolkit.decision_model` — the public, library-first surface.

``DecisionModel`` resolves an identifier (catalog id, cache entry, local path or
``org/name`` repo), picks the right backend, and answers System One requests:
typed questions in, calibrated probabilities out, on CPU.

The CLI and the REST server are thin adapters over this class; nothing here
knows about argparse or FastAPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backends import Backend, create_backend
from .errors import BackendNotAvailableError
from .models import ModelStore, ResolvedModel
from .results import Answer, Classification, Decision, wrap_answers
from .schema import Choice, normalize_questions

logger = logging.getLogger(__name__)


class DecisionModel:
    """A local decision model, loaded lazily on first use.

    Parameters
    ----------
    identifier:
        A catalog id (``laya``, ``julia-1``, ``von``...), a model in the
        toolkit cache, a local directory, or a hub repo (``org/name``). When
        omitted, the model in use from ``config.json`` is taken.
    backend:
        Backend name (``laya``, ``julia``, ``von``, ``systemone``) or a
        ready-made backend instance (useful for tests and custom runtimes).
    backend_options:
        Extra keyword arguments for the backend (e.g. ``{"subfolder": ...}``,
        ``{"base_url": "http://localhost:8000"}`` for the ``systemone``
        backend).
    home:
        Toolkit home override (models cache + config).
    device:
        ``auto`` (default), ``cpu``, ``cuda``, ``mps``... Backends interpret it.
    """

    def __init__(
        self,
        identifier: str | None = None,
        *,
        backend: Backend | str | None = None,
        backend_options: Mapping[str, Any] | None = None,
        home: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self._identifier = identifier
        self._store = ModelStore(home=home)
        self._device = device or self._store.config.device
        self._backend: Backend | None = backend if backend is not None and not isinstance(backend, str) else None
        self._backend_name: str | None = backend if isinstance(backend, str) else None
        self._backend_options = dict(backend_options or {})
        self._resolved: ResolvedModel | None = None

    # ------------------------------------------------------------------ info
    @property
    def identifier(self) -> str:
        return self._identifier or self._store.current() or "(none)"

    @property
    def backend_name(self) -> str:
        if self._backend is not None:
            return getattr(self._backend, "name", type(self._backend).__name__)
        if self._backend_name:
            return self._backend_name
        resolved = self._resolve()
        if resolved.catalog is not None:
            return resolved.catalog.backend
        raise BackendNotAvailableError(
            f"cannot infer a backend for {self.identifier!r}; pass backend=<name>"
        )

    @property
    def resolved(self) -> ResolvedModel:
        return self._resolve()

    # ------------------------------------------------------------- lifecycle
    def _resolve(self) -> ResolvedModel:
        if self._resolved is None:
            self._resolved = self._store.resolve(self._identifier)
        return self._resolved

    def _ensure_backend(self) -> Backend:
        if self._backend is None:
            name = self._backend_name or self._infer_backend_name()
            if name == "systemone":
                # a server backend owns the model; no local resolution is needed
                resolved = ResolvedModel(
                    identifier=self._identifier or "(server)",
                    path=None,
                    repo=None,
                    subfolder=None,
                    catalog=None,
                    origin="path",
                )
            else:
                resolved = self._resolve()
                if name is None:
                    name = resolved.catalog.backend if resolved.catalog else None
            if name is None:
                raise BackendNotAvailableError(
                    f"cannot infer a backend for {self.identifier!r}; pass backend=<name>"
                )
            logger.info("loading model=%s backend=%s device=%s", self.identifier, name, self._device)
            self._backend = create_backend(
                name, resolved, device=self._device, options=self._backend_options
            )
        return self._backend

    def _infer_backend_name(self) -> str | None:
        if self._identifier is None:
            return None
        resolved = self._resolve()
        return resolved.catalog.backend if resolved.catalog else None

    def warmup(self) -> None:
        """Load the model (and its runtime) ahead of the first request."""
        backend = self._ensure_backend()
        warmup = getattr(backend, "warmup", None)
        if callable(warmup):
            warmup()

    def close(self) -> None:
        """Release the backend (and any server it owns)."""
        if self._backend is not None:
            close = getattr(self._backend, "close", None)
            if callable(close):
                close()
            self._backend = None

    # ------------------------------------------------------------ inference
    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
    ) -> Decision:
        """Answer typed questions about ``state`` (System One, in and out)."""
        normalized = normalize_questions(questions)
        backend = self._ensure_backend()
        raw_answers = dict(backend.system_one(state, normalized))
        answers: dict[str, Answer] = wrap_answers(raw_answers, normalized)
        return Decision(answers=answers, model=self.identifier, raw=raw_answers)

    def classify(
        self,
        text: str,
        categories: Mapping[str, str] | Sequence[str],
        *,
        instructions: str | None = None,
        threshold: float = 0.5,
    ) -> Classification:
        """Pick the best-fitting label for ``text`` with a calibrated confidence.

        ``categories`` is either ``{label: description}`` or a sequence of
        labels (the label doubles as its description). The call always returns
        the best label plus ``below_threshold`` — informative, not prescriptive.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("cannot classify an empty text")
        if isinstance(categories, Mapping):
            criteria = {str(label): str(description) for label, description in categories.items()}
        else:
            criteria = {str(label): str(label) for label in categories}
        if not criteria:
            raise ValueError("classify needs at least one category")

        question = Choice(
            instructions=instructions or "Which category fits this text best?",
            criteria=criteria,
        )
        decision = self.decide(text, {"label": question})
        answer = decision["label"]
        scores = {label: float(p) for label, p in answer.probabilities.items()}
        if not scores:
            chosen = answer.choice
            confidence = answer.confidence
            if chosen is None or confidence is None:
                raise ValueError(f"backend returned no probabilities for choice: {answer.data!r}")
            scores = {label: (confidence if label == chosen else 0.0) for label in criteria}

        label, confidence = max(scores.items(), key=lambda item: item[1])
        return Classification(
            label=label,
            confidence=confidence,
            below_threshold=confidence < threshold,
            scores=scores,
        )


__all__ = ["DecisionModel"]
