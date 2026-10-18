"""`decision_toolkit.backends.laya` — Convai's Laya, in-process on CPU.

Laya is a non-autoregressive System One encoder (ModernBERT-large): one forward
pass answers ``choice`` / ``noul`` / ``score`` questions and returns calibrated
probabilities. The official ``laya`` package owns the checkpoint download and
the device placement; this backend only wires it into the toolkit.

The constructor accepts an injected ``agent`` so tests never import the real
runtime:

>>> class FakeAgent:
...     def predict(self, state, questions): return {"answers": {"q": {"type": "noul", "noul": 0.9}}}
>>> backend = LayaBackend(resolved, agent=FakeAgent())
"""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import BackendError, BackendNotAvailableError
from ..models import ResolvedModel
from .base import BaseBackend, state_to_text


class LayaBackend(BaseBackend):
    """System One over the ``laya`` runtime (pip install "decision-toolkit[laya]")."""

    name = "laya"

    def __init__(
        self,
        resolved: ResolvedModel,
        *,
        device: str | None = None,
        agent: Any | None = None,
        subfolder: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(resolved, device=device, **options)
        self._subfolder = subfolder if subfolder is not None else resolved.subfolder
        self._agent = agent
        self._fast = bool(options.get("fast", False))

    # ------------------------------------------------------------------ load
    def _load(self) -> Any:
        if self._agent is None:
            try:
                import laya
            except ImportError as error:  # pragma: no cover - depends on environment
                raise BackendNotAvailableError(
                    "the Laya runtime is not installed: pip install \"decision-toolkit[laya]\""
                ) from error

            target = str(self._resolved.path) if self._resolved.path else self._resolved.repo
            if target is None:  # pragma: no cover - resolve() enforces this
                raise BackendError("Laya needs a local path or a hub repo; nothing was resolved")
            device = None if self._device in (None, "auto") else self._device
            self._agent = laya.load(
                target,
                device=device,
                subfolder=self._subfolder,
                fast=self._fast,
            )
        return self._agent

    def warmup(self) -> None:
        self._load()

    # ------------------------------------------------------------- inference
    def system_one(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        agent = self._load()
        result = agent.predict(state_to_text(state), dict(questions))
        if not isinstance(result, Mapping) or "answers" not in result:
            raise BackendError(f"unexpected Laya response shape: {type(result).__name__}")
        return dict(result["answers"])


__all__ = ["LayaBackend"]
