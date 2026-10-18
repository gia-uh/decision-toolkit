"""`decision_toolkit.backends.base` — the backend protocol and shared helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, runtime_checkable

from ..errors import BackendError
from ..models import ResolvedModel


@runtime_checkable
class Backend(Protocol):
    """What :class:`~decision_toolkit.decision_model.DecisionModel` needs."""

    name: str

    def system_one(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]: ...


class BaseBackend:
    """Common plumbing: lazy loading, option handling, no-op lifecycle hooks."""

    name = "base"

    def __init__(self, resolved: ResolvedModel, *, device: str | None = None, **options: Any) -> None:
        self._resolved = resolved
        self._device = device
        self._options = dict(options)

    def system_one(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        raise NotImplementedError

    def warmup(self) -> None:  # pragma: no cover - overridden when meaningful
        """Load the model ahead of the first request (optional)."""

    def close(self) -> None:  # pragma: no cover - overridden when meaningful
        """Release the runtime (optional)."""


def state_to_text(state: Any) -> str:
    """Render a System One ``state`` as text for backends that take strings."""
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping) or isinstance(state, (list, tuple)):
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    if state is None:
        raise BackendError("state cannot be None")
    return str(state)


__all__ = ["Backend", "BaseBackend", "state_to_text"]
