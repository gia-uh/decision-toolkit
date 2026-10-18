"""`decision_toolkit.backends` — the inference seams, one per model family.

A backend's whole contract is:

* give it the resolved model (a local path and/or a hub reference) and its
  options;
* call :meth:`system_one` with a state and canonical System One questions;
* get back the canonical System One ``answers`` mapping.

Production backends import their heavy runtimes lazily (``import laya`` only
when a real decision is requested), and every constructor accepts an injected
runtime object so the test suite runs without models, network or GPUs.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..catalog import IMPLEMENTED_BACKENDS
from ..errors import BackendNotAvailableError
from ..models import ResolvedModel
from .base import Backend, BaseBackend, state_to_text
from .julia import JuliaBackend
from .laya import LayaBackend
from .systemone import SystemOneBackend
from .von import VonBackend

_REGISTRY: dict[str, type[BaseBackend]] = {
    "laya": LayaBackend,
    "julia": JuliaBackend,
    "von": VonBackend,
    "systemone": SystemOneBackend,
}

_MILESTONES = {
    "gguf": "GGUF/llama.cpp decoding (M1)",
    "onnx": "ONNX Runtime encoding (M1)",
    "gliner2": "the gliner2 runtime (M1)",
    "transformers": "a Transformers runtime (M1)",
}


def create_backend(
    name: str,
    resolved: ResolvedModel,
    *,
    device: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> BaseBackend:
    """Instantiate the backend ``name`` for a resolved model.

    Raises :class:`BackendNotAvailableError` for catalogued-but-not-yet-runnable
    backends, with the milestone that will bring them and the model's notes.
    """
    options = dict(options or {})
    if name == "systemone":
        # server-based backend: the model reference is the server
        base_url = options.pop("base_url", None)
        if not base_url:
            raise BackendNotAvailableError(
                "the 'systemone' backend talks to a running server; pass "
                "backend_options={'base_url': 'http://host:port'}"
            )
        return SystemOneBackend(resolved, base_url=base_url, **options)

    backend_type = _REGISTRY.get(name)
    if backend_type is None:
        milestone = _MILESTONES.get(name, "a later milestone")
        notes = resolved.catalog.notes if resolved.catalog else ""
        detail = f" ({notes})" if notes else ""
        raise BackendNotAvailableError(
            f"backend {name!r} is catalogued but not runnable yet — planned with {milestone}.{detail}"
        )
    return backend_type(resolved, device=device, **options)


__all__ = [
    "Backend",
    "BaseBackend",
    "JuliaBackend",
    "LayaBackend",
    "SystemOneBackend",
    "VonBackend",
    "create_backend",
    "IMPLEMENTED_BACKENDS",
    "state_to_text",
]
