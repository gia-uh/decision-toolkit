"""`decision_toolkit.adapters` — the adapter registry.

An adapter is written **only** when a real `.onnx` export demands it, and it is
named after the model contract it implements (not after a brand or a runtime).
Adding a model with an existing contract means adding a catalog variant, not an
adapter.
"""

from __future__ import annotations

from typing import Any

from ..errors import AdapterNotAvailableError
from .base import Adapter
from .gliner2 import Gliner2Adapter

ADAPTERS: dict[str, type] = {
    "gliner2": Gliner2Adapter,
}


def create_adapter(name: str, **options: Any) -> Adapter:
    """Instantiate the adapter ``name``.

    Raises :class:`AdapterNotAvailableError` when the adapter is catalogued but
    not written yet (no real export has demanded it so far).
    """
    adapter_type = ADAPTERS.get(name)
    if adapter_type is None:
        raise AdapterNotAvailableError(
            f"adapter {name!r} is not implemented yet; it is written when its real ONNX export "
            "is inspected (no speculative adapters)"
        )
    return adapter_type(**options)


__all__ = ["Adapter", "ADAPTERS", "create_adapter"]
