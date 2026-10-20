"""`decision_toolkit.adapters` — the adapter registry.

An adapter is written **only** when a real `.onnx` export demands it, and it is
named after the model contract it implements (not after a brand or a runtime).
Adding a model with an existing contract means adding a catalog variant, not an
adapter.

Third-party adapters can be registered in-process with
:func:`register_adapter` or discovered through the ``decision_toolkit.adapters``
entry-point group.  A custom adapter may declare ``required_roles`` (an
iterable of file roles) so the core can validate a local/custom model before
loading it.
"""

from __future__ import annotations

import logging
from typing import Any

from ..errors import AdapterNotAvailableError, AdapterNotImplementedError
from .base import Adapter
from .gliner2 import Gliner2Adapter
from .gliner2_decide import Gliner2DecideAdapter
from .julia import JuliaAdapter
from .laya import LayaAdapter
from .opendecider import OpendeciderAdapter

logger = logging.getLogger(__name__)

ADAPTERS: dict[str, Any] = {
    "gliner2": Gliner2Adapter,
    "gliner2-decide": Gliner2DecideAdapter,
    "julia": JuliaAdapter,
    "laya": LayaAdapter,
    "opendecider": OpendeciderAdapter,
}

#: Fixed role requirements for the adapters that ship with the toolkit.
#: ``gliner2`` has an alternative layout (one fused ``model`` or split
#: ``encoder`` + ``classifier``), handled separately below.
_REQUIRED_ROLES: dict[str, frozenset[str]] = {
    "gliner2-decide": frozenset({"model", "tokenizer"}),
    "julia": frozenset({"model", "tokenizer"}),
    "laya": frozenset({"encoder", "head", "tokenizer", "config"}),
    "opendecider": frozenset({"model", "tokenizer", "config"}),
}

_ENTRY_POINTS_LOADED = False


def register_adapter(name: str, factory: Any, *, replace: bool = False) -> None:
    """Register an adapter factory under ``name``.

    ``factory`` is called with the adapter options as keyword arguments and must
    return an object implementing :class:`~decision_toolkit.adapters.base.Adapter`.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("adapter name must be a non-empty string")
    if not callable(factory):
        raise TypeError(f"adapter factory for {name!r} must be callable")
    if name in ADAPTERS and not replace:
        raise AdapterNotAvailableError(
            f"adapter {name!r} is already registered; pass replace=True to override it"
        )
    ADAPTERS[name] = factory
    logger.debug("registered adapter %s -> %r", name, factory)


def unregister_adapter(name: str) -> None:
    """Remove a registered adapter (mostly useful in tests)."""
    ADAPTERS.pop(name, None)


def _load_entry_points() -> None:
    """Discover third-party adapters once, lazily, without breaking imports."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.10
        return
    try:
        discovered = entry_points(group="decision_toolkit.adapters")
    except Exception:  # noqa: BLE001 - defensive for odd metadata backends  # pragma: no cover
        return
    for entry_point in discovered:
        try:
            register_adapter(entry_point.name, entry_point.load())
        except Exception as error:  # noqa: BLE001 - a bad plugin must not break the toolkit
            logger.warning(
                "could not load adapter entry point %s: %s", entry_point.name, error
            )


def create_adapter(name: str, **options: Any) -> Adapter:
    """Instantiate the adapter ``name``.

    Raises :class:`AdapterNotImplementedError` when the adapter is catalogued but
    not written yet (no real export has demanded it).
    """
    _load_entry_points()
    adapter_type = ADAPTERS.get(name)
    if adapter_type is None:
        raise AdapterNotImplementedError(
            f"adapter {name!r} is not implemented yet; it is written when its real ONNX export "
            "is inspected (no speculative adapters)"
        )
    return adapter_type(**options)


def required_roles(name: str, files: Any = None) -> frozenset[str]:
    """The file roles an adapter needs, for early custom-model validation."""
    factory = ADAPTERS.get(name)
    declared = getattr(factory, "required_roles", None)
    if declared is not None:
        return frozenset(str(role) for role in declared)
    return _REQUIRED_ROLES.get(name, frozenset())


def validate_adapter_files(name: str, files: Any) -> None:
    """Validate a role -> path mapping against the adapter's contract.

    Only role presence and basic shape are checked here; the engine later checks
    that each path exists and opens.  For ``gliner2`` either a fused ``model``
    file or the split ``encoder`` + ``classifier`` pair is accepted.
    """
    if not isinstance(files, dict) or not files:
        raise AdapterNotAvailableError(
            f"adapter {name!r} needs a role -> path mapping (files=...)"
        )
    for role, path in files.items():
        if not isinstance(role, str) or not role.strip():
            raise AdapterNotAvailableError(f"adapter {name!r}: file roles must be non-empty strings")
        if not isinstance(path, str) or not path.strip():
            raise AdapterNotAvailableError(f"adapter {name!r}: file path for {role!r} is empty")

    roles = set(files)
    if name == "gliner2":
        if "tokenizer" not in roles:
            raise AdapterNotAvailableError("adapter 'gliner2' needs a 'tokenizer' file")
        if "model" not in roles and not {"encoder", "classifier"} <= roles:
            raise AdapterNotAvailableError(
                "adapter 'gliner2' needs either a fused 'model' file or 'encoder' + "
                "'classifier' files"
            )
        return

    missing = required_roles(name, files) - roles
    if missing:
        raise AdapterNotAvailableError(
            f"adapter {name!r} needs file roles {sorted(missing)}; got {sorted(roles)}"
        )


__all__ = [
    "ADAPTERS",
    "Adapter",
    "create_adapter",
    "register_adapter",
    "required_roles",
    "unregister_adapter",
    "validate_adapter_files",
]
