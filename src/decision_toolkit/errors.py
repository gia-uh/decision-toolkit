"""`decision_toolkit.errors` — the toolkit's error hierarchy.

Every error raised on purpose by the toolkit derives from
:class:`DecisionToolkitError`, so applications can catch one type and the CLI
can turn any of them into a clean exit code + message.
"""

from __future__ import annotations


class DecisionToolkitError(Exception):
    """Base class for every error the toolkit raises on purpose."""


class ModelNotFoundError(DecisionToolkitError):
    """An identifier is neither in the catalog, nor the cache, nor a local path."""


class AdapterNotAvailableError(DecisionToolkitError):
    """The adapter for a model is not implemented yet, or its runtime is missing."""


class InvalidQuestionError(DecisionToolkitError, ValueError):
    """A question violates the System One contract (type, criteria, limits...)."""


class AdapterError(DecisionToolkitError):
    """The inference adapter failed (server error, malformed answer...)."""


#: Backwards-compatible aliases (the first scaffold used "backend" names).
BackendNotAvailableError = AdapterNotAvailableError
BackendError = AdapterError


__all__ = [
    "DecisionToolkitError",
    "ModelNotFoundError",
    "AdapterNotAvailableError",
    "AdapterError",
    "InvalidQuestionError",
    "BackendNotAvailableError",
    "BackendError",
]
