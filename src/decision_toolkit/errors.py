"""`decision_toolkit.errors` — the toolkit's error hierarchy.

Every error raised on purpose by the toolkit derives from
:class:`DecisionToolkitError`, so applications can catch one type, the CLI can
turn any of them into a clean exit code + message, and the REST layer can map
them to a stable HTTP contract without guessing from strings.

Each error exposes a stable ``code`` (snake_case, wire-friendly) and a
``to_dict()`` envelope.  The hierarchy also separates the failures callers
usually need to tell apart: a model that is not downloaded (recoverable by
downloading it), an adapter that is not implemented, a capability the selected
model does not have, a truncated input, and a backend/runtime failure.
"""

from __future__ import annotations

from typing import Any


class DecisionToolkitError(Exception):
    """Base class for every error the toolkit raises on purpose."""

    code = "decision_toolkit_error"

    def to_dict(self) -> dict[str, Any]:
        """Stable, JSON-friendly error envelope."""
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": str(self),
        }


class ModelNotFoundError(DecisionToolkitError):
    """An identifier is neither in the catalog, nor the cache, nor a local path."""

    code = "model_not_found"


class AdapterNotAvailableError(DecisionToolkitError):
    """The adapter for a model is not available, or its runtime is missing."""

    code = "adapter_not_available"


class ModelNotDownloadedError(AdapterNotAvailableError):
    """A catalog model is known, but its selected variant is not in the cache."""

    code = "model_not_downloaded"


class AdapterNotImplementedError(AdapterNotAvailableError):
    """A catalogued adapter has not been written yet (no real export demands it)."""

    code = "adapter_not_implemented"


class InvalidQuestionError(DecisionToolkitError, ValueError):
    """A question violates the System One contract (type, criteria, limits...)."""

    code = "invalid_question"


class InvalidRequestError(DecisionToolkitError, ValueError):
    """A request is malformed outside the question itself (state, options...)."""

    code = "invalid_request"


class UnsupportedCapabilityError(DecisionToolkitError):
    """The selected model does not support a requested question type or input."""

    code = "unsupported_capability"


class MultimodalNotSupportedError(UnsupportedCapabilityError):
    """The selected model is text-only; image/audio input is not available."""

    code = "multimodal_not_supported"


class TruncationError(DecisionToolkitError):
    """Strict mode was requested and the model had to truncate its input."""

    code = "truncated_input"


class AdapterError(DecisionToolkitError):
    """The inference adapter failed (server error, malformed answer...)."""

    code = "adapter_error"


class BackendFailureError(AdapterError):
    """The runtime (ONNX Runtime, tokenizer...) failed while executing."""

    code = "backend_failure"


class ServerBusyError(DecisionToolkitError):
    """The server concurrency gate is full; the caller should retry later."""

    code = "server_busy"

    def __init__(self, message: str = "server is at capacity", retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = max(0.0, float(retry_after))


class VerificationError(DecisionToolkitError):
    """A cached model failed verification (missing files, bad hashes, graph...)."""

    code = "verification_failed"


#: Backwards-compatible aliases (the first scaffold used "backend" names).
BackendNotAvailableError = AdapterNotAvailableError
BackendError = AdapterError


__all__ = [
    "AdapterError",
    "AdapterNotAvailableError",
    "AdapterNotImplementedError",
    "BackendError",
    "BackendFailureError",
    "BackendNotAvailableError",
    "DecisionToolkitError",
    "InvalidQuestionError",
    "InvalidRequestError",
    "ModelNotDownloadedError",
    "ModelNotFoundError",
    "MultimodalNotSupportedError",
    "ServerBusyError",
    "TruncationError",
    "UnsupportedCapabilityError",
    "VerificationError",
]
