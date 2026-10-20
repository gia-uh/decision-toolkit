"""`decision_toolkit.observability` — request correlation and structured events.

The library **emits** events; it never configures logging handlers.  A CLI, a
REST server, or an application decides where those events go and in which
format.  This keeps the toolkit usable as a library and gives every surface the
same event vocabulary.

Every event carries a ``request_id`` when one is available.  Callers can bind a
context around a unit of work (``with request_context("req-123"): ...``) or pass
``request_id=`` explicitly to ``DecisionModel.decide``.  Events deliberately do
not include state or question text by default: only counts, model identity,
timings, token counts and error codes.  Logging user content is an explicit
decision for the application, not a library default.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("decision_toolkit_request_id", default=None)


def new_request_id() -> str:
    """Generate an opaque request id (32 hex characters, no PII)."""
    return uuid.uuid4().hex


def get_request_id() -> str | None:
    """The request id bound to the current context, if any."""
    return _request_id.get()


def ensure_request_id(request_id: str | None = None) -> str:
    """Return ``request_id``, the context one, or a freshly generated one."""
    if request_id:
        return str(request_id)
    current = get_request_id()
    return current if current else new_request_id()


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind ``request_id`` to the current context and restore it afterwards."""
    value = ensure_request_id(request_id)
    token = _request_id.set(value)
    try:
        yield value
    finally:
        _request_id.reset(token)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one structured event through the standard logging hierarchy.

    The message is a compact JSON object so it is useful with the default
    formatter and trivially parseable by a configured JSON formatter later.
    ``None`` values are omitted to keep lines stable.
    """
    if not logger.isEnabledFor(level):
        return
    payload: dict[str, Any] = {"event": event, "request_id": get_request_id()}
    payload.update({key: value for key, value in fields.items() if value is not None})
    logger.log(level, "%s", json.dumps(payload, default=str, sort_keys=True, ensure_ascii=False))


__all__ = [
    "ensure_request_id",
    "get_request_id",
    "log_event",
    "new_request_id",
    "request_context",
]
