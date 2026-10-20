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
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("decision_toolkit_request_id", default=None)


class JsonLogFormatter(logging.Formatter):
    """Render each record as one JSON object.

    ``log_event`` already emits a JSON message; it is merged into the envelope
    instead of being nested as an escaped string.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
        }
        message = record.getMessage()
        try:
            parsed = json.loads(message)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            payload.update(parsed)
        else:
            payload["message"] = message
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)


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


def configure_logging(
    level: str | int = "INFO",
    log_format: str = "text",
    *,
    logger_name: str = "decision_toolkit",
) -> logging.Logger:
    """Configure a stderr handler for the toolkit logger (server/CLI use).

    The library itself never calls this. ``log_format="json"`` renders the
    structured events as one JSON object per line.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = (
        JsonLogFormatter()
        if log_format == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handler = next(
        (candidate for candidate in logger.handlers if getattr(candidate, "_dtk_handler", False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler._dtk_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    handler.setFormatter(formatter)
    logger.propagate = False
    return logger


__all__ = [
    "JsonLogFormatter",
    "configure_logging",
    "ensure_request_id",
    "get_request_id",
    "log_event",
    "new_request_id",
    "request_context",
]
