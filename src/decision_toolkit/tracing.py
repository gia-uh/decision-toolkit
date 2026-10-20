"""Optional OpenTelemetry tracing.

OpenTelemetry is deliberately optional: the library and local server must work
without any tracing package. When enabled and installed, the server creates
spans around HTTP requests, decisions and model loads. Exporter/endpoint setup
belongs to the application, not to decision-toolkit.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


def configure_tracing(enabled: bool = False) -> Any | None:
    """Return an OTel tracer, or ``None`` when disabled/unavailable."""
    if not enabled:
        return None
    try:
        from opentelemetry import trace
    except ImportError:
        logger.warning(
            "tracing requested but opentelemetry-api is not installed; "
            "install decision-toolkit[tracing]"
        )
        return None
    return trace.get_tracer("decision_toolkit")


@contextmanager
def tracing_span(tracer: Any | None, name: str, **attributes: Any) -> Iterator[Any | None]:
    """Start a span when a tracer exists; otherwise yield ``None``."""
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


__all__ = ["configure_tracing", "tracing_span"]
