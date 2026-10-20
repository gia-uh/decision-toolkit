"""Shared wire envelopes for the REST projections.

These models are deliberately **not** part of the core. They are Pydantic
models used only by the ``[server]`` extra, so the library remains usable
without FastAPI/Pydantic installed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """OpenAI-compatible error detail (also used for System One errors)."""

    model_config = ConfigDict(extra="forbid")

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """Standard error envelope: ``{"error": {...}}``."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


class WireUsage(BaseModel):
    """The common token-usage fields exposed by both projections."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


def error_payload(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly error envelope without instantiating the handler."""
    return ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code,
        )
    ).model_dump()


__all__ = ["ErrorDetail", "ErrorResponse", "WireUsage", "error_payload"]
