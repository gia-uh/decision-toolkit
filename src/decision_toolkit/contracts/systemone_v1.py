"""System One wire contract, pinned to TypeSafe spec **1.14**.

The canonical reference used for this projection is the TypeSafe-compatible
System One API implemented by llama.cpp (``POST /v1/systemone``):

* request: ``state`` + ``questions`` (+ ``files``/``images``)
* questions: ``choice`` / ``noul`` / ``score`` (and the future ``box``)
* response: ``model`` + ``answers`` + ``usage``
* ``usage.output_tokens`` is always ``0`` (decisions generate no text)
* invalid requests return ``400``; unsupported image input returns ``501``

Toolkit-specific metadata lives under ``x_decision_toolkit`` so the standard
fields keep their meaning and shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SYSTEMONE_SPEC = "1.14"
MAX_QUESTIONS = 128


class SystemOneQuestion(BaseModel):
    """One typed System One question.

    ``instructions`` accepts a string, object or array in System One; the
    projection canonicalizes objects/arrays to JSON text before calling the
    core, whose adapters consume text.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["choice", "noul", "score", "box"]
    instructions: Any
    criteria: Any = None

    @model_validator(mode="after")
    def _validate(self) -> SystemOneQuestion:
        instructions = self.instructions
        if instructions is None:
            raise ValueError("instructions is required")
        if isinstance(instructions, str) and not instructions.strip():
            raise ValueError("instructions cannot be empty")
        if not isinstance(instructions, (str, dict, list)):
            raise ValueError(  # noqa: TRY004 - Pydantic validation, not a Python type error
                "instructions must be a string, object or array"
            )

        if self.type == "choice":
            criteria = self.criteria
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("a choice needs a non-empty criteria object")
            if len(criteria) > 255:
                raise ValueError("a choice supports at most 255 options")
            for label, description in criteria.items():
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("choice labels must be non-empty strings")
                if description is not None and not isinstance(description, str):
                    raise ValueError("choice descriptions must be strings or null")
        elif self.type == "noul":
            criteria = self.criteria
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise ValueError("noul criteria must be an object")
                unknown = set(criteria) - {"false", "true"}
                if unknown:
                    raise ValueError(
                        f"noul criteria keys must be 'false' and/or 'true', got {sorted(unknown)}"
                    )
        elif self.type == "score":
            criteria = self.criteria
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("a score needs an array of 2 to 10 levels")
            if any(not isinstance(level, str) or not level.strip() for level in criteria):
                raise ValueError("score levels must be non-empty strings")
        elif self.type == "box":
            criteria = self.criteria
            if criteria is not None:
                if not isinstance(criteria, list) or not criteria:
                    raise ValueError("box criteria must be a non-empty array of labels")
                if any(not isinstance(label, str) or not label.strip() for label in criteria):
                    raise ValueError("box labels must be non-empty strings")
        return self


class SystemOneRequest(BaseModel):
    """System One request body.

    ``model`` and ``variant`` are routing extensions: System One itself does not
    require a model field, but a multi-model local server does. They are
    optional and standard clients can omit them.
    """

    model_config = ConfigDict(extra="forbid")

    state: Any = None
    questions: dict[str, SystemOneQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)
    files: list[str] | None = None
    images: list[str] | None = None
    model: str | None = None
    variant: str | None = None

    @model_validator(mode="after")
    def _validate_state(self) -> SystemOneRequest:
        if self.state is None and not self.files and not self.images:
            raise ValueError("state is required unless files/images are provided")
        return self


class SystemOneUsage(BaseModel):
    """Usage exactly as System One 1.14 defines it."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(default=0, ge=0, le=0)


class SystemOneResponse(BaseModel):
    """System One response, plus a namespaced toolkit extension."""

    model_config = ConfigDict(extra="forbid")

    model: str
    answers: dict[str, dict[str, Any]]
    usage: SystemOneUsage
    x_decision_toolkit: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "MAX_QUESTIONS",
    "SYSTEMONE_SPEC",
    "SystemOneQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "SystemOneUsage",
]
