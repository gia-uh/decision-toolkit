"""System One wire contract, pinned to TypeSafe's published API schema 0.2.0.

The reference is the TypeSafe OpenAPI document at
``https://api.typesafe.ai/openapi.json`` (OpenAPI version 0.2.0) and its
documentation at ``https://docs.typesafe.ai/api``:

* request: ``model`` + ``state`` + ``questions`` (+ optional ``assets``)
* questions: ``choice`` / ``noul`` / ``score`` / ``box``
* response: ``model`` + ``answers`` + ``usage``
* ``usage.output_tokens`` is a non-negative token count; local decision
  adapters currently report 0 because they do not generate text
* request-schema failures use ``422`` with a FastAPI-style ``detail`` array
* unsupported image/asset input returns ``501`` because the current catalog is
  text-only

``model`` remains an optional routing extension for the local
multi-model server; standard clients send ``model``. Toolkit-specific metadata
lives under ``x_decision_toolkit`` so standard fields keep their meaning.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SYSTEMONE_SPEC = "0.2.0"
MAX_QUESTIONS = 128


class SystemOneQuestion(BaseModel):
    """One typed System One question.

    ``instructions`` accepts a string, object or array in System One; the
    projection canonicalizes objects/arrays to JSON text before calling the
    core, whose adapters consume text.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["choice", "noul", "score", "box"]
    instructions: Any = None
    criteria: Any = None

    @model_validator(mode="after")
    def _validate(self) -> SystemOneQuestion:
        instructions = self.instructions
        if isinstance(instructions, str) and not instructions.strip():
            raise ValueError("instructions cannot be an empty string")
        if instructions is not None and not isinstance(instructions, (str, dict, list)):
            raise ValueError(
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
                if description is not None and not isinstance(description, (str, dict, list)):
                    raise ValueError(
                        "choice descriptions must be strings, objects, arrays or null"
                    )
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
                for meaning in criteria.values():
                    if meaning is not None and not isinstance(meaning, (str, dict, list)):
                        raise ValueError(
                            "noul criteria values must be strings, objects, arrays or null"
                        )
        elif self.type == "score":
            criteria = self.criteria
            if not isinstance(criteria, list) or not 1 <= len(criteria) <= 10:
                raise ValueError("a score needs an array of 1 to 10 levels")
            if any(
                not isinstance(level, (str, dict, list))
                or (isinstance(level, str) and not level.strip())
                for level in criteria
            ):
                raise ValueError("score levels must be non-empty strings, objects or arrays")
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

    ``model`` is a routing extension: System One itself does not require a model
    field, but a multi-model local server does. It is optional and standard
    clients can omit it.
    """

    model_config = ConfigDict(extra="forbid")

    state: Any = None
    questions: dict[str, SystemOneQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)
    assets: dict[str, Any] | None = None
    # Legacy local extensions; standard clients use state + assets.
    files: list[str] | None = None
    images: list[str] | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _validate_state(self) -> SystemOneRequest:
        if self.state is None and not self.files and not self.images and not self.assets:
            raise ValueError("state is required unless files/images/assets are provided")
        return self


class SystemOneUsage(BaseModel):
    """Usage as defined by the TypeSafe System One schema."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(default=0, ge=0)


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
