"""OpenAI Decisions wire contract (beta).

The projection mirrors the dialect already used by the toolkit's
``openai_to_systemone`` translator:

* request: ``input`` (string or message list) + ``questions`` list
* questions: ``predicate`` / ``choice`` / ``score``
* response: ``id``, ``object``, ``created_at``, ``model``, ``answers``, ``usage``
* errors: the OpenAI ``{"error": {message, type, param, code}}`` envelope

The upstream Decisions API is still beta; this module pins the shape and is the
only place that needs to change when the upstream spec changes. Standard fields
keep their meaning; toolkit metadata lives under ``x_decision_toolkit``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OPENAI_DECISIONS_SPEC = "beta"
MAX_QUESTIONS = 128


class OpenAIChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: Any
    description: str | None = None


class OpenAIScoreLevel(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str | None = None
    description: str | None = None


class OpenAIDecisionQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["predicate", "choice", "score"]
    name: str | None = None
    instructions: Any = None
    choices: list[OpenAIChoice] | None = None
    levels: list[OpenAIScoreLevel] | None = None

    @model_validator(mode="after")
    def _validate(self) -> OpenAIDecisionQuestion:
        if self.type == "choice":
            if not self.choices or len(self.choices) < 2:
                raise ValueError("a choice needs at least 2 choices")
            if len(self.choices) > 255:
                raise ValueError("a choice supports at most 255 choices")
        elif self.type == "score":
            if not self.levels or len(self.levels) < 2:
                raise ValueError("a score needs at least 2 levels")
            if len(self.levels) > 10:
                raise ValueError("a score supports at most 10 levels")
            for level in self.levels:
                if level.label is None or not str(level.label).strip():
                    raise ValueError("every score level needs a non-empty label")
        return self


class OpenAIDecisionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Any
    questions: list[OpenAIDecisionQuestion] = Field(
        min_length=1, max_length=MAX_QUESTIONS
    )
    model: str | None = None
    variant: str | None = None
    safety_identifier: str | None = None

    @model_validator(mode="after")
    def _validate_input(self) -> OpenAIDecisionsRequest:
        if self.input is None:
            raise ValueError("input is required")
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input cannot be empty")
        if isinstance(self.input, list) and not self.input:
            raise ValueError("input cannot be an empty message list")
        return self


class OpenAIDecisionAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["predicate", "choice", "score", "refusal"]
    name: str
    probability: float | None = None
    choice: Any = None
    probabilities: list[dict[str, Any]] | None = None
    score: float | None = None
    confidence: float | None = None


class OpenAIUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class OpenAIDecisionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed"] = "completed"
    model: str
    answers: list[OpenAIDecisionAnswer]
    usage: OpenAIUsage
    safety_identifier: str | None = None
    x_decision_toolkit: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "MAX_QUESTIONS",
    "OPENAI_DECISIONS_SPEC",
    "OpenAIChoice",
    "OpenAIDecisionAnswer",
    "OpenAIDecisionQuestion",
    "OpenAIDecisionsRequest",
    "OpenAIDecisionsResponse",
    "OpenAIScoreLevel",
    "OpenAIUsage",
]
