"""Versioned wire contracts for the REST surface.

The programmatic core is the superset. These modules are the conformance-tested
projections of a subset of it:

* ``systemone_v1`` — TypeSafe System One API (OpenAPI 0.2.0 subset).
* ``openai_decisions_beta`` — OpenAI Decisions beta dialect.

Each module owns its request/response models. ``mapping`` contains only pure
projections; the FastAPI server contains no inference or contract logic.
"""

from .common import ErrorDetail, ErrorResponse, WireUsage, error_payload
from .mapping import (
    canonical_instructions,
    openai_response,
    project_systemone_answer,
    systemone_questions,
    systemone_response,
)
from .openai_decisions_beta import (
    OPENAI_DECISIONS_SPEC,
    OpenAIDecisionAnswer,
    OpenAIDecisionQuestion,
    OpenAIDecisionsRequest,
    OpenAIDecisionsResponse,
    OpenAIUsage,
)
from .systemone_v1 import (
    SYSTEMONE_SPEC,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneUsage,
)

__all__ = [
    "OPENAI_DECISIONS_SPEC",
    "SYSTEMONE_SPEC",
    "ErrorDetail",
    "ErrorResponse",
    "OpenAIDecisionAnswer",
    "OpenAIDecisionQuestion",
    "OpenAIDecisionsRequest",
    "OpenAIDecisionsResponse",
    "OpenAIUsage",
    "SystemOneQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "SystemOneUsage",
    "WireUsage",
    "canonical_instructions",
    "error_payload",
    "openai_response",
    "project_systemone_answer",
    "systemone_questions",
    "systemone_response",
]
