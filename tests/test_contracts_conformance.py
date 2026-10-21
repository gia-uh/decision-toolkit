"""Conformance checks against the published wire schemas.

References:
* TypeSafe System One: https://docs.typesafe.ai/api (OpenAPI 0.2.0)
* OpenAI Decisions: https://developers.openai.com/api/docs/guides/decisions
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from decision_toolkit.contracts.openai_decisions_beta import (
    OpenAIDecisionsRequest,
    OpenAIDecisionsResponse,
    OpenAIUsage,
)
from decision_toolkit.contracts.systemone_v1 import (
    SYSTEMONE_SPEC,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneUsage,
)


def test_systemone_is_pinned_to_the_published_schema():
    assert SYSTEMONE_SPEC == "0.2.0"


def test_systemone_usage_allows_nonzero_output_tokens():
    usage = SystemOneUsage(input_tokens=120, output_tokens=12)
    assert usage.output_tokens == 12


def test_systemone_accepts_rich_question_values_and_standard_assets():
    question = SystemOneQuestion.model_validate(
        {
            "type": "choice",
            "instructions": {"task": "route the message"},
            "criteria": {
                "billing": {"meaning": "payments"},
                "other": ["anything else"],
            },
        }
    )
    assert question.instructions == {"task": "route the message"}

    request = SystemOneRequest.model_validate(
        {
            "model": "jev-latest",
            "state": {"message": "I was charged twice."},
            "assets": {
                "photo": {
                    "type": "image",
                    "format": "png",
                    "data_base64": "AAAA",
                }
            },
            "questions": {"billing": question.model_dump()},
        }
    )
    assert request.assets is not None


def test_openai_decision_response_uses_the_public_envelope():
    response = OpenAIDecisionsResponse.model_validate(
        {
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": "refund", "probability": 0.9}],
            "usage": {
                "input_tokens": 42,
                "output_tokens": 0,
                "total_tokens": 42,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )
    assert response.model == "gpt-6-luna"
    assert response.usage.total_tokens == 42
    assert set(response.model_dump()) == {
        "model",
        "answers",
        "usage",
        "x_decision_toolkit",
    }


def test_openai_request_rejects_unverified_response_metadata_fields():
    with pytest.raises(ValidationError):
        OpenAIDecisionsRequest.model_validate(
            {
                "model": "gpt-6-luna",
                "input": "x",
                "questions": [{"type": "predicate", "name": "q", "instructions": "Q?"}],
                "safety_identifier": "not-part-of-the-confirmed-request-shape",
            }
        )


def test_openai_usage_accepts_published_token_details():
    usage = OpenAIUsage(
        input_tokens=42,
        output_tokens=0,
        total_tokens=42,
        input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
        output_tokens_details={"reasoning_tokens": 0},
    )
    assert usage.input_tokens_details["cached_tokens"] == 0
