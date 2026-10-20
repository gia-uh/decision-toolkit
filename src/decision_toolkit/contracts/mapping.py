"""Pure mappings between core decisions and the two wire projections.

No business logic lives here: the core has already decided. This module only
selects the standard fields each contract exposes and puts toolkit-only
metadata in the namespaced ``x_decision_toolkit`` extension.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..results import Answer, Decision
from ..schema import systemone_answers_to_openai
from .openai_decisions_beta import OPENAI_DECISIONS_SPEC
from .systemone_v1 import (
    SYSTEMONE_SPEC,
    SystemOneQuestion,
    SystemOneResponse,
    SystemOneUsage,
)


def canonical_instructions(value: Any) -> str:
    """Turn a System One instructions value into text for the core adapters."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def systemone_questions(
    questions: Mapping[str, SystemOneQuestion],
) -> dict[str, dict[str, Any]]:
    """Project wire questions to the canonical core question dictionaries."""
    projected: dict[str, dict[str, Any]] = {}
    for name, question in questions.items():
        # Do not use exclude_none here: choice descriptions are explicitly
        # allowed to be null and dropping them would drop the option itself.
        data = question.model_dump()
        if data.get("criteria") is None:
            data.pop("criteria", None)
        data["instructions"] = canonical_instructions(data["instructions"])
        projected[name] = data
    return projected


def _answer_metadata(decision: Decision) -> dict[str, Any]:
    return {
        name: {
            "confidence_kind": answer.confidence_kind,
            "truncated": answer.truncated,
            "warnings": list(answer.warnings),
            "action": dict(answer.action) if answer.action is not None else None,
        }
        for name, answer in decision.answers.items()
    }


def project_systemone_answer(answer: Answer) -> dict[str, Any]:
    """Project one core answer onto the System One 1.14 fields."""
    if answer.type == "choice":
        projected: dict[str, Any] = {
            "type": "choice",
            "choice": answer.choice,
            "probabilities": {str(key): float(value) for key, value in answer.probabilities.items()},
        }
        if answer.confidence is not None:
            projected["confidence"] = float(answer.confidence)
        return projected
    if answer.type == "score":
        projected = {
            "type": "score",
            "score": answer.score,
            "legend": {str(key): str(value) for key, value in answer.legend.items()},
            "probabilities": {str(key): float(value) for key, value in answer.probabilities.items()},
        }
        if answer.confidence is not None:
            projected["confidence"] = float(answer.confidence)
        return projected
    if answer.type == "noul":
        return {"type": "noul", "noul": answer.noul}
    if answer.type == "box":
        projected = {"type": "box", "box": answer.data.get("box", answer.value)}
        if answer.confidence is not None:
            projected["confidence"] = float(answer.confidence)
        return projected
    return {"type": "refusal"}


def systemone_response(decision: Decision) -> SystemOneResponse:
    """Build the exact System One response plus a namespaced extension."""
    answers = {
        name: project_systemone_answer(answer)
        for name, answer in decision.answers.items()
    }
    return SystemOneResponse(
        model=decision.model or "unknown",
        answers=answers,
        usage=SystemOneUsage(
            input_tokens=decision.usage.input_tokens,
            output_tokens=0,
        ),
        x_decision_toolkit={
            "spec": SYSTEMONE_SPEC,
            "request_id": decision.request_id,
            "variant": decision.variant,
            "adapter": decision.adapter,
            "latency_ms": decision.latency_ms,
            "truncated": decision.truncated,
            "warnings": list(decision.warnings),
            "usage": decision.usage.to_dict(),
            "answers": _answer_metadata(decision),
        },
    )


def openai_response(
    decision: Decision,
    names: Sequence[str],
    *,
    safety_identifier: str | None = None,
    values_by_name: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Build the OpenAI Decisions response envelope."""
    answers = systemone_answers_to_openai(
        decision.answer_payload(),
        names,
        values_by_name=values_by_name,
    )
    return {
        "id": decision.id or f"dec_{int(time.time() * 1000)}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": decision.model or "unknown",
        "answers": answers,
        "usage": {
            "input_tokens": decision.usage.input_tokens,
            "output_tokens": 0,
            "total_tokens": decision.usage.input_tokens,
        },
        "safety_identifier": safety_identifier,
        "x_decision_toolkit": {
            "spec": OPENAI_DECISIONS_SPEC,
            "request_id": decision.request_id,
            "variant": decision.variant,
            "adapter": decision.adapter,
            "latency_ms": decision.latency_ms,
            "truncated": decision.truncated,
            "warnings": list(decision.warnings),
            "usage": decision.usage.to_dict(),
            "answers": _answer_metadata(decision),
        },
    }


__all__ = [
    "canonical_instructions",
    "openai_response",
    "project_systemone_answer",
    "systemone_questions",
    "systemone_response",
]
