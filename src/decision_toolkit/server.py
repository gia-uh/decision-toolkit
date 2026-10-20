"""`decision_toolkit.server` — the REST surface (extra ``[server]``).

A thin FastAPI adapter over :class:`~decision_toolkit.decision_model.DecisionModel`:

* ``POST /v1/systemone`` — the System One contract (``state`` + typed ``questions``).
* ``POST /v1/decisions`` — OpenAI's Decisions dialect (``input`` + ``questions`` list),
  translated to System One and back.
* ``GET /v1/models`` — the curated catalog, adapters and cache state.
* ``GET /v1/info``, ``GET /v1/health``, ``GET /v1/ready``.

Anything that talks to Jev talks to ``dtk serve`` by changing the base URL.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import __version__, catalog
from .decision_model import DecisionModel
from .errors import (
    AdapterNotAvailableError,
    DecisionToolkitError,
    InvalidQuestionError,
    ModelNotFoundError,
)
from .models import ModelStore
from .schema import (
    normalize_questions,
    openai_to_systemone,
    systemone_answers_to_openai,
)


def create_app(
    *,
    model: DecisionModel | None = None,
    model_id: str | None = None,
    variant: str | None = None,
    home: str | None = None,
    threads: int = 4,
) -> Any:
    """Build the FastAPI app. Importing this needs the ``[server]`` extra."""
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            'the REST server needs the [server] extra: pip install "decision-toolkit[server]"'
        ) from error

    app = FastAPI(title="decision-toolkit", version=__version__)
    state: dict[str, DecisionModel | None] = {"model": model}

    def get_model() -> DecisionModel:
        if state["model"] is None:
            state["model"] = DecisionModel(
                model_id, variant=variant, home=home, threads=threads
            )
        return state["model"]  # type: ignore[return-value]

    def translate(error: DecisionToolkitError) -> HTTPException:
        if isinstance(error, (InvalidQuestionError, ModelNotFoundError)):
            return HTTPException(status_code=400, detail=str(error))
        if isinstance(error, AdapterNotAvailableError):
            return HTTPException(status_code=503, detail=str(error))
        return HTTPException(status_code=500, detail=str(error))

    # ------------------------------------------------------------------ meta
    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready")
    def ready() -> dict[str, Any]:
        try:
            bound = get_model()
            return {
                "ready": True,
                "model": bound.identifier,
                "adapter": bound.adapter_name,
                "variant": bound.variant,
            }
        except DecisionToolkitError as error:
            return {"ready": False, "detail": str(error)}

    @app.get("/v1/info")
    def info() -> dict[str, Any]:
        bound = get_model()
        return {
            "name": "decision-toolkit",
            "version": __version__,
            "model": bound.identifier,
            "adapter": bound.adapter_name,
            "variant": bound.variant,
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        store = ModelStore(home=home)
        cached = {(entry.identifier, entry.variant) for entry in store.entries()}
        return {
            "current": store.current(),
            "catalog": [
                {
                    "id": entry.id,
                    "name": entry.name,
                    "adapter": entry.adapter,
                    "license": entry.license,
                    "params": entry.params,
                    "status": entry.status,
                    "variants": [
                        {
                            "name": variant.name,
                            "quant": variant.quant,
                            "size_mb": variant.size_mb,
                            "role": variant.role,
                            "latency_ms_p50": variant.latency_ms_p50,
                            "verified_at": variant.verified_at,
                            "parity": dict(variant.parity) if variant.parity else None,
                            "cached": (entry.id, variant.name) in cached,
                        }
                        for variant in entry.variants
                    ],
                }
                for entry in catalog.CATALOG
            ],
        }

    # ------------------------------------------------------------- inference
    @app.post("/v1/systemone")
    def systemone(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
        state_value = payload.get("state")
        questions = payload.get("questions")
        if state_value is None or not isinstance(questions, Mapping):
            raise HTTPException(status_code=400, detail="'state' and 'questions' are required")
        try:
            normalized = normalize_questions(questions)
            decision = get_model().decide(state_value, normalized)
        except DecisionToolkitError as error:
            raise translate(error) from error
        return {"model": get_model().identifier, "answers": decision.to_dict()}

    @app.post("/v1/decisions")
    def decisions(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
        input_value = payload.get("input")
        questions = payload.get("questions")
        if input_value is None or not isinstance(questions, list):
            raise HTTPException(status_code=400, detail="'input' and a 'questions' list are required")
        try:
            state_text, systemone_questions, names = openai_to_systemone(input_value, questions)
            decision = get_model().decide(state_text, systemone_questions)
            answers = systemone_answers_to_openai(decision.raw, names)
        except DecisionToolkitError as error:
            raise translate(error) from error
        return {"model": get_model().identifier, "answers": answers}

    return app


__all__ = ["create_app"]
