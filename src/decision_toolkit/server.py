"""`decision_toolkit.server` — the REST projection (extra ``[server]``).

The server is deliberately a **thin, versioned projection** of the core:

* ``POST /v1/systemone`` — System One 1.14 (TypeSafe compatible).
* ``POST /v1/decisions`` — OpenAI Decisions beta dialect.
* ``GET /v1/models`` / ``GET /v1/models/{id}`` — OpenAI model-list shape.
* ``GET /v1/decision-models`` — rich catalog/capability metadata.
* ``GET /health`` / ``GET /ready`` and ``/v1`` aliases.

The wire models live in :mod:`decision_toolkit.contracts`; the server contains
no inference or contract logic. It adds transport semantics: Bearer auth,
request ids, access logs, body limits, CORS, timeouts and a standard error
envelope. The core remains the superset.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any

from starlette.requests import Request

from . import __version__, catalog
from .contracts import (
    OPENAI_DECISIONS_SPEC,
    SYSTEMONE_SPEC,
    ErrorResponse,
    OpenAIDecisionsRequest,
    OpenAIDecisionsResponse,
    SystemOneRequest,
    SystemOneResponse,
    canonical_instructions,
    error_payload,
    openai_response,
    systemone_questions,
    systemone_response,
)
from .errors import (
    AdapterError,
    AdapterNotAvailableError,
    AdapterNotImplementedError,
    DecisionToolkitError,
    InvalidQuestionError,
    InvalidRequestError,
    ModelNotDownloadedError,
    ModelNotFoundError,
    MultimodalNotSupportedError,
    TruncationError,
    UnsupportedCapabilityError,
)
from .models import ModelStore
from .observability import (
    configure_logging,
    get_request_id,
    log_event,
    new_request_id,
    request_context,
)
from .registry import ModelRegistry
from .schema import openai_to_systemone

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("decision_toolkit.server.access")

_MAX_FILES = 8


class _BodyTooLarge(BaseException):
    """Internal control-flow signal for the ASGI body limiter.

    It deliberately does not derive from ``Exception`` so FastAPI's generic
    exception handler cannot turn a chunked over-limit body into a 500 before
    the middleware can emit the standard 413 envelope.
    """


class _RequestMiddleware:
    """Request id, access log and body-size limit as one ASGI middleware."""

    def __init__(self, app: Any, *, max_body_bytes: int, logger_: logging.Logger) -> None:
        self.app = app
        self.max_body_bytes = max(1, int(max_body_bytes))
        self.logger = logger_

    @staticmethod
    def _request_id(scope: dict[str, Any]) -> str:
        for key, value in scope.get("headers", []):
            if key.lower() == b"x-request-id":
                candidate = value.decode("latin-1", errors="replace").strip()
                if candidate and len(candidate) <= 128:
                    return candidate
        return new_request_id()

    async def _send_json(
        self,
        send: Any,
        status: int,
        payload: dict[str, Any],
        request_id: str,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"x-request-id", request_id.encode("ascii", errors="replace")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._request_id(scope)
        started = time.perf_counter()
        status = 500
        response_started = False
        total_body = 0

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status, response_started
            if message.get("type") == "http.response.start":
                response_started = True
                status = int(message.get("status", 500))
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii", errors="replace")))
                message["headers"] = headers
            await send(message)

        async def receive_wrapper() -> dict[str, Any]:
            nonlocal total_body
            message = await receive()
            if message.get("type") == "http.request":
                total_body += len(message.get("body", b""))
                if total_body > self.max_body_bytes:
                    raise _BodyTooLarge
            return message

        content_length = 0
        for key, value in scope.get("headers", []):
            if key.lower() == b"content-length":
                try:
                    content_length = int(value)
                except ValueError:
                    content_length = 0
                break

        if content_length > self.max_body_bytes:
            await self._send_json(
                send,
                413,
                error_payload(
                    f"request body exceeds {self.max_body_bytes} bytes",
                    error_type="invalid_request_error",
                    code="request_too_large",
                ),
                request_id,
            )
            status = 413
            self._log_access(scope, request_id, status, started)
            return

        try:
            with request_context(request_id):
                await self.app(scope, receive_wrapper, send_wrapper)
        except _BodyTooLarge:
            if not response_started:
                await self._send_json(
                    send,
                    413,
                    error_payload(
                        f"request body exceeds {self.max_body_bytes} bytes",
                        error_type="invalid_request_error",
                        code="request_too_large",
                    ),
                    request_id,
                )
                status = 413
        finally:
            self._log_access(scope, request_id, status, started, total_body)

    def _log_access(
        self,
        scope: dict[str, Any],
        request_id: str,
        status: int,
        started: float,
        body_bytes: int = 0,
    ) -> None:
        log_event(
            self.logger,
            "http.request",
            request_id=request_id,
            method=scope.get("method"),
            path=scope.get("path"),
            status=status,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            body_bytes=body_bytes or None,
        )


def _status_for(error: DecisionToolkitError) -> int:
    if isinstance(error, MultimodalNotSupportedError):
        return 501
    if isinstance(error, UnsupportedCapabilityError):
        return 400
    if isinstance(error, ModelNotFoundError):
        return 404
    if isinstance(error, ModelNotDownloadedError):
        return 503
    if isinstance(error, AdapterNotImplementedError):
        return 501
    if isinstance(error, AdapterNotAvailableError):
        return 503
    if isinstance(error, (InvalidQuestionError, InvalidRequestError, TruncationError)):
        return 400
    if isinstance(error, AdapterError):
        return 500
    return 500


def _error_type(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status == 429:
        return "rate_limit_error"
    if status >= 500:
        return "service_unavailable_error" if status in {501, 503, 504} else "server_error"
    return "invalid_request_error"


def _error_body(error: DecisionToolkitError) -> dict[str, Any]:
    status = _status_for(error)
    return error_payload(
        str(error),
        error_type=_error_type(status),
        code=getattr(error, "code", "decision_toolkit_error"),
    )


def _has_media(value: Any) -> bool:
    """Detect image/audio parts embedded in a System One state or OpenAI input."""
    if isinstance(value, dict):
        if value.get("type") in {"image_url", "input_image", "input_audio", "audio"}:
            return True
        if "image_url" in value or "input_image" in value:
            return True
        return any(_has_media(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_media(item) for item in value)
    return False


def _validate_files(files: Sequence[str] | None, *, kind: str) -> None:
    if not files:
        return
    if len(files) > _MAX_FILES:
        raise InvalidRequestError(f"at most {_MAX_FILES} {kind} can be sent")
    for item in files:
        if not isinstance(item, str) or not item.startswith("data:"):
            raise InvalidRequestError(f"{kind} must be data URLs")
    raise MultimodalNotSupportedError(
        f"{kind} input is not supported by the text-only models in this catalog"
    )


def _rich_model_payload(
    model: catalog.CatalogModel,
    cached: set[tuple[str, str]],
    *,
    current: str | None,
) -> dict[str, Any]:
    payload = model.info.to_dict()
    payload["cached_variants"] = [
        variant.name for variant in model.variants if (model.id, variant.name) in cached
    ]
    payload["current"] = model.id == current
    payload["object"] = "model"
    payload["owned_by"] = "decision-toolkit"
    payload["created"] = 0
    return payload


def _cached_custom_payload(entry: Any) -> dict[str, Any]:
    return {
        "id": entry.identifier,
        "object": "model",
        "created": 0,
        "owned_by": "decision-toolkit",
        "name": entry.identifier,
        "adapter": entry.adapter or "unknown",
        "license": "custom",
        "params": "",
        "architecture": "custom ONNX",
        "languages": "unknown",
        "status": "custom",
        "capabilities": {
            "question_types": ["choice", "noul", "score"],
            "max_options": 255,
            "max_score_levels": 10,
            "max_context_tokens": None,
            "languages": ["unknown"],
            "multimodal": False,
            "multi_label": False,
            "batch": False,
            "usage": True,
        },
        "variants": [],
        "cached_variants": [entry.variant],
        "current": False,
        "notes": "local import",
        "x_decision_toolkit": {"cached_variants": [entry.variant]},
    }


def create_app(
    *,
    model: Any | None = None,
    model_id: str | None = None,
    variant: str | None = None,
    home: str | None = None,
    threads: int = 4,
    adapter: Any = None,
    files: dict[str, str] | None = None,
    api_key: str | None = None,
    cors_origins: Sequence[str] = (),
    max_body_mb: float = 16.0,
    timeout_seconds: float = 120.0,
    max_models: int = 2,
    warmup: bool = False,
    log_level: str | int | None = None,
    log_format: str = "text",
) -> Any:
    """Build the FastAPI app. Importing this needs the ``[server]`` extra.

    ``model`` binds a fixed model (tests/embedding). Otherwise requests may
    select ``model`` and ``variant`` and the registry loads them lazily.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.exceptions import RequestValidationError
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from starlette.exceptions import HTTPException as StarletteHTTPException
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            'the REST server needs the [server] extra: pip install "decision-toolkit[server]"'
        ) from error

    if log_level is not None:
        configure_logging(log_level, log_format)
    if isinstance(cors_origins, str):
        cors_origins = [cors_origins]

    registry = ModelRegistry(
        model=model,
        model_id=model_id,
        variant=variant,
        home=home,
        threads=threads,
        max_loaded=max_models,
        adapter=adapter,
        files=files,
    )
    if warmup:
        registry.warmup_default()

    @asynccontextmanager
    async def lifespan(_app: Any) -> Any:
        yield
        log_event(logger, "server.shutdown", model=registry.default_identifier)
        registry.close()

    app = FastAPI(
        title="decision-toolkit",
        version=__version__,
        description=(
            "Local-first decision models. System One 1.14 and OpenAI Decisions "
            "beta are projections of the same programmatic core."
        ),
        lifespan=lifespan,
    )
    app.state.registry = registry
    app.state.settings = {
        "auth": bool(api_key),
        "contracts": {
            "systemone": SYSTEMONE_SPEC,
            "openai_decisions": OPENAI_DECISIONS_SPEC,
        },
        "max_body_mb": max_body_mb,
        "timeout_seconds": timeout_seconds,
    }

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not api_key:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    protected = [Depends(require_auth)]

    def request_id(request: Request) -> str:
        return getattr(request.state, "request_id", None) or get_request_id() or new_request_id()

    async def run(func: Any, *args: Any, **kwargs: Any) -> Any:
        if timeout_seconds and timeout_seconds > 0:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs), timeout=timeout_seconds
                )
            except TimeoutError as error:
                raise HTTPException(status_code=504, detail="decision timed out") from error
        return await asyncio.to_thread(func, *args, **kwargs)

    def catalog_state() -> tuple[ModelStore, set[tuple[str, str]], str | None]:
        store = registry.store
        cached = {(entry.identifier, entry.variant) for entry in store.entries()}
        return store, cached, store.current()

    # ------------------------------------------------------------- meta/health
    @app.get("/health", tags=["meta"])
    @app.get("/v1/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["meta"])
    @app.get("/v1/ready", tags=["meta"])
    def ready() -> Any:
        is_ready, detail = registry.is_ready()
        payload = {"ready": is_ready, **detail}
        return JSONResponse(status_code=200 if is_ready else 503, content=payload)

    @app.get("/v1/info", tags=["meta"], dependencies=protected)
    def info() -> dict[str, Any]:
        return {
            "name": "decision-toolkit",
            "version": __version__,
            "contracts": {
                "systemone": SYSTEMONE_SPEC,
                "openai_decisions": OPENAI_DECISIONS_SPEC,
            },
            "registry": registry.describe(),
            "auth": bool(api_key),
        }

    # ------------------------------------------------------------------ models
    @app.get("/v1/models", tags=["models"], dependencies=protected)
    def models() -> dict[str, Any]:
        _store, cached, current = catalog_state()
        data = [
            {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "decision-toolkit",
                "x_decision_toolkit": _rich_model_payload(model, cached, current=current),
            }
            for model in catalog.CATALOG
        ]
        known = {model.id for model in catalog.CATALOG}
        for entry in _store.entries():
            if entry.identifier not in known:
                data.append(_cached_custom_payload(entry))
                known.add(entry.identifier)
        return {"object": "list", "data": data}

    @app.get("/v1/models/{model_name:path}", tags=["models"], dependencies=protected)
    def model_detail(model_name: str) -> dict[str, Any]:
        _store, cached, current = catalog_state()
        model = catalog.get(model_name)
        if model is not None:
            rich = _rich_model_payload(model, cached, current=current)
            return {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "decision-toolkit",
                "x_decision_toolkit": rich,
            }
        entry = _store.get(model_name)
        if entry is not None:
            return _cached_custom_payload(entry)
        raise ModelNotFoundError(f"unknown model {model_name!r}")

    @app.get("/v1/decision-models", tags=["models"], dependencies=protected)
    def decision_models() -> dict[str, Any]:
        _store, cached, current = catalog_state()
        data = [_rich_model_payload(model, cached, current=current) for model in catalog.CATALOG]
        known = {model.id for model in catalog.CATALOG}
        for entry in _store.entries():
            if entry.identifier not in known:
                data.append(_cached_custom_payload(entry))
                known.add(entry.identifier)
        return {"object": "list", "data": data}

    # --------------------------------------------------------------- inference
    @app.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        dependencies=protected,
        tags=["decisions"],
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            501: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def systemone(payload: SystemOneRequest, request: Request) -> SystemOneResponse:
        _validate_files([*(payload.files or []), *(payload.images or [])], kind="images")
        if _has_media(payload.state):
            raise MultimodalNotSupportedError(
                "image/audio parts in state are not supported by the text-only models "
                "in this catalog"
            )
        questions = systemone_questions(payload.questions)
        with registry.acquire(payload.model, payload.variant) as selected:
            decision = await run(
                selected.decide,
                payload.state,
                questions,
                request_id=request_id(request),
            )
        return systemone_response(decision)

    @app.post(
        "/v1/decisions",
        response_model=OpenAIDecisionsResponse,
        response_model_exclude_none=True,
        dependencies=protected,
        tags=["decisions"],
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            501: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def decisions(
        payload: OpenAIDecisionsRequest, request: Request
    ) -> OpenAIDecisionsResponse:
        if _has_media(payload.input):
            raise MultimodalNotSupportedError(
                "image/audio input is not supported by the text-only models in this catalog"
            )
        wire_questions = [question.model_dump(exclude_none=True) for question in payload.questions]
        for question in wire_questions:
            instructions = question.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                question["instructions"] = canonical_instructions(instructions)
        names = [
            str(question.get("name") or f"q{index}")
            for index, question in enumerate(wire_questions)
        ]
        if len(set(names)) != len(names):
            raise InvalidRequestError("question names must be unique")
        state_text, core_questions, names = openai_to_systemone(payload.input, wire_questions)
        with registry.acquire(payload.model, payload.variant) as selected:
            decision = await run(
                selected.decide,
                state_text,
                core_questions,
                request_id=request_id(request),
            )
        values_by_name = {
            name: question["_values"]
            for name, question in core_questions.items()
            if "_values" in question
        }
        return openai_response(
            decision,
            names,
            safety_identifier=payload.safety_identifier,
            values_by_name=values_by_name,
        )

    # --------------------------------------------------------------- handlers
    @app.exception_handler(DecisionToolkitError)
    async def toolkit_error_handler(request: Request, error: DecisionToolkitError) -> JSONResponse:
        status = _status_for(error)
        log_event(
            logger,
            "rest.error",
            level=logging.ERROR if status >= 500 else logging.WARNING,
            request_id=request_id(request),
            path=request.url.path,
            status=status,
            error_code=getattr(error, "code", "decision_toolkit_error"),
            error=type(error).__name__,
        )
        return JSONResponse(status_code=status, content=_error_body(error))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, error: RequestValidationError) -> JSONResponse:
        errors = error.errors()
        first = errors[0] if errors else {}
        location = ".".join(str(item) for item in first.get("loc", ()) if item != "body")
        message = first.get("msg", "invalid request")
        if location:
            message = f"{location}: {message}"
        payload = error_payload(
            message,
            error_type="invalid_request_error",
            param=location or None,
            code="invalid_request",
        )
        log_event(
            logger,
            "rest.validation_error",
            level=logging.WARNING,
            request_id=request_id(request),
            path=request.url.path,
            status=400,
            error_code="invalid_request",
        )
        return JSONResponse(status_code=400, content=payload)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
        status = int(error.status_code)
        detail = error.detail if isinstance(error.detail, str) else "request failed"
        code = "http_error"
        if status == 401:
            code = "authentication_error"
        elif status == 504:
            code = "timeout"
        payload = error_payload(detail, error_type=_error_type(status), code=code)
        log_event(
            logger,
            "rest.http_error",
            level=logging.ERROR if status >= 500 else logging.WARNING,
            request_id=request_id(request),
            path=request.url.path,
            status=status,
            error_code=code,
        )
        headers = dict(error.headers or {})
        return JSONResponse(status_code=status, content=payload, headers=headers)

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unhandled REST error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "internal server error",
                error_type="server_error",
                code="internal_error",
            ),
        )

    # -------------------------------------------------------------- middleware
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["x-request-id"],
        )
    app.add_middleware(
        _RequestMiddleware,
        max_body_bytes=int(max_body_mb * 1024 * 1024),
        logger_=access_logger,
    )

    log_event(
        logger,
        "server.startup",
        version=__version__,
        model=registry.default_identifier,
        variant=registry.default_variant,
        auth=bool(api_key),
        systemone=SYSTEMONE_SPEC,
        openai_decisions=OPENAI_DECISIONS_SPEC,
    )
    return app


__all__ = ["create_app"]
