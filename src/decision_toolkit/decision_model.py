"""`decision_toolkit.decision_model` — the public, library-first surface.

``DecisionModel`` resolves a curated model (or a custom local directory), picks
the catalog's adapter, opens the one ONNX engine, and answers System One
requests: typed questions in, calibrated probabilities out, on CPU.

This is the **superset** layer.  The CLI, the REST projection and third-party
applications all consume this API; no wire-specific concept lives here.

Phase 1 additions over the first scaffold:

* context-manager lifecycle and idempotent ``close()``;
* a lock around lazy loading and inference, so concurrent callers are safe;
* typed ``Usage``, latency, truncation warnings and normalized confidence on
  every ``Decision``;
* batch and async methods;
* capability validation before loading a model;
* request correlation through ``request_id`` and structured events;
* custom adapters (in-process registry + entry points) and ``from_onnx``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .adapters import Adapter, create_adapter, validate_adapter_files
from .adapters.base import state_to_text
from .catalog import Capabilities, ModelInfo
from .engine import OnnxEngine, encode
from .errors import (
    AdapterNotAvailableError,
    InvalidRequestError,
    ModelNotDownloadedError,
    TruncationError,
    UnsupportedCapabilityError,
)
from .models import ModelStore, ResolvedModel
from .observability import ensure_request_id, log_event
from .results import Answer, Classification, Decision, Usage, wrap_answers
from .schema import State, normalize_questions, normalize_state

logger = logging.getLogger(__name__)


class DecisionModel:
    """A local ONNX decision model, loaded lazily on first use.

    Parameters
    ----------
    identifier:
        A catalog id (``gliner2.5-multi``...), a cached variant, a local
        directory, or a hub repo (``org/name``). When omitted, the model in
        use from ``config.json`` is taken.
    adapter:
        Adapter name (``gliner2``...) or a ready-made adapter instance (tests,
        custom runtimes). When omitted, the catalog's adapter is used.
    adapter_options:
        Extra keyword arguments for the adapter (e.g. ``{"task": "categoria"}``).
    variant:
        Quantization variant (``fp16``, ``fp32``, ``q8``...). Defaults to the
        catalog's ``default`` variant, or the variant in use from config.
    files:
        Role -> file mapping for **custom** local models (``encoder``,
        ``classifier``, ``tokenizer``...). Catalog models come with their map.
    home / threads:
        Toolkit home override and ONNX Runtime thread count.
    strict:
        When true, a decision whose adapter had to truncate input raises
        ``TruncationError`` instead of returning a truncated answer. Can be
        overridden per call.
    """

    def __init__(
        self,
        identifier: str | None = None,
        *,
        adapter: Adapter | str | None = None,
        adapter_options: Mapping[str, Any] | None = None,
        variant: str | None = None,
        files: Mapping[str, str] | None = None,
        home: str | Path | None = None,
        threads: int = 4,
        strict: bool = False,
    ) -> None:
        self._identifier = identifier
        self._store = ModelStore(home=home)
        self._variant = variant
        self._files_override = dict(files or {})
        self._injected_adapter: Adapter | None = (
            adapter if adapter is not None and not isinstance(adapter, str) else None
        )
        self._adapter: Adapter | None = None
        self._adapter_name: str | None = adapter if isinstance(adapter, str) else None
        self._adapter_options = dict(adapter_options or {})
        self._threads = int(threads)
        self._strict = bool(strict)
        self._resolved: ResolvedModel | None = None
        self._engine: OnnxEngine | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ info
    @property
    def identifier(self) -> str:
        return self._identifier or self._store.current() or "(none)"

    @property
    def variant(self) -> str | None:
        if self._resolved is not None and self._resolved.variant is not None:
            return self._resolved.variant.name
        if self._variant is not None:
            return self._variant
        if self._identifier is None:
            return self._store.current_variant()
        return None

    @property
    def adapter_name(self) -> str:
        if self._injected_adapter is not None:
            return getattr(self._injected_adapter, "name", type(self._injected_adapter).__name__)
        if self._adapter_name:
            return self._adapter_name
        resolved = self._resolve()
        if resolved.catalog is not None:
            return resolved.catalog.adapter
        if resolved.adapter:
            return resolved.adapter
        raise AdapterNotAvailableError(
            f"cannot infer an adapter for {self.identifier!r}; pass adapter=<name>"
        )

    @property
    def resolved(self) -> ResolvedModel:
        return self._resolve()

    def _safe_adapter_name(self) -> str | None:
        """Adapter label for diagnostics; never masks the original error."""
        try:
            return self.adapter_name
        except Exception:  # noqa: BLE001 - diagnostics must be side-effect free
            if self._injected_adapter is not None:
                return getattr(self._injected_adapter, "name", type(self._injected_adapter).__name__)
            return self._adapter_name

    @property
    def capabilities(self) -> Capabilities:
        """Semantic capabilities of the selected model (validated before use)."""
        adapter = self._injected_adapter or self._adapter
        if adapter is not None:
            declared = getattr(adapter, "capabilities", None)
            if isinstance(declared, Capabilities):
                return declared
            supported = getattr(adapter, "supported_question_types", None)
            if supported is not None:
                return Capabilities(question_types=tuple(str(item) for item in supported))
            return Capabilities()

        resolved = self._resolve()
        if resolved.catalog is not None and self._adapter_name is None:
            return resolved.catalog.capabilities
        # A catalog model with an adapter override, or a custom/imported model:
        # capabilities are unknown until the adapter is instantiated.
        return Capabilities()

    def model_info(self) -> ModelInfo:
        """Typed metadata for catalog models; a minimal descriptor otherwise."""
        if self._identifier is not None or self._store.current() is not None:
            try:
                resolved = self._resolve()
            except Exception:  # noqa: BLE001 - metadata should never hide a model error
                resolved = None
            if resolved is not None and resolved.catalog is not None:
                return resolved.catalog.info
        return ModelInfo(
            id=self.identifier,
            name=self.identifier,
            adapter=self._safe_adapter_name() or "custom",
            license="custom",
            params="",
            architecture="custom ONNX",
            languages="unknown",
            status="custom",
            capabilities=self.capabilities,
        )

    # ------------------------------------------------------------- lifecycle
    def _resolve(self) -> ResolvedModel:
        with self._lock:
            if self._resolved is None:
                variant = self._variant
                if self._identifier is None and variant is None:
                    variant = self._store.current_variant()
                self._resolved = self._store.resolve(self._identifier, variant)
            return self._resolved

    def _ensure(self) -> tuple[OnnxEngine | None, Adapter]:
        with self._lock:
            if self._adapter is not None:
                return self._engine, self._adapter

            if self._injected_adapter is not None:
                adapter = self._injected_adapter
                if self._engine is None and self._identifier is not None and self._files_override:
                    resolved = self._resolve()
                    if resolved.path is not None:
                        name = self._adapter_name or getattr(adapter, "name", "custom")
                        validate_adapter_files(name, self._files_override)
                        self._engine = OnnxEngine(
                            resolved.path, self._files_override, threads=self._threads
                        )
                self._adapter = adapter
                log_event(
                    logger,
                    "model.load",
                    model=self.identifier,
                    variant=self.variant,
                    adapter=getattr(adapter, "name", type(adapter).__name__),
                    origin="injected",
                )
                return self._engine, self._adapter

            resolved = self._resolve()
            name = self._adapter_name or resolved.adapter
            if name is None and resolved.catalog is not None:
                name = resolved.catalog.adapter
            if name is None:
                raise AdapterNotAvailableError(
                    f"cannot infer an adapter for {self.identifier!r}; pass adapter=<name>"
                )
            files = self._files_override or dict(resolved.files)
            if resolved.path is None:
                if resolved.origin == "catalog":
                    model = resolved.catalog
                    variant = resolved.variant
                    hint = f"dtk models download {resolved.identifier}"
                    if variant is not None:
                        hint += f" --variant {variant.name}"
                    raise ModelNotDownloadedError(
                        f"{resolved.identifier!r} is not downloaded yet; run `{hint}` "
                        f"(status: {model.status if model else 'custom'})"
                    )
                raise AdapterNotAvailableError(
                    "custom ONNX models need a local directory plus adapter and files="
                )
            if not files:
                raise AdapterNotAvailableError(
                    "the model does not declare its files; pass files={'encoder': ..., 'tokenizer': ...}"
                )
            validate_adapter_files(name, files)

            if self._engine is None:
                self._engine = OnnxEngine(resolved.path, files, threads=self._threads)
            try:
                self._adapter = create_adapter(name, **self._adapter_options)
            except Exception:
                if self._engine is not None:
                    self._engine.close()
                    self._engine = None
                raise
            log_event(
                logger,
                "model.load",
                model=self.identifier,
                variant=self.variant,
                adapter=name,
                origin=resolved.origin,
            )
            return self._engine, self._adapter

    def warmup(self) -> DecisionModel:
        """Load the adapter (and, through it, the ONNX sessions) ahead of use."""
        with self._lock:
            engine, adapter = self._ensure()
            warmup = getattr(adapter, "warmup", None)
            if callable(warmup):
                warmup(engine)
        log_event(
            logger,
            "model.warmup",
            model=self.identifier,
            variant=self.variant,
            adapter=self.adapter_name,
        )
        return self

    def close(self) -> None:
        """Release ONNX sessions and the loaded adapter. Idempotent."""
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            self._adapter = None
        log_event(logger, "model.close", model=self.identifier, variant=self.variant)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------ inference
    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        strict: bool | None = None,
        request_id: str | None = None,
    ) -> Decision:
        """Answer typed questions about ``state`` (System One, in and out).

        ``strict`` overrides the constructor default.  ``request_id`` is
        propagated to structured events and to the returned ``Decision``.
        """
        started = time.perf_counter()
        rid = ensure_request_id(request_id)
        strict_mode = self._strict if strict is None else bool(strict)
        try:
            normalized = normalize_questions(questions)
            state_value = normalize_state(state)
            capabilities = self.capabilities
            self._validate_capabilities(state_value, normalized, capabilities)
            log_event(
                logger,
                "decision.start",
                request_id=rid,
                model=self.identifier,
                variant=self.variant,
                adapter=self._safe_adapter_name(),
                questions=len(normalized),
            )
            with self._lock:
                engine, adapter = self._ensure()
                raw = dict(adapter.system_one(engine, state_value, normalized))
                answers: dict[str, Answer] = wrap_answers(raw, normalized)
                usage = self._usage(engine, state_value, normalized, raw)
                latency_ms = (time.perf_counter() - started) * 1000.0
                warnings = tuple(
                    dict.fromkeys(
                        warning for answer in answers.values() for warning in answer.warnings
                    )
                )
                decision = Decision(
                    answers=answers,
                    model=self.identifier,
                    raw=raw,
                    variant=self.variant,
                    adapter=self.adapter_name,
                    usage=usage,
                    latency_ms=latency_ms,
                    warnings=warnings,
                    request_id=rid,
                    id=rid,
                )
                if strict_mode and decision.truncated:
                    raise TruncationError(
                        f"{self.identifier!r} truncated the input; "
                        + ("; ".join(warnings) if warnings else "no truncation details available")
                    )
            log_event(
                logger,
                "decision.end",
                request_id=rid,
                model=self.identifier,
                variant=self.variant,
                adapter=self._safe_adapter_name(),
                questions=len(normalized),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                usage_estimated=usage.estimated,
                latency_ms=round(latency_ms, 3),
                truncated=decision.truncated,
            )
            return decision
        except BaseException as error:
            log_event(
                logger,
                "decision.error",
                level=logging.ERROR,
                request_id=rid,
                model=self.identifier,
                variant=self.variant,
                adapter=self._safe_adapter_name(),
                error_code=getattr(error, "code", type(error).__name__),
                error=type(error).__name__,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            raise

    def classify(
        self,
        text: str,
        categories: Mapping[str, str] | Sequence[str],
        *,
        instructions: str | None = None,
        threshold: float = 0.5,
        request_id: str | None = None,
    ) -> Classification:
        """Pick the best-fitting label for ``text`` with a calibrated confidence."""
        if not isinstance(text, str) or not text.strip():
            raise InvalidRequestError("cannot classify an empty text")
        if isinstance(categories, Mapping):
            criteria = {str(label): str(description) for label, description in categories.items()}
        else:
            criteria = {str(label): str(label) for label in categories}
        if not criteria:
            raise InvalidRequestError("classify needs at least one category")

        question = {
            "type": "choice",
            "instructions": instructions or "Which category fits this text best?",
            "criteria": criteria,
        }
        decision = self.decide(text, {"label": question}, request_id=request_id)
        answer = decision["label"]
        scores = {label: float(p) for label, p in answer.probabilities.items()}
        if not scores:
            chosen = answer.choice
            confidence = answer.confidence
            if chosen is None or confidence is None:
                raise InvalidRequestError(
                    f"adapter returned no probabilities for choice: {answer.data!r}"
                )
            scores = {label: (confidence if label == chosen else 0.0) for label in criteria}

        label, confidence = max(scores.items(), key=lambda item: item[1])
        return Classification(
            label=label,
            confidence=confidence,
            below_threshold=confidence < threshold,
            scores=scores,
            model=decision.model,
            variant=decision.variant,
            usage=decision.usage,
            latency_ms=decision.latency_ms,
            warnings=decision.warnings,
            request_id=decision.request_id,
        )

    # ------------------------------------------------------------- batching
    def decide_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        strict: bool | None = None,
        request_id: str | None = None,
    ) -> list[Decision]:
        """Answer the same questions for many states, loading the model once.

        One ``questions`` mapping is reused for every state; a sequence must
        match the number of states one-to-one.  The model is loaded once; the
        current adapters execute sequentially, which is the stable contract
        even when an adapter later supports true ONNX batching.
        """
        if isinstance(states, (str, bytes)) or not isinstance(states, Iterable):
            raise InvalidRequestError("states must be an iterable of states, not a string")
        state_list = list(states)
        if isinstance(questions, Mapping):
            question_list: list[Mapping[str, Any]] = [questions] * len(state_list)
        else:
            question_list = list(questions)
            if len(question_list) != len(state_list):
                raise InvalidRequestError(
                    f"got {len(state_list)} states and {len(question_list)} question mappings"
                )
        rid = ensure_request_id(request_id)
        return [
            self.decide(
                state,
                question,
                strict=strict,
                request_id=f"{rid}:{index}",
            )
            for index, (state, question) in enumerate(zip(state_list, question_list))
        ]

    def classify_many(
        self,
        texts: Iterable[str],
        categories: Mapping[str, str] | Sequence[str],
        *,
        instructions: str | None = None,
        threshold: float = 0.5,
        request_id: str | None = None,
    ) -> list[Classification]:
        """One-label classification for many texts, loading the model once."""
        if isinstance(texts, (str, bytes)):
            raise InvalidRequestError("texts must be an iterable of strings, not a string")
        rid = ensure_request_id(request_id)
        return [
            self.classify(
                text,
                categories,
                instructions=instructions,
                threshold=threshold,
                request_id=f"{rid}:{index}",
            )
            for index, text in enumerate(texts)
        ]

    # ---------------------------------------------------------------- async
    async def adecide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        strict: bool | None = None,
        request_id: str | None = None,
    ) -> Decision:
        """Async wrapper around :meth:`decide` (runs in a worker thread)."""
        return await asyncio.to_thread(
            self.decide, state, questions, strict=strict, request_id=request_id
        )

    async def adecide_many(
        self,
        states: Iterable[Any],
        questions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        strict: bool | None = None,
        request_id: str | None = None,
    ) -> list[Decision]:
        """Async wrapper around :meth:`decide_many`."""
        return await asyncio.to_thread(
            self.decide_many, states, questions, strict=strict, request_id=request_id
        )

    async def aclassify(
        self,
        text: str,
        categories: Mapping[str, str] | Sequence[str],
        *,
        instructions: str | None = None,
        threshold: float = 0.5,
        request_id: str | None = None,
    ) -> Classification:
        """Async wrapper around :meth:`classify`."""
        return await asyncio.to_thread(
            self.classify,
            text,
            categories,
            instructions=instructions,
            threshold=threshold,
            request_id=request_id,
        )

    async def aclassify_many(
        self,
        texts: Iterable[str],
        categories: Mapping[str, str] | Sequence[str],
        *,
        instructions: str | None = None,
        threshold: float = 0.5,
        request_id: str | None = None,
    ) -> list[Classification]:
        """Async wrapper around :meth:`classify_many`."""
        return await asyncio.to_thread(
            self.classify_many,
            texts,
            categories,
            instructions=instructions,
            threshold=threshold,
            request_id=request_id,
        )

    # ---------------------------------------------------------- custom models
    @classmethod
    def from_onnx(
        cls,
        path: str | Path,
        *,
        adapter: Adapter | str,
        files: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> DecisionModel:
        """Open a local ONNX model with an explicit adapter and file-role map."""
        target = Path(path).expanduser()
        if not target.is_dir():
            raise AdapterNotAvailableError(f"custom ONNX model directory not found: {target}")
        if not files:
            raise AdapterNotAvailableError(
                "from_onnx needs files={'role': 'relative/path.onnx', ...}"
            )
        return cls(str(target), adapter=adapter, files=files, **kwargs)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _validate_capabilities(
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        capabilities: Capabilities,
    ) -> None:
        assets: Sequence[Any] = ()
        if isinstance(state, State):
            assets = state.assets
        elif isinstance(state, Mapping) and state.get("assets"):
            assets = state["assets"]  # type: ignore[assignment]
        if assets and not capabilities.multimodal:
            raise UnsupportedCapabilityError(
                "the selected model is text-only; it does not support state assets/images"
            )

        supported = set(capabilities.question_types)
        for name, question in questions.items():
            qtype = str(question.get("type"))
            if qtype not in supported:
                raise UnsupportedCapabilityError(
                    f"question {name!r} has type {qtype!r}, but the selected model supports "
                    f"{sorted(supported)}"
                )
            if question.get("multi_label") and not capabilities.multi_label:
                raise UnsupportedCapabilityError(
                    f"question {name!r} requests multi-label answers, which this model does "
                    "not declare"
                )
            if qtype == "choice":
                criteria = question.get("criteria") or {}
                if capabilities.max_options is not None and len(criteria) > capabilities.max_options:
                    raise UnsupportedCapabilityError(
                        f"question {name!r} has {len(criteria)} options; the model supports at "
                        f"most {capabilities.max_options}"
                    )
            elif qtype == "score":
                levels = question.get("criteria") or []
                if (
                    capabilities.max_score_levels is not None
                    and len(levels) > capabilities.max_score_levels
                ):
                    raise UnsupportedCapabilityError(
                        f"question {name!r} has {len(levels)} levels; the model supports at "
                        f"most {capabilities.max_score_levels}"
                    )

    @staticmethod
    def _usage(
        engine: OnnxEngine | None,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        answers: Mapping[str, Mapping[str, Any]],
    ) -> Usage:
        try:
            state_text = state_to_text(state)
        except Exception:  # noqa: BLE001 - usage must never break a successful decision
            state_text = ""
        questions_text = json.dumps(questions, ensure_ascii=False, sort_keys=True, default=str)
        answers_text = json.dumps(answers, ensure_ascii=False, sort_keys=True, default=str)

        tokenizer = None
        if engine is not None:
            try:
                tokenizer = engine.tokenizer
            except Exception:  # noqa: BLE001 - tokenizer is optional for accounting
                tokenizer = None

        if tokenizer is not None:
            try:
                input_tokens = len(
                    encode(tokenizer, f"{state_text}\n{questions_text}")
                )
                output_tokens = len(encode(tokenizer, answers_text))
                return Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated=True,
                )
            except Exception as error:  # noqa: BLE001 - fall back to a rough count
                logger.debug("token counting fallback: %s", error)

        input_tokens = len(f"{state_text} {questions_text}".split())
        output_tokens = len(answers_text.split()) if answers_text else 0
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens, estimated=True)


__all__ = ["DecisionModel"]
