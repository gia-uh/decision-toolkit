"""`decision_toolkit.decision_model` — the public, library-first surface.

``DecisionModel`` resolves a curated model (or a custom local directory), picks
the catalog's adapter, opens the one ONNX engine, and answers System One
requests: typed questions in, calibrated probabilities out, on CPU.

The CLI and the REST server are thin adapters over this class; nothing here
knows about argparse or FastAPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters import Adapter, create_adapter
from .engine import OnnxEngine
from .errors import AdapterNotAvailableError
from .models import ModelStore, ResolvedModel
from .results import Answer, Classification, Decision, wrap_answers
from .schema import Choice, normalize_questions

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
        catalog's ``default`` variant.
    files:
        Role -> file mapping for **custom** local models (``encoder``,
        ``classifier``, ``tokenizer``...). Catalog models come with their map.
    home / threads:
        Toolkit home override and ONNX Runtime thread count.
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
    ) -> None:
        self._identifier = identifier
        self._store = ModelStore(home=home)
        self._variant = variant
        self._files_override = dict(files or {})
        self._adapter: Adapter | None = adapter if adapter is not None and not isinstance(adapter, str) else None
        self._adapter_name: str | None = adapter if isinstance(adapter, str) else None
        self._adapter_options = dict(adapter_options or {})
        self._threads = int(threads)
        self._resolved: ResolvedModel | None = None
        self._engine: OnnxEngine | None = None

    # ------------------------------------------------------------------ info
    @property
    def identifier(self) -> str:
        return self._identifier or self._store.current() or "(none)"

    @property
    def variant(self) -> str | None:
        if self._resolved is not None and self._resolved.variant is not None:
            return self._resolved.variant.name
        return self._variant

    @property
    def adapter_name(self) -> str:
        if self._adapter is not None:
            return getattr(self._adapter, "name", type(self._adapter).__name__)
        if self._adapter_name:
            return self._adapter_name
        resolved = self._resolve()
        if resolved.catalog is not None:
            return resolved.catalog.adapter
        raise AdapterNotAvailableError(
            f"cannot infer an adapter for {self.identifier!r}; pass adapter=<name>"
        )

    @property
    def resolved(self) -> ResolvedModel:
        return self._resolve()

    # ------------------------------------------------------------- lifecycle
    def _resolve(self) -> ResolvedModel:
        if self._resolved is None:
            self._resolved = self._store.resolve(self._identifier, self._variant)
        return self._resolved

    def _ensure(self) -> tuple[OnnxEngine | None, Adapter]:
        if self._adapter is not None:
            return self._engine, self._adapter

        resolved = self._resolve()
        name = self._adapter_name or (resolved.catalog.adapter if resolved.catalog else None)
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
                raise AdapterNotAvailableError(
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

        if self._engine is None:
            self._engine = OnnxEngine(resolved.path, files, threads=self._threads)
        logger.info(
            "loading model=%s variant=%s adapter=%s", self.identifier, self.variant, name
        )
        self._adapter = create_adapter(name, **self._adapter_options)
        return self._engine, self._adapter

    def warmup(self) -> None:
        """Load the adapter (and, through it, the ONNX sessions) ahead of use."""
        engine, adapter = self._ensure()
        warmup = getattr(adapter, "warmup", None)
        if callable(warmup):
            warmup(engine)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._adapter = None

    # ------------------------------------------------------------ inference
    def decide(self, state: Any, questions: Mapping[str, Any]) -> Decision:
        """Answer typed questions about ``state`` (System One, in and out)."""
        normalized = normalize_questions(questions)
        engine, adapter = self._ensure()
        raw_answers = dict(adapter.system_one(engine, state, normalized))
        answers: dict[str, Answer] = wrap_answers(raw_answers, normalized)
        return Decision(answers=answers, model=self.identifier, raw=raw_answers)

    def classify(
        self,
        text: str,
        categories: Mapping[str, str] | Sequence[str],
        *,
        instructions: str | None = None,
        threshold: float = 0.5,
    ) -> Classification:
        """Pick the best-fitting label for ``text`` with a calibrated confidence."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("cannot classify an empty text")
        if isinstance(categories, Mapping):
            criteria = {str(label): str(description) for label, description in categories.items()}
        else:
            criteria = {str(label): str(label) for label in categories}
        if not criteria:
            raise ValueError("classify needs at least one category")

        question = Choice(
            instructions=instructions or "Which category fits this text best?",
            criteria=criteria,
        )
        decision = self.decide(text, {"label": question})
        answer = decision["label"]
        scores = {label: float(p) for label, p in answer.probabilities.items()}
        if not scores:
            chosen = answer.choice
            confidence = answer.confidence
            if chosen is None or confidence is None:
                raise ValueError(f"adapter returned no probabilities for choice: {answer.data!r}")
            scores = {label: (confidence if label == chosen else 0.0) for label in criteria}

        label, confidence = max(scores.items(), key=lambda item: item[1])
        return Classification(
            label=label,
            confidence=confidence,
            below_threshold=confidence < threshold,
            scores=scores,
        )


__all__ = ["DecisionModel"]
