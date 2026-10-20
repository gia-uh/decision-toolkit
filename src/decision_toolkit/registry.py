"""`decision_toolkit.registry` — the server's lazy multi-model registry.

One REST process can serve the whole catalog. The registry resolves a model id
and variant per request, keeps a small LRU of loaded ``DecisionModel``
instances, and reports readiness based on what is actually warm. It contains no
HTTP or wire-contract code, so it is reusable and testable on its own.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decision_model import DecisionModel
from .errors import ModelNotFoundError
from .models import ModelStore
from .observability import log_event

logger = logging.getLogger(__name__)


@dataclass
class _Slot:
    model: DecisionModel
    in_use: int = 0


class ModelRegistry:
    """Lazy, process-local registry of decision models.

    ``model`` injects a fixed model (tests, embedding); otherwise the registry
    creates ``DecisionModel`` instances on demand from the toolkit cache.
    ``max_loaded`` bounds how many idle models remain loaded; a model in use is
    never evicted.
    """

    def __init__(
        self,
        *,
        model: DecisionModel | None = None,
        model_id: str | None = None,
        variant: str | None = None,
        home: str | Path | None = None,
        threads: int = 4,
        max_loaded: int = 2,
        adapter: Any = None,
        files: Mapping[str, str] | None = None,
    ) -> None:
        self._fixed = model
        self._model_id = model_id
        self._variant = variant
        self._home = home
        self._threads = int(threads)
        self._max_loaded = max(1, int(max_loaded))
        self._adapter = adapter
        self._files = dict(files or {}) or None
        self._store = ModelStore(home=home)
        self._slots: OrderedDict[str, _Slot] = OrderedDict()
        self._lock = threading.RLock()
        self._warmup_error: str | None = None

    # ------------------------------------------------------------------ info
    @property
    def store(self) -> ModelStore:
        return self._store

    @property
    def default_identifier(self) -> str | None:
        if self._fixed is not None:
            return self._fixed.identifier
        return self._model_id or self._store.current()

    @property
    def default_variant(self) -> str | None:
        if self._fixed is not None:
            return self._fixed.variant
        if self._variant is not None:
            return self._variant
        if self._model_id is None:
            return self._store.current_variant()
        return None

    @property
    def warmup_error(self) -> str | None:
        return self._warmup_error

    @property
    def fixed(self) -> bool:
        return self._fixed is not None

    @property
    def loaded_count(self) -> int:
        with self._lock:
            loaded = 1 if self._fixed is not None and self._fixed.is_loaded else 0
            return loaded + sum(1 for slot in self._slots.values() if slot.model.is_loaded)

    @staticmethod
    def _key(identifier: str, variant: str | None) -> str:
        return f"{identifier}@{variant or ''}"

    def describe(self) -> dict[str, object]:
        with self._lock:
            return {
                "default_model": self.default_identifier,
                "default_variant": self.default_variant,
                "max_loaded": self._max_loaded,
                "loaded": [
                    {
                        "key": key,
                        "model": slot.model.identifier,
                        "variant": slot.model.variant,
                        "in_use": slot.in_use,
                        "loaded": slot.model.is_loaded,
                    }
                    for key, slot in self._slots.items()
                ],
            }

    # ------------------------------------------------------------ selection
    def _fixed_or_error(self, model_id: str | None, variant: str | None) -> DecisionModel | None:
        if self._fixed is None:
            return None
        fixed_id = self._fixed.identifier
        if model_id and model_id != fixed_id:
            raise ModelNotFoundError(
                f"this server is bound to {fixed_id!r}; requested model {model_id!r}"
            )
        if variant and self._fixed.variant and variant != self._fixed.variant:
            raise ModelNotFoundError(
                f"this server is bound to variant {self._fixed.variant!r}; "
                f"requested {variant!r}"
            )
        return self._fixed

    def _new_slot(self, identifier: str, variant: str | None) -> _Slot:
        return _Slot(
            DecisionModel(
                identifier,
                variant=variant,
                home=self._home,
                threads=self._threads,
                adapter=self._adapter,
                files=self._files,
            )
        )

    def _get_slot(self, model_id: str | None, variant: str | None) -> _Slot:
        fixed = self._fixed_or_error(model_id, variant)
        if fixed is not None:
            return _Slot(fixed, in_use=1)

        identifier = model_id or self.default_identifier
        if not identifier:
            raise ModelNotFoundError(
                "no model configured; pass `model` or set one with `dtk models use`"
            )
        effective_variant = variant
        if effective_variant is None and identifier == self.default_identifier:
            effective_variant = self.default_variant
        key = self._key(identifier, effective_variant)

        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = self._new_slot(identifier, effective_variant)
                self._slots[key] = slot
                log_event(
                    logger,
                    "registry.add",
                    model=identifier,
                    variant=effective_variant,
                    loaded=len(self._slots),
                )
                self._evict_locked()
            else:
                self._slots.move_to_end(key)
            slot.in_use += 1
            return slot

    def _evict_locked(self) -> None:
        while len(self._slots) > self._max_loaded:
            victim_key = next(
                (key for key, slot in self._slots.items() if slot.in_use == 0),
                None,
            )
            if victim_key is None:
                logger.warning(
                    "registry at max_loaded=%d but every model is in use; keeping all",
                    self._max_loaded,
                )
                return
            victim = self._slots.pop(victim_key)
            victim.model.close()
            log_event(
                logger,
                "registry.evict",
                model=victim.model.identifier,
                variant=victim.model.variant,
            )

    @contextmanager
    def acquire(
        self,
        model_id: str | None = None,
        variant: str | None = None,
    ) -> Iterator[DecisionModel]:
        """Acquire a model for one request and release it afterwards."""
        slot = self._get_slot(model_id, variant)
        try:
            yield slot.model
        finally:
            if self._fixed is None:
                with self._lock:
                    slot.in_use = max(0, slot.in_use - 1)
                    self._evict_locked()

    # -------------------------------------------------------------- readiness
    def is_ready(self) -> tuple[bool, dict[str, object]]:
        if self._fixed is not None:
            return True, {
                "model": self._fixed.identifier,
                "variant": self._fixed.variant,
                "adapter": self._fixed.adapter_name,
                "loaded": self._fixed.is_loaded,
            }
        identifier = self.default_identifier
        if not identifier:
            return False, {"detail": "no model configured"}
        variant = self.default_variant
        key = self._key(identifier, variant)
        with self._lock:
            slot = self._slots.get(key)
            loaded = bool(slot and slot.model.is_loaded)
        if not loaded:
            detail = "model is configured but not warm"
            if self._warmup_error:
                detail = f"model warmup failed: {self._warmup_error}"
            return False, {
                "model": identifier,
                "variant": variant,
                "detail": detail,
            }
        return True, {
            "model": identifier,
            "variant": variant,
            "adapter": slot.model.adapter_name,  # type: ignore[union-attr]
            "loaded": True,
        }

    def warmup_default(self) -> bool:
        """Warm the configured default model; record the failure for /ready."""
        try:
            with self.acquire() as model:
                model.warmup()
            self._warmup_error = None
            return True
        except Exception as error:  # noqa: BLE001 - readiness must report, not crash
            self._warmup_error = str(error)
            log_event(
                logger,
                "registry.warmup_error",
                level=logging.ERROR,
                error_code=getattr(error, "code", type(error).__name__),
                error=type(error).__name__,
            )
            return False

    def close(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                slot.model.close()
            self._slots.clear()
            if self._fixed is not None:
                self._fixed.close()


__all__ = ["ModelRegistry"]
