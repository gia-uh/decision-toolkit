"""`decision_toolkit.backends.julia` — Supersonic Labs' Julia 1, in-process on CPU.

Julia 1 (144M, mmBERT-small) ships its own Python runtime inside the model
repository:

    python -c "from huggingface_hub import snapshot_download; snapshot_download('SupersonicLabs/Julia-1', local_dir='Julia-1')"
    python -m pip install -e ./Julia-1

which provides ``julia.load_model(...)`` and ``engine.predict(state=..., questions=...)``.
The runtime module is named ``julia`` and may collide with PyJulia, so this
backend checks for the right entry points and explains the situation when the
wrong package is installed.

The constructor accepts an injected ``engine`` so tests never import the
runtime:

>>> class FakeEngine:
...     def predict(self, state, questions): return {"answers": {"q": {"type": "noul", "noul": 0.9}}}
>>> backend = JuliaBackend(resolved, engine=FakeEngine())
"""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import BackendError, BackendNotAvailableError
from ..models import ResolvedModel
from .base import BaseBackend

_INSTALL_HINT = (
    "install Julia 1's official runtime from its model repository:\n"
    "  dtk models download julia-1\n"
    "  python -m pip install -e <toolkit-cache>/models/SupersonicLabs--Julia-1"
)


class JuliaBackend(BaseBackend):
    """System One over Julia 1's official runtime."""

    name = "julia"

    def __init__(
        self,
        resolved: ResolvedModel,
        *,
        device: str | None = None,
        engine: Any | None = None,
        strict_encoding: bool = True,
        max_length: int = 8192,
        head_length: int = 512,
        **options: Any,
    ) -> None:
        super().__init__(resolved, device=device, **options)
        self._engine = engine
        self._strict_encoding = strict_encoding
        self._max_length = max_length
        self._head_length = head_length

    # ------------------------------------------------------------------ load
    def _load(self) -> Any:
        if self._engine is None:
            try:
                import julia  # type: ignore[import-not-found]
            except ImportError as error:  # pragma: no cover - depends on environment
                raise BackendNotAvailableError(
                    f"the Julia 1 runtime is not importable.\n{_INSTALL_HINT}"
                ) from error

            if not hasattr(julia, "load_model"):
                raise BackendNotAvailableError(
                    "`import julia` resolved to a different package (PyJulia?), not Julia 1's "
                    f"runtime. Remove it from this environment and install the model runtime.\n{_INSTALL_HINT}"
                )

            if self._resolved.path is None:
                raise BackendNotAvailableError(
                    "Julia 1 needs a local model directory (its runtime is installed from the "
                    f"repository itself).\n{_INSTALL_HINT}"
                )
            device = "cpu" if self._device in (None, "auto") else self._device
            self._engine = julia.load_model(
                str(self._resolved.path),
                device=device,
                strict_encoding=self._strict_encoding,
                max_length=self._max_length,
                head_length=self._head_length,
            )
        return self._engine

    def warmup(self) -> None:
        self._load()

    # ------------------------------------------------------------- inference
    def system_one(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        engine = self._load()
        result = engine.predict(state=state, questions=dict(questions))
        if not isinstance(result, Mapping) or "answers" not in result:
            raise BackendError(f"unexpected Julia response shape: {type(result).__name__}")
        return dict(result["answers"])


__all__ = ["JuliaBackend"]
