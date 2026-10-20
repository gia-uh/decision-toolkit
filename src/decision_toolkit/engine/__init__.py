"""`decision_toolkit.engine` — the one ONNX runtime.

The engine owns what is common to every ONNX model: opening `onnxruntime`
sessions and loading the tokenizer. It knows **nothing** about prompts,
markers, labels or probability readouts — that belongs to the adapters.

Files are declared by role (``encoder``, ``classifier``, ``tokenizer``...), so
an adapter asks for ``engine.session("encoder")`` and never hardcodes claim
paths.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..errors import AdapterError, AdapterNotAvailableError


class OnnxEngine:
    """Sessions + tokenizer for one downloaded model variant."""

    def __init__(
        self,
        model_dir: str | Path,
        files: Mapping[str, str],
        *,
        threads: int = 4,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.files = dict(files)
        self.threads = int(threads)
        self._sessions: dict[str, Any] = {}
        self._tokenizer: Any | None = None

    # ------------------------------------------------------------------ files
    def path(self, role: str) -> Path:
        """The local path of the file declared for ``role``."""
        relative = self.files.get(role)
        if not relative:
            raise AdapterError(f"the variant does not declare a {role!r} file")
        path = self.model_dir / relative
        if not path.is_file():
            raise AdapterError(f"file for {role!r} is missing: {path}")
        return path

    def read_json(self, role: str) -> dict[str, Any]:
        return json.loads(self.path(role).read_text(encoding="utf-8"))

    # --------------------------------------------------------------- runtime
    def session(self, role: str) -> Any:
        """Open (once) the onnxruntime session for ``role``."""
        if role not in self._sessions:
            self._sessions[role] = open_session(self.path(role), self.threads)
        return self._sessions[role]

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.path("tokenizer"))
        return self._tokenizer

    def close(self) -> None:
        self._sessions.clear()
        self._tokenizer = None


def open_session(path: str | Path, threads: int = 4) -> Any:
    """Open an ONNX Runtime session on CPU (lazy import)."""
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - depends on environment
        raise AdapterNotAvailableError(
            'the ONNX engine needs onnxruntime: pip install "decision-toolkit[onnx]"'
        ) from error

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def load_tokenizer(path: str | Path) -> Any:
    """Load a tokenizer.json, preferring the transformers wrapper when present."""
    target = Path(path)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(target.parent))
    except Exception:  # noqa: BLE001, S110 - fall back to the tokenizers runtime
        pass
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(target))
    except ImportError as error:  # pragma: no cover - depends on environment
        raise AdapterNotAvailableError(
            'the ONNX engine needs a tokenizer runtime: pip install "decision-toolkit[onnx]"'
        ) from error


def encode(tokenizer: Any, text: str) -> list[int]:
    """Tokenize without special tokens, accepting transformers or tokenizers."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    return [int(token) for token in ids]


__all__ = ["OnnxEngine", "encode", "load_tokenizer", "open_session"]
