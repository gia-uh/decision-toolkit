"""`decision_toolkit.models` — the toolkit's own model cache.

Everything the toolkit runs lives here, or is a local path you provide. The
HuggingFace cache is unrelated: this toolkit never reads it. The only network
access in the whole toolkit is :meth:`ModelStore.download`.

A downloaded model is stored **per variant**, so quantizations coexist::

    <home>/models/
        laya-multilingual@q8/
            model.json          # identifier, variant, repo, role -> file map
            encoder.onnx ...    # only the files the variant declares

``config.json`` is created on first use with the defaults already written.
There are no environment variables.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from . import catalog
from .config import Config, load_config, save_config
from .errors import ModelNotFoundError

logger = logging.getLogger(__name__)

_ENTRY_META = "model.json"


@dataclass(frozen=True)
class ModelEntry:
    """A model variant stored in the toolkit cache."""

    identifier: str
    variant: str
    path: Path
    repo: str
    files: Mapping[str, str]
    source: str
    origin: str
    created_at: str
    size_bytes: int


@dataclass(frozen=True)
class ResolvedModel:
    """What the engine needs: a local directory and the role -> file mapping."""

    identifier: str
    path: Path | None
    catalog: catalog.CatalogModel | None
    variant: catalog.Variant | None
    files: Mapping[str, str]
    origin: str  # "cache" | "catalog" | "path"


def _dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _dirname(identifier: str, variant: str) -> str:
    return f"{identifier}@{variant}"


def _snapshot_download(
    repo_id: str,
    local_dir: Path,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Download only the declared files into ``local_dir`` (isolated for tests)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        revision=revision,
        allow_patterns=allow_patterns,
    )


class ModelStore:
    """The toolkit's model cache: list, download, import, use, delete, resolve."""

    def __init__(self, home: str | Path | None = None) -> None:
        self.config: Config = load_config(home)

    # ------------------------------------------------------------------ disk
    @property
    def models_dir(self) -> Path:
        return self.config.cache_dir

    def entries(self) -> list[ModelEntry]:
        """Every downloaded model variant in the toolkit cache."""
        if not self.models_dir.is_dir():
            return []
        found: list[ModelEntry] = []
        for directory in sorted(self.models_dir.iterdir()):
            meta_path = directory / _ENTRY_META
            if not directory.is_dir() or not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            found.append(
                ModelEntry(
                    identifier=meta["identifier"],
                    variant=meta.get("variant", "default"),
                    path=directory,
                    repo=meta.get("repo", ""),
                    files=dict(meta.get("files") or {}),
                    source=meta.get("source", "download"),
                    origin=meta.get("origin", "?"),
                    created_at=meta.get("created_at", "?"),
                    size_bytes=_dir_size(directory),
                )
            )
        return found

    def get(self, identifier: str, variant: str | None = None) -> ModelEntry | None:
        """The cache entry for ``identifier`` (optionally a specific variant)."""
        candidates = [entry for entry in self.entries() if entry.identifier == identifier]
        if variant is not None:
            candidates = [entry for entry in candidates if entry.variant == variant]
        if not candidates:
            return None
        # prefer the catalog default when several variants are cached
        model = catalog.get(identifier)
        if model is not None and model.default_variant is not None:
            for entry in candidates:
                if entry.variant == model.default_variant.name:
                    return entry
        return candidates[0]

    def delete(self, identifier: str, variant: str | None = None) -> None:
        """Delete a cached variant; refuses the model in use."""
        entry = self.get(identifier, variant)
        if entry is None:
            raise ModelNotFoundError(f"{identifier!r} is not in the toolkit cache")
        if self.config.current == identifier:
            raise ModelNotFoundError(
                f"{identifier!r} is the model in use; run `dtk models use <other>` first"
            )
        shutil.rmtree(entry.path)
        logger.info("deleted %s (%d bytes)", entry.path, entry.size_bytes)

    # -------------------------------------------------------------- network
    def download(
        self, identifier: str, variant: str | None = None, revision: str | None = None
    ) -> ModelEntry:
        """Download one variant of a catalog model (the only network command)."""
        model = catalog.get(identifier)
        if model is None:
            raise ModelNotFoundError(
                f"{identifier!r} is not a catalog model; the toolkit only downloads curated "
                "models (custom ONNX files: import them or point at a local directory)"
            )
        chosen = catalog.select_variant(model, variant)
        local_dir = self.models_dir / _dirname(identifier, chosen.name)
        if (local_dir / _ENTRY_META).is_file():
            raise ModelNotFoundError(
                f"{identifier}@{chosen.name} is already in the cache ({local_dir}); "
                "delete it first to re-download"
            )

        allow_patterns = list(chosen.files.values()) + list(chosen.extra_files)
        logger.info(
            "downloading %s@%s (%s, %.1f MB) from %s",
            identifier, chosen.name, chosen.quant, chosen.size_mb, chosen.repo,
        )
        _snapshot_download(
            chosen.repo, local_dir, revision=revision or chosen.revision,
            allow_patterns=allow_patterns,
        )

        meta = {
            "identifier": identifier,
            "variant": chosen.name,
            "repo": chosen.repo,
            "files": dict(chosen.files),
            "extra_files": list(chosen.extra_files),
            "source": "download",
            "origin": f"huggingface:{chosen.repo}@{revision or chosen.revision or 'main'}",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (local_dir / _ENTRY_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        size = _dir_size(local_dir)
        logger.info("downloaded %s@%s into %s (%.1f MB)", identifier, chosen.name, local_dir, size / 1e6)
        return self.get(identifier, chosen.name)  # type: ignore[return-value]

    def import_model(self, path: str | Path, name: str | None = None) -> ModelEntry:
        """Copy a local model directory into the toolkit cache (custom models)."""
        source = Path(path).expanduser().resolve()
        if not source.is_dir():
            raise ModelNotFoundError(f"local model path not found: {source}")
        identifier = name or source.name
        local_dir = self.models_dir / _dirname(identifier, "local")
        if local_dir.exists():
            raise ModelNotFoundError(f"{identifier}@local is already in the toolkit cache")
        shutil.copytree(source, local_dir)
        meta = {
            "identifier": identifier,
            "variant": "local",
            "repo": "",
            "files": {},
            "extra_files": [],
            "source": "import",
            "origin": str(source),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (local_dir / _ENTRY_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return self.get(identifier, "local")  # type: ignore[return-value]

    # ------------------------------------------------------------------ use
    def use(self, identifier: str, variant: str | None = None) -> None:
        """Set the model in use (persistent)."""
        if catalog.get(identifier) is None and self.get(identifier, variant) is None and "/" not in identifier:
            raise ModelNotFoundError(
                f"{identifier!r} is neither in the toolkit cache nor the catalog; "
                "run `dtk models list` / `dtk models download <id>`"
            )
        self.config.current = identifier
        save_config(self.config)

    def current(self) -> str | None:
        return self.config.current

    # -------------------------------------------------------------- resolve
    def resolve(self, identifier: str | None = None, variant: str | None = None) -> ResolvedModel:
        """Resolve an identifier into the directory and file map an adapter needs."""
        if identifier is None:
            identifier = self.config.current
        if identifier is None:
            raise ModelNotFoundError(
                "no model given and none in use; pass a model id or run `dtk models use <id>`"
            )

        # 1. a local path (custom model; the caller supplies adapter + files)
        candidate = Path(identifier).expanduser()
        if candidate.exists():
            return ResolvedModel(
                identifier=identifier,
                path=candidate.resolve(),
                catalog=catalog.get(identifier),
                variant=None,
                files={},
                origin="path",
            )

        # 2. the toolkit cache
        entry = self.get(identifier, variant)
        if entry is not None:
            model = catalog.get(identifier)
            chosen = None
            if model is not None:
                try:
                    chosen = catalog.select_variant(model, entry.variant)
                except ModelNotFoundError:
                    chosen = None
            return ResolvedModel(
                identifier=identifier,
                path=entry.path,
                catalog=model,
                variant=chosen,
                files=entry.files,
                origin="cache",
            )

        # 3. the catalog
        model = catalog.get(identifier)
        if model is not None:
            chosen = None
            if model.variants:
                chosen = catalog.select_variant(model, variant)
            return ResolvedModel(
                identifier=identifier,
                path=None,
                catalog=model,
                variant=chosen,
                files=dict(chosen.files) if chosen else {},
                origin="catalog",
            )

        # 4. a bare hub reference (custom; the caller supplies adapter + files)
        repo = identifier[3:] if identifier.startswith("hf:") else identifier
        if "/" in repo:
            return ResolvedModel(
                identifier=identifier,
                path=None,
                catalog=None,
                variant=None,
                files={},
                origin="catalog",
            )

        suggestions = ", ".join(
            model.id for model in catalog.CATALOG if model.status in {"m0", "m1"}
        )
        raise ModelNotFoundError(
            f"unknown model {identifier!r}. Runnable today: {suggestions or '(none yet)'}. "
            "See `dtk models list` for the full catalog."
        )


__all__ = ["ModelStore", "ModelEntry", "ResolvedModel"]
