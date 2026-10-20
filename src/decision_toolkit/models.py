"""`decision_toolkit.models` — the toolkit's own model cache.

Everything the toolkit runs lives here, or is a local path you provide. The
HuggingFace cache is unrelated: this toolkit never reads it. The only network
access in the whole toolkit is :meth:`ModelStore.download`.

A downloaded model is stored **per variant**, so quantizations coexist::

    <home>/models/
        laya@q8/
            model.json          # identifier, variant, repo, role -> file map
            encoder.onnx ...    # only the files the variant declares

``config.json`` is created on first use with the defaults already written.
There are no environment variables.

Phase 1 hardening adds: a process-local download lock, an atomic
download-then-rename flow (a failed download never leaves a half-usable cache
entry), gzip decompression for ``.gz`` external-data files, SHA256 recording
and verification, and custom imports that can carry an adapter and file map.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import catalog
from .config import Config, load_config, save_config
from .errors import ModelNotFoundError

logger = logging.getLogger(__name__)

_ENTRY_META = "model.json"
_DOWNLOAD_LOCK = threading.RLock()


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
    adapter: str | None = None
    sha256: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedModel:
    """What the engine needs: a local directory and the role -> file mapping."""

    identifier: str
    path: Path | None
    catalog: catalog.CatalogModel | None
    variant: catalog.Variant | None
    files: Mapping[str, str]
    origin: str  # "cache" | "catalog" | "path"
    adapter: str | None = None


def _dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _dirname(identifier: str, variant: str) -> str:
    return f"{identifier}@{variant}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gunzip_tree(root: Path) -> None:
    """Decompress any ``*.gz`` file under ``root`` and remove the archive."""
    for archive in sorted(root.rglob("*.gz")):
        target = archive.with_name(archive.name[:-3])
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive, "rb") as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        archive.unlink()


def _strip_gz(path: str) -> str:
    return path.removesuffix(".gz")


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
            if directory.name.startswith("."):
                continue  # an in-flight atomic download, never a cache entry
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
                    adapter=meta.get("adapter"),
                    sha256=dict(meta.get("sha256") or {}),
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
        """Delete a cached variant; refuses the variant in use."""
        entry = self.get(identifier, variant)
        if entry is None:
            raise ModelNotFoundError(f"{identifier!r} is not in the toolkit cache")
        if self.config.current == identifier and (
            self.config.current_variant is None or entry.variant == self.config.current_variant
        ):
            raise ModelNotFoundError(
                f"{identifier!r} is the model in use; run `dtk models use <other>` first"
            )
        shutil.rmtree(entry.path)
        logger.info("deleted %s (%d bytes)", entry.path, entry.size_bytes)

    def verify(self, identifier: str, variant: str | None = None) -> dict[str, Any]:
        """Check a cached variant: files, recorded SHA256 and, if available, ONNX sessions."""
        entry = self.get(identifier, variant)
        if entry is None:
            raise ModelNotFoundError(f"{identifier!r} is not in the toolkit cache")
        missing = [path for path in entry.files.values() if not (entry.path / path).is_file()]
        hashes: dict[str, dict[str, Any]] = {}
        for role, expected in entry.sha256.items():
            relative = entry.files.get(role)
            if relative is None:
                hashes[role] = {"expected": expected, "actual": None, "ok": False}
                continue
            target = entry.path / relative
            actual = _sha256(target) if target.is_file() else None
            hashes[role] = {
                "expected": expected,
                "actual": actual,
                "ok": actual == expected,
            }
        report: dict[str, Any] = {
            "identifier": entry.identifier,
            "variant": entry.variant,
            "files": dict(entry.files),
            "missing": missing,
            "hashes": hashes,
            "sessions": {},
            "ok": not missing and all(item["ok"] for item in hashes.values()),
        }
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            report["note"] = "onnxruntime not installed; sessions not checked"
            return report

        from .engine import open_session

        for role, relative in entry.files.items():
            if not relative.endswith(".onnx"):
                continue
            try:
                open_session(entry.path / relative, threads=1)
                report["sessions"][role] = "ok"
            except Exception as error:  # noqa: BLE001 - report the real runtime error
                report["sessions"][role] = f"error: {error}"
                report["ok"] = False
        return report

    # -------------------------------------------------------------- network
    def download(
        self,
        identifier: str,
        variant: str | None = None,
        revision: str | None = None,
        *,
        force: bool = False,
    ) -> ModelEntry:
        """Download one variant of a catalog model (the only network command).

        The download happens in a temporary sibling directory and is renamed
        into place only after the metadata is written, so an interrupted or
        failed download never appears as a usable cache entry.
        """
        model = catalog.get(identifier)
        if model is None:
            raise ModelNotFoundError(
                f"{identifier!r} is not a catalog model; the toolkit only downloads curated "
                "models (custom ONNX files: import them or point at a local directory)"
            )
        chosen = catalog.select_variant(model, variant)
        final_dir = self.models_dir / _dirname(identifier, chosen.name)

        with _DOWNLOAD_LOCK:
            if final_dir.exists() and not force:
                raise ModelNotFoundError(
                    f"{identifier}@{chosen.name} is already in the cache ({final_dir}); "
                    "delete it first or pass force=True to re-download"
                )
            partial = final_dir.with_name(
                f".{final_dir.name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            shutil.rmtree(partial, ignore_errors=True)

            allow_patterns = list(chosen.files.values()) + list(chosen.extra_files)
            logger.info(
                "downloading %s@%s (%s, %.1f MB) from %s",
                identifier, chosen.name, chosen.quant, chosen.size_mb, chosen.repo,
            )
            try:
                _snapshot_download(
                    chosen.repo,
                    partial,
                    revision=revision or chosen.revision,
                    allow_patterns=allow_patterns,
                )
                _gunzip_tree(partial)

                files = {role: _strip_gz(relative) for role, relative in chosen.files.items()}
                extra_files = [_strip_gz(relative) for relative in chosen.extra_files]
                sha256 = {
                    role: _sha256(partial / relative)
                    for role, relative in files.items()
                    if (partial / relative).is_file()
                }
                meta = {
                    "identifier": identifier,
                    "variant": chosen.name,
                    "repo": chosen.repo,
                    "files": files,
                    "extra_files": extra_files,
                    "sha256": sha256,
                    "adapter": model.adapter,
                    "source": "download",
                    "origin": f"huggingface:{chosen.repo}@{revision or chosen.revision or 'main'}",
                    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                (partial / _ENTRY_META).write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                partial.rename(final_dir)
            except Exception:
                shutil.rmtree(partial, ignore_errors=True)
                raise

            size = _dir_size(final_dir)
            logger.info(
                "downloaded %s@%s into %s (%.1f MB)",
                identifier, chosen.name, final_dir, size / 1e6,
            )
            return self.get(identifier, chosen.name)  # type: ignore[return-value]

    def import_model(
        self,
        path: str | Path,
        name: str | None = None,
        *,
        adapter: str | None = None,
        files: Mapping[str, str] | None = None,
    ) -> ModelEntry:
        """Copy a local model directory into the toolkit cache (custom models).

        ``adapter`` and ``files`` make the imported model directly runnable by
        ``DecisionModel`` without repeating the role map at every call.
        """
        source = Path(path).expanduser().resolve()
        if not source.is_dir():
            raise ModelNotFoundError(f"local model path not found: {source}")
        identifier = name or source.name
        local_dir = self.models_dir / _dirname(identifier, "local")
        if local_dir.exists():
            raise ModelNotFoundError(f"{identifier}@local is already in the toolkit cache")
        role_map = dict(files or {})
        for role, relative in role_map.items():
            if not isinstance(role, str) or not role.strip():
                raise ModelNotFoundError("import files need non-empty role names")
            if not isinstance(relative, str) or not relative.strip():
                raise ModelNotFoundError(f"import file for {role!r} needs a non-empty path")
        partial = local_dir.with_name(f".{local_dir.name}.partial-{os.getpid()}")
        shutil.rmtree(partial, ignore_errors=True)
        try:
            shutil.copytree(source, partial)
            sha256 = {
                role: _sha256(partial / relative)
                for role, relative in role_map.items()
                if (partial / relative).is_file()
            }
            meta = {
                "identifier": identifier,
                "variant": "local",
                "repo": "",
                "files": role_map,
                "extra_files": [],
                "sha256": sha256,
                "adapter": adapter,
                "source": "import",
                "origin": str(source),
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            (partial / _ENTRY_META).write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            partial.rename(local_dir)
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return self.get(identifier, "local")  # type: ignore[return-value]

    # ------------------------------------------------------------------ use
    def use(self, identifier: str, variant: str | None = None) -> None:
        """Set the model (and optional variant) in use, persistently."""
        model = catalog.get(identifier)
        cached = self.get(identifier, variant)
        if model is None and cached is None and "/" not in identifier:
            raise ModelNotFoundError(
                f"{identifier!r} is neither in the toolkit cache nor the catalog; "
                "run `dtk models list` / `dtk models download <id>`"
            )
        if variant is not None:
            if model is not None:
                catalog.select_variant(model, variant)
            elif cached is None:
                raise ModelNotFoundError(
                    f"{identifier!r} is not cached with variant {variant!r}"
                )
        self.config.current = identifier
        self.config.current_variant = variant
        save_config(self.config)

    def current(self) -> str | None:
        return self.config.current

    def current_variant(self) -> str | None:
        return self.config.current_variant

    # -------------------------------------------------------------- resolve
    def resolve(self, identifier: str | None = None, variant: str | None = None) -> ResolvedModel:
        """Resolve an identifier into the directory and file map an adapter needs."""
        if identifier is None:
            identifier = self.config.current
            if variant is None:
                variant = self.config.current_variant
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
                adapter=entry.adapter or (model.adapter if model is not None else None),
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
                adapter=model.adapter,
            )

        # 4. a bare hub reference (custom; the caller supplies adapter + files)
        repo = identifier.removeprefix("hf:")
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


__all__ = ["ModelEntry", "ModelStore", "ResolvedModel"]
