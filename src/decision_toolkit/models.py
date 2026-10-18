"""`decision_toolkit.models` — the toolkit's own model cache.

Everything the toolkit uses lives here, or is a local path you provide. The
HuggingFace cache is unrelated: this toolkit never reads it. The only network
access in the whole toolkit is :func:`ModelStore.download` (the
``dtk models download`` command).

Layout::

    <home>/                      # default: ~/.local/share/decision-toolkit
    ├── config.json              # {"cache_dir": "...", "current": "...", "device": "..."}
    └── models/
        └── convaiinnovations--laya@multilingual/
            ├── model.json       # provenance: identifier, repo, subfolder, source, date
            └── ...              # model files (or the requested subfolder)

``config.json`` is created on first use with the defaults already written, so
you can see and edit them (e.g. point ``cache_dir`` at another disk).
There are no environment variables.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import catalog
from .config import Config, load_config, save_config
from .errors import ModelNotFoundError

logger = logging.getLogger(__name__)

_ENTRY_META = "model.json"


@dataclass(frozen=True)
class ModelEntry:
    """A model stored in the toolkit cache."""

    identifier: str
    path: Path
    source: str  # "download" | "import"
    origin: str  # "huggingface:<repo>@<rev>" | "<absolute path>"
    created_at: str
    size_bytes: int
    repo: str | None = None
    subfolder: str | None = None


@dataclass(frozen=True)
class ResolvedModel:
    """What a backend needs to run a model: a local path and/or a hub reference."""

    identifier: str
    path: Path | None
    repo: str | None
    subfolder: str | None
    catalog: catalog.CatalogModel | None
    origin: str  # "cache" | "catalog" | "path"


def _dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _safe_dirname(identifier: str, subfolder: str | None) -> str:
    name = identifier.replace("/", "--")
    return f"{name}@{subfolder}" if subfolder else name


def _snapshot_download(
    repo_id: str,
    local_dir: Path,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Download a hub snapshot into ``local_dir`` (isolated for tests)."""
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
        """Every model stored in the toolkit cache."""
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
                    path=directory,
                    source=meta.get("source", "download"),
                    origin=meta.get("origin", "?"),
                    created_at=meta.get("created_at", "?"),
                    size_bytes=_dir_size(directory),
                    repo=meta.get("repo"),
                    subfolder=meta.get("subfolder"),
                )
            )
        return found

    def get(self, identifier: str) -> ModelEntry | None:
        """The cache entry for ``identifier`` (or ``None``)."""
        for entry in self.entries():
            if entry.identifier == identifier:
                return entry
        return None

    def delete(self, identifier: str) -> None:
        """Delete a cached model; refuses the model in use."""
        entry = self.get(identifier)
        if entry is None:
            raise ModelNotFoundError(f"{identifier!r} is not in the toolkit cache")
        if self.config.current == identifier:
            raise ModelNotFoundError(
                f"{identifier!r} is the model in use; run `dtk models use <other>` first"
            )
        shutil.rmtree(entry.path)
        logger.info("deleted %s (%d bytes)", entry.path, entry.size_bytes)

    # -------------------------------------------------------------- network
    def download(self, identifier: str, revision: str | None = None) -> ModelEntry:
        """Download a catalog model (or ``hf:org/name``) into the toolkit cache.

        This is the only command that uses the network.
        """
        repo, subfolder, catalog_entry = self._parse_identifier(identifier)
        local_dir = self.models_dir / _safe_dirname(identifier, subfolder)
        if (local_dir / _ENTRY_META).is_file():
            raise ModelNotFoundError(
                f"{identifier!r} is already in the toolkit cache ({local_dir}); "
                "delete it first to re-download"
            )

        allow_patterns = [f"{subfolder}/*"] if subfolder else None
        logger.info("downloading %s%s", repo, f" subfolder={subfolder}" if subfolder else "")
        _snapshot_download(repo, local_dir, revision=revision, allow_patterns=allow_patterns)

        meta = {
            "identifier": identifier,
            "repo": repo,
            "subfolder": subfolder,
            "source": "download",
            "origin": f"huggingface:{repo}@{revision or 'main'}",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "catalog_name": catalog_entry.name if catalog_entry else None,
        }
        (local_dir / _ENTRY_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        size = _dir_size(local_dir)
        logger.info("downloaded %s into %s (%.1f MB)", repo, local_dir, size / 1e6)
        return self.get(identifier)  # type: ignore[return-value]

    def import_model(self, path: str | Path, name: str | None = None) -> ModelEntry:
        """Copy a local model directory into the toolkit cache."""
        source = Path(path).expanduser().resolve()
        if not source.is_dir():
            raise ModelNotFoundError(f"local model path not found: {source}")
        identifier = name or source.name
        local_dir = self.models_dir / _safe_dirname(identifier, None)
        if local_dir.exists():
            raise ModelNotFoundError(f"{identifier!r} is already in the toolkit cache")
        shutil.copytree(source, local_dir)
        meta = {
            "identifier": identifier,
            "repo": None,
            "subfolder": None,
            "source": "import",
            "origin": str(source),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "catalog_name": None,
        }
        (local_dir / _ENTRY_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return self.get(identifier)  # type: ignore[return-value]

    # ------------------------------------------------------------------ use
    def use(self, identifier: str) -> None:
        """Set the model in use (persistent)."""
        if self.get(identifier) is None and catalog.get(identifier) is None and "/" not in identifier:
            raise ModelNotFoundError(
                f"{identifier!r} is neither in the toolkit cache nor the catalog; "
                "run `dtk models list` / `dtk models download <id>`"
            )
        self.config.current = identifier
        save_config(self.config)

    def current(self) -> str | None:
        """The model in use, if any."""
        return self.config.current

    # -------------------------------------------------------------- resolve
    def resolve(self, identifier: str | None = None) -> ResolvedModel:
        """Resolve an identifier into what a backend needs to run it.

        Resolution order: local path, cached entry, catalog entry. Raises
        :class:`ModelNotFoundError` with suggestions otherwise.
        """
        if identifier is None:
            identifier = self.config.current
        if identifier is None:
            raise ModelNotFoundError(
                "no model given and none in use; pass a model id or run `dtk models use <id>`"
            )

        # 1. a local path
        candidate = Path(identifier).expanduser()
        if candidate.exists():
            catalog_entry = catalog.find_by_repo(candidate.name) if "/" not in identifier else None
            return ResolvedModel(
                identifier=identifier,
                path=candidate.resolve(),
                repo=None,
                subfolder=None,
                catalog=catalog_entry,
                origin="path",
            )

        # 2. the toolkit cache
        entry = self.get(identifier)
        if entry is not None:
            return ResolvedModel(
                identifier=identifier,
                path=entry.path,
                repo=entry.repo,
                subfolder=entry.subfolder,
                catalog=catalog.get(identifier),
                origin="cache",
            )

        # 3. the catalog
        catalog_entry = catalog.get(identifier)
        if catalog_entry is not None:
            return ResolvedModel(
                identifier=identifier,
                path=None,
                repo=catalog_entry.repo,
                subfolder=catalog_entry.subfolder,
                catalog=catalog_entry,
                origin="catalog",
            )

        # 4. a bare hub reference ("org/name" or "hf:org/name")
        repo = identifier[3:] if identifier.startswith("hf:") else identifier
        if "/" in repo:
            catalog_entry = catalog.find_by_repo(repo)
            return ResolvedModel(
                identifier=identifier,
                path=None,
                repo=repo,
                subfolder=catalog_entry.subfolder if catalog_entry else None,
                catalog=catalog_entry,
                origin="catalog",
            )

        suggestions = ", ".join(model.id for model in catalog.CATALOG if model.status in {"m0", "variant"})
        raise ModelNotFoundError(
            f"unknown model {identifier!r}. Runnable today: {suggestions}. "
            "See `dtk models list` for the full catalog."
        )

    # -------------------------------------------------------------- helpers
    def _parse_identifier(self, identifier: str) -> tuple[str, str | None, catalog.CatalogModel | None]:
        """Split an identifier into (repo, subfolder, catalog entry)."""
        entry = catalog.get(identifier)
        if entry is not None:
            return entry.repo, entry.subfolder, entry
        repo = identifier[3:] if identifier.startswith("hf:") else identifier
        if "/" in repo:
            found = catalog.find_by_repo(repo)
            return repo, found.subfolder if found else None, found
        raise ModelNotFoundError(
            f"unknown model {identifier!r}; pass a catalog id (see `dtk models list`) "
            "or a hub repo 'org/name'"
        )


__all__ = ["ModelStore", "ModelEntry", "ResolvedModel"]
