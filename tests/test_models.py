"""The model cache: download (faked), import, use/current, delete, resolve."""

from __future__ import annotations

import json

import pytest

import decision_toolkit.models as models_module
from decision_toolkit.errors import ModelNotFoundError
from decision_toolkit.models import ModelStore


@pytest.fixture
def store(home) -> ModelStore:
    return ModelStore(home=home)


@pytest.fixture
def fake_snapshot(monkeypatch):
    """Replace the hub download with a local fake; record its arguments."""
    calls: list[dict] = []

    def _fake(repo_id, local_dir, revision=None, allow_patterns=None):
        calls.append(
            {"repo_id": repo_id, "local_dir": local_dir, "revision": revision, "allow_patterns": allow_patterns}
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(models_module, "_snapshot_download", _fake)
    return calls


def test_empty_cache(store):
    assert store.entries() == []
    assert store.current() is None


def test_download_writes_provenance(store, fake_snapshot):
    entry = store.download("laya")
    assert entry.identifier == "laya"
    assert entry.repo == "convaiinnovations/laya"
    assert entry.subfolder is None
    assert fake_snapshot[0]["repo_id"] == "convaiinnovations/laya"
    assert fake_snapshot[0]["allow_patterns"] is None
    meta = json.loads((entry.path / "model.json").read_text())
    assert meta["source"] == "download"
    assert meta["origin"].startswith("huggingface:convaiinnovations/laya@")


def test_download_subfolder_variant_uses_allow_patterns(store, fake_snapshot):
    entry = store.download("laya-multilingual")
    assert entry.subfolder == "multilingual"
    assert fake_snapshot[0]["allow_patterns"] == ["multilingual/*"]


def test_download_is_idempotent_guarded(store, fake_snapshot):
    store.download("von")
    with pytest.raises(ModelNotFoundError):
        store.download("von")


def test_download_unknown_identifier(store, fake_snapshot):
    with pytest.raises(ModelNotFoundError):
        store.download("nope")


def test_download_hub_reference(store, fake_snapshot):
    entry = store.download("hf:someorg/somemodel")
    assert entry.repo == "someorg/somemodel"
    assert fake_snapshot[0]["repo_id"] == "someorg/somemodel"


def test_import_and_delete(store, tmp_path):
    source = tmp_path / "my-model"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"1234")

    entry = store.import_model(source, name="my-finetune")
    assert entry.identifier == "my-finetune"
    assert (entry.path / "weights.bin").read_bytes() == b"1234"
    assert store.get("my-finetune") is not None

    store.delete("my-finetune")
    assert store.get("my-finetune") is None


def test_delete_refuses_the_model_in_use(store, fake_snapshot):
    store.download("laya")
    store.use("laya")
    with pytest.raises(ModelNotFoundError):
        store.delete("laya")


def test_use_accepts_catalog_and_cache_ids(store, fake_snapshot):
    store.use("julia-1")  # catalog id, not downloaded
    assert store.current() == "julia-1"
    store.download("laya")
    store.use("laya")
    assert store.current() == "laya"


def test_use_rejects_unknown(store):
    with pytest.raises(ModelNotFoundError):
        store.use("nope")


def test_resolve_prefers_cache_then_catalog(store, fake_snapshot):
    resolved = store.resolve("julia-1")
    assert resolved.origin == "catalog"
    assert resolved.repo == "SupersonicLabs/Julia-1"
    assert resolved.path is None

    store.download("julia-1")
    resolved = store.resolve("julia-1")
    assert resolved.origin == "cache"
    assert resolved.path is not None


def test_resolve_accepts_local_paths_and_hub_refs(store, tmp_path):
    local = tmp_path / "handmade"
    local.mkdir()
    resolved = store.resolve(str(local))
    assert resolved.origin == "path"
    assert resolved.path == local.resolve()

    resolved = store.resolve("hf:org/name")
    assert resolved.repo == "org/name"


def test_resolve_unknown_gives_suggestions(store):
    with pytest.raises(ModelNotFoundError) as excinfo:
        store.resolve("definitely-not-a-model")
    assert "laya" in str(excinfo.value)


def test_resolve_none_uses_current(store, fake_snapshot):
    with pytest.raises(ModelNotFoundError):
        store.resolve(None)
    store.download("laya")
    store.use("laya")
    assert store.resolve(None).identifier == "laya"
