"""The model cache: variants, allow_patterns, provenance, resolve."""

from __future__ import annotations

import json

import pytest

from decision_toolkit.errors import ModelNotFoundError
from decision_toolkit.models import ModelStore


@pytest.fixture
def store(home) -> ModelStore:
    return ModelStore(home=home)


def test_empty_cache(store):
    assert store.entries() == []
    assert store.current() is None


def test_download_default_variant_only_fetches_declared_files(store, fake_snapshot):
    entry = store.download("gliner2.5-multi")
    assert entry.identifier == "gliner2.5-multi"
    assert entry.variant == "fp16"
    assert entry.path.name == "gliner2.5-multi@fp16"
    call = fake_snapshot[0]
    assert call["repo_id"] == "jugaadsrl/gliner2.5-multi-v1-onnx"
    assert call["allow_patterns"] == [
        "fp16_25/encoder_fp16.onnx",
        "fp32_25/classifier_fp32.onnx",
        "fp16_25/tokenizer.json",
    ]
    meta = json.loads((entry.path / "model.json").read_text())
    assert meta["variant"] == "fp16"
    assert meta["repo"] == "jugaadsrl/gliner2.5-multi-v1-onnx"


def test_download_explicit_variant(store, fake_snapshot):
    entry = store.download("gliner2.5-multi", variant="fp32")
    assert entry.variant == "fp32"
    assert fake_snapshot[0]["allow_patterns"] == [
        "fp32_25/encoder_fp32.onnx",
        "fp32_25/classifier_fp32.onnx",
        "fp32_25/tokenizer.json",
    ]


def test_download_laya_variant_includes_external_data(store, fake_snapshot):
    store.download("laya-multilingual")
    patterns = fake_snapshot[0]["allow_patterns"]
    assert "v1/encoder_q8.onnx" in patterns
    assert "v1/encoder_q8.onnx.data" in patterns
    assert "v1/head_q8.onnx.data" in patterns


def test_download_unknown_model_is_rejected(store, fake_snapshot):
    with pytest.raises(ModelNotFoundError):
        store.download("nope")


def test_download_unknown_variant_is_rejected(store, fake_snapshot):
    with pytest.raises(ModelNotFoundError) as excinfo:
        store.download("gliner2.5-multi", variant="q4")
    assert "fp16" in str(excinfo.value)


def test_download_without_variants_is_rejected(store, fake_snapshot):
    with pytest.raises(ModelNotFoundError):
        store.download("laya")  # planned, no variant yet


def test_download_is_idempotent_guarded(store, fake_snapshot):
    store.download("gliner2.5-multi")
    with pytest.raises(ModelNotFoundError):
        store.download("gliner2.5-multi")


def test_import_and_delete(store, tmp_path):
    source = tmp_path / "my-model"
    source.mkdir()
    (source / "weights.onnx").write_bytes(b"1234")

    entry = store.import_model(source, name="my-finetune")
    assert entry.identifier == "my-finetune"
    assert entry.variant == "local"
    assert (entry.path / "weights.onnx").read_bytes() == b"1234"

    store.delete("my-finetune")
    assert store.get("my-finetune") is None


def test_delete_refuses_the_model_in_use(store, fake_snapshot):
    store.download("gliner2.5-multi")
    store.use("gliner2.5-multi")
    with pytest.raises(ModelNotFoundError):
        store.delete("gliner2.5-multi")


def test_verify_reports_files_and_session_note(store, fake_snapshot):
    store.download("gliner2.5-multi")
    report = store.verify("gliner2.5-multi")
    assert report["identifier"] == "gliner2.5-multi"
    assert report["variant"] == "fp16"
    assert report["missing"] == []
    assert report["ok"] is True
    # the fake files are not real ONNX graphs; without onnxruntime the sessions are not checked
    if "note" in report:
        assert "onnxruntime" in report["note"]


def test_use_accepts_catalog_and_cache_ids(store, fake_snapshot):
    store.use("julia-1")  # catalog id, not downloaded
    assert store.current() == "julia-1"
    store.download("gliner2.5-multi")
    store.use("gliner2.5-multi")
    assert store.current() == "gliner2.5-multi"


def test_use_rejects_unknown(store):
    with pytest.raises(ModelNotFoundError):
        store.use("nope")


def test_resolve_catalog_before_download(store):
    resolved = store.resolve("gliner2.5-multi")
    assert resolved.origin == "catalog"
    assert resolved.path is None
    assert resolved.variant.name == "fp16"
    assert set(resolved.files) == {"encoder", "classifier", "tokenizer"}


def test_resolve_cache_after_download(store, fake_snapshot):
    store.download("gliner2.5-multi")
    resolved = store.resolve("gliner2.5-multi")
    assert resolved.origin == "cache"
    assert resolved.path is not None
    assert resolved.variant.name == "fp16"
    assert (resolved.path / "fp16_25/encoder_fp16.onnx").is_file()


def test_resolve_variant_override(store, fake_snapshot):
    store.download("gliner2.5-multi", variant="fp32")
    resolved = store.resolve("gliner2.5-multi", variant="fp32")
    assert resolved.variant.name == "fp32"


def test_resolve_accepts_local_paths_and_hub_refs(store, tmp_path):
    local = tmp_path / "handmade"
    local.mkdir()
    resolved = store.resolve(str(local))
    assert resolved.origin == "path"
    assert resolved.path == local.resolve()

    resolved = store.resolve("hf:org/name")
    assert resolved.catalog is None
    assert resolved.path is None


def test_resolve_unknown_gives_suggestions(store):
    with pytest.raises(ModelNotFoundError) as excinfo:
        store.resolve("definitely-not-a-model")
    assert "gliner2.5-multi" in str(excinfo.value)


def test_resolve_none_uses_current(store, fake_snapshot):
    with pytest.raises(ModelNotFoundError):
        store.resolve(None)
    store.download("gliner2.5-multi")
    store.use("gliner2.5-multi")
    assert store.resolve(None).identifier == "gliner2.5-multi"
