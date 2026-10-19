"""The curated library: only runnable models, real variants, measured evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_toolkit import catalog
from decision_toolkit.errors import ModelNotFoundError

EXPECTED_IDS = {"gliner2.5-multi", "gliner2.5-decide", "laya", "julia-1", "opendecider-nano"}


def test_catalog_contains_only_available_models():
    assert {model.id for model in catalog.CATALOG} == EXPECTED_IDS
    assert all(model.status == "m0" for model in catalog.CATALOG)


def test_ids_are_unique():
    ids = [model.id for model in catalog.CATALOG]
    assert len(ids) == len(set(ids))


def test_every_entry_has_adapter_and_license():
    for model in catalog.CATALOG:
        assert model.adapter, model.id
        assert model.license, model.id
        assert model.adapter in catalog.IMPLEMENTED_ADAPTERS, model.id


def test_variants_are_real_and_complete():
    for model in catalog.CATALOG:
        for variant in model.variants:
            assert "/" in variant.repo, (model.id, variant.name)
            assert variant.files, (model.id, variant.name)
            assert variant.quant, (model.id, variant.name)
            assert variant.size_mb > 0, (model.id, variant.name)
            assert variant.role in {"default", "quality", "edge"}, (model.id, variant.name)
            for role, path in variant.files.items():
                assert role and path and not path.startswith("/"), (model.id, variant.name)


def test_gliner2_multi_has_two_real_variants():
    model = catalog.get("gliner2.5-multi")
    assert {variant.name for variant in model.variants} == {"fp16", "fp32"}
    assert model.default_variant.name == "fp16"
    assert model.default_variant.quant == "fp16"


def test_laya_q8_variant_declares_split_files_and_external_data():
    model = catalog.get("laya")
    variant = model.default_variant
    assert variant.name == "q8"
    assert set(variant.files) == {"encoder", "head", "tokenizer", "config"}
    assert any(name.endswith(".onnx.data") for name in variant.extra_files)


def test_select_variant_by_name_and_default():
    model = catalog.get("gliner2.5-multi")
    assert catalog.select_variant(model).name == "fp16"
    assert catalog.select_variant(model, "fp32").quant == "fp32"
    with pytest.raises(ModelNotFoundError):
        catalog.select_variant(model, "q4")


def test_select_variant_without_variants_errors():
    model = catalog.CatalogModel(
        id="x", name="X", adapter="gliner2", license="apache-2.0",
        params="1", architecture="x",
    )
    with pytest.raises(ModelNotFoundError):
        catalog.select_variant(model)


def test_m0_variants_have_measured_evidence():
    for model in catalog.CATALOG:
        for variant in model.variants:
            if variant.role != "default":
                continue
            assert variant.verified_at, (model.id, variant.name)
            assert variant.latency_ms_p50 is not None, (model.id, variant.name)
            assert variant.parity and variant.parity.get("baseline"), (model.id, variant.name)


def test_every_m0_model_has_a_card():
    for model in catalog.CATALOG:
        assert Path("docs/models", f"{model.id}.md").is_file(), model.id
