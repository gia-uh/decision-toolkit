"""The curated catalog: adapters, real variants, statuses and variants selection."""

from __future__ import annotations

import pytest

from decision_toolkit import catalog
from decision_toolkit.errors import ModelNotFoundError


def test_ids_are_unique():
    ids = [model.id for model in catalog.CATALOG]
    assert len(ids) == len(set(ids))


def test_statuses_are_known():
    assert {model.status for model in catalog.CATALOG} <= {
        "m0", "m1", "planned", "phase2", "waitlist"
    }


def test_every_entry_has_adapter_and_license():
    for model in catalog.CATALOG:
        assert model.adapter, model.id
        assert model.license, model.id


def test_m1_entries_use_an_implemented_adapter():
    for model in catalog.CATALOG:
        if model.status == "m1":
            assert model.adapter in catalog.IMPLEMENTED_ADAPTERS, model.id


def test_m0_entries_use_an_implemented_adapter():
    for model in catalog.CATALOG:
        if model.status == "m0":
            assert model.adapter in catalog.IMPLEMENTED_ADAPTERS, model.id
            assert model.default_variant is not None, model.id


def test_laya_q8_is_verified_m0():
    model = catalog.get("laya")
    assert model.status == "m0"
    assert model.default_variant.repo == "nvkudva/laya-onnx-q8"


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
    assert model is not None and model.status == "m0"
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
    with pytest.raises(ModelNotFoundError):
        catalog.select_variant(catalog.get("gliner2.5-decide"))


def test_waitlist_entries_flag_their_issue():
    for model in catalog.CATALOG:
        if model.status == "waitlist":
            assert model.license in {"unverified", "other (LFM Open License)"} or model.notes, model.id


def test_phase2_entries_have_no_variants():
    for model in catalog.CATALOG:
        if model.status == "phase2":
            assert not model.variants, model.id
