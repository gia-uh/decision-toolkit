"""`decision_toolkit.catalog` — the curated model list.

The catalog is **ours**, not a mirror of HuggingFace. A model enters only when
it is understood and verified: clean license, and adapter whose contract we can
implement, a real ONNX export on the Hub, measured parity against the original,
and CPU-viable latency. If any of those is missing, the model stays out.

Every variant points to files that **exist on the Hub** (no placeholders). A
variant is written only when there is a real export that demands it.

Status values:

``m0``        runnable today (adapter + verified export)
``m1``        adapter validated; the public export is identified but still to verify
``planned``   real export identified; its adapter is not written yet
``phase2``    no ONNX yet: conversion from the original weights is planned
``waitlist``  meets most requirements; pending license confirmation or recipe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .errors import ModelNotFoundError


@dataclass(frozen=True)
class Variant:
    """One downloadable build of a model (quantization + layout)."""

    name: str
    repo: str
    files: Mapping[str, str]      # role -> path inside the repo
    quant: str                    # fp32 | fp16 | int8 | q4 ...
    size_mb: float
    role: str = "default"         # default | quality | edge
    revision: str | None = None
    extra_files: tuple[str, ...] = ()   # external data (.onnx.data), checksums...
    notes: str = ""


@dataclass(frozen=True)
class CatalogModel:
    """One curated model, with its real downloadable variants."""

    id: str
    name: str
    adapter: str
    license: str
    params: str
    architecture: str
    languages: str = "en"
    status: str = "planned"
    variants: tuple[Variant, ...] = ()
    notes: str = ""

    @property
    def default_variant(self) -> Variant | None:
        for variant in self.variants:
            if variant.role == "default":
                return variant
        return self.variants[0] if self.variants else None


CATALOG: tuple[CatalogModel, ...] = (
    # ------------------------------------------------------------------ ONNX
    CatalogModel(
        id="gliner2.5-multi",
        name="GLiNER2.5 multi (v1 + classifier)",
        adapter="gliner2",
        license="apache-2.0",
        params="287M",
        architecture="GLiNER2.5 multi-v1",
        languages="multi",
        status="m0",
        notes="Base v1 export with classifier head, verified end-to-end: public fp16 export "
              "downloaded and run (9/15 routing, 353 ms CPU), same accuracy as the local "
              "int8 build.",
        variants=(
            Variant(
                name="fp16",
                repo="jugaadsrl/gliner2.5-multi-v1-onnx",
                files={
                    "encoder": "fp16_25/encoder_fp16.onnx",
                    "classifier": "fp32_25/classifier_fp32.onnx",
                    "tokenizer": "fp16_25/tokenizer.json",
                },
                quant="fp16",
                size_mb=577.1,
                role="default",
            ),
            Variant(
                name="fp32",
                repo="jugaadsrl/gliner2.5-multi-v1-onnx",
                files={
                    "encoder": "fp32_25/encoder_fp32.onnx",
                    "classifier": "fp32_25/classifier_fp32.onnx",
                    "tokenizer": "fp32_25/tokenizer.json",
                },
                quant="fp32",
                size_mb=1131.9,
                role="quality",
            ),
        ),
    ),
    CatalogModel(
        id="gliner2.5-decide",
        name="GLiNER2.5-Decide",
        adapter="gliner2",
        license="apache-2.0",
        params="340M",
        architecture="DeBERTa-v3-large",
        status="planned",
        notes="Fused export exists (onnx-community/GLiNER2.5-Decide-ONNX; fp32 1.74 GB, "
              "fp16 872 MB, q4 888 MB, q4f16 523 MB) but its signature is not verified against "
              "the gliner2 adapter; if it differs, it needs its own adapter.",
    ),
    CatalogModel(
        id="gliner2.5-multi-decide",
        name="GLiNER2.5-multi-Decide",
        adapter="gliner2",
        license="apache-2.0",
        params="287M",
        architecture="GLiNER2.5 multi",
        languages="multi",
        status="planned",
        notes="Decide fine-tune; no compatible split export identified yet.",
    ),
    CatalogModel(
        id="laya",
        name="Laya (English)",
        adapter="laya",
        license="apache-2.0",
        params="421M",
        architecture="ModernBERT-large",
        status="m0",
        notes="English checkpoint. Split q8 export verified: billing 0.9612 vs torch 0.9598, "
              "urgent 0.056 vs 0.061, tone 1.025 vs 1.026 (within the export's stated "
              "≤0.0158 probability shift); 514 ms/decision on CPU.",
        variants=(
            Variant(
                name="q8",
                repo="nvkudva/laya-onnx-q8",
                files={
                    "encoder": "v1/encoder_q8.onnx",
                    "head": "v1/head_q8.onnx",
                    "tokenizer": "v1/tokenizer.json",
                    "config": "v1/rl_agent_config.json",
                },
                extra_files=("v1/encoder_q8.onnx.data", "v1/head_q8.onnx.data"),
                quant="int8",
                size_mb=527.9,
                role="default",
            ),
        ),
    ),
    CatalogModel(
        id="laya-multilingual",
        name="Laya multilingual",
        adapter="laya",
        license="apache-2.0",
        params="~150M",
        architecture="mmBERT-base",
        languages="100+",
        status="planned",
        notes="No split multilingual export is published yet (mizchi 681 MB and androidli "
              "int4 428 MB are fused single-file layouts). The split multilingual build was "
              "validated locally (same answers as the torch runtime).",
    ),
    CatalogModel(
        id="julia-1",
        name="Julia 1",
        adapter="julia",
        license="apache-2.0",
        params="141M",
        architecture="mmBERT-small",
        languages="52 locales",
        status="m0",
        notes="Official ONNX export verified: 100/100 argmax parity against the published "
              "PyTorch logits, max absolute logit error 8.5e-05; 329 ms/decision on CPU.",
        variants=(
            Variant(
                name="fp32",
                repo="SupersonicLabs/Julia-1-ONNX",
                files={
                    "model": "model.onnx",
                    "tokenizer": "tokenizer.json",
                },
                extra_files=("model.onnx.data",),
                quant="fp32",
                size_mb=618.2,
                role="default",
            ),
        ),
    ),
    CatalogModel(
        id="opendecider-nano",
        name="OpenDecider nano",
        adapter="opendecider",
        license="apache-2.0",
        params="395M",
        architecture="Ettin encoder 400M",
        status="planned",
        notes="Official ONNX build with its own config; adapter to be written from the real "
              "export.",
        variants=(
            Variant(
                name="q8f16",
                repo="manjunathshiva/opendecider-nano-ONNX",
                files={
                    "model": "onnx/model_q8f16.onnx",
                    "tokenizer": "tokenizer.web.json",
                    "config": "opendecider.json",
                },
                quant="int8/fp16",
                size_mb=468.8,
                role="default",
            ),
            Variant(
                name="fp16",
                repo="manjunathshiva/opendecider-nano-ONNX",
                files={
                    "model": "onnx/model_fp16.onnx",
                    "tokenizer": "tokenizer.web.json",
                    "config": "opendecider.json",
                },
                quant="fp16",
                size_mb=791.9,
                role="quality",
            ),
        ),
    ),
    CatalogModel(
        id="intern-decision-0.8b",
        name="Intern-Decision 0.8B",
        adapter="decoder",
        license="apache-2.0",
        params="0.85B",
        architecture="Qwen3.5-0.8B",
        status="planned",
        notes="Community ONNX export; decoder-style readout, external data is gzip-compressed.",
        variants=(
            Variant(
                name="fp16",
                repo="Wortou/Intern-Decision-0.8B-ONNX-FP16",
                files={
                    "model": "onnx/decoder_model_merged_fp16.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "decision-manifest.json",
                },
                extra_files=(
                    "onnx/decoder_model_merged_fp16.onnx_data.gz",
                    "onnx/embed_tokens_fp16.onnx_data.gz",
                ),
                quant="fp16",
                size_mb=1350.7,
                role="default",
            ),
        ),
    ),
    CatalogModel(
        id="jevk5-lite",
        name="JevK5-Lite",
        adapter="jevk5",
        license="apache-2.0",
        params="~0.1B",
        architecture="JevK5 Lite",
        status="waitlist",
        notes="Export broken upstream: the .onnx files reference external model*.onnx.data files "
              "that are not published (only listed in SHA256SUMS). Not adoptable until fixed.",
    ),
    # -------------------------------------------------------------- phase 2
    CatalogModel(
        id="d3-lite",
        name="Decision 3.0 lite",
        adapter="decoder",
        license="apache-2.0",
        params="0.85B",
        architecture="Qwen3.5-0.8B",
        status="phase2",
        notes="No ONNX published yet; conversion from the original weights is planned for v2.",
    ),
    CatalogModel(
        id="kev-0.8b",
        name="Kev 0.8B",
        adapter="decoder",
        license="apache-2.0",
        params="~0.8B",
        architecture="Qwen3.5-0.8B (LoRA)",
        status="phase2",
        notes="No ONNX published yet; v2 conversion planned.",
    ),
    CatalogModel(
        id="sieve-0.8b",
        name="Sieve 0.8B",
        adapter="decoder",
        license="apache-2.0",
        params="~0.8B",
        architecture="Qwen3.5-0.8B",
        status="phase2",
        notes="No ONNX published yet; v2 conversion planned.",
    ),
    CatalogModel(
        id="strands-decider-2b",
        name="Strands Decider 2B (hobson-v21)",
        adapter="decoder",
        license="apache-2.0",
        params="2B",
        architecture="Qwen3.5-2B (adapter)",
        status="phase2",
        notes="Agent-oriented decider; no ONNX yet — v2 conversion.",
    ),
    CatalogModel(
        id="thx-01",
        name="THX-01",
        adapter="encoder-readout",
        license="apache-2.0",
        params="322M",
        architecture="mmBERT-base",
        languages="az / ru / en / tr",
        status="phase2",
        notes="No ONNX published; export planned for v2.",
    ),
    # --------------------------------------------------------------- waitlist
    CatalogModel(
        id="bosun-0.6b",
        name="Bosun v3.1 0.6B",
        adapter="decoder",
        license="apache-2.0",
        params="0.6B",
        architecture="Qwen3-0.6B (adapter)",
        status="waitlist",
        notes="Tiny decoder; no ONNX yet — v2 conversion planned.",
    ),
    CatalogModel(
        id="d3-edge",
        name="Decision 3.0 edge",
        adapter="decoder",
        license="apache-2.0",
        params="0.6B",
        architecture="Qwen3.5-0.8B",
        status="waitlist",
        notes="Same family as d3-lite; no ONNX yet — v2 conversion. Not a licensing issue.",
    ),
    CatalogModel(
        id="tev1-0.8b",
        name="Tev1 0.8B",
        adapter="decoder",
        license="unverified",
        params="0.87B",
        architecture="Qwen3.5-0.8B",
        status="waitlist",
        notes="License not declared on the model card; experimental.",
    ),
    CatalogModel(
        id="d1-omni-600m",
        name="Liquid d1-omni 600M",
        adapter="decoder",
        license="other (LFM Open License)",
        params="587M",
        architecture="LFM2.5-Encoder-350M",
        status="waitlist",
        notes="Text + image + audio decisions; license needs review before core.",
    ),
    CatalogModel(
        id="bekko-400m",
        name="bekko system-one 400M",
        adapter="encoder-readout",
        license="unverified",
        params="400M",
        architecture="Ettin reranker 400M",
        status="waitlist",
        notes="Tiny Japanese-authored family; license pending.",
    ),
)

CATALOG_BY_ID: dict[str, CatalogModel] = {model.id: model for model in CATALOG}

#: Adapters that ship with the toolkit today.
IMPLEMENTED_ADAPTERS = frozenset({"gliner2", "julia", "laya"})


def get(model_id: str) -> CatalogModel | None:
    """Return the catalog entry for ``model_id`` (or ``None``)."""
    return CATALOG_BY_ID.get(model_id)


def select_variant(model: CatalogModel, name: str | None = None) -> Variant:
    """Choose a variant by name, or the model's default."""
    if name is None:
        variant = model.default_variant
        if variant is None:
            raise ModelNotFoundError(
                f"{model.id!r} has no downloadable ONNX variant yet "
                f"(status {model.status}); see `dtk models list`"
            )
        return variant
    for variant in model.variants:
        if variant.name == name:
            return variant
    available = ", ".join(variant.name for variant in model.variants) or "(none)"
    raise ModelNotFoundError(f"{model.id!r} has no variant {name!r}; available: {available}")


__all__ = [
    "CatalogModel",
    "Variant",
    "CATALOG",
    "CATALOG_BY_ID",
    "IMPLEMENTED_ADAPTERS",
    "get",
    "select_variant",
]
