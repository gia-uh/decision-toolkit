"""`decision_toolkit.catalog` — the curated library of runnable models.

The catalog contains **only models that are runnable today** (`m0`): a clean
license, an adapter whose contract we implement from a real export, a verified
ONNX build on the Hub, and measured parity/latency. Researched-but-unavailable
models (planned, conversion pending, broken exports) live in ``docs/ROADMAP.md``
and are deliberately absent here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ModelNotFoundError


@dataclass(frozen=True)
class Variant:
    """One downloadable build of a model (quantization + layout).

    ``parity`` holds the measured correctness evidence against a baseline, e.g.
    ``{"baseline": "torch ...", "cases": 100, "top1": 1.0,
    "max_abs_logit_error": 8.5e-05}``. It is data, not prose: the default choice
    and its audit trail live in the catalog.
    """

    name: str
    repo: str
    files: Mapping[str, str]      # role -> path inside the repo
    quant: str                    # fp32 | fp16 | int8 | q4 ...
    size_mb: float
    role: str = "default"         # default | quality | edge
    revision: str | None = None
    extra_files: tuple[str, ...] = ()   # external data (.onnx.data), checksums...
    notes: str = ""
    verified_at: str | None = None
    latency_ms_p50: float | None = None
    parity: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CatalogModel:
    """One runnable model, with its real downloadable variants."""

    id: str
    name: str
    adapter: str
    license: str
    params: str
    architecture: str
    languages: str = "en"
    status: str = "m0"
    variants: tuple[Variant, ...] = ()
    notes: str = ""

    @property
    def default_variant(self) -> Variant | None:
        for variant in self.variants:
            if variant.role == "default":
                return variant
        return self.variants[0] if self.variants else None


CATALOG: tuple[CatalogModel, ...] = (
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
                verified_at="2026-10-19",
                latency_ms_p50=353.0,
                parity={
                    "baseline": "mininvox int8 build",
                    "cases": 15,
                    "accuracy": 0.6,
                    "note": "same 9/15 routing accuracy as the local int8 build; per-case agreement not recorded",
                },
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
        adapter="gliner2-decide",
        license="apache-2.0",
        params="340M",
        architecture="DeBERTa-v3-large",
        status="m0",
        notes="Fused ONNX export verified on CPU: refund example → refund_request 0.944; "
              "flight change → seat_change 0.593 (change second, 0.334); the sentiment example "
              "resolves positive (the card's potential output said mixed — documented, not "
              "hidden). English-only; q4f16 build; 912 ms/decision.",
        variants=(
            Variant(
                name="q4f16",
                repo="onnx-community/GLiNER2.5-Decide-ONNX",
                files={
                    "model": "onnx/model_q4f16.onnx",
                    "tokenizer": "tokenizer.json",
                },
                extra_files=("onnx/model_q4f16.onnx_data",),
                quant="q4/fp16",
                size_mb=523.5,
                role="default",
                verified_at="2026-10-19",
                latency_ms_p50=912.0,
                parity={
                    "baseline": "model card examples",
                    "cases": 2,
                    "note": "refund 0.944 matches the card; seat_change 0.593 vs card's change; sentiment positive vs card's mixed",
                },
            ),
        ),
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
                verified_at="2026-10-19",
                latency_ms_p50=514.0,
                parity={
                    "baseline": "torch convaiinnovations/laya",
                    "cases": 3,
                    "max_probability_delta": 0.005,
                    "choices_match": True,
                },
            ),
        ),
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
                verified_at="2026-10-19",
                latency_ms_p50=329.0,
                parity={
                    "baseline": "pytorch logits (official parity-cases)",
                    "cases": 100,
                    "top1": 1.0,
                    "max_abs_logit_error": 8.5e-05,
                },
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
        status="m0",
        notes="Official ONNX build. Prompt/readout ported from the pinned packaging code "
              "(nano_ids + softmax over [MASK]); runs on CPU: billing 0.907, urgent 0.572, "
              "tone 1.235, 530 ms/decision. The official benchmark inputs are hashed, so no "
              "per-case torch parity is possible.",
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
                verified_at="2026-10-19",
                latency_ms_p50=530.0,
                parity={
                    "baseline": "pinned packaging code",
                    "cases": 0,
                    "note": "official benchmark inputs are hashed; prompt/readout ported from the pinned code",
                },
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
)

CATALOG_BY_ID: dict[str, CatalogModel] = {model.id: model for model in CATALOG}

#: Adapters that ship with the toolkit today.
IMPLEMENTED_ADAPTERS = frozenset({"gliner2", "gliner2-decide", "julia", "laya", "opendecider"})


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
                f"(available today: {sorted(CATALOG_BY_ID)})"
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
