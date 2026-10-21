"""`decision_toolkit.catalog` — the curated library of runnable models.

The catalog contains **only models that are runnable today** (`m0`): a clean
license, an adapter whose contract we implement from a real export, a verified
ONNX build on the Hub, and measured parity/latency. Researched-but-unavailable
models (planned, conversion pending, broken exports) live in ``docs/ROADMAP.md``
and are deliberately absent here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ModelNotFoundError


@dataclass(frozen=True)
class Capabilities:
    """What a model can actually do, independent of any wire contract.

    These are semantic capabilities, not REST fields.  The core validates
    requests against them before loading a model, so an unsupported question
    fails with ``UnsupportedCapabilityError`` instead of a backend error.
    """

    question_types: tuple[str, ...] = ("choice", "noul", "score")
    max_options: int | None = 255
    max_score_levels: int | None = 10
    max_context_tokens: int | None = None
    languages: tuple[str, ...] = ("en",)
    multimodal: bool = False
    multi_label: bool = False
    batch: bool = False
    usage: bool = True

    def supports(self, question_type: str) -> bool:
        return str(question_type) in self.question_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_types": list(self.question_types),
            "max_options": self.max_options,
            "max_score_levels": self.max_score_levels,
            "max_context_tokens": self.max_context_tokens,
            "languages": list(self.languages),
            "multimodal": self.multimodal,
            "multi_label": self.multi_label,
            "batch": self.batch,
            "usage": self.usage,
        }


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
    sha256: Mapping[str, str] = field(default_factory=dict)        # role -> sha256
    extra_sha256: Mapping[str, str] = field(default_factory=dict)  # relative path -> sha256


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
    capabilities: Capabilities = field(default_factory=Capabilities)

    @property
    def default_variant(self) -> Variant | None:
        for variant in self.variants:
            if variant.role == "default":
                return variant
        return self.variants[0] if self.variants else None

    @property
    def info(self) -> ModelInfo:
        """Typed metadata for the public API (catalog + capabilities)."""
        return ModelInfo(
            id=self.id,
            name=self.name,
            adapter=self.adapter,
            license=self.license,
            params=self.params,
            architecture=self.architecture,
            languages=self.languages,
            status=self.status,
            capabilities=self.capabilities,
            variants=self.variants,
            notes=self.notes,
        )


@dataclass(frozen=True)
class ModelInfo:
    """Public model metadata: identity, provenance and capabilities."""

    id: str
    name: str
    adapter: str
    license: str
    params: str
    architecture: str
    languages: str
    status: str
    capabilities: Capabilities
    variants: tuple[Variant, ...] = ()
    notes: str = ""

    @property
    def default_variant(self) -> Variant | None:
        for variant in self.variants:
            if variant.role == "default":
                return variant
        return self.variants[0] if self.variants else None

    def to_dict(self) -> dict[str, Any]:
        default = self.default_variant
        return {
            "id": self.id,
            "name": self.name,
            "adapter": self.adapter,
            "license": self.license,
            "params": self.params,
            "architecture": self.architecture,
            "languages": self.languages,
            "status": self.status,
            "capabilities": self.capabilities.to_dict(),
            "default_variant": default.name if default else None,
            "variants": [
                {
                    "name": variant.name,
                    "quant": variant.quant,
                    "size_mb": variant.size_mb,
                    "role": variant.role,
                    "repo": variant.repo,
                    "revision": variant.revision,
                    "verified_at": variant.verified_at,
                    "latency_ms_p50": variant.latency_ms_p50,
                    "parity": dict(variant.parity) if variant.parity else None,
                    "sha256": dict(variant.sha256),
                    "extra_sha256": dict(variant.extra_sha256),
                }
                for variant in self.variants
            ],
            "notes": self.notes,
        }


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
        capabilities=Capabilities(languages=("multi",)),
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
                revision="3de2c11f55853b7919f985a59bfc8df27b5ec991",
                sha256={
                    "encoder": "b389910b27261d4999bd41c164bfa0334f76b3bc9fd3d890527b6f07fa1bfb94",
                    "classifier": "3c06934244a4763a7411d9dceb002ba2d5c84d1bb7f7d66aa64acd5653314454",
                    "tokenizer": "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a",
                },
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
                revision="3de2c11f55853b7919f985a59bfc8df27b5ec991",
                sha256={
                    "encoder": "cba26576eef7e14eba87f3485395d5b24e5047f819c859a9d667a63915ed404f",
                    "classifier": "3c06934244a4763a7411d9dceb002ba2d5c84d1bb7f7d66aa64acd5653314454",
                    "tokenizer": "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a",
                },
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
        capabilities=Capabilities(languages=("en",)),
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
                revision="18271032588167fe9a3fbb8c5a4900b1231f8119",
                sha256={
                    "model": "93f3e0d4ed8bbd4858840f8a2e66028a70e86a482e1ff9822d2ac726381684f6",
                    "tokenizer": "3ad87d9ffe669147063e70850927dd2da90249e2acc5c8527f1eb65df467bcc8",
                },
                extra_sha256={
                    "onnx/model_q4f16.onnx_data": "df2b05dc97ff875e0d2b51ec21de0049b7b3e7feddaa5224d318dc79e66210e8",
                },
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
        capabilities=Capabilities(languages=("en",), max_context_tokens=1024),
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
                revision="cccc9a01e06189cbb1c153a51ea32642102e306d",
                sha256={
                    "encoder": "5267b578a536bf12808ec84c1f354e87efd150d3c38d44099ef3dd29c5f43124",
                    "head": "e86c14c89a18b14f6115a7ccbfd039fce7ff97178fa0c07c14af936690a1f688",
                    "tokenizer": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
                    "config": "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
                },
                extra_sha256={
                    "v1/encoder_q8.onnx.data": "6375a3ae8f3ebfa4ca196e813fc7bdaf2e8ad8cf709218ef844ae8814dfffc59",
                    "v1/head_q8.onnx.data": "cfdf7c199378b07758c69f58935871cf517c97c29914328d4b11fcfbe50b7f8b",
                },
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
        capabilities=Capabilities(languages=("mul",), max_context_tokens=1024),
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
                revision="82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8",
                sha256={
                    "model": "97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9",
                    "tokenizer": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
                },
                extra_sha256={
                    "model.onnx.data": "fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1",
                },
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
        capabilities=Capabilities(languages=("en",), max_context_tokens=2048),
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
                revision="5137ae7b4f1cf8c55f1d05d3f5cdd3694d92da3e",
                sha256={
                    "model": "80436dbed1a284548b5e6c3975bb26df81fb493fc4e64629e5a6ef490bf0e0ae",
                    "tokenizer": "fd4146f0082d98fd1fc03bf41349f4d1eed478987ce9a64addc1654dfcad14ee",
                    "config": "28eea9d70afba02740fdd871b148004437cd4f8c31b416776bb864d7895cf83f",
                },
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
                revision="5137ae7b4f1cf8c55f1d05d3f5cdd3694d92da3e",
                sha256={
                    "model": "ce3b92ed327c394570123867d48a73383e748b015c08ca5b735670d6601ab44c",
                    "tokenizer": "fd4146f0082d98fd1fc03bf41349f4d1eed478987ce9a64addc1654dfcad14ee",
                    "config": "28eea9d70afba02740fdd871b148004437cd4f8c31b416776bb864d7895cf83f",
                },
            ),
        ),
    ),
    CatalogModel(
        id="laya-multilingual",
        name="Laya multilingual",
        adapter="laya",
        license="apache-2.0",
        params="322M",
        architecture="mmBERT-base",
        languages="multi",
        status="m0",
        capabilities=Capabilities(languages=("multi",), max_context_tokens=1024),
        notes="Multilingual Laya decision model (100+ languages) exported as one fused "
              "ONNX graph by onnx-community; the graph contains encoder and decision head.",
        variants=(
            Variant(
                name="fp16",
                repo="onnx-community/laya-multilingual-ONNX",
                files={
                    "model": "onnx/model_fp16.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "config.json",
                },
                quant="fp16",
                size_mb=681.8,
                role="default",
                revision="46b77bbf5642fec5f14e540570228a8cbe8ab81f",
                extra_files=("onnx/model_fp16.onnx_data",),
                sha256={
                    "model": "81f641e21289ddb2cd8ac774c1cddebfdb0fa7e4b183e49ad476d2faee46d98a",
                    "tokenizer": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
                    "config": "cc32109b1fded91ec6734127ffb1827f80cdcf8ffdd4eeda75a1b0f4d3687dcc",
                },
                extra_sha256={
                    "onnx/model_fp16.onnx_data": "3c9eb52f5f795f76d7a681197228131fac7e92c3e5a4c90b68ec4065699ee4cf",
                },
                verified_at="2026-10-21",
                latency_ms_p50=37.4,
                parity={
                    "baseline": "upstream ONNX vs PyTorch (model card)",
                    "max_probability_delta": 0.035,
                    "note": "upstream reports fp32 max |dp| 1e-4 and fp16 0.035; local fp16/fp32 agree on answers",
                },
            ),
            Variant(
                name="fp32",
                repo="onnx-community/laya-multilingual-ONNX",
                files={
                    "model": "onnx/model.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "config.json",
                },
                quant="fp32",
                size_mb=1325.6,
                role="quality",
                revision="46b77bbf5642fec5f14e540570228a8cbe8ab81f",
                extra_files=("onnx/model.onnx_data",),
                sha256={
                    "model": "22461d988f675dd6c2b89ff8feb9ed7ffcf7e37854e5ed512f8b275178cf2f9f",
                    "tokenizer": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
                    "config": "cc32109b1fded91ec6734127ffb1827f80cdcf8ffdd4eeda75a1b0f4d3687dcc",
                },
                extra_sha256={
                    "onnx/model.onnx_data": "f01238e183ace77661282e4eb2dd9b872c7044cb9678f904cf409e6cc1146d09",
                },
                verified_at="2026-10-21",
                latency_ms_p50=34.1,
            ),
        ),
    ),
    CatalogModel(
        id="open-jev-deberta-v3-large",
        name="Open-Jev DeBERTa v3 large",
        adapter="open-jev",
        license="apache-2.0",
        params="435M",
        architecture="DeBERTa-v3-large + span head",
        languages="en",
        status="m0",
        capabilities=Capabilities(languages=("en",), max_context_tokens=512),
        notes="Open, Jev-shaped typed-decision model: one state plus typed questions in "
              "one DeBERTa sequence, with a calibrated span-pooling readout.",
        variants=(
            Variant(
                name="q4f16",
                repo="onnx-community/open-jev-deberta-v3-large-ONNX",
                files={
                    "model": "onnx/model_q4f16.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "config.json",
                },
                quant="q4/fp16",
                size_mb=357.0,
                role="edge",
                revision="7c79f25b5ac496089f448a969c801872ad59d31c",
                extra_files=("onnx/model_q4f16.onnx_data",),
                sha256={
                    "model": "f9f014e96682fa0a9ac3c7b3b41786f58a2996dbff890b8ed18eed3869cc495d",
                    "tokenizer": "cd119378b0160677b7a1e561ba29ada83918c1d420b326516a585382e83d9d39",
                    "config": "2ec35432332ee6b5880509eefe44e6279fd9d3543f6ba96098119ffe0b0c2d5e",
                },
                extra_sha256={
                    "onnx/model_q4f16.onnx_data": "89bd9c644c621e3d79fbd1de6bb03e622c38e1dd42e11ab8acece8a236c3481f",
                },
                verified_at="2026-10-21",
                latency_ms_p50=219.4,
                parity={
                    "baseline": "upstream public validation + local variant check",
                    "in_domain_accuracy": 0.854,
                    "ood_accuracy": 0.690,
                    "note": "upstream measured; local fp16/fp32 agree on published example",
                },
            ),
            Variant(
                name="fp16",
                repo="onnx-community/open-jev-deberta-v3-large-ONNX",
                files={
                    "model": "onnx/model_fp16.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "config.json",
                },
                quant="fp16",
                size_mb=883.5,
                role="default",
                revision="7c79f25b5ac496089f448a969c801872ad59d31c",
                extra_files=("onnx/model_fp16.onnx_data",),
                sha256={
                    "model": "2fd22c68fef1c74ac8d4541ea4cbcfbd38553c15552a7bd0a09280c196662e9b",
                    "tokenizer": "cd119378b0160677b7a1e561ba29ada83918c1d420b326516a585382e83d9d39",
                    "config": "2ec35432332ee6b5880509eefe44e6279fd9d3543f6ba96098119ffe0b0c2d5e",
                },
                extra_sha256={
                    "onnx/model_fp16.onnx_data": "f5cc34a4e6cc82483fbaa8bac85f38b008758b13c4dfbb2ac1b637a92da3fd14",
                },
                verified_at="2026-10-21",
                latency_ms_p50=241.2,
                parity={
                    "baseline": "upstream open-jev validation (public gold labels)",
                    "in_domain_accuracy": 0.854,
                    "ood_accuracy": 0.690,
                    "note": "local fp16 matches fp32 on the published example; q4 variants are edge builds",
                },
            ),
            Variant(
                name="q4",
                repo="onnx-community/open-jev-deberta-v3-large-ONNX",
                files={
                    "model": "onnx/model_q4.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "config.json",
                },
                quant="q4",
                size_mb=486.7,
                role="edge",
                revision="7c79f25b5ac496089f448a969c801872ad59d31c",
                extra_files=("onnx/model_q4.onnx_data",),
                sha256={
                    "model": "5be15f8076cc2c16d76fb2e9c8703a0d7d355950422ce47e8dae072ce9ca3b8e",
                    "tokenizer": "cd119378b0160677b7a1e561ba29ada83918c1d420b326516a585382e83d9d39",
                    "config": "2ec35432332ee6b5880509eefe44e6279fd9d3543f6ba96098119ffe0b0c2d5e",
                },
                extra_sha256={
                    "onnx/model_q4.onnx_data": "886cf56af0dd14b6938415af724eb48e5a5d5db3c728c05b50702585c0ea29ab",
                },
                verified_at="2026-10-21",
                latency_ms_p50=56.0,
            ),
            Variant(
                name="fp32",
                repo="onnx-community/open-jev-deberta-v3-large-ONNX",
                files={
                    "model": "onnx/model.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "config.json",
                },
                quant="fp32",
                size_mb=1757.9,
                role="quality",
                revision="7c79f25b5ac496089f448a969c801872ad59d31c",
                extra_files=("onnx/model.onnx_data", "onnx/model.onnx_data_1"),
                sha256={
                    "model": "ceaa072f23425cbc6ed0234453499f4de4bd7beee43499ea984db4eb67409621",
                    "tokenizer": "cd119378b0160677b7a1e561ba29ada83918c1d420b326516a585382e83d9d39",
                    "config": "2ec35432332ee6b5880509eefe44e6279fd9d3543f6ba96098119ffe0b0c2d5e",
                },
                extra_sha256={
                    "onnx/model.onnx_data": "73b7b54307a768c9b2afaf1da70b7be3ff7b9da72cbbdfc1d2d32b27f08d4f55",
                    "onnx/model.onnx_data_1": "a15119f55a8e90de0194a23675817fc63605bbfa209a6f1175e0368e396c97a2",
                },
                verified_at="2026-10-21",
                latency_ms_p50=78.2,
            ),
        ),
    ),
    CatalogModel(
        id="strands-decider-2b",
        name="Strands Decider 2B",
        adapter="strands-decider",
        license="apache-2.0",
        params="2B",
        architecture="Qwen3.5-2B + pointer head",
        languages="multi",
        status="m0",
        capabilities=Capabilities(languages=("multi",), max_context_tokens=4096),
        notes="Qwen3.5-2B decision model with a pointer readout: state once, one prompt "
              "per typed question, logits over the numbered options.",
        variants=(
            Variant(
                name="q4f16",
                repo="onnx-community/strands-decider-2B-hobson-v19-ONNX",
                files={
                    "model": "onnx/model_q4f16.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "config.json",
                },
                quant="q4/fp16",
                size_mb=1113.6,
                role="default",
                revision="31a83e0244b26d939ae33f87189d6c8204003e53",
                extra_files=("onnx/model_q4f16.onnx_data",),
                sha256={
                    "model": "8ee20666f274ebb3774bc98e0026f3bab9cffd3796440c73d7700169e756b68a",
                    "tokenizer": "a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86",
                    "config": "707931b0fef6a8581964f6dc5f6ae97ba8cab078cfb5c9cfff958272e8146343",
                },
                extra_sha256={
                    "onnx/model_q4f16.onnx_data": "d442842f1c5db3dd5b84ac35f683b408a1330075dae6f8f51dcdc5154b64f920",
                },
                verified_at="2026-10-21",
                latency_ms_p50=719.3,
                parity={
                    "baseline": "v19 reference fixture (upstream)",
                    "cases": 3,
                    "max_probability_delta": 0.018,
                    "note": "local q4f16 matches the published fixture answers; q8 is closer in some probabilities",
                },
            ),
            Variant(
                name="q8",
                repo="onnx-community/strands-decider-2B-hobson-v19-ONNX",
                files={
                    "model": "onnx/model_quantized.onnx",
                    "tokenizer": "tokenizer.json",
                    "config": "config.json",
                },
                quant="int8",
                size_mb=1821.4,
                role="quality",
                revision="31a83e0244b26d939ae33f87189d6c8204003e53",
                extra_files=("onnx/model_quantized.onnx_data",),
                sha256={
                    "model": "d156ecf442aaceccd891864b709a7acbea558e382fb1253a7eb8e53c48f46b23",
                    "tokenizer": "a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86",
                    "config": "707931b0fef6a8581964f6dc5f6ae97ba8cab078cfb5c9cfff958272e8146343",
                },
                extra_sha256={
                    "onnx/model_quantized.onnx_data": "0b92e9b7ff19b21c478ae056af397addd0f674180637b4850967fbe8849b21cf",
                },
                verified_at="2026-10-21",
                latency_ms_p50=1467.2,
            ),
        ),
    ),
)

CATALOG_BY_ID: dict[str, CatalogModel] = {model.id: model for model in CATALOG}

#: Adapters that ship with the toolkit today.
IMPLEMENTED_ADAPTERS = frozenset({"gliner2", "gliner2-decide", "julia", "laya", "opendecider", "open-jev", "strands-decider"})


def get(model_id: str) -> CatalogModel | None:
    """Return the catalog entry for ``model_id`` (or ``None``)."""
    return CATALOG_BY_ID.get(model_id)


def parse_alias(model_id: str) -> tuple[str, str | None]:
    """Split a REST model alias ``catalog-id@variant``.

    Plain ids and non-catalog paths are returned unchanged. This is the
    standard-compatible way to select a quantization: the OpenAI/TypeSafe
    ``model`` field remains a single string; no non-standard ``variant`` body
    field is needed.
    """
    if "@" not in model_id:
        return model_id, None
    base, variant_name = model_id.rsplit("@", 1)
    model = get(base)
    if model is not None and any(variant.name == variant_name for variant in model.variants):
        return base, variant_name
    return model_id, None


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
    available = ", ".join(
        f"{variant.name} (default)" if variant.role == "default" else variant.name
        for variant in model.variants
    ) or "(none)"
    raise ModelNotFoundError(f"{model.id!r} has no variant {name!r}; available: {available}")


__all__ = [
    "CATALOG",
    "CATALOG_BY_ID",
    "IMPLEMENTED_ADAPTERS",
    "Capabilities",
    "CatalogModel",
    "ModelInfo",
    "Variant",
    "get",
    "parse_alias",
    "select_variant",
]
