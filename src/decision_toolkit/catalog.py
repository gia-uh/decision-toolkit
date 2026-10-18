"""`decision_toolkit.catalog` — the curated, versioned model list.

The toolkit never picks a model for you: this catalog is the informational,
tested shortlist of open-weights decision models that fit the project's
requirements (commercial-friendly license, CPU-viable, System One-compatible or
with a public adapter). It grows the way ``fastembed`` grows: a few well-tested
entries per release, not an unfiltered mirror of the ecosystem.

Status values:

``m0``        runnable now (its backend ships with the toolkit)
``planned``   catalogued, backend arrives in a later milestone
``variant``   an alternate checkpoint/subfolder of a runnable entry
``waitlist``  meets most requirements; pending license confirmation or CPU recipe
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogModel:
    """One curated model (or variant) of the catalog."""

    id: str
    name: str
    repo: str
    backend: str
    license: str
    params: str
    approx_size: str
    architecture: str = ""
    languages: str = "en"
    tasks: str = "choice / noul / score"
    status: str = "planned"
    subfolder: str | None = None
    notes: str = ""


CATALOG: tuple[CatalogModel, ...] = (
    # ------------------------------------------------------------------ M0
    CatalogModel(
        id="laya",
        name="Laya",
        repo="convaiinnovations/laya",
        backend="laya",
        license="apache-2.0",
        params="421M",
        approx_size="≈0.7 GB",
        architecture="ModernBERT-large",
        tasks="choice / noul / score",
        status="m0",
        notes="English checkpoint (repo root); see laya-multilingual and laya-typed-decisions.",
    ),
    CatalogModel(
        id="laya-multilingual",
        name="Laya multilingual",
        repo="convaiinnovations/laya",
        backend="laya",
        license="apache-2.0",
        params="421M",
        approx_size="≈0.7 GB",
        architecture="ModernBERT-large",
        languages="100+",
        status="variant",
        subfolder="multilingual",
        notes="Same repo, multilingual checkpoint; 100+ languages.",
    ),
    CatalogModel(
        id="laya-typed-decisions",
        name="Laya typed-decisions",
        repo="convaiinnovations/laya",
        backend="laya",
        license="apache-2.0",
        params="421M",
        approx_size="≈0.7 GB",
        architecture="ModernBERT-large",
        status="variant",
        subfolder="typed-decisions",
        notes="Fine-tuned on the typed-decisions benchmark split.",
    ),
    CatalogModel(
        id="julia-1",
        name="Julia 1",
        repo="SupersonicLabs/Julia-1",
        backend="julia",
        license="apache-2.0",
        params="141M",
        approx_size="≈0.55 GB",
        architecture="mmBERT-small",
        languages="52 locales (MASSIVE)",
        status="m0",
        notes="Ships its own Python runtime (pip install -e the model repo).",
    ),
    CatalogModel(
        id="von",
        name="Von 1.3",
        repo="wfzyx/von",
        backend="von",
        license="apache-2.0",
        params="395M",
        approx_size="≈3 GB",
        architecture="ModernBERT-large",
        status="m0",
        notes="von-sdk (pip install von-sdk); weights/SDK frozen — repo archived 2026-10-17.",
    ),
    # ------------------------------------------------------------- planned (M1)
    CatalogModel(
        id="thx-01",
        name="THX-01",
        repo="doofz/THX-01",
        backend="transformers",
        license="apache-2.0",
        params="322M",
        approx_size="≈0.6 GB",
        architecture="mmBERT-base",
        languages="az / ru / en / tr",
        status="planned",
        notes="'Provisional' release; tiny multilingual classifier.",
    ),
    CatalogModel(
        id="gliner2.5-decide",
        name="GLiNER2.5-Decide",
        repo="fastino/GLiNER2.5-Decide",
        backend="gliner2",
        license="apache-2.0",
        params="340M",
        approx_size="≈0.5 GB",
        architecture="DeBERTa-v3-large",
        status="planned",
        notes="Labels at request time; single and multi-label classification.",
    ),
    CatalogModel(
        id="gliner2.5-multi-decide",
        name="GLiNER2.5-multi-Decide",
        repo="fastino/GLiNER2.5-multi-Decide",
        backend="gliner2",
        license="apache-2.0",
        params="287M",
        approx_size="≈0.5 GB",
        architecture="GLiNER2.5 multi",
        languages="multi",
        status="planned",
        notes="Multilingual sibling of GLiNER2.5-Decide.",
    ),
    CatalogModel(
        id="d3-lite",
        name="Decision 3.0 lite",
        repo="vllm-sr/d3-lite",
        backend="gguf",
        license="apache-2.0",
        params="0.85B",
        approx_size="≈1.7 GB (bf16)",
        architecture="Qwen3.5-0.8B",
        status="planned",
        notes="vLLM Semantic Router Decision 3.0 family.",
    ),
    CatalogModel(
        id="intern-decision-0.8b",
        name="Intern-Decision 0.8B",
        repo="internlm/Intern-Decision-0.8B",
        backend="gguf",
        license="apache-2.0",
        params="0.85B",
        approx_size="≈1.7 GB (bf16)",
        architecture="Qwen3.5-0.8B",
        status="planned",
    ),
    CatalogModel(
        id="kev-0.8b",
        name="Kev 0.8B",
        repo="jaredpalmer/kev-0.8b",
        backend="gguf",
        license="apache-2.0",
        params="~0.8B",
        approx_size="≈1.6 GB (bf16)",
        architecture="Qwen3.5-0.8B (LoRA)",
        status="planned",
        notes="Training recipe published alongside the weights.",
    ),
    CatalogModel(
        id="sieve-0.8b",
        name="Sieve 0.8B",
        repo="sthanika-ai/Sieve-0.8B",
        backend="gguf",
        license="apache-2.0",
        params="~0.8B",
        approx_size="≈1.6 GB (bf16)",
        architecture="Qwen3.5-0.8B",
        status="planned",
    ),
    CatalogModel(
        id="strands-decider-2b",
        name="Strands Decider 2B (hobson-v21)",
        repo="StrandsAgents/strands-decider-2B-hobson-v21",
        backend="gguf",
        license="apache-2.0",
        params="2B",
        approx_size="≈4 GB (bf16)",
        architecture="Qwen3.5-2B (adapter)",
        status="planned",
        notes="Agent-oriented decider (tool calls, guardrails).",
    ),
    CatalogModel(
        id="opendecider-nano",
        name="OpenDecider nano",
        repo="manjunathshiva/opendecider-nano",
        backend="onnx",
        license="apache-2.0",
        params="395M",
        approx_size="≈0.8 GB",
        architecture="Ettin encoder 400M",
        status="planned",
        notes="Distilled encoder; ONNX builds proven in-browser.",
    ),
    # --------------------------------------------------------------- waitlist
    CatalogModel(
        id="bosun-0.6b",
        name="Bosun v3.1 0.6B",
        repo="Hanno-Labs/bosun-v3.1-0.6b",
        backend="gguf",
        license="apache-2.0",
        params="0.6B",
        approx_size="≈0.4 GB (Q4)",
        architecture="Qwen3-0.6B (adapter)",
        status="waitlist",
        notes="Tiny decoder; ready to promote once the GGUF backend lands.",
    ),
    CatalogModel(
        id="d3-edge",
        name="Decision 3.0 edge",
        repo="vllm-sr/d3-edge",
        backend="gguf",
        license="apache-2.0",
        params="0.6B",
        approx_size="≈1.2 GB (bf16)",
        architecture="Qwen3.5-0.8B",
        status="waitlist",
        notes="Same family as d3-lite; waiting on the GGUF backend (M1), not on licensing.",
    ),
    CatalogModel(
        id="tev1-0.8b",
        name="Tev1 0.8B",
        repo="togethercomputer/Tev1-0.8B-experimental",
        backend="gguf",
        license="unverified",
        params="0.87B",
        approx_size="≈1.7 GB (bf16)",
        architecture="Qwen3.5-0.8B",
        status="waitlist",
        notes="License not declared on the model card; experimental.",
    ),
    CatalogModel(
        id="d1-omni-600m",
        name="Liquid d1-omni 600M",
        repo="LiquidAI/d1-omni-600M",
        backend="transformers",
        license="other (LFM Open License)",
        params="587M",
        approx_size="≈2.4 GB (F32)",
        architecture="LFM2.5-Encoder-350M",
        status="waitlist",
        notes="Text + image + audio decisions; license needs review before core.",
    ),
    CatalogModel(
        id="bekko-400m",
        name="bekko system-one 400M",
        repo="hotchpotch/bekko-system-one-v0-400m",
        backend="onnx",
        license="unverified",
        params="400M",
        approx_size="≈1.6 GB",
        status="waitlist",
        notes="Tiny Japanese-authored family (17M / 68M / 400M); license pending.",
    ),
)

CATALOG_BY_ID: dict[str, CatalogModel] = {model.id: model for model in CATALOG}
CATALOG_BY_REPO: dict[str, CatalogModel] = {}
for _model in CATALOG:
    # first entry wins: repo-level defaults (laya -> English root entry)
    CATALOG_BY_REPO.setdefault(_model.repo, _model)

#: Backends shipped by the toolkit today. Everything else is catalogued only.
IMPLEMENTED_BACKENDS = frozenset({"laya", "julia", "von", "systemone"})


def get(model_id: str) -> CatalogModel | None:
    """Return the catalog entry for ``model_id`` (or ``None``)."""
    return CATALOG_BY_ID.get(model_id)


def find_by_repo(repo: str) -> CatalogModel | None:
    """Return the catalog entry for an ``org/name`` repo (or ``None``)."""
    return CATALOG_BY_REPO.get(repo)


__all__ = [
    "CatalogModel",
    "CATALOG",
    "CATALOG_BY_ID",
    "CATALOG_BY_REPO",
    "IMPLEMENTED_BACKENDS",
    "get",
    "find_by_repo",
]
