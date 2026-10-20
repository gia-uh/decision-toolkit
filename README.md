# decision-toolkit

Local-first toolkit around **open-weights decision models**: typed **choice /
yes-no (`noul`) / score** questions over a state (text or JSON), answered with a
**calibrated probability for every option** — no generated text, no parsing.

One **ONNX engine** (`onnxruntime`), in-process, on CPU. One **adapter per model
contract** (how a model is fed and read). Three thin surfaces over the same API:
**library**, **CLI** and **REST**.

Co-developed by [GIA-UH](https://github.com/gia-uh) and [Syalia S.R.L.](https://github.com/syalia-srl).

**Examples:** [docs/EXAMPLES.md](https://github.com/gia-uh/decision-toolkit/blob/main/docs/EXAMPLES.md) covers the library, REST and
CLI. `dtk --help` and every subcommand help also include runnable examples.

## Installation

```bash
pip install decision-toolkit              # library + CLI (0.1.1 on PyPI)
pip install "decision-toolkit[onnx]"      # + onnxruntime, tokenizers (the model runtime)
pip install "decision-toolkit[server]"    # + REST server (FastAPI)
```

PyPI: <https://pypi.org/project/decision-toolkit/0.1.1/>. See
[docs/RELEASE.md](https://github.com/gia-uh/decision-toolkit/blob/main/docs/RELEASE.md)
for the Trusted Publishing flow and clean-install checks.

## Three ways to use it

**1. As a library (primary interface)**

```python
from decision_toolkit import DecisionModel, choice, noul, score

# download once: dtk models download gliner2.5-multi
model = DecisionModel("gliner2.5-multi")

answer = model.decide(
    state="Meeting with Acme: we agreed a 10% discount and will sign next week.",
    questions={
        "cat": choice("What kind of record is this?", {
            "meeting": "a meeting or call with topics, decisions and actions",
            "tasks": "things to do, with dates or owners",
            "contact": "a person's details: name, role, company, phone, email",
            "idea": "an idea, reflection, observation or question",
            "other": "a greeting, small talk or anything else",
        }),
        "urgent": noul("Does it need a reply today?"),
    },
)
answer["cat"].choice          # 'meeting'
answer["cat"].confidence      # 0.77
answer["urgent"].noul         # probability of "true"
```

```python
verdict = model.classify(
    "Remind me: dentist next Tuesday at 9 am.",
    {"reminder": "an event with a date or time",
     "task": "a list of things to do",
     "idea": "an idea or reflection"},
    threshold=0.6,
)
verdict.label, verdict.confidence, verdict.below_threshold
```

The core is the **superset** of the surfaces. It also exposes lifecycle,
usage, batch/async, capabilities and structured events:

```python
from decision_toolkit import DecisionModel, request_context, noul

with DecisionModel("gliner2.5-multi", strict=True) as model:
    model.warmup()
    with request_context("req-42"):
        decision = model.decide("Customer wants a refund today.",
                                {"urgent": noul("Does it need a reply today?")})
    batch = model.decide_many(["first state", "second state"],
                              {"urgent": noul("Today?")})

decision["urgent"].confidence       # normalized probability
decision.usage.to_dict()            # input/output tokens + estimated flag
decision.latency_ms                 # per-decision latency
decision.request_id                 # 'req-42'
decision.truncated, decision.warnings
```

`strict=True` turns silent truncation into `TruncationError`. Capabilities are
validated before inference (`box`/assets are rejected by text-only models).
Custom adapters can be registered with `register_adapter`; local ONNX models
can be opened with `DecisionModel.from_onnx(...)`. See
[docs/CORE.md](https://github.com/gia-uh/decision-toolkit/blob/main/docs/CORE.md) for the full phase 1 contract, error taxonomy,
logging events and the expressiveness inventory.

**2. As a CLI**

```bash
dtk models list --adapter --quant                # catalog, variants and cache state
dtk models variants gliner2.5-multi --json       # real variants (quant, size, role)
dtk models download gliner2.5-multi --dry-run    # show the download first
dtk models use gliner2.5-multi --variant fp16
dtk models current --json                        # includes the current variant
dtk decide --model gliner2.5-multi \
    --state "Meeting with Acme: 10% discount, sign next week." \
    --questions-file questions.yaml --json
dtk decide --model gliner2.5-multi \
    --input-jsonl states.jsonl --questions-file questions.yaml \
    --output-jsonl decisions.jsonl --batch-size 32
dtk classify --model gliner2.5-multi \
    --text "Dentist next Tuesday at 9" --labels "reminder,task,idea" \
    --threshold 0.6 --json
dtk config list --json
dtk models verify --all --json
```

`dtk` has consistent exit codes (`0` ok, `1` runtime, `2` usage, `3`
verification), `--json` everywhere, quiet/verbose modes and no tracebacks for
expected failures. See [docs/CLI.md](https://github.com/gia-uh/decision-toolkit/blob/main/docs/CLI.md) for questions files, JSONL
batch, custom `--adapter/--files`, config and completion.

**3. As a REST service (optional extra)**

```bash
pip install "decision-toolkit[server]"
dtk serve --model gliner2.5-multi --warmup \
  --api-key "$DECISION_TOOLKIT_API_KEY" \
  --log-format json
```

Endpoints: `POST /v1/systemone` (System One 1.14), `POST /v1/decisions`
(OpenAI Decisions beta), `GET /v1/models` (OpenAI shape),
`GET /v1/decision-models` (rich catalog), `GET /v1/info`, `GET /health`,
`GET /ready`. Requests may select `model` and `variant`; failures use the OpenAI
`{"error": {...}}` envelope and every response carries `x-request-id`.
`TYPESAFE_BASE_URL=http://localhost:8000` makes existing System One clients work.
See [docs/REST.md](https://github.com/gia-uh/decision-toolkit/blob/main/docs/REST.md) for the pinned contracts, limits, auth and
the standard-evolution process.

## The catalog is curated

The catalog is **ours**, not a mirror of HuggingFace. A model enters only when it
is understood and verified:

1. clean commercial license;
2. an adapter whose contract we can implement **from a real export**;
3. a real ONNX export on the Hub;
4. measured parity against the original on a fixed sample;
5. CPU-viable latency.

Every variant points to files that exist. No placeholders, no speculative
adapters (there is no generic "any ONNX" adapter).

| id | adapter | status | default variant |
| --- | --- | --- | --- |
| `gliner2.5-multi` | `gliner2` | **m0** — verified end-to-end (9/15, 353 ms) | `fp16` (577 MB) |
| `gliner2.5-decide` | `gliner2-decide` | **m0** — fused export verified (refund 0.944, 912 ms; English-only) | `q4f16` (524 MB) |
| `laya` | `laya` | **m0** — verified against torch (Δp ≤ 0.005) | `q8` (528 MB) |
| `julia-1` | `julia` | **m0** — verified: 100/100 argmax parity, 329 ms | `fp32` (618 MB) |
| `opendecider-nano` | `opendecider` | **m0** — official build; prompt ported from pinned code (530 ms) | `q8f16` (469 MB) |

**Only `m0` models are in the catalog.** Researched-but-unavailable models
(planned, conversion pending, broken exports) are listed in
[docs/ROADMAP.md](https://github.com/gia-uh/decision-toolkit/blob/main/docs/ROADMAP.md) and are not downloadable through the toolkit.

Per-model cards with the contract, measured evidence and known limitations:
[docs/models](https://github.com/gia-uh/decision-toolkit/tree/main/docs/models).

## Variants and quantization

A quantization is a **variant** of a model, with its real repo, files and role:

```yaml
gliner2.5-multi:
  fp16:  encoder_fp16.onnx + classifier_fp32.onnx      # 577 MB  default
  fp32:  encoder_fp32.onnx + classifier_fp32.onnx      # 1132 MB quality
```

- `dtk models download <id>` gets the **default** variant; `--variant fp32` gets
  another.
- A variant becomes default only after passing the parity gate (top-1 ≥ 99%,
  mean |Δp| ≤ 0.02, accuracy delta ≤ 1 pp against the fp32 build on a fixed
  sample) and measuring latency on CPU.
- Unverified variants are never default. No silent fallback.

## Operations

The server has bounded inference concurrency (`--max-concurrent`), an optional
per-key rate limiter (`--rate-limit`, 429 + `Retry-After`), Prometheus metrics
at `/metrics`, and optional OpenTelemetry spans (`--tracing`). Saturation
returns 503 `server_busy` instead of building an unbounded queue. State and
question text never become log/metric labels.

```bash
dtk serve --model gliner2.5-multi --warmup --max-concurrent 4 \
  --rate-limit 120 --rate-window 60
PYTHONPATH=src python bench/load.py --base-url http://127.0.0.1:8000 \
  --concurrency 10 --requests 30
```

See [docs/OPERATIONS.md](https://github.com/gia-uh/decision-toolkit/blob/main/docs/OPERATIONS.md) for metric names, alerts, the
runbook and the multi-worker caveat.

## Reproducibility

Every catalog variant is pinned to a real HuggingFace **commit SHA** and carries
a **SHA256 per file**, including external `.onnx.data` files. Downloads reject
mismatches and verification treats the catalog hashes as the source of truth:

```bash
dtk models verify --all --json    # exit 3 if any file/hash/session fails
```

Latency/parity/calibration evidence is reproducible with the harness in
[`bench/`](https://github.com/gia-uh/decision-toolkit/tree/main/bench) and documented in [docs/BENCH.md](https://github.com/gia-uh/decision-toolkit/blob/main/docs/BENCH.md). CI runs
lint, unit/contract tests and a CLI smoke from source on Python 3.12/3.13.

There is **no wheel or PyPI release yet**; packaging is a later milestone. See
[CHANGELOG.md](https://github.com/gia-uh/decision-toolkit/blob/main/CHANGELOG.md), [CONTRIBUTING.md](https://github.com/gia-uh/decision-toolkit/blob/main/CONTRIBUTING.md),
[SECURITY.md](https://github.com/gia-uh/decision-toolkit/blob/main/SECURITY.md) and
[THIRD_PARTY_NOTICES.md](https://github.com/gia-uh/decision-toolkit/blob/main/THIRD_PARTY_NOTICES.md).

## Custom ONNX (at your own risk)

You can run any ONNX file with an adapter you trust — outside the curated
catalog:

```python
model = DecisionModel.from_onnx(
    "./my-model",
    adapter="gliner2",              # the layout you declare
    files={"encoder": "encoder.onnx", "classifier": "classifier.onnx",
           "tokenizer": "tokenizer.json"},
)
# no parity, no license check, no support
```

Third-party layouts can be registered in-process:

```python
from decision_toolkit import register_adapter
register_adapter("my-layout", MyAdapter)
```

Custom models do not appear in `dtk models list`; the toolkit never guesses a
layout. If a model deserves support, it goes through curation.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install pytest ruff PyYAML huggingface_hub fastapi httpx pydantic numpy
PYTHONPATH=src pytest -q             # fast suite (no model downloads)
PYTHONPATH=src pytest -m integration -o addopts=""   # real model, opt-in
```

No wheel or editable install is required while the project is pre-distribution.
CI runs the same source-tree commands on Python 3.12/3.13. See
[CONTRIBUTING.md](https://github.com/gia-uh/decision-toolkit/blob/main/CONTRIBUTING.md).

## License

MIT — see [LICENSE](https://github.com/gia-uh/decision-toolkit/blob/main/LICENSE).
