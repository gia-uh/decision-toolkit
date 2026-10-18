# decision-toolkit

Local-first toolkit around **open-weights decision models** (System One): typed
**choice / yes-no (`noul`) / score** questions over a state (text or JSON), answered
with a **calibrated probability for every option** — no generated text, no parsing,
no prompt output to clean up.

It runs on **CPU** by default (GPU is used when available, never required), ships as a
**library first**, and its **CLI and REST server are thin adapters over the same public
API**. It is not a gateway, not a hosting service and not a new wire protocol: the
toolkit speaks **TypeSafe's System One contract** (`state` + typed `questions`) and the
OpenAI Decisions dialect alongside it.

Co-developed by [GIA-UH](https://github.com/gia-uh) and [Syalia S.R.L.](https://github.com/syalia-srl).

## Installation

```bash
pip install decision-toolkit              # library + CLI (light base: no torch)
pip install "decision-toolkit[laya]"      # + the Laya runtime
pip install "decision-toolkit[von]"       # + the Von SDK
pip install "decision-toolkit[server]"    # + REST server (FastAPI)
```

The base install deliberately avoids heavy dependencies (`torch`, `onnxruntime`,
`llama.cpp`). Backend extras pull only what the models you use need.

## Three ways to use it

**1. As a library (primary interface)**

```python
from decision_toolkit import DecisionModel, choice, noul, score

model = DecisionModel("laya")                 # or "laya-multilingual", "julia-1", "von", …

answer = model.decide(
    state="I was charged twice for the March invoice and nobody has replied.",
    questions={
        "team":   choice("Which team should handle this?", {
                      "billing": "charges, invoices, refunds",
                      "technical": "bugs and outages",
                  }),
        "urgent": noul("Does the customer need a reply today?"),
        "tone":   score("How upset is the customer?", ["calm", "annoyed", "furious"]),
    },
)
answer["team"].choice            # 'billing'
answer["team"].confidence        # 0.93
answer["urgent"].noul            # 0.91  (probability of "true")
answer["tone"].score             # 1.2   (expected level)

# One-label classification with a threshold (informative, not prescriptive)
verdict = model.classify("Dentist next Tuesday at 9", {
    "reminder": "an event with a date or time",
    "task": "a list of things to do",
    "idea": "an idea or reflection",
}, threshold=0.6)
verdict.label, verdict.confidence, verdict.below_threshold
```

**2. As a CLI**

```bash
dtk models list                                  # the curated catalog (12 models + waitlist)
dtk models download laya-multilingual
dtk models use laya-multilingual
dtk decide --state "Refund the duplicate charge today." \
    --q team='choice:billing|cancellation' --q urgent='noul' --q tone='score:calm|annoyed|furious'
dtk classify --text "Dentist next Tuesday at 9" --labels "reminder,task,idea" --threshold 0.6
```

**3. As a REST service (optional extra)**

```bash
pip install "decision-toolkit[server]"
dtk serve --model laya-multilingual --host 0.0.0.0 --port 8000
```

Endpoints: `POST /v1/systemone`, `POST /v1/decisions` (OpenAI Decisions dialect),
`GET /v1/models`, `GET /v1/info`, `GET /v1/health`, `GET /v1/ready`.
Anything that talks to Jev talks to `dtk serve` by changing the base URL
(`TYPESAFE_BASE_URL=http://localhost:8000`).

## Model management (the toolkit's own cache)

Models live in the toolkit's **own cache**, independent from HuggingFace's:

```bash
dtk models                            # your cache (★ model in use, origin, size)
dtk models list                       # the curated catalog: status, license, params, backend
dtk models download THX-01            # into the toolkit cache (the only command that uses the network)
dtk models use laya-multilingual      # set the model in use (persistent)
dtk models import ./my-model --name my-laya-finetune
dtk models delete THX-01              # refuses the model in use
```

Configuration lives in `<home>/config.json` (default home
`~/.local/share/decision-toolkit/`, created on first use):

```json
{
  "cache_dir": "~/.local/share/decision-toolkit/models",
  "current": "laya-multilingual",
  "device": "auto"
}
```

There are **no environment variables**; for non-default installations pass
`--home DIR` (global option, before the command). `device` is `auto`
(CPU unless you ask otherwise), or `cpu`, `cuda`, `mps`, `cuda:1`, …

## The catalog (M0)

Twelve models were selected to start (all Apache-2.0, all CPU-viable). The toolkit
adds models the way `fastembed` does: a curated, versioned, tested list that grows
release by release. Backends marked `M1` are catalogued but not yet runnable.

| id | model | params | backend | status |
| --- | --- | --- | --- | --- |
| `laya` / `laya-multilingual` / `laya-typed-decisions` | Laya (Convai Innovations) | 421M | laya | **M0** |
| `julia-1` | Julia 1 (Supersonic Labs) | 141M | julia | **M0** |
| `von` | Von (wfzyx) | 395M | von | **M0** (weights/SDK frozen; repo archived 2026-10-17) |
| `thx-01` | THX-01 (HAL-X AI) | 322M | transformers | M1 |
| `gliner2.5-decide` / `gliner2.5-multi-decide` | GLiNER2.5-Decide (Fastino) | 340M / 287M | gliner2 | M1 |
| `d3-lite` | Decision 3.0 lite (vLLM-SR) | 0.85B | gguf | M1 |
| `intern-decision-0.8b` | Intern-Decision (InternLM) | 0.85B | gguf | M1 |
| `kev-0.8b` | Kev (Jared Palmer) | ~0.8B | gguf | M1 |
| `sieve-0.8b` | Sieve (Sthānika AI) | ~0.8B | gguf | M1 |
| `strands-decider-2b` | Strands Decider (agentic) | 2B | gguf | M1 |
| `opendecider-nano` | OpenDecider nano | 395M | onnx | M1 |

A waitlist (Bosun, d3-edge, Tev1, d1-omni-600M, OneJev, bekko, …) is tracked in
`definition order` inside the catalog; entries join the runnable set when a backend
and a license check are complete. **Non-commercial licenses do not enter the core.**

## Highlights

- **Library-first design**: the CLI and the REST server are thin adapters over the same public API.
- **CPU-first**: a decision is one scoring pass; encoders run on ONNX/OpenVINO-class runtimes, decoders on GGUF.
- **System One canonical**: `state` + `questions` (`choice` / `noul` / `score`) in and out, with the OpenAI Decisions dialect translated at the edges (`/v1/decisions`).
- **Calibration is first-class**: answers carry probabilities and confidence; `decision_toolkit.calibration` ships Brier/ECE and threshold sweeps so a threshold means the same across models.
- **Own model cache**: only `models download` touches the network; everything else works with what you already have.
- **USB-stick friendly**: no daemon, no account, no API key. `pip install` → decision on CPU in minutes.

> **Status:** M0 scaffold (October 2026). The public API is being stabilised before the
> first release. Backends for the GGUF/ONNX families land in M1; `dtk serve` is complete
> for the M0 backends.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest                       # fast suite (no model downloads)
uv run pytest -m integration        # loads a real model (CPU, needs its runtime installed)
```

## License

MIT — see [LICENSE](LICENSE).
