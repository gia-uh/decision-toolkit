# decision-toolkit

Local-first toolkit around **open-weights decision models**: typed **choice /
yes-no (`noul`) / score** questions over a state (text or JSON), answered with a
**calibrated probability for every option** — no generated text, no parsing.

One **ONNX engine** (`onnxruntime`), in-process, on CPU. One **adapter per model
contract** (how a model is fed and read). Three thin surfaces over the same API:
**library**, **CLI** and **REST**.

Co-developed by [GIA-UH](https://github.com/gia-uh) and [Syalia S.R.L.](https://github.com/syalia-srl).

## Installation

```bash
pip install decision-toolkit              # library + CLI (light base)
pip install "decision-toolkit[onnx]"      # + onnxruntime, tokenizers (the model runtime)
pip install "decision-toolkit[server]"    # + REST server (FastAPI)
```

## Three ways to use it

**1. As a library (primary interface)**

```python
from decision_toolkit import DecisionModel, choice, noul, score

# download once: dtk models download gliner2.5-multi
model = DecisionModel("gliner2.5-multi")

answer = model.decide(
    state="Reunión con Acme: acordamos un descuento del 10% y firmar la semana que viene.",
    questions={
        "cat": choice("¿Qué tipo de registro describe el texto?", {
            "reunion": "una reunión o llamada con temas, decisiones y acciones",
            "tareas": "cosas pendientes por hacer, con fechas o responsables",
            "contacto": "datos de una persona: nombre, rol, empresa, teléfono, correo",
            "idea": "una idea, reflexión, observación o pregunta",
            "otras": "saludo, charla suelta o cualquier cosa que no sea ninguna",
        }),
        "urgent": noul("¿Necesita respuesta hoy?"),
    },
)
answer["cat"].choice          # 'reunion'
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

**2. As a CLI**

```bash
dtk models list                                  # curated catalog, adapters, variants
dtk models variants gliner2.5-multi              # real variants (quant, size, role)
dtk models download gliner2.5-multi              # the default variant only
dtk decide --model gliner2.5-multi \
    --state "Reunión con Acme: descuento del 10%, firmar la semana que viene." \
    --q cat='choice:reunion|tareas|contacto|idea|otras' --pretty
dtk classify --model gliner2.5-multi \
    --text "Dentist next Tuesday at 9" --labels "reminder,task,idea" --threshold 0.6
```

**3. As a REST service (optional extra)**

```bash
pip install "decision-toolkit[server]"
dtk serve --model gliner2.5-multi --host 0.0.0.0 --port 8000
```

Endpoints: `POST /v1/systemone`, `POST /v1/decisions` (OpenAI dialect),
`GET /v1/models`, `GET /v1/info`, `GET /v1/health`, `GET /v1/ready`.
`TYPESAFE_BASE_URL=http://localhost:8000` makes existing System One clients work.

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
| `gliner2.5-multi-decide` | `gliner2` | planned — no compatible export yet | — |
| `laya` | `laya` | **m0** — verified against torch (Δp ≤ 0.005) | `q8` (528 MB) |
| `laya-multilingual` | `laya` | planned — no split export published | — |
| `julia-1` | `julia` | **m0** — verified: 100/100 argmax parity, 329 ms | `fp32` (618 MB) |
| `opendecider-nano` | `opendecider` | **m0** — official build; prompt ported from pinned code (530 ms) | `q8f16` (469 MB) |
| `intern-decision-0.8b` | `decoder` | planned — 3-graph export; needs upstream prompt compiler; parity unverified | — |
| `jevk5-lite` | `jevk5` | planned — tiny 9 MB export | `int8` (9 MB) |
| `d3-lite`, `kev-0.8b`, `sieve-0.8b`, `strands-decider-2b`, `thx-01` | — | phase 2 (conversion) | — |

**Statuses**: `m0` runnable today · `m1` adapter validated, export to verify ·
`planned` real export, adapter pending · `phase2` conversion planned · `waitlist`
license pending.

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

## Custom ONNX (at your own risk)

You can run any ONNX file with an adapter you trust — outside the curated
catalog:

```python
model = DecisionModel(
    "./mi-modelo",
    adapter="gliner2",              # the layout you declare
    files={"encoder": "encoder.onnx", "classifier": "classifier.onnx",
           "tokenizer": "tokenizer.json"},
)
# no parity, no license check, no support
```

Custom models do not appear in `dtk models list`; the toolkit never guesses a
layout. If a model deserves support, it goes through curation.

## Development

```bash
uv venv && uv pip install -e ".[dev,onnx,server]"
uv run pytest                       # fast suite (no model downloads)
uv run pytest -m integration        # loads a real model (needs its download/runtime)
```

## License

MIT — see [LICENSE](LICENSE).
