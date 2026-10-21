# GLiNER2.5 multi (v1 + classifier)

- **Adapter:** `gliner2` — split encoder + classifier, schema with `[P]` / `[C]` / `[DESCRIPTION]` / `[SEP_TEXT]`
- **License:** Apache-2.0
- **Parameters:** 287M · **Architecture:** GLiNER2.5 multi-v1 · **Languages:** multilingual

## Default variant

| | |
|---|---|
| Repo | `jugaadsrl/gliner2.5-multi-v1-onnx` |
| Files | `fp16_25/encoder_fp16.onnx`, `fp32_25/classifier_fp32.onnx`, `fp16_25/tokenizer.json` |
| Quantization | fp16 encoder + fp32 classifier |
| Download | 577 MB |
| Role | default; `fp32` is a quality variant |

## Variants

| Variant | Quantization | Size | Role | Install |
| --- | --- | ---: | --- | --- |
| `fp16` * | fp16 encoder + fp32 classifier | 577.1 MB | default | `dtk models download gliner2.5-multi --variant fp16` |
| `fp32` | fp32 encoder + fp32 classifier | 1131.9 MB | quality | `dtk models download gliner2.5-multi --variant fp32` |

`dtk models download gliner2.5-multi` gets `fp16`; `--variant fp32` downloads
the quality build alongside it. `dtk models use gliner2.5-multi --variant fp32`
makes it persistent. See [Variants and quantization](../VARIANTS.md).

## Reproducibility

- **Revision:** `3de2c11f55853b7919f985a59bfc8df27b5ec991`
- **Default file hashes (SHA256):**
  - `fp16_25/encoder_fp16.onnx` — `b389910b27261d4999bd41c164bfa0334f76b3bc9fd3d890527b6f07fa1bfb94`
  - `fp32_25/classifier_fp32.onnx` — `3c06934244a4763a7411d9dceb002ba2d5c84d1bb7f7d66aa64acd5653314454`
  - `fp16_25/tokenizer.json` — `c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a`
- **Verify:** `dtk models verify gliner2.5-multi --variant fp16`
- **Bench:** `uv run python bench/run.py --suite bench/suites/smoke.jsonl --models gliner2.5-multi`
- The `fp32` variant is pinned to the same revision; its hashes live in `catalog.py`.

## Evidence (2026-10-19)

- **9/15** on the 15-case Spanish routing suite (meeting / tasks / contact / idea / other categories).
- **353 ms/decision** on CPU.
- Same accuracy as the local int8 build used during development; per-case agreement with that build was not recorded.

## Contract notes

- The classifier input name is read from the graph (`choice_states` in this export, `hidden_state` in others), so the adapter does not hardcode it.
- Labels and descriptions travel in the request; no prompt template is generated.
- `task` defaults to `classify`; override per question with a raw `task` field.

## Known limitations

- This is the **base multi-v1 export**, not the `multi-Decide` fine-tune (no compatible export published yet).
- English/multilingual tokenizer fallback: the run above used the raw `tokenizers` fallback; a `transformers` tokenizer may change scores slightly.
- Benchmarked on a synthetic Spanish routing set; validate on your own data.

## Loading each variant

REST selects the quantization through the standard `model` field using the
alias `gliner2.5-multi@variant`; there is no `variant` field in the request body.

### `fp16` (fp16, default)

**Library**

```python
from decision_toolkit import DecisionModel

model = DecisionModel("gliner2.5-multi", variant="fp16")
```

**CLI**

```bash
dtk models download gliner2.5-multi --variant fp16
dtk models use gliner2.5-multi --variant fp16
dtk decide --model gliner2.5-multi --variant fp16 \
  --state "..." --q 'urgent=noul:Is this urgent?'
```

**REST**

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model": "gliner2.5-multi@fp16", "state": "...",
       "questions": {"urgent": {"type": "noul",
                     "instructions": "Is this urgent?"}}}}'
```

### `fp32` (fp32, quality)

**Library**

```python
from decision_toolkit import DecisionModel

model = DecisionModel("gliner2.5-multi", variant="fp32")
```

**CLI**

```bash
dtk models download gliner2.5-multi --variant fp32
dtk models use gliner2.5-multi --variant fp32
dtk decide --model gliner2.5-multi --variant fp32 \
  --state "..." --q 'urgent=noul:Is this urgent?'
```

**REST**

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model": "gliner2.5-multi@fp32", "state": "...",
       "questions": {"urgent": {"type": "noul",
                     "instructions": "Is this urgent?"}}}}'
```

## Usage

```bash
dtk models download gliner2.5-multi                       # fp16 default
dtk models download gliner2.5-multi --variant fp32        # quality variant
dtk decide --model gliner2.5-multi --state "Meeting with Acme: 10% discount…" \
    --q cat='choice:meeting|tasks|contact|idea|other'

# use the quality variant for one request
dtk decide --model gliner2.5-multi --variant fp32 \
    --state "Meeting with Acme: 10% discount…" \
    --q cat='choice:meeting|tasks|contact|idea|other'
```

## Provenance

Conversion published by `jugaadsrl`; base model `fastino/gliner2.5-multi-v1`.
