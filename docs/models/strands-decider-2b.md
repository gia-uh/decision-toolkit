# Strands Decider 2B

- **Status:** `m0` (ONNX export verified end-to-end)
- **Adapter:** `strands-decider` — Qwen3.5-2B torso + pointer head
- **License:** Apache-2.0
- **Parameters:** 2B · **Architecture:** Qwen3.5-2B-Base + LoRA + pointer head
- **Languages:** multilingual · **Context:** 4,096 tokens

## Default variant

| | |
|---|---|
| Repo | `onnx-community/strands-decider-2B-hobson-v19-ONNX` |
| Files | `onnx/model_q4f16.onnx` + `.onnx_data`, `tokenizer.json`, `config.json` |
| Quantization | 4-bit weights, fp16 activations |
| Download | 1.11 GB |
| Role | default |

## Variants

| Variant | Quantization | Size | Role | Install |
| --- | --- | ---: | --- | --- |
| `q4f16` * | 4-bit + fp16 | 1.11 GB | default | `dtk models download strands-decider-2b --variant q4f16` |
| `q8` | int8 block | 1.82 GB | quality | `dtk models download strands-decider-2b --variant q8` |

`q8` is the closer-to-reference build; `q4f16` is smaller and faster on CPU.
Select one with `dtk models use strands-decider-2b --variant <name>`. See
[Variants and quantization](../VARIANTS.md).

## Reproducibility

- **Revision:** `31a83e0244b26d939ae33f87189d6c8204003e53`
- **Default file hashes (SHA256):**
  - `onnx/model_q4f16.onnx` — `8ee20666f274ebb3774bc98e0026f3bab9cffd3796440c73d7700169e756b68a`
  - `onnx/model_q4f16.onnx_data` — `d442842f1c5db3dd5b84ac35f683b408a1330075dae6f8f51dcdc5154b64f920`
  - `tokenizer.json` — `a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86`
  - `config.json` — `707931b0fef6a8581964f6dc5f6ae97ba8cab078cfb5c9cfff958272e8146343`
- **Verify:** `dtk models verify strands-decider-2b --variant q4f16`
- **Bench:** `uv run python bench/run.py --suite bench/suites/smoke.jsonl --models strands-decider-2b`
- The `q8` hashes live in `catalog.py`.

## Evidence (2026-10-21)

Reference fixture from the upstream export (`Help! My payouts have been failing
for 3 days!`):

| Variant | urgent P(true) | team billing | frustration score |
|---|---:|---:|---:|
| upstream reference | 0.829 | 0.844 | 1.10 |
| `q4f16` | 0.808 | 0.845 | 1.11 |
| `q8` | 0.816 | 0.836 | 1.10 |

- Max probability delta on the three reference questions: **0.018** for
  `q4f16`, smaller for `q8` on most fields.
- CPU p50 after warmup: **719 ms** q4f16, **1,467 ms** q8.
- The upstream card reports q4f16 max |Δp| 0.072 on 11 questions and q8 0.016
  on its WebGPU checks.

## Contract notes

- Prompt: `<state>…</state>` plus one `<question type="…">` block ending in
  `<answer>`; options are numbered and each option's last token is a pointer.
- Graph inputs: `input_ids`, `attention_mask`, `answer_pos`, `option_pos`.
- Logits are divided by `decider.temperature_by_kind` and softmaxed per question.
- `noul` uses false/true slots; `score` uses level indices with descriptions.

## Known limitations

- 2B model: CPU latency is seconds, not milliseconds; q8 is slower than q4f16.
- The upstream project warns that behavior under concurrent requests is not
  verified and that its server is experimental.
- The ONNX export is a WebGPU-oriented quantized build; use `q8` when
  probabilities need to be closest to the reference.
- No image/audio support in this text-only projection.

## Loading each variant

REST selects the quantization through the standard `model` field using the
alias `strands-decider-2b@variant`; there is no `variant` field in the request body.

### `q4f16` (q4/fp16, default)

**Library**

```python
from decision_toolkit import DecisionModel

model = DecisionModel("strands-decider-2b", variant="q4f16")
```

**CLI**

```bash
dtk models download strands-decider-2b --variant q4f16
dtk models use strands-decider-2b --variant q4f16
dtk decide --model strands-decider-2b --variant q4f16 \
  --state "..." --q 'urgent=noul:Is this urgent?'
```

**REST**

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model": "strands-decider-2b@q4f16", "state": "...",
       "questions": {"urgent": {"type": "noul",
                     "instructions": "Is this urgent?"}}}}'
```

### `q8` (int8, quality)

**Library**

```python
from decision_toolkit import DecisionModel

model = DecisionModel("strands-decider-2b", variant="q8")
```

**CLI**

```bash
dtk models download strands-decider-2b --variant q8
dtk models use strands-decider-2b --variant q8
dtk decide --model strands-decider-2b --variant q8 \
  --state "..." --q 'urgent=noul:Is this urgent?'
```

**REST**

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model": "strands-decider-2b@q8", "state": "...",
       "questions": {"urgent": {"type": "noul",
                     "instructions": "Is this urgent?"}}}}'
```

## Usage

```bash
dtk models download strands-decider-2b
dtk decide --model strands-decider-2b \
  --state "Help! My payouts have been failing for 3 days!" \
  --q team='choice:billing=refunds|technical=bugs|sales=pricing' \
  --q urgent='noul:Does this convey urgency?' \
  --q tone='score:calm|frustrated|depressed'
```

## Provenance

ONNX conversion by `onnx-community` of
`StrandsAgents/strands-decider-2B-hobson-v19`; Apache-2.0, built from
Qwen3.5-2B-Base plus a LoRA adapter and pointer head.
