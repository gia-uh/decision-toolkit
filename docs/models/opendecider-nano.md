# OpenDecider nano

- **Adapter:** `opendecider` — single graph, softmax over `[MASK]` marker logits
- **License:** Apache-2.0
- **Parameters:** 395M · **Architecture:** Ettin encoder 400M

## Default variant

| | |
|---|---|
| Repo | `manjunathshiva/opendecider-nano-ONNX` |
| Files | `onnx/model_q8f16.onnx`, `tokenizer.web.json`, `opendecider.json` |
| Quantization | q8f16 |
| Download | 468.8 MB |
| Role | default; `fp16` is a quality variant |

## Variants

| Variant | Quantization | Size | Role | Install |
| --- | --- | ---: | --- | --- |
| `q8f16` * | int8/fp16 | 468.8 MB | default | `dtk models download opendecider-nano --variant q8f16` |
| `fp16` | fp16 | 791.9 MB | quality | `dtk models download opendecider-nano --variant fp16` |

`dtk models download opendecider-nano` gets `q8f16`; `--variant fp16` downloads
the quality build. Select one with
`dtk models use opendecider-nano --variant <name>`. See
[Variants and quantization](../VARIANTS.md).

## Reproducibility

- **Revision:** `5137ae7b4f1cf8c55f1d05d3f5cdd3694d92da3e`
- **Default file hashes (SHA256):**
  - `onnx/model_q8f16.onnx` — `80436dbed1a284548b5e6c3975bb26df81fb493fc4e64629e5a6ef490bf0e0ae`
  - `tokenizer.web.json` — `fd4146f0082d98fd1fc03bf41349f4d1eed478987ce9a64addc1654dfcad14ee`
  - `opendecider.json` — `28eea9d70afba02740fdd871b148004437cd4f8c31b416776bb864d7895cf83f`
- **Verify:** `dtk models verify opendecider-nano --variant q8f16`
- **Bench:** `uv run python bench/run.py --suite bench/suites/smoke.jsonl --models opendecider-nano`
- The `fp16` variant is pinned to the same revision; its hashes live in `catalog.py`.

## Evidence (2026-10-19)

- Example: charged-twice ticket → **billing 0.907**, urgent 0.572, tone 1.235.
- **530 ms/decision** on CPU.
- The prompt/readout are a literal port of the pinned packaging code (`opendecider/prompt.py::nano_ids`) and `opendecider.json` (special ids, max_len, readout definition).
- The official benchmark inputs are **hashed**, so no per-case torch parity is possible from the published artifacts.

## Contract notes

- Sequence: `[CLS] question: <instructions> [SEP] ([MASK] opt)… [SEP] input: <state> [SEP]`.
- `special_ids` come from `opendecider.json` (cls 50281, sep 50282, mask 50284); `max_len` 2048.
- The answer is `softmax(logits at [MASK] positions)`; no temperature.

## Known limitations

- q8f16 build; quality vs fp16 was not measured.
- No independent parity dataset (the official benchmark inputs are hashed).
- The export's tokenizer is a web tokenizer (`tokenizer.web.json`), loaded through the `tokenizers` fallback.

## Loading each variant

REST selects the quantization through the standard `model` field using the
alias `opendecider-nano@variant`; there is no `variant` field in the request body.

### `q8f16` (int8/fp16, default)

**Library**

```python
from decision_toolkit import DecisionModel

model = DecisionModel("opendecider-nano", variant="q8f16")
```

**CLI**

```bash
dtk models download opendecider-nano --variant q8f16
dtk models use opendecider-nano --variant q8f16
dtk decide --model opendecider-nano --variant q8f16 \
  --state "..." --q 'urgent=noul:Is this urgent?'
```

**REST**

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model": "opendecider-nano@q8f16", "state": "...",
       "questions": {"urgent": {"type": "noul",
                     "instructions": "Is this urgent?"}}}}'
```

### `fp16` (fp16, quality)

**Library**

```python
from decision_toolkit import DecisionModel

model = DecisionModel("opendecider-nano", variant="fp16")
```

**CLI**

```bash
dtk models download opendecider-nano --variant fp16
dtk models use opendecider-nano --variant fp16
dtk decide --model opendecider-nano --variant fp16 \
  --state "..." --q 'urgent=noul:Is this urgent?'
```

**REST**

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model": "opendecider-nano@fp16", "state": "...",
       "questions": {"urgent": {"type": "noul",
                     "instructions": "Is this urgent?"}}}}'
```

## Usage

```bash
dtk models download opendecider-nano                      # q8f16 default
dtk models download opendecider-nano --variant fp16       # quality variant
dtk decide --model opendecider-nano --state "I was charged twice…" \
    --q team='choice:billing|technical|other' --q urgent='noul:Needs a reply today?'

dtk decide --model opendecider-nano --variant fp16 \
    --state "I was charged twice…" \
    --q team='choice:billing|technical|other' --q urgent='noul:Needs a reply today?'
```

## Provenance

Official ONNX build by the OpenDecider author; base model `manjunathshiva/opendecider-nano` at revision `beeeb640…`.
