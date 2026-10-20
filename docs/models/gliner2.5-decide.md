# GLiNER2.5-Decide

- **Status:** `m0` (verified end-to-end)
- **Adapter:** `gliner2-decide` — fused graph with `[L]` label markers and `[SEP_STRUCT]`
- **License:** Apache-2.0
- **Parameters:** 340M · **Architecture:** DeBERTa-v3-large · **Language:** English only

## Variant

| | |
|---|---|
| Repo | `onnx-community/GLiNER2.5-Decide-ONNX` |
| Files | `onnx/model_q4f16.onnx`, `onnx/model_q4f16.onnx_data`, `tokenizer.json` |
| Quantization | q4f16 (4-bit MatMul, fp16 elsewhere) |
| Download | 523.5 MB |
| Role | default |

## Reproducibility

- **Revision:** `18271032588167fe9a3fbb8c5a4900b1231f8119`
- **Default file hashes (SHA256):**
  - `onnx/model_q4f16.onnx` — `93f3e0d4ed8bbd4858840f8a2e66028a70e86a482e1ff9822d2ac726381684f6`
  - `onnx/model_q4f16.onnx_data` — `df2b05dc97ff875e0d2b51ec21de0049b7b3e7feddaa5224d318dc79e66210e8`
  - `tokenizer.json` — `3ad87d9ffe669147063e70850927dd2da90249e2acc5c8527f1eb65df467bcc8`
- **Verify:** `dtk models verify gliner2.5-decide --variant q4f16`
- **Bench:** `uv run python bench/run.py --suite bench/suites/smoke.jsonl --models gliner2.5-decide`

## Evidence (2026-10-19)

- Model-card refund example → **`refund_request` 0.944** (the card expects `refund_request`).
- Flight-change example → **`seat_change` 0.593**; `change` second at 0.334 (the card's potential output was `change`).
- Sentiment example → **`positive` 0.99**; the card's potential output was `mixed`. Documented as a known divergence, not hidden.
- **912 ms/decision** on CPU.

## Contract notes

- One fused graph: `input_ids`, `attention_mask`, `marker_positions` → `logits` per label.
- Sequence: `( [P] prompt [DESCRIPTION] … ( [L] label … ) ) [SEP_STRUCT] [SEP_TEXT] words`.
- The state is lowercased and tokenized word by word; labels keep their case.
- `noul` uses labels `yes` / `no`; `score` uses the level strings as labels.

## Known limitations

- **English only** (declared by the export).
- Quantized q4f16; the fp16/fp32 builds may resolve the sentiment example differently.
- The model card's examples are "potential outputs", not guarantees; validate on your data.

## Usage

```bash
dtk models download gliner2.5-decide
dtk decide --model gliner2.5-decide --state "My subscription renewed after the outage…" \
    --q intent='choice:refund_request|order_status|cancel_subscription|other'
```

## Provenance

ONNX Community export of `fastino/GLiNER2.5-Decide`; classification path only (extraction heads are not exported).
