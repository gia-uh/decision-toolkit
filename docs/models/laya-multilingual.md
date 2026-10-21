# Laya multilingual

- **Status:** `m0` (fused ONNX export verified end-to-end)
- **Adapter:** `laya` — fused encoder + decision head, `marker_pos` / `qtype`, calibrated temperatures
- **License:** Apache-2.0
- **Parameters:** 322M · **Architecture:** mmBERT-base · **Languages:** 100+ languages
- **Context:** 1,024 tokens in this export (the upstream checkpoint supports up to 8,192)

## Default variant

| | |
|---|---|
| Repo | `onnx-community/laya-multilingual-ONNX` |
| Files | `onnx/model_fp16.onnx` + `.onnx_data`, `tokenizer.json`, `config.json` |
| Quantization | fp16 weights and activations, fp32 inputs/outputs |
| Download | 682 MB |
| Role | default |

## Variants

| Variant | Quantization | Size | Role | Install |
| --- | --- | ---: | --- | --- |
| `fp16` * | fp16 | 681.8 MB | default | `dtk models download laya-multilingual --variant fp16` |
| `fp32` | fp32 | 1.33 GB | quality | `dtk models download laya-multilingual --variant fp32` |

`dtk models download laya-multilingual` gets `fp16`; `--variant fp32` downloads
the full-precision build. Select one with
`dtk models use laya-multilingual --variant <name>`. See
[Variants and quantization](../VARIANTS.md).

## Reproducibility

- **Revision:** `46b77bbf5642fec5f14e540570228a8cbe8ab81f`
- **Default file hashes (SHA256):**
  - `onnx/model_fp16.onnx` — `81f641e21289ddb2cd8ac774c1cddebfdb0fa7e4b183e49ad476d2faee46d98a`
  - `onnx/model_fp16.onnx_data` — `3c9eb52f5f795f76d7a681197228131fac7e92c3e5a4c90b68ec4065699ee4cf`
  - `tokenizer.json` — `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f`
  - `config.json` — `cc32109b1fded91ec6734127ffb1827f80cdcf8ffdd4eeda75a1b0f4d3687dcc`
- **Verify:** `dtk models verify laya-multilingual --variant fp16`
- **Bench:** `uv run python bench/run.py --suite bench/suites/smoke.jsonl --models laya-multilingual`
- The `fp32` variant hashes live in `catalog.py`.

## Evidence (2026-10-21)

- CPU p50: **37.4 ms** fp16, **34.1 ms** fp32 on the audit machine after warmup.
- Local example (`Help! My payouts have been failing for 3 days!`):
  team **billing 1.00**, urgency 0.756, frustration 1.26.
- The upstream conversion reports max |Δp| **1e-4** fp32 and **0.035** fp16
  against the PyTorch checkpoint on typed-decision sequences.
- fp16 and fp32 agree on the answers in the local smoke/audit cases.

## Contract notes

- Sequence: `[CLS] <type> question: <instructions> [SEP] ([MASK] option) … [SEP] state [SEP]`.
- One fused graph: `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`
  → `logits` per option + `act_logits`.
- Temperatures come from the `laya` section of `config.json` and are clamped to
  `[0.5, 5]`, matching the reference runtime.
- The adapter also supports the split `encoder` + `head` Laya layout; the
  multilingual export is the fused layout.

## Known limitations

- Low-resource languages remain weak in the upstream checkpoint; validate
  your language and label set before relying on calibrated confidence.
- This export's `max_len` is 1,024; the upstream checkpoint supports 8,192 with
  a larger setting, but the pinned export config uses 1,024.
- The adapter uses the fused graph path; the English split checkpoint remains a
  separate catalog entry (`laya`).

## Usage

```bash
dtk models download laya-multilingual
dtk decide --model laya-multilingual --state "Help! My payouts have been failing for 3 days!" \
  --q team='choice:billing=refunds|technical=bugs|sales=pricing' \
  --q urgent='noul:Does this convey urgency?'
```

## Provenance

Converted by `onnx-community` from `convaiinnovations/laya-multilingual`;
the upstream model is the multilingual Laya checkpoint. Apache-2.0.
