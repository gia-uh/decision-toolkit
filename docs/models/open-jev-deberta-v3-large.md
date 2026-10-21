# Open-Jev DeBERTa v3 large

- **Status:** `m0` (ONNX export verified end-to-end)
- **Adapter:** `open-jev` — one sequence, span-pooled option scoring
- **License:** Apache-2.0
- **Parameters:** ~435M · **Architecture:** DeBERTa-v3-large + span head
- **Language:** English · **Context:** 512 tokens

## Default variant

| | |
|---|---|
| Repo | `onnx-community/open-jev-deberta-v3-large-ONNX` |
| Files | `onnx/model_fp16.onnx` + `.onnx_data`, `tokenizer.json`, `config.json` |
| Quantization | fp16 |
| Download | 884 MB |
| Role | default |

## Variants

| Variant | Quantization | Size | Role | Install |
| --- | --- | ---: | --- | --- |
| `fp16` * | fp16 | 883.5 MB | default | `dtk models download open-jev-deberta-v3-large --variant fp16` |
| `fp32` | fp32 | 1.76 GB | quality | `dtk models download open-jev-deberta-v3-large --variant fp32` |
| `q4` | 4-bit | 486.7 MB | edge | `dtk models download open-jev-deberta-v3-large --variant q4` |
| `q4f16` | 4-bit + fp16 | 357.0 MB | edge | `dtk models download open-jev-deberta-v3-large --variant q4f16` |

Start with `fp16`; use `q4`/`q4f16` when download size or CPU work matters,
and re-check thresholds. See [Variants and quantization](../VARIANTS.md).

## Reproducibility

- **Revision:** `7c79f25b5ac496089f448a969c801872ad59d31c`
- **Default file hashes (SHA256):**
  - `onnx/model_fp16.onnx` — `2fd22c68fef1c74ac8d4541ea4cbcfbd38553c15552a7bd0a09280c196662e9b`
  - `onnx/model_fp16.onnx_data` — `f5cc34a4e6cc82483fbaa8bac85f38b008758b13c4dfbb2ac1b637a92da3fd14`
  - `tokenizer.json` — `cd119378b0160677b7a1e561ba29ada83918c1d420b326516a585382e83d9d39`
  - `config.json` — `2ec35432332ee6b5880509eefe44e6279fd9d3543f6ba96098119ffe0b0c2d5e`
- **Verify:** `dtk models verify open-jev-deberta-v3-large --variant fp16`
- **Bench:** `uv run python bench/run.py --suite bench/suites/smoke.jsonl --models open-jev-deberta-v3-large`
- Hashes for the other variants live in `catalog.py`.

## Evidence (2026-10-21)

- CPU p50 after warmup: **241.2 ms** fp16, **78.2 ms** fp32, **56.0 ms** q4,
  **219.4 ms** q4f16. The 4-bit paths are size/edge choices, not automatic wins.
- Published example (`I was charged twice…`): product area **fees & charges**
  (0.70), sentiment **0.819**, refund **0.941** with fp16/fp32.
- The upstream model card reports **0.854** in-domain accuracy, **0.690** on its
  OOD question/option set, and ECE **0.022** in-domain.
- Local fp16 and fp32 agree on the published example; q4 variants are lossy and
  should be validated against your thresholds.

## Contract notes

- Sequence: `[CLS] [STATE] state [Q] instructions [OPT] option … [SEP]`.
- Graph inputs: `input_ids`, `attention_mask`, `seg`, `pair_q`, `pair_opt`.
- One forward pass scores every option; softmax is per question group with the
  checkpoint temperature (`1.05`).
- The state is capped to 256 tokens and the full sequence to 512.

## Known limitations

- English only; 512-token context; it reads the question only partially
  (upstream in-domain 0.85 vs OOD 0.69).
- Score on unseen level sets is weak in the upstream evaluation.
- It chooses among supplied options; it cannot generate explanations.
- This is an independent open-jev reproduction, not an official TypeSafe model.

## Usage

```bash
dtk models download open-jev-deberta-v3-large
dtk decide --model open-jev-deberta-v3-large \
  --state "I was charged twice and want a refund." \
  --q team='choice:billing=payments|technical=bugs|other=anything else' \
  --q refund='noul:Is the customer asking for a refund?'
```

## Provenance

ONNX conversion by `onnx-community` of
`com-kotobalabs/open-jev-deberta-v3-large`, an independent open reproduction of
the Jev-shaped typed-decision contract. Apache-2.0.
