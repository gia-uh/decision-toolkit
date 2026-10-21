# Catalog proposal: three ONNX decision models

**Date:** 2026-10-21  
**Status:** implemented and verified in the working tree  
**Scope:** add Laya multilingual, Open-Jev DeBERTa v3 large and Strands Decider 2B

## Executive summary

Three public ONNX decision models meet the catalog gates for this iteration:

| Model | License | Adapter | Variants | Default |
|---|---|---|---|---:|
| Laya multilingual | Apache-2.0 | existing `laya` + fused branch | fp16, fp32 | fp16 |
| Open-Jev DeBERTa v3 large | Apache-2.0 | new `open-jev` | fp16, fp32, q4, q4f16 | fp16 |
| Strands Decider 2B | Apache-2.0 | new `strands-decider` | q4f16, q8 | q4f16 |

They have a real public ONNX export, complete external-data files, a pinned
revision and a known prompt contract. The local machine verified every file
hash and opened every ONNX session. No speculative or unavailable model was
added.

## Why these three

- **Laya multilingual** closes the multilingual gap in the existing Laya
  family. The ONNX export is fused, but the prompt and calibrated temperatures
  are the same as the split Laya contract. It needed one adapter branch, not a
  new model family.
- **Open-Jev DeBERTa v3 large** is an independent, open reproduction of the
  typed-decision shape. It has a complete fp16/fp32/q4/q4f16 family and a
  documented span-pooling graph.
- **Strands Decider 2B** is the first larger decision model in the catalog. It
  has a published reference fixture and a clear pointer-head graph. It is more
  expensive on CPU, so it is marked as a quality/agentic option rather than a
  replacement for the small models.

`gliner2.5-multi-decide` is **not** in this batch. Its ONNX export is real, but
the prompt uses `[L]` markers instead of the existing GLiNER2 `[C]` layout; it
needs a separate prompt adapter and its own parity run.

## Verification performed

- `dtk models verify --all --json`: **15/15 variants OK** (SHA256, files and
  ONNX sessions).
- Real inference for every new variant, using `choice`, `noul` and `score`.
- Smoke parity suite: all added cases matched their expected labels.
- `dtk models variants` exposes all variants and install commands.
- Full test suite: **247 passed, 1 deselected**.
- Ruff and MkDocs `--strict`: clean.
- Python 3.12/3.13 CI matrix and wheel clean-install job: green.

### Measured CPU latency

Post-warmup p50, one simple choice question, audit machine (Python 3.13.2,
ONNX Runtime 1.31.0, CPU):

| Model | Variant | p50 |
|---|---|---:|
| Laya multilingual | fp16 | 37.4 ms |
| Laya multilingual | fp32 | 34.1 ms |
| Open-Jev | fp16 | 241.2 ms |
| Open-Jev | fp32 | 78.2 ms |
| Open-Jev | q4 | 56.0 ms |
| Open-Jev | q4f16 | 219.4 ms |
| Strands Decider 2B | q4f16 | 719.3 ms |
| Strands Decider 2B | q8 | 1467.2 ms |

The Open-Jev 4-bit paths are size/edge choices; they are not assumed to be
faster or more accurate. Strands is a 2B model and is intentionally slower
than the other catalog entries.

### Evidence / parity

- **Laya multilingual:** upstream conversion reports fp32 max |Δp| 1e-4 and
  fp16 0.035 against PyTorch; local fp16/fp32 agree on the audit answers.
- **Open-Jev:** upstream reports 0.854 in-domain and 0.690 OOD accuracy, plus
  0.022 in-domain ECE. Local fp16/fp32 agree on the published example; q4
  variants are explicitly edge builds.
- **Strands Decider 2B:** reference fixture matches within 0.018 max
  probability delta; q8 is closer to the reference on most fields.

These are upstream parity claims plus local runtime/answer checks, not an
independent reproduction of the full upstream training evaluation.

## Quantization policy

- Default variants are chosen for a balance of quality, size and CPU latency:
  Laya fp16, Open-Jev fp16, Strands q4f16.
- `fp32` remains a quality variant where published.
- `q4`/`q4f16`/`q8` are exposed, but their thresholds must be validated on the
  user's data; no silent fallback exists.
- `variant` remains a **decision-toolkit extension** in REST, not a TypeSafe or
  OpenAI field.

## Risks and limitations

1. **Upstream evidence vs local evidence.** Full PyTorch parity was not
   re-run for these three models; the catalog records upstream parity and local
   runtime checks. A larger acceptance run is still desirable.
2. **Strands CPU cost.** q8 takes ~1.5 s per simple question; this is suitable
   for local/agentic workflows, not high-throughput serving.
3. **Quantization drift.** Open-Jev q4 and Strands q4f16 can move near-ties;
   the model cards call this out.
4. **Coverage.** Laya multilingual is weak on low-resource languages; Open-Jev
   is English-only and weak on unseen score scales; Strands is a 2B model with
   its own upstream limitations.
5. **No image/box support.** The catalog remains text-only.

## Recommendation

Accept the three models as `m0` entries with the variants listed above. Keep
`gliner2.5-multi-decide` in the roadmap until its `[L]` adapter is implemented
and parity is measured. The next model batch should prioritize either that
adapter or a larger calibration/parity run for the new entries.

## Related documentation

- [Model cards](models/README.md)
- [Variants and quantization](VARIANTS.md)
- [Benchmarks](BENCH.md)
- [REST](REST.md)
