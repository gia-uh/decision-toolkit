# Benchmarks and calibration

The reproducible harness lives in [`bench/`](https://github.com/gia-uh/decision-toolkit/tree/main/bench). It is intentionally
outside the installed package.

## What it measures

1. **Parity/regression** — each case has an `expected` answer; the harness fails
   if the pinned model stops matching.
2. **Steady-state latency** — p50/p95 after a warmup decision, not first-call
   initialization.
3. **Calibration** — Brier score, ECE and threshold coverage on cases with
   `labels`.

The included suite is a small fixed smoke/regression suite, not a general
quality benchmark.

## Reproduce

```bash
uv run python bench/run.py \
  --suite bench/suites/smoke.jsonl \
  --repeats 3 \
  --output bench/results/2026-10-20-smoke.json

uv run python bench/run.py \
  --suite bench/suites/calibration.jsonl \
  --repeats 1 \
  --output bench/results/2026-10-20-calibration.json
```

Models must already be cached; the harness never downloads.

## Current evidence (2026-10-20)

Environment: Python 3.12.7, ONNX Runtime 1.23.2, tokenizers 0.22.1, numpy
2.3.4, Linux x86_64, Intel i7-13620H.

| Model | Variant | p50 ms | p95 ms | Parity |
|---|---|---:|---:|---:|
| `gliner2.5-multi` | fp16 | 177.1 | 186.8 | 1/1 |
| `gliner2.5-decide` | q4f16 | 292.9 | 294.8 | 1/1 |
| `laya` | q8 | 262.7 | 290.1 | 1/1 |
| `julia-1` | fp32 | 16.6 | 16.7 | 1/1 |
| `opendecider-nano` | q8f16 | 273.9 | 279.0 | 1/1 |

Calibration smoke (`gliner2.5-multi`, 10 synthetic routing cases): accuracy
`0.60`, Brier `0.173`, ECE `0.247`.

Full evidence, per-case answers and environment metadata:
[`bench/results/2026-10-20-smoke.json`](https://github.com/gia-uh/decision-toolkit/blob/main/bench/results/2026-10-20-smoke.json)
and
[`bench/results/2026-10-20-calibration.json`](https://github.com/gia-uh/decision-toolkit/blob/main/bench/results/2026-10-20-calibration.json).

## Interpreting latency

The model cards retain the historical cold-start measurements used when each
variant was admitted. The harness reports **post-warmup steady-state** latency,
which is lower. Compare like with like: use the harness when comparing commits
or machines; use the card values only as historical admission evidence.

## Adding a case

Add a line to the suite with `id`, `model`, `state`, `questions`, and either
`expected` (regression) and/or `labels` (calibration). Keep cases stable; if an
expected model behavior changes, update the case only with a documented reason.
