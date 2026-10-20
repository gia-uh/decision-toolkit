# Bench harness

Reproducible CPU measurements for the pinned catalog variants. This is a
**regression/smoke suite and a calibration harness**, not a claim of general
model quality: the included cases are small and fixed so results can be
compared across commits.

## Run

```bash
# parity + steady-state latency (one pinned case per model)
PYTHONPATH=src python bench/run.py \
  --suite bench/suites/smoke.jsonl \
  --repeats 3 \
  --output bench/results/2026-10-20-smoke.json

# ECE/Brier on the synthetic Spanish routing set
PYTHONPATH=src python bench/run.py \
  --suite bench/suites/calibration.jsonl \
  --repeats 1 \
  --output bench/results/2026-10-20-calibration.json
```

The models must already be in the toolkit cache (`dtk models download ...`).
The harness never downloads; it loads each model once, warms sessions, then
measures steady-state decisions.

## Suite format

JSONL, one case per line:

```json
{
  "id": "case-1",
  "model": "gliner2.5-multi",
  "state": "text or JSON state",
  "questions": {"cat": {"type": "choice", "instructions": "...", "criteria": {"a": null, "b": null}}},
  "expected": {"cat": "a"},
  "labels": {"cat": "a"}
}
```

- `expected` is a regression assertion: a choice label, a score level, or a
  boolean for `noul` (compared through `Answer.value`).
- `labels` feeds Brier/ECE/threshold calibration. Only choice and `noul`
  questions currently contribute calibration pairs.
- `ok` is false if any `expected` case does not match or a model fails to load.

## Current results (2026-10-20)

Environment: Python 3.12.7, ONNX Runtime 1.23.2, tokenizers 0.22.1, numpy
2.3.4, Linux x86_64, Intel i7-13620H. See the JSON files for full per-case data.

| Model | Variant | p50 ms | p95 ms | Parity |
|---|---|---:|---:|---:|
| `gliner2.5-multi` | fp16 | 177.1 | 186.8 | 1/1 |
| `gliner2.5-decide` | q4f16 | 292.9 | 294.8 | 1/1 |
| `laya` | q8 | 262.7 | 290.1 | 1/1 |
| `julia-1` | fp32 | 16.6 | 16.7 | 1/1 |
| `opendecider-nano` | q8f16 | 273.9 | 279.0 | 1/1 |

Calibration smoke (`gliner2.5-multi`, 10 synthetic Spanish routing cases):
accuracy `0.60`, Brier `0.173`, ECE `0.247`. This set is deliberately small and
synthetic; use your own labelled data for a real calibration decision.

## Notes

- Latency is measured **after** a warmup decision, so it excludes the first
  session/tokenizer initialization. It is not directly comparable to the
  historical cold-start numbers in the model cards.
- `bench/results/` is the versioned evidence. Re-run with a different machine
  and record the new environment; do not overwrite old results silently.
- The harness lives outside the package on purpose: the runtime library stays
  free of benchmark tooling.
