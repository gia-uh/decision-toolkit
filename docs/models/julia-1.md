# Julia 1

- **Adapter:** `julia` — single graph with `marker_pos` / `qtype`
- **License:** Apache-2.0
- **Parameters:** 141M · **Architecture:** mmBERT-small · **Languages:** 52 locales (MASSIVE)

## Default variant

| | |
|---|---|
| Repo | `SupersonicLabs/Julia-1-ONNX` |
| Files | `model.onnx`, `model.onnx.data`, `tokenizer.json` |
| Quantization | fp32 |
| Download | 618.2 MB |
| Role | default |

## Variants

| Variant | Quantization | Size | Role | Install |
| --- | --- | ---: | --- | --- |
| `fp32` * | fp32 | 618.2 MB | default | `dtk models download julia-1 --variant fp32` |

The official fp32 build is the only verified variant. A community int8 export
exists but was not validated and is not catalogued; see
[Variants and quantization](../VARIANTS.md).

## Reproducibility

- **Revision:** `82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8`
- **File hashes (SHA256):**
  - `model.onnx` — `97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9`
  - `model.onnx.data` — `fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1`
  - `tokenizer.json` — `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f`
- **Verify:** `dtk models verify julia-1 --variant fp32`
- **Bench:** `uv run python bench/run.py --suite bench/suites/smoke.jsonl --models julia-1`

## Evidence (2026-10-19)

- **100/100 argmax matches** against the repository's published `pytorch_logits` (100 parity cases).
- **Max absolute logit error: 8.5e-05**.
- **329 ms/decision** on CPU.
- Example: charged-twice ticket → `billing` 0.9999; urgent 0.993; tone 1.995.

## Contract notes

- Sequence: `[CLS] <type> question: <instructions> [SEP] ([MASK] opt)… [SEP] state [SEP]` (same builder as the official `julia/data.py`, ported token for token).
- `qtype`: choice=0, score=1, noul=2; `noul` options are ordered `[false, true]`.
- Default `max_length=1024`, `head_length=256` (the ONNX browser runtime's defaults; the model supports up to 8192).
- No temperature is applied (the graph scores options directly).

## Known limitations

- Only the official fp32 export is verified; a community int8 export exists but was not validated.
- Historical benchmarks used 1024 tokens; accuracy at 8k is not established by the export.
- The tokenizer uses `<mask>/<bos>/<eos>`; both conventions are handled, but this checkpoint's tokenizer is the `<...>` one.

## Loading each variant

REST selects the quantization through the standard `model` field using the
alias `julia-1@variant`; there is no `variant` field in the request body.

### `fp32` (fp32, default)

**Library**

```python
from decision_toolkit import DecisionModel

model = DecisionModel("julia-1", variant="fp32")
```

**CLI**

```bash
dtk models download julia-1 --variant fp32
dtk models use julia-1 --variant fp32
dtk decide --model julia-1 --variant fp32 \
  --state "..." --q 'urgent=noul:Is this urgent?'
```

**REST**

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"model": "julia-1@fp32", "state": "...",
       "questions": {"urgent": {"type": "noul",
                     "instructions": "Is this urgent?"}}}}'
```

## Usage

```bash
dtk models download julia-1
dtk decide --model julia-1 --state "I was charged twice…" \
    --q team='choice:billing|technical|other' --q urgent='noul:Needs a reply today?'
```

## Provenance

Official export by Supersonic Labs; base model `SupersonicLabs/Julia-1`.
