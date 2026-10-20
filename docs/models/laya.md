# Laya (English)

- **Status:** `m0` (verified against the torch runtime)
- **Adapter:** `laya` — split encoder + head with `marker_pos` / `qtype`
- **License:** Apache-2.0
- **Parameters:** 421M · **Architecture:** ModernBERT-large · **Language:** English

## Default variant

| | |
|---|---|
| Repo | `nvkudva/laya-onnx-q8` |
| Files | `v1/encoder_q8.onnx` + `.data`, `v1/head_q8.onnx` + `.data`, `v1/tokenizer.json`, `v1/rl_agent_config.json` |
| Quantization | int8 (q8) |
| Download | 527.9 MB |
| Role | default |

## Variants

| Variant | Quantization | Size | Role | Install |
| --- | --- | ---: | --- | --- |
| `q8` * | int8 | 527.9 MB | default | `dtk models download laya --variant q8` |

This is the only catalogued build. The upstream repository does not publish an
fp32 build; see [Variants and quantization](../VARIANTS.md).

## Reproducibility

- **Revision:** `cccc9a01e06189cbb1c153a51ea32642102e306d`
- **File hashes (SHA256):**
  - `v1/encoder_q8.onnx` — `5267b578a536bf12808ec84c1f354e87efd150d3c38d44099ef3dd29c5f43124`
  - `v1/encoder_q8.onnx.data` — `6375a3ae8f3ebfa4ca196e813fc7bdaf2e8ad8cf709218ef844ae8814dfffc59`
  - `v1/head_q8.onnx` — `e86c14c89a18b14f6115a7ccbfd039fce7ff97178fa0c07c14af936690a1f688`
  - `v1/head_q8.onnx.data` — `cfdf7c199378b07758c69f58935871cf517c97c29914328d4b11fcfbe50b7f8b`
  - `v1/tokenizer.json` — `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30`
  - `v1/rl_agent_config.json` — `ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd`
- **Verify:** `dtk models verify laya --variant q8`
- **Bench:** `uv run python bench/run.py --suite bench/suites/smoke.jsonl --models laya`

## Evidence (2026-10-19)

| Question | ONNX q8 | torch | Δ |
|---|---|---|---|
| team (choice) | **billing 0.9612** | billing 0.9598 | 0.0014 |
| urgent (noul) | 0.0560 | 0.0610 | 0.0050 |
| tone (score) | 1.0250 | 1.0262 | 0.0012 |

- Max probability delta: **0.005**; all choices match.
- **514 ms/decision** on CPU.
- The export's own README reports ≤0.0158 worst-case shift over its 26-question parity set.

## Contract notes

- Sequence: `[CLS] <type> question: <instructions> [SEP] ([MASK] opt)… [SEP] state [SEP]`.
- Supports both tokenizer serializations found in the wild: `<mask>/<bos>/<eos>` (multilingual) and `[MASK]/[CLS]/[SEP]` (English).
- Head input names (`hidden_states` vs `hidden`) and `qtype` rank (1-D vs 2-D) are read from the graph.
- Temperature is taken from `rl_agent_config.json` and clamped to [0.5, 5] like the torch runtime.

## Known limitations

- **English checkpoint.** The multilingual checkpoint is a different model (mmBERT-base) and has no split public export yet.
- q8 quantization; the fp32 build was not published in this repo.

## Usage

```bash
dtk models download laya
dtk decide --model laya --state "I was charged twice…" \
    --q team='choice:billing|technical|other' --q urgent='noul:Needs a reply today?'
```

## Provenance

Quantization and browser runtime by `nvkudva`; base model `convaiinnovations/laya` (English root).
