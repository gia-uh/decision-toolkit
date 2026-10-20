# Model cards

Verified `m0` models. Each card documents the adapter contract, the real download,
the measured evidence and the known limitations. Models that are not `m0` have no
card on purpose: they are not part of the curated library yet.

| Model | Adapter | Variants (default *) | Pinned | Latency (CPU) | Evidence |
|---|---|---|---|---|---|
| [GLiNER2.5 multi](gliner2.5-multi.md) | `gliner2` | fp16* · 577 MB; fp32 · 1132 MB | `3de2c11f` | 353 ms | 9/15 routing; same as local int8 |
| [GLiNER2.5-Decide](gliner2.5-decide.md) | `gliner2-decide` | q4f16* · 524 MB | `18271032` | 912 ms | refund 0.944 (card example) |
| [Laya (English)](laya.md) | `laya` | q8* · 528 MB | `cccc9a01` | 514 ms | torch parity: Δp ≤ 0.005 |
| [Julia 1](julia-1.md) | `julia` | fp32* · 618 MB | `82a2fadf` | 329 ms | 100/100 argmax; 8.5e-05 max logit error |
| [OpenDecider nano](opendecider-nano.md) | `opendecider` | q8f16* · 469 MB; fp16 · 792 MB | `5137ae7b` | 530 ms | official build; prompt ported from pinned code |

A `*` marks the default. To inspect sizes, hashes and exact commands, run
`dtk models variants <id>` or read [Variants and quantization](../VARIANTS.md).

Latencies above are the historical admission measurements. Reproducible
post-warmup measurements and the exact command are in
[BENCH.md](../BENCH.md).

## What a card must contain

1. Status and adapter (the model contract).
2. Every catalogued variant: repo, files, quantization, download size and role.
3. Measured evidence: latency on CPU, parity/accuracy against a named baseline.
4. Known limitations (language, export layout, quantization, benchmark scope).
5. A runnable usage example.
6. Provenance of the export.
7. The pinned revision and SHA256 per file, plus the verify/bench command.
