# Model cards

These are the models available through `decision-toolkit`. Each card explains
how to load and use the model, its variants, the measured evidence and its
known limitations.

| Model | Adapter | Variants (default *) | Pinned | Latency (CPU) | Evidence |
|---|---|---|---|---|---|
| [GLiNER2.5 multi](gliner2.5-multi.md) | `gliner2` | fp16* · 577 MB; fp32 · 1132 MB | `3de2c11f` | 353 ms | 9/15 routing; same as local int8 |
| [GLiNER2.5-Decide](gliner2.5-decide.md) | `gliner2-decide` | q4f16* · 524 MB | `18271032` | 912 ms | refund 0.944 (card example) |
| [Laya (English)](laya.md) | `laya` | q8* · 528 MB | `cccc9a01` | 514 ms | torch parity: Δp ≤ 0.005 |
| [Julia 1](julia-1.md) | `julia` | fp32* · 618 MB | `82a2fadf` | 329 ms | 100/100 argmax; 8.5e-05 max logit error |
| [OpenDecider nano](opendecider-nano.md) | `opendecider` | q8f16* · 469 MB; fp16 · 792 MB | `5137ae7b` | 530 ms | official build; prompt ported from pinned code |
| [Laya multilingual](laya-multilingual.md) | `laya` | fp16* · 682 MB; fp32 · 1326 MB | `46b77bbf` | 37 ms | fused ONNX; upstream Δp ≤ 0.035 |
| [Open-Jev DeBERTa v3 large](open-jev-deberta-v3-large.md) | `open-jev` | fp16* · 884 MB; fp32 · 1758 MB; q4 · 487 MB; q4f16 · 357 MB | `7c79f25b` | 241 ms | 0.854 in-domain; 0.690 OOD (upstream) |
| [Strands Decider 2B](strands-decider-2b.md) | `strands-decider` | q4f16* · 1114 MB; q8 · 1821 MB | `31a83e02` | 719 ms | reference fixture max Δp 0.018 |

A `*` marks the default. To inspect sizes, hashes and exact commands, run
`dtk models variants <id>` or read [Variants and quantization](../VARIANTS.md).

Latencies above are CPU measurements. Reproducible post-warmup measurements
and the exact command are in [BENCH.md](../BENCH.md).
