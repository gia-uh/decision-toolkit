# Roadmap (not in the catalog)

The catalog only contains models that are **runnable today** (`m0`). The entries
below are researched but not available through the toolkit: they have no ONNX
export, no verified parity, a broken export, or an adapter still to be written.
They are deliberately absent from `CATALOG`, `dtk models list` and `/v1/models`.

| id | status | reason |
|---|---|---|
| `gliner2.5-multi-decide` | planned | ONNX split export exists (`DanKau/gliner2.5-multi-decide-onnx`), but the `[L]` prompt layout needs an adapter |
| `intern-decision-0.8b` | planned | ONNX exists (`Wortou/Intern-Decision-0.8B-ONNX-FP16`), but it is split and needs the upstream prompt compiler; `parityVerified: false` |
| `d3-lite` | phase2 | Web ONNX torso/readout files exist, but the repo declares no license/README; not eligible |
| `kev-0.8b` | phase2 | ONNX exists (`midudev/kev-0.8b-ONNX`) but external data is missing and it uses custom ops |
| `sieve-0.8b` | phase2 | No ONNX published; conversion planned |
| `thx-01` | phase2 | No ONNX published; conversion planned |
| `jevk5-lite` | waitlist | Export references external `.onnx.data` files that are not published |
| `bosun-0.6b` | waitlist | No ONNX yet |
| `d3-edge` | waitlist | Web ONNX files exist, but no license/README; not eligible |
| `tev1-0.8b` | waitlist | ONNX exists, but the upstream licence is `other` and needs review |
| `d1-omni-600m` | waitlist | ONNX exists, but the LFM Open License needs review and it is multimodal |
| `bekko-400m` | waitlist | License pending |

## How a model moves into the catalog

1. A real, runnable ONNX export exists on the Hub.
2. Its contract is understood and an adapter runs it (written only from the real files).
3. Parity against the original is measured on a fixed sample.
4. License is clean and CPU latency is measured.
5. A model card is written; the variant carries `verified_at`, `latency_ms_p50`
   and `parity`.

Conversion work (safetensors → ONNX, quantization, manifests) is a separate
project planned for the next phase.
