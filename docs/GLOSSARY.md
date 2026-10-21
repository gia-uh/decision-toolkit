# Glossary

| Term | Meaning |
|---|---|
| **State** | The text or JSON value the model reasons over. In the library it can be a string, `State(text=...)` or a JSON-compatible object. |
| **Question** | A typed request for one decision: `choice`, `noul` or `score`. |
| **`choice`** | A question with a list of options and one selected option. The core has a multi-label flag, but no model in the current catalog declares it. |
| **`noul`** | A yes/no question. The result is a probability of `true`, not a hard boolean. |
| **`score`** | A question with ordered levels. The result includes the expected score, the most likely level and the level legend. |
| **Criteria** | Descriptions or option definitions attached to a question. For `choice`, they tell the model what each option means. |
| **Adapter** | The code that formats a state and typed questions for a specific model layout and reads its outputs. One adapter per model contract. |
| **Catalog** | The models and pinned variants shipped with the toolkit. `dtk models list` shows it. |
| **Variant** | One real, pinned ONNX build of a model. The variant name is passed to `--variant` or `variant=`. |
| **Quantization** | The numeric format of a build, such as `fp16`, `fp32`, `q8`, `q4` or `q4f16`. It is related to, but not always identical to, the variant name. |
| **Model alias** | The REST identifier `id@variant` used in the standard `model` field, for example `gliner2.5-multi@fp32`. |
| **Default variant** | The catalog's preferred variant for a model, marked with `*` in `dtk models list`. Used when no explicit or persisted selection exists. |
| **Current selection** | The model and variant stored by `dtk models use`; reported by `dtk models current`. |
| **Model cache** | The local directory holding downloaded or imported model variants. It is independent from the HuggingFace cache. |
| **Toolkit home** | The parent directory containing `config.json` and the model cache, by default `~/.local/share/decision-toolkit/`. |
| **`cache_dir`** | The `config.json` value that points at the model cache. It may be absolute or relative to the toolkit home. |
| **Model store** | The `ModelStore` API that owns download, import, selection, verification and resolution. |
| **Strict mode** | A mode where truncation raises `TruncationError` instead of returning a truncated answer. |
| **Truncation** | Shortening an input or output to fit a model context window. Answers expose `truncated` and `warnings`. |
| **Confidence** | The normalized probability reported for the chosen answer. Its meaning is documented in [Core API](CORE.md#confidence). |
| **Calibration** | How well predicted probabilities match observed frequencies. Evidence is measured per model and variant; see [Benchmarks](BENCH.md). |
| **Warmup** | Loading the ONNX session before the first real request. Available in the library and with `dtk serve --warmup`. |
| **Usage** | Token accounting attached to a decision: input, output, total and whether the count is estimated. |
| **Request ID** | A correlation identifier attached to a decision and emitted in logs. Set it with `request_context(...)` or the CLI/REST request metadata. |
| **System One** | The TypeSafe REST schema used by `POST /v1/systemone`. |
| **OpenAI Decisions beta** | The OpenAI Decisions schema used by `POST /v1/decisions`. |
| **Local extension** | A field or response extension namespaced by decision-toolkit, outside the public TypeSafe/OpenAI shapes. |
| **`top_k`** | In one-label classification, the number of best alternatives to return (when the surface exposes it). |
| **Threshold** | A probability cutoff used by `classify` to decide whether to return a label or mark the result as below threshold. |
| **Custom model** | A local ONNX directory supplied by the user with an explicit adapter and file map. It is not a catalog entry and has no catalog guarantees. |
| **REST server** | The optional FastAPI process started by `dtk serve`; it projects a subset of the core over HTTP. |
| **CLI** | The `dtk` command, a thin operational surface over the core. |
