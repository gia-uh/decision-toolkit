# Core API

The programmatic API is the **superset**: the most expressive layer of the
toolkit. The REST server is a documented, conformance-tested projection of a subset of
this API, and the CLI is our own operational subset. Nothing in this document
belongs to a wire contract.

```
Programmatic core   ← superset, semantic source of truth
   ├── REST           ← subset projected onto System One / OpenAI, versioned
   └── CLI            ← own workflows, no external standard
```

## Public surface

```python
from decision_toolkit import (
    DecisionModel, ModelStore,
    State, Choice, Noul, Score, Box,
    choice, noul, score, box,
    Usage, Capabilities, ModelInfo,
    register_adapter,
    request_context,
)
```

| API | Purpose |
| --- | --- |
| `DecisionModel(...)` | A local model, loaded lazily, one ONNX engine. |
| `decide(state, questions)` | Typed System One decision. |
| `classify(text, categories)` | One-label convenience over `choice`. |
| `decide_many(...)` / `classify_many(...)` | Batch with one model load. |
| `adecide(...)` / `adecide_many(...)` / `aclassify(...)` | Async wrappers. |
| `warmup()` / `close()` / `with DecisionModel(...)` | Lifecycle. |
| `capabilities` | What the selected model can actually do. |
| `model_info()` | Typed catalog/custom metadata. |
| `DecisionModel.from_onnx(...)` | Custom local ONNX with an explicit adapter and file map. |
| `register_adapter(name, factory)` | In-process third-party adapter. |
| `ModelStore(...)` | Cache: download, import, verify, use, resolve. |

Variants (quantizations) are selected by name; see
[Variants and quantization](VARIANTS.md) for the catalog and commands.

## Types and serialization

- `State(text="...", assets=())` or a plain string/JSON object.
- `Choice`, `Noul`, `Score`, `Box` (typed question builders) or canonical raw
  question mappings.
- `Answer` exposes `kind`, `value`, `confidence`, `confidence_kind`,
  `truncated`, `warnings`, `action`, `refusal` and the raw `data`.
- `Decision` exposes `answers`, `model`, `variant`, `adapter`, `usage`,
  `latency_ms`, `warnings`, `request_id`, `id`, `truncated` and `refused`.
- `Usage` exposes `input_tokens`, `output_tokens`, `total_tokens`,
  `estimated`.
- `Capabilities` exposes `question_types`, option/level/context limits,
  languages, `multimodal`, `multi_label`, `batch`, `usage`.
- `ModelInfo` exposes identity, license, variants and capabilities.

Serialization is stable and JSON-friendly:

```python
decision.to_dict()      # raw System One answers (CLI/REST compatibility)
decision.to_payload()   # answers + identity + usage + latency + warnings
usage.to_dict()
capabilities.to_dict()
model_info().to_dict()
```

## Lifecycle and concurrency

```python
with DecisionModel("gliner2.5-multi", variant="fp16") as model:
    model.warmup()
    decision = model.decide("...", {"urgent": noul("Today?")})
# sessions released here
```

- Lazy load under an `RLock`; concurrent callers do not load the model twice.
- Inference is serialized per model instance; ONNX sessions are shared, adapter
  caches are not assumed to be thread-safe. This is the documented v0.1
  contract. Use one `DecisionModel` per process or per worker.
- `close()` is idempotent. An injected adapter can be reused after `close()`.
- `warmup()` loads sessions before the first request.

## Execution: single, batch, async

```python
decisions = model.decide_many(
    ["state one", "state two"],
    {"urgent": noul("Today?")},
    request_id="batch-42",
)

result = await model.adecide("state", {"urgent": noul("Today?")})
```

- `decide_many` loads the model once. Current adapters execute sequentially;
  `Capabilities.batch` is `False` until an adapter declares true ONNX batch
  support. The API is stable either way.
- A sequence of question mappings is accepted, one per state.
- Async methods run the synchronous core in a worker thread (`asyncio.to_thread`).

## Usage accounting

Every `Decision` carries `Usage`:

- `input_tokens`: tokenizer count of the state plus the normalized questions.
- `output_tokens`: tokenizer count of the serialized answers.
- `estimated`: **always explicit**. The core does not reproduce each model's
  exact internal prompt, and falls back to a whitespace estimate when no
  tokenizer is available. An estimate is never presented as exact.

## Truncation

Adapters that must fit a context window (Laya, Julia, OpenDecider) mark each
answer with:

```python
answer.truncated   # bool
answer.warnings    # ("input was truncated to fit max_len=1024",)
decision.truncated # any answer truncated
```

No adapter truncates silently. `strict=True` (constructor or per call) raises
`TruncationError` instead of returning the truncated decision:

```python
DecisionModel("julia-1", strict=True).decide(long_state, questions)
```

Context limits are configured through adapter options
(`adapter_options={"max_length": 512, "head_length": 128}` where the adapter
supports them); `Capabilities.max_context_tokens` states the known limit.

## Confidence

Confidence is normalized in the core, once:

- `choice`: probability of the chosen option;
- `score`: probability of the most likely level (the expected `score` remains
  available separately);
- `noul`: `max(p_true, p_false)`.

`Answer.confidence_kind` is `"probability"` for all current adapters. Laya also
exposes its entropy-based distribution confidence as
`answer.data["distribution_confidence"]`; it does not replace the normalized
`confidence`. Callers that need a single comparable number can use
`confidence` across models.

## Capabilities

```python
model.capabilities.question_types
model.capabilities.max_context_tokens
model.capabilities.multimodal
```

The core validates before loading/inference:

- unsupported question type (`box` today) → `UnsupportedCapabilityError`;
- assets/images on a text-only model → `UnsupportedCapabilityError`;
- option/level counts above the model's declared limits;
- `multi_label=True` when the model does not declare it.

Catalog capabilities are data in `decision_toolkit.catalog`; a custom adapter
may declare its own `Capabilities` or `supported_question_types`.

## Errors

All deliberate errors derive from `DecisionToolkitError` and expose
`code` + `to_dict()`:

| Error | Code |
| --- | --- |
| `ModelNotFoundError` | `model_not_found` |
| `ModelNotDownloadedError` | `model_not_downloaded` |
| `AdapterNotAvailableError` | `adapter_not_available` |
| `AdapterNotImplementedError` | `adapter_not_implemented` |
| `AdapterError` | `adapter_error` |
| `BackendFailureError` | `backend_failure` |
| `InvalidQuestionError` | `invalid_question` |
| `InvalidRequestError` | `invalid_request` |
| `UnsupportedCapabilityError` | `unsupported_capability` |
| `TruncationError` | `truncated_input` |
| `VerificationError` | `verification_failed` |

The REST layer will map these codes to HTTP without parsing messages.

## Logging and request correlation

The library **emits** structured events and never configures handlers:

```python
from decision_toolkit import request_context

with request_context("req-123"):
    decision = model.decide("...", {"urgent": noul("Today?")})

assert decision.request_id == "req-123"
```

Events emitted by `decision_toolkit.*` loggers:

- `model.load`, `model.warmup`, `model.close`
- `decision.start`, `decision.end`, `decision.error`

Each event is a compact JSON object with `event`, `request_id`, model identity,
variant, adapter, question count, token counts, latency, truncation and error
code. State and question text are **not** logged by default. The CLI/REST
layers configure formatting and destinations separately.

## Extensibility

### Custom local ONNX

```python
model = DecisionModel.from_onnx(
    "./my-model",
    adapter="julia",
    files={"model": "model.onnx", "tokenizer": "tokenizer.json"},
)
```

File roles are validated before the engine opens the files. `gliner2` accepts a
fused `model` or split `encoder` + `classifier`; `laya`, `julia` and
`opendecider` have fixed role sets.

### Third-party adapters

```python
from decision_toolkit import register_adapter

register_adapter("my-layout", MyAdapter)
model = DecisionModel("./my-model", adapter="my-layout", files={...})
```

Adapters can also be discovered from the `decision_toolkit.adapters` entry-point
group. A custom adapter may declare `required_roles` or `supported_question_types`.
Registration is in-process (no daemon, no plugin server).

## Model store

`ModelStore` owns the toolkit cache (`~/.local/share/decision-toolkit/models`):

- `download(id, variant=..., force=...)`: temporary download directory, gzip
  decompression, SHA256 recording, atomic rename; a failed download never
  leaves a usable half-entry.
- `import_model(path, name=..., adapter=..., files=...)`: custom model with its
  adapter and role map.
- `use(id, variant=...)` / `current()` / `current_variant()`: persistent model
  selection.
- `verify(id, variant=...)`: declared files, SHA256 checks, ONNX session checks
  (when `onnxruntime` is installed); returns a structured report.
- `resolve(id, variant=...)`: local directory + role → file map + adapter for
  the core.

## Capabilities and limits

The core exposes the following capabilities:

| Capability | Core status |
| --- | --- |
| `choice` / `noul` / `score` | implemented |
| `box` | typed + validated, unsupported by current models |
| Text state / JSON state | implemented |
| Assets/images | typed (`State.assets`), capability-gated |
| Per-request model/variant | constructor + `ModelStore.resolve` |
| Usage tokens | implemented, `estimated` explicit |
| Latency | per decision |
| Warnings / truncation | implemented; strict mode |
| Normalized confidence | implemented |
| Refusals/action metadata | typed accessors |
| Batch | `decide_many` / `classify_many` (sequential execution today) |
| Async | `adecide` / `aclassify` / batch wrappers |
| Context manager / close / warmup | implemented |
| Concurrency lock | implemented (serialized inference) |
| Capability validation | implemented |
| Custom adapters | `register_adapter`, entry points, `from_onnx` |
| Structured logging | emitted, request-id correlated |
| Model store | atomic download, hashes, verify, import metadata |
| Multi-label | capability flag + validation; no adapter yet |

## Not part of the core

- REST contract projection and standard compliance are implemented in the
  server; see [REST.md](REST.md).
- CLI workflows across the catalog are implemented in [CLI.md](CLI.md).
- Catalog pinning and source CI are implementation details; packaging is
  documented in [RELEASE.md](RELEASE.md).
- Metrics, rate limiting and load/fuzz testing are server-side features; see
  [OPERATIONS.md](OPERATIONS.md).
