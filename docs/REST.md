# REST API

REST is a **documented projection** of the programmatic core, not a second
implementation. The core remains the superset; the server selects a supported
subset, maps it to the published wire contracts, and adds transport concerns
(auth, request ids, limits, CORS, logging).

```
Programmatic core (superset)
        │  pure projection + conformance tests
        ▼
REST (supported subset; TypeSafe System One + OpenAI Decisions beta)
```

Compatibility is tested against the published shapes, not against a local
invention. The current models are text-only, so image/assets and `box` requests
are explicitly rejected rather than silently approximated.

The wire models live in `decision_toolkit/contracts/`, the multi-model registry
in `decision_toolkit/registry.py`, and `server.py` contains only transport and
FastAPI glue.

## Pinned contracts

| Contract | Version | Reference | Module |
| --- | --- | --- | --- |
| TypeSafe System One | API schema `0.2.0` | <https://docs.typesafe.ai/api> | `contracts/systemone_v1.py` |
| OpenAI Decisions | `beta` | <https://developers.openai.com/api/docs/guides/decisions> | `contracts/openai_decisions_beta.py` |

`GET /v1/info` exposes both versions. "System One 1.14" was an incorrect label
in earlier documentation; the published TypeSafe API schema is `0.2.0`. The
implementation targets that schema and the published OpenAI beta shape, with the
limitations listed below.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/systemone` | TypeSafe System One decision. |
| `POST` | `/v1/decisions` | OpenAI Decisions beta decision. |
| `GET` | `/v1/models` | OpenAI model-list shape. |
| `GET` | `/v1/models/{id}` | One model in the OpenAI shape. |
| `GET` | `/v1/decision-models` | Rich catalog: variants, licenses, capabilities, cache. |
| `GET` | `/v1/info` | Version, contracts, registry state, auth mode. |
| `GET` | `/health`, `/v1/health` | Liveness, always `200`. |
| `GET` | `/ready`, `/v1/ready` | `200` only when the default model is warm; `503` otherwise. |

`/v1/models` and `/v1/models/{id}` are the OpenAI-compatible surface.
Toolkit-only metadata is under `x_decision_toolkit`, never mixed into standard
fields. `/v1/decision-models` is the rich, non-standard catalog endpoint.

## Variants as model aliases

Neither TypeSafe nor OpenAI has a `variant` request field. The toolkit therefore
publishes each quantization as a **model alias**:

```json
{"model": "gliner2.5-multi@fp32", "state": "...", "questions": {...}}
```

The `model` field is the standard field. `gliner2.5-multi` uses the server's
default (or the persisted `dtk models use` selection); `gliner2.5-multi@fp32`
selects that exact variant. Both aliases are listed by `/v1/models`, and the
registry loads them in separate slots, so two variants of one base model can be
used by the same server process.

`/v1/models/gliner2.5-multi@fp32`, `/v1/decision-models`, `/v1/info` and `/ready`
expose the alias, variant, default, cached state and current selection. The
response metadata under `x_decision_toolkit.variant` is an extension; the
request body remains standard. See [Variants and quantization](VARIANTS.md).

## System One (TypeSafe API schema 0.2.0)

Request (standard fields, plus the optional local `model` routing extension):

```json
{
  "model": "gliner2.5-multi",
  "state": "I was charged twice and nobody replied.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "payments", "shipping": null, "technical": null}
    },
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    "urgency": {
      "type": "score",
      "instructions": "How urgent?",
      "criteria": ["can wait", "this week", "today"]
    }
  }
}
```

Response:

```json
{
  "model": "gliner2.5-multi",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.82, "shipping": 0.11, "technical": 0.07},
      "confidence": 0.82
    },
    "angry": {"type": "noul", "noul": 0.61},
    "urgency": {
      "type": "score",
      "score": 1.2,
      "legend": {"0": "can wait", "1": "this week", "2": "today"},
      "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
      "confidence": 0.4
    }
  },
  "usage": {"input_tokens": 239, "output_tokens": 12},
  "x_decision_toolkit": {
    "spec": "0.2.0",
    "request_id": "req-123",
    "variant": "fp16",
    "adapter": "gliner2",
    "latency_ms": 353.2,
    "truncated": false,
    "warnings": [],
    "usage": {"input_tokens": 239, "output_tokens": 12, "total_tokens": 251, "estimated": true},
    "answers": {"route": {"confidence_kind": "probability", "truncated": false}}
  }
}
```

Standard fields:

- `usage.output_tokens` is a non-negative count. The current core reports the
  tokenizer count of the serialized answers (it may be non-zero); do not assume
  it is always zero.
- `usage.input_tokens` comes from the core token accounting.
- The full core usage, including the explicit `estimated` flag, is under
  `x_decision_toolkit.usage`.
- `choice` requires a non-empty criteria object; System One `score` accepts
  `1`–`10` ordered levels; `noul` may omit criteria.
- Request-schema and question-validation failures return `422` with the
  FastAPI-style `{"detail": [...]}` body used by the TypeSafe schema. Model
  lookup, capability and transport failures keep their HTTP status and the
  documented error envelope.
- Standard `assets` are accepted by the schema but return `501` because the
  current catalog is text-only. Legacy `files`/`images` fields are local
  extensions; valid data URLs also return `501`, malformed values return `422`.
- `model` is optional only as a local convenience; standard clients send it.
  `variant` is a local extension, not part of the TypeSafe schema.

## OpenAI Decisions beta

Request:

```json
{
  "input": "Customer wants a refund.",
  "questions": [
    {"type": "predicate", "name": "refund", "instructions": "Is it a refund?"},
    {"type": "choice", "name": "team", "instructions": "Which team?",
     "choices": [{"value": "billing"}, {"value": "tech"}]},
    {"type": "score", "name": "urgency", "instructions": "How urgent?",
     "levels": [{"label": "low"}, {"label": "high"}]}
  ],
  "model": "gliner2.5-multi"
}
```

Response:

```json
{
  "model": "gliner2.5-multi",
  "answers": [
    {"type": "predicate", "name": "refund", "probability": 0.93},
    {"type": "choice", "name": "team", "choice": "billing",
     "probabilities": [{"value": "billing", "probability": 0.91},
                       {"value": "tech", "probability": 0.09}],
     "confidence": 0.91},
    {"type": "score", "name": "urgency", "score": 0.8,
     "probabilities": [{"value": 0, "label": "low", "probability": 0.2},
                       {"value": 1, "label": "high", "probability": 0.8}],
     "confidence": 0.8}
  ],
  "usage": {
    "input_tokens": 92,
    "output_tokens": 12,
    "total_tokens": 104,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  },
  "x_decision_toolkit": {
    "spec": "beta",
    "request_id": "req-123",
    "variant": "fp16",
    "adapter": "gliner2",
    "latency_ms": 350.1,
    "truncated": false,
    "warnings": [],
    "usage": {"input_tokens": 92, "output_tokens": 12, "total_tokens": 104, "estimated": true}
  }
}
```

- The public beta response fields are `model`, `answers` and `usage`; there is
  no invented `id`, `object`, `created_at`, `status` or `safety_identifier`.
- Original choice values are echoed back, not the internal System One labels.
  The published beta accepts string/boolean values; the toolkit also accepts
  numbers as a local extension.
- Duplicate question names return `400`.
- Images in `input` return `501` (`multimodal_not_supported`); the published
  beta supports inline images, but the current catalog has no vision model.
- `variant` is a local extension; standard clients omit it.
- OpenAI SDK clients can list models (`client.models.list()`) and parse the
  error envelope (`{"error": {message, type, param, code}}`).

## Auth

Bearer auth is optional and disabled by default for local use:

```bash
dtk serve --model gliner2.5-multi --api-key "$DECISION_TOOLKIT_API_KEY"
```

- Every protected `/v1` endpoint (models, catalog, info, decisions) requires
  `Authorization: Bearer <key>` when configured.
- `/health` and `/ready` stay public.
- Missing/invalid keys return `401` with `WWW-Authenticate: Bearer` and the
  standard error envelope.
- Comparison is constant-time (`hmac.compare_digest`).

## Readiness and warmup

A model is not "ready" merely because it is configured:

- `/ready` returns `503` with `{"ready": false, "detail": ...}` until the default
  model has been warmed.
- `dtk serve --warmup` warms it at startup.
- The first successful decision also warms it.
- `--max-models N` keeps at most N idle models loaded (LRU). Models in use are
  never evicted.

## Limits and transport

- 1–128 questions (local limit; the TypeSafe schema has no maximum).
- 1–255 choice options; 1–10 System One score levels; OpenAI score levels
  require at least 2.
- 8 legacy files/images maximum (the current catalog supports 0); standard
  `assets` and images are rejected with `501`.
- Request body limit (`--max-body-mb`, default 16) returns `413`.
- Per-decision timeout (`--timeout`, default 120 s) returns `504`.
- CORS origins are explicit (`--cors-origin`, repeatable).
- Every response carries `x-request-id` (accepted from the client or generated).
- `dtk serve --log-format json` emits one JSON object per line; `text` is the
  default. State/question content is not logged. `--log-level` controls the
  toolkit logger.

## Error mapping

`/v1/systemone` request-validation failures follow the TypeSafe schema:
`422` with `{"detail": [...]}`. `/v1/decisions` validation and toolkit domain
errors use the OpenAI envelope
`{"error": {"message", "type", "param", "code"}}`.

| Condition | System One | OpenAI Decisions |
| --- | --- | --- |
| Request-schema validation | `422` + `detail` | `400` + error envelope |
| Toolkit question validation | `422` + `detail` | `400` + error envelope |
| Unknown model | `404` | `404` |
| Model not downloaded | `503` | `503` |
| Unsupported `box` / assets / images | `501` | `501` |
| Truncation in strict mode | `400` | `400` |
| Body too large | `413` | `413` |
| Missing/invalid bearer | `401` | `401` |
| Timeout | `504` | `504` |

The CLI/REST never parse error strings to decide behavior; they use the stable
`code`.

## Versioning and evolving standards

The standards are still moving. The projection is designed to change without
touching the core or the adapters:

1. **Detect upstream change**: diff the System One / OpenAI spec against the
   pinned module.
2. **Map impact**: update only `contracts/<spec version>.py` and its
   `mapping.py` projection.
3. **Core impact check**: if the new standard needs a capability the core does
   not have, add it to the core first; never implement it in the server.
4. **Conformance**: run the reference-client and core-vs-HTTP equivalence
   tests, update the compatibility matrix, and publish the version in
   `/v1/info`.
5. **Compatibility**: keep the old projection while a deprecation window is
   open; version endpoints or media types on a breaking change. Extensions
   always live under `x_decision_toolkit`.

Adding a new spec version means adding a new module, not editing the old one.

## Compatibility matrix (current)

| Feature | System One (TypeSafe 0.2.0) | OpenAI Decisions beta |
| --- | --- | --- |
| `choice` / `noul` / `score` | yes | `choice` / `predicate` / `score` |
| `box` | typed by core; rejected by current text models | no |
| Text state | yes | `input` string |
| JSON state | yes | message/string `input` |
| Images/assets | `501` for current models | `501` for current models |
| Usage | `input_tokens` + non-negative `output_tokens` | input/output/total + details |
| Per-request model | standard field; optional local default | standard field; optional local default |
| `variant` | local extension | local extension |
| Validation errors | `422` + `detail` | `400` + envelope |
| Auth / CORS / limits | server-level | server-level |

## Running

```bash
dtk serve --model gliner2.5-multi --warmup \
  --api-key "$DECISION_TOOLKIT_API_KEY" \
  --cors-origin http://localhost:3000 \
  --log-format json --log-level INFO
```

Then point a System One client at `http://localhost:8000` or an OpenAI client
at `http://localhost:8000/v1`.

## Testing

- System One contract tests cover the published request/response fields,
  non-negative usage, the invalid request matrix and image/asset status codes.
- OpenAI contract tests use the official `openai` Python SDK for
  `models.list()` and error-envelope parsing.
- Equivalence tests send the same core decision through HTTP and compare
  answers, usage and metadata.
- OpenAPI tests assert the real Pydantic request/response/error schemas.
- Transport tests cover auth, request ids, 413, CORS, readiness and per-request
  model selection.

## Not implemented by this server

- Metrics, rate limiting and load testing are implemented as server-side
  features; see [OPERATIONS.md](OPERATIONS.md).
- CLI batch/JSONL and custom-model workflows are in the CLI; see [CLI.md](CLI.md).
- Pinned catalog hashes and CI are implementation details; packaging is
  documented in [RELEASE.md](RELEASE.md).
- Images/assets and `box` are rejected because the current catalog is text-only.
