# REST projection (phase 2)

REST is a **standards-compliant projection** of the programmatic core, not a
second implementation. The core remains the superset; the server selects a
subset, maps it to a pinned wire contract, and adds transport concerns (auth,
request ids, limits, CORS, logging).

```
Núcleo programático (superset)
        │  pure projection + conformance
        ▼
REST (subset; System One 1.14 + OpenAI Decisions beta)
```

The wire models live in `decision_toolkit/contracts/`, the multi-model registry
in `decision_toolkit/registry.py`, and `server.py` contains only transport and
FastAPI glue.

## Pinned contracts

| Contract | Version | Module |
| --- | --- | --- |
| System One (TypeSafe-compatible) | `1.14` | `contracts/systemone_v1.py` |
| OpenAI Decisions | `beta` | `contracts/openai_decisions_beta.py` |

`GET /v1/info` exposes both versions. The reference for System One is the
TypeSafe 1.14 API as implemented by llama.cpp's `POST /v1/systemone`; the
OpenAI dialect is the beta Decisions shape already translated by the toolkit.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/systemone` | System One 1.14 decision. |
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

## System One 1.14

Request:

```json
{
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
  "usage": {"input_tokens": 239, "output_tokens": 0},
  "x_decision_toolkit": {
    "spec": "1.14",
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

- `usage.output_tokens` is **always 0** (decisions generate no text).
- `usage.input_tokens` comes from the core token accounting.
- The full core usage, including the explicit `estimated` flag, is under
  `x_decision_toolkit.usage`.
- `choice` requires a non-empty criteria object; `score` needs 2–10 levels;
  `noul` may omit criteria. Invalid bodies return `400`.
- `files`/`images` accept data URLs only. Text-only models return `501` with
  `code: "multimodal_not_supported"`; malformed/non-data URLs return `400`.
- `model` and `variant` are optional routing extensions for a multi-model local
  server. An unqualified request uses the configured default model.

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
  "safety_identifier": "user-hash",
  "model": "gliner2.5-multi",
  "variant": "fp16"
}
```

Response:

```json
{
  "id": "req-123",
  "object": "response",
  "created_at": 1790000000,
  "status": "completed",
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
  "usage": {"input_tokens": 92, "output_tokens": 0, "total_tokens": 92},
  "safety_identifier": "user-hash",
  "x_decision_toolkit": {
    "spec": "beta",
    "request_id": "req-123",
    "variant": "fp16",
    "adapter": "gliner2",
    "latency_ms": 350.1,
    "truncated": false,
    "warnings": [],
    "usage": {"input_tokens": 92, "output_tokens": 10, "total_tokens": 102, "estimated": true}
  }
}
```

- Original choice values (numbers, booleans, strings) are echoed back, not the
  internal System One labels.
- Duplicate question names return `400`.
- Images in `input` return `501` (`multimodal_not_supported`).
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

- 1–128 questions.
- 1–255 choice options; 2–10 score levels.
- 8 files/images maximum (the current catalog supports 0).
- Request body limit (`--max-body-mb`, default 16) returns `413`.
- Per-decision timeout (`--timeout`, default 120 s) returns `504`.
- CORS origins are explicit (`--cors-origin`, repeatable).
- Every response carries `x-request-id` (accepted from the client or generated).
- `dtk serve --log-format json` emits one JSON object per line; `text` is the
  default. State/question content is not logged. `--log-level` controls the
  toolkit logger.

## Error mapping

All errors use `{"error": {"message", "type", "param", "code"}}`.

| Condition | HTTP | code |
| --- | --- | --- |
| Invalid question/body | `400` | `invalid_question` / `invalid_request` |
| Unsupported capability (`box`, assets) | `400` | `unsupported_capability` |
| Unknown model | `404` | `model_not_found` |
| Model not downloaded | `503` | `model_not_downloaded` |
| Adapter unavailable | `503` | `adapter_not_available` |
| Adapter not implemented | `501` | `adapter_not_implemented` |
| Images/audio unsupported | `501` | `multimodal_not_supported` |
| Truncation in strict mode | `400` | `truncated_input` |
| Body too large | `413` | `request_too_large` |
| Missing/invalid bearer | `401` | `authentication_error` |
| Timeout | `504` | `timeout` |
| Backend failure | `500` | `backend_failure` / `adapter_error` |

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

| Feature | System One 1.14 | OpenAI Decisions beta |
| --- | --- | --- |
| `choice` / `noul` / `score` | yes | `choice` / `predicate` / `score` |
| `box` | typed by core; rejected by current text models | no |
| Text state | yes | `input` string |
| JSON state | yes | message/string `input` |
| Images | `501` for current models | `501` for current models |
| Usage | `input_tokens` + `output_tokens=0` | input/output/total |
| Per-request model/variant | extension | `model` + `variant` extension |
| Errors | OpenAI-style envelope | OpenAI envelope |
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

- System One contract tests cover the exact request/response fields, usage
  `output_tokens=0`, invalid request matrix and image status codes.
- OpenAI contract tests use the official `openai` Python SDK for
  `models.list()` and error-envelope parsing.
- Equivalence tests send the same core decision through HTTP and compare
  answers, usage and metadata.
- OpenAPI tests assert the real Pydantic request/response/error schemas.
- Transport tests cover auth, request ids, 413, CORS, readiness and per-request
  model selection.

## Out of scope for phase 2

- Metrics/Prometheus, rate limiting and load tests are implemented in phase 5
  (see [OPERATIONS.md](OPERATIONS.md)).
- CLI batch/JSONL and custom-model workflows are implemented in phase 3
  (see [CLI.md](CLI.md)).
- Pinned catalog hashes and source-only CI are implemented in phase 4; wheel,
  PyPI and distribution tagging are deferred to phase 6.
