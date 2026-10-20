# Operations (phase 5)

This document covers the operational behavior added in phase 5: rate limits,
backpressure, metrics, tracing, load testing and the PII contract. Packaging is
still a later milestone.

## Server flags

```bash
dtk serve \
  --model gliner2.5-multi --warmup \
  --host 127.0.0.1 --port 8000 \
  --max-concurrent 4 \
  --rate-limit 120 --rate-window 60 --rate-burst 20 \
  --api-key "$DECISION_TOOLKIT_API_KEY" \
  --log-format json
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-concurrent` | `4` | Inference calls allowed at once; extra calls get `503`. |
| `--rate-limit` | `0` | Requests per key per window; `0` disables. |
| `--rate-window` | `60` | Window in seconds. |
| `--rate-burst` | `--rate-limit` | Token-bucket capacity. |
| `--no-metrics` | off | Disable `/metrics`. |
| `--tracing` | off | Enable optional OpenTelemetry spans. |
| `--timeout` | `120` | Per-decision timeout. |
| `--max-body-mb` | `16` | Request body limit. |
| `--max-models` | `2` | Idle models kept loaded (LRU). |

## Rate limiting and backpressure

- **Rate limit key**: SHA256 of the `Authorization` header when present,
  otherwise the client IP. A token/key is never logged.
- **Limits**: token bucket per key. Sustained rate `rate_limit / rate_window`,
  burst capacity `rate_burst` (defaults to `rate_limit`).
- **Rejection**: `429`, `code: "rate_limit_exceeded"`, `Retry-After`.
- **Concurrency gate**: at most `max_concurrent` inference calls. When full,
  the request is rejected immediately with `503`, `code: "server_busy"`,
  `Retry-After`, instead of building an unbounded queue.
- **Executor**: inference runs in a bounded `ThreadPoolExecutor` with
  `max_concurrent` workers. A timeout returns `504`; the worker may finish in
  the background, but the pool cannot grow beyond its configured size.
- `/health`, `/ready` and `/metrics` are exempt from rate limiting.
- Rate limits are **per process**. With `--workers N`, each worker has its own
  bucket; do not treat the limit as global. This is intentional for a
  local-first deployment without Redis.

## Metrics

`GET /metrics` exposes Prometheus text (no auth by default; bind the server to
loopback or put it behind a reverse proxy for remote deployments).

| Metric | Type | Labels |
| --- | --- | --- |
| `dtk_http_requests_total` | counter | method, path, status |
| `dtk_http_request_duration_seconds` | histogram | method, path |
| `dtk_decisions_total` | counter | model, variant, status |
| `dtk_decision_duration_seconds` | histogram | model, variant |
| `dtk_input_tokens_total` | counter | model, variant |
| `dtk_output_tokens_total` | counter | model, variant |
| `dtk_model_load_seconds` | histogram | model, variant, result |
| `dtk_rate_limited_total` | counter | path |
| `dtk_server_busy_total` | counter | — |
| `dtk_inflight_requests` | gauge | — |
| `dtk_inference_inflight` | gauge | — |
| `dtk_model_cache_entries` | gauge | — |

Paths are normalized to route templates (`/v1/models/{id}`), never raw ids.
State and question text are never labels.

Example alerts:

- `rate(dtk_http_requests_total{status=~"5.."}[5m]) > 0`
- `histogram_quantile(0.95, rate(dtk_decision_duration_seconds_bucket[5m]))`
  above a model-specific threshold
- `dtk_server_busy_total` increasing continuously
- `dtk_rate_limited_total` increasing for a single client

## Tracing

`--tracing` enables optional spans when `opentelemetry-api` is installed:

- `dtk.http.request`
- `dtk.decision`
- model load is covered by the decision span and the load metric.

Tracing is a no-op when the package is absent. Exporters are configured by the
application/environment, not by decision-toolkit. Install with:

```bash
pip install "decision-toolkit[tracing]"
```

(No wheel exists yet; this extra applies once packaging is available. For source
work, install `opentelemetry-api` directly.)

## Load testing

```bash
# start a server in another terminal
dtk serve --model gliner2.5-multi --warmup --port 8011 --max-concurrent 4

# probe it
PYTHONPATH=src python bench/load.py \
  --base-url http://127.0.0.1:8011 \
  --model gliner2.5-multi \
  --concurrency 10 --requests 30 \
  --output bench/results/2026-10-20-load.json
```

The committed result (`bench/results/2026-10-20-load.json`) shows the expected
backpressure behavior with four inference slots: 4 requests succeeded, 26 were
rejected with `503 server_busy` + `Retry-After`, none became a `500`, and the
probe measured 48.9 req/s (including very fast rejections). This is a
single-machine probe, not an SLO.

## PII and logging

- Access logs: method, path, status, duration, request id, body size.
- Decision logs: model, variant, adapter, question count, token counts, latency,
  truncation, error code.
- Never logged by default: state, question text, instructions, options, API
  key/token, raw error payloads.
- Tests use sentinel values and assert they never appear in logs, metrics,
  responses or error envelopes.
- If you enable a custom logger/handler, keep this redaction; don't log request
  bodies.

## Dependency scan

CI runs `pip-audit` over the runtime/test dependency set in the `audit` job.
A local audit on 2026-10-20 reported no known vulnerabilities. Re-run it after
changing dependency ranges:

```bash
pip install pip-audit
pip-audit
```

## Runbook

### `429 rate_limit_exceeded`

1. Check the client's sustained rate and `Retry-After`.
2. Raise `--rate-limit`/`--rate-burst` only if CPU headroom exists.
3. For multiple clients behind one IP without auth, use API keys so each key has
   its own bucket.

### `503 server_busy`

1. Check `dtk_inference_inflight` and CPU.
2. Lower client concurrency; raise `--max-concurrent` only if the machine has
   headroom.
3. `--threads` controls ONNX intra-op threads; it is not a per-request knob.
4. `--max-models` affects memory, not throughput.

### `504 timeout`

1. Inspect the model's p95 and the request's context length.
2. Check truncation warnings; very long states may hit the model limit.
3. Raise `--timeout` only as a last resort; a timed-out worker may still be
   using a slot until it finishes.

### Metrics unavailable

- `/metrics` returns 404 when `--no-metrics` is set.
- The endpoint is intentionally not rate limited; protect it at the network
  layer if exposed.

## Multi-worker and shutdown

- `--workers N` starts N processes; each has its own cache, limiter and metrics.
- A reverse proxy should balance across workers. Do not expect a global rate
  limit without an external store.
- On shutdown the registry closes models and the inference executor is asked to
  cancel queued work; in-flight ONNX calls cannot be interrupted safely, so
  allow the process to drain.
