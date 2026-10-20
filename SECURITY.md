# Security policy

## Supported versions

The project is pre-release (`0.1.x` repository milestone). Security fixes are
applied to `main`; no packaged distribution is published yet.

## Reporting a vulnerability

Please do **not** open a public issue for a vulnerability. Use GitHub's
**private security advisory** flow for this repository:

1. Security → Advisories → New draft advisory.
2. Include the affected layer (core/REST/CLI), a minimal reproduction and the
   impact.
3. Allow time for a fix before public disclosure.

## Security model

- **Local-first**: no daemon, no telemetry, no network access during inference.
- **Network access** occurs only on an explicit `dtk models download` or
  `ModelStore.download`.
- **Supply chain**: catalog variants are pinned to commit SHA and SHA256 per
  file. `dtk models verify` fails on a mismatch; downloads reject mismatched
  files. Do not run unpinned custom ONNX models without understanding their
  provenance.
- **REST auth**: Bearer auth is optional and disabled by default for local use.
  Enable it with `--api-key` or `DECISION_TOOLKIT_API_KEY` for any non-loopback
  deployment. Built-in per-key rate limiting and bounded inference concurrency
  are available via `--rate-limit` and `--max-concurrent`; TLS/WAF still belong
  in a reverse proxy.
- **Logging**: states and question text are not logged by default. Access logs
  contain method, path, status, duration and request id.
- **Untrusted inputs**: the core validates question shapes, limits, capabilities
  and body sizes. ONNX Runtime itself parses model files; only run verified
  catalog builds or models you trust.

## Operational recommendations

- Bind to `127.0.0.1` unless you need remote access.
- Set an API key and CORS origins explicitly for any shared deployment.
- Keep the model cache private; it may contain downloaded proprietary or
  licensed model weights.
- Re-run `dtk models verify --all` after copying a cache between machines.
- `/metrics` is public by default on the bound interface; restrict it at the
  network layer when exposing the server remotely, or use `--no-metrics`.
