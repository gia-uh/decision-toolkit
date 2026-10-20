# Changelog

All notable changes to this project are documented here. Version `0.1.1` is
published on PyPI.

## [0.1.1] — 2026-10-20

Metadata-only release: refreshed the PyPI project description after the final
README correction. No code or model behavior changed.

## [0.1.0] — 2026-10-20

### Phase 6 — packaging

- Added `py.typed`, project URLs, co-author metadata and sdist inclusion.
- Wheel and sdist build successfully; `twine check` passes with current tooling.
- Wheel installs in a clean venv with `[onnx,server,tracing]`; `dtk` and the
  REST server work from the installed artifact.
- Added a Trusted Publishing release workflow for TestPyPI → PyPI and GitHub
  Releases.
- Published `0.1.0` to PyPI:
  <https://pypi.org/project/decision-toolkit/0.1.0/>.
- Trusted Publishing workflow is available for future TestPyPI/PyPI releases.
- Added `docs/RELEASE.md`.

### Phase 5 — operations

- Per-key rate limiting with `429` + `Retry-After` (disabled by default).
- Bounded inference concurrency (`--max-concurrent`) with `503 server_busy`
  instead of unbounded queueing.
- Prometheus `/metrics`: HTTP, decision, token, model-load, rate-limit, busy,
  inflight and model-cache metrics with low-cardinality labels.
- Optional OpenTelemetry spans (`--tracing`) with a no-op fallback.
- `bench/load.py` concurrent load probe and a versioned backpressure result.
- Deterministic REST fuzz/negative corpus and PII sentinel tests.
- `pip-audit` CI job; `docs/OPERATIONS.md` runbook.
- Still no wheel/PyPI/tag.

### Phase 4 — reproducibility

- Pinned every catalog variant to a HuggingFace commit SHA.
- Recorded SHA256 for every declared model file and external-data file.
- `dtk models verify` now treats the catalog hashes as the source of truth and
  also verifies `.onnx.data` files.
- Model downloads reject files that do not match the pinned hashes.
- Added a reproducible benchmark/calibration harness in `bench/` with versioned
  results and a synthetic calibration suite.
- Added source-only CI for Python 3.12/3.13 (lint, tests, CLI smoke; no wheel).
- Added model-card reproducibility sections, `docs/BENCH.md`, contributing,
  security and third-party notices.

### Phase 3 — CLI

- Consistent `--json`, quiet/verbose modes and exit codes (`0/1/2/3/130`).
- JSON/YAML questions files, JSONL batching, custom `--adapter/--files`.
- Variant-aware `use/current/delete` and strict `verify --all`.
- `config get/set/list`, `version`, `completion`.
- `serve --workers/--reload`, API key, CORS, log format and warmup.

### Phase 2 — REST

- Versioned System One 1.14 and OpenAI Decisions beta projections.
- Bearer auth, per-request model/variant, usage, standard error envelope.
- OpenAI-shaped `/v1/models`, rich `/v1/decision-models`, readiness `503`.
- Request ids, access logs, body limits, CORS and timeouts.

### Phase 1 — core

- Typed `DecisionModel` API with lifecycle, batching, async, usage, warnings,
  normalized confidence, capabilities, structured events and atomic model store.

[0.1.1]: https://pypi.org/project/decision-toolkit/0.1.1/
[0.1.0]: https://pypi.org/project/decision-toolkit/0.1.0/
