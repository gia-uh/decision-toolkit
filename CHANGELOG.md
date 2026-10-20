# Changelog

All notable changes to this project are documented here. Version `0.1.6` is
published on PyPI.

## [0.1.6] — 2026-10-20

Variant catalog and selection release. `dtk models list` now exposes every
quantization, `dtk models variants` prints the exact install command, and a
variant selected with `dtk models use` is honored by later commands. Release
verification now installs the published wheel URL from the registry JSON API,
avoiding simple-index cache delays.

### Fixed

- `dtk models list` now shows every catalogued variant, marks the default with
  `*` and lists cached variants. `dtk models variants` prints the exact download
  command for each variant, and `download --dry-run` reports all available
  variants.
- A variant selected with `dtk models use <id> --variant <name>` is now used by
  later commands that name the model without repeating `--variant`.
- Added a complete variants/quantization table and download/select examples to
  the README, CLI reference, model cards and the new variants guide.
- Release verification resolves the published wheel URL from the TestPyPI/PyPI
  JSON API and installs it with uv, avoiding simple-index cache delays.

## [0.1.5] — 2026-10-20

Release verification fix for the uv migration: TestPyPI installation now uses
uv's `unsafe-best-match` index strategy and retries while the TestPyPI simple
index propagates. No runtime or model changes.

## [0.1.4] — 2026-10-20

TestPyPI-only packaging validation build for the uv migration. Package
management, dependency resolution, builds, CI, release verification and
documentation deployment moved to uv; development and docs dependencies moved
to PEP 735 dependency groups, and `uv.lock` is committed. PyPI publication
stopped when TestPyPI verification could not resolve the new version; `0.1.5`
fixes that verification step. No runtime or model changes.

## [0.1.3] — 2026-10-20

Infrastructure release: CI builds and clean-installs the wheel on every PR;
release verification installs from TestPyPI before PyPI and from PyPI before
the GitHub release; MkDocs documentation is published to GitHub Pages; PyPI
environment protection and updated GitHub Actions. No runtime or model changes.

## [0.1.2] — 2026-10-20

Trusted Publishing validation release. No functional or model changes.

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

[0.1.6]: https://pypi.org/project/decision-toolkit/0.1.6/
[0.1.5]: https://pypi.org/project/decision-toolkit/0.1.5/
[0.1.4]: https://test.pypi.org/project/decision-toolkit/0.1.4/
[0.1.3]: https://pypi.org/project/decision-toolkit/0.1.3/
[0.1.2]: https://pypi.org/project/decision-toolkit/0.1.2/
[0.1.1]: https://pypi.org/project/decision-toolkit/0.1.1/
[0.1.0]: https://pypi.org/project/decision-toolkit/0.1.0/
