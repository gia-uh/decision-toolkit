# Changelog

All notable changes to this project are documented here. Version `0.4.1` is
published on PyPI.

## [0.4.1] — 2026-10-21

Catalog presentation patch: every variant now carries its REST alias
(`id@variant`) in `dtk models list`, `--json` and `dtk models variants`. Model
cards include copy-paste loading examples for library, CLI and REST for every
variant. No model or runtime changes.

## [0.4.0] — 2026-10-21

REST variant-selection correction. Quantizations are selected through standard
model aliases (`id@variant`), not a non-standard `variant` body field. The same
REST process can load two variants of one base model in separate registry slots.

### Fixed

- Removed `variant` from System One and OpenAI request bodies; standard clients
  now see only the public request fields.
- Added `catalog-id@variant` model aliases to `/v1/models` and
  `/v1/models/{id}`. The standard `model` field selects the alias.
- Registry keys and slots are per alias, so `laya-multilingual@fp16` and
  `laya-multilingual@fp32` load independently and concurrently.
- Updated README, REST, variants, examples and CLI help so REST never shows a
  variant field in a standard request body.

## [0.3.1] — 2026-10-21

Catalog UX patch: `dtk models list` now prints one row per variant, with
`variant`, quantization, role, size, cache state and the exact download/select
commands. `dtk models variants` prints both commands for every variant. No
model, adapter or REST contract changes.

## [0.3.0] — 2026-10-21

Catalog expansion: three verified ONNX decision models and eight variants.
Laya multilingual, Open-Jev DeBERTa v3 large and Strands Decider 2B are now
available through the library, CLI and REST projection.

### Added

- `laya-multilingual` with fused Laya ONNX support (`fp16` default, `fp32`).
- `open-jev-deberta-v3-large` with a span-pooling adapter (`fp16` default,
  `fp32`, `q4`, `q4f16`).
- `strands-decider-2b` with a pointer-head adapter (`q4f16` default, `q8`).
- New `open-jev` and `strands-decider` adapters and a fused branch in `laya`.
- Model cards, benchmark cases, a catalog proposal and documentation for every
  added variant.
- All 15 catalogued variants verified for SHA256, files and ONNX sessions.

## [0.2.1] — 2026-10-20

Metadata-only release: this project's documentation and project URLs now point
to `https://gia-uh.github.io/decision-toolkit/` instead of using the
`uhgia.org` custom domain owned by the `gia-uh/gia-uh.github.io` repository.
No runtime, model, CLI or REST behavior changed.

## [0.2.0] — 2026-10-20

REST conformance correction. The previous documentation called the TypeSafe
contract "System One 1.14"; that version was not real. The public TypeSafe API
schema is `0.2.0`. This release aligns the request/response shapes and error
behavior with the published schemas, documents unsupported vision/assets as an
explicit text-only limitation, and removes invented OpenAI response fields.

### Fixed

- System One is now pinned to the published TypeSafe API schema `0.2.0`, not
  the invented `1.14` label. `/v1/info` reports `0.2.0`.
- System One accepts rich instructions/criteria, the standard `assets` field,
  one-level scores and non-negative `output_tokens`; schema validation returns
  `422` with a FastAPI-style `detail` array. Assets/images return `501` because
  the catalog is text-only.
- OpenAI Decisions responses now expose the public `model`, `answers` and
  `usage` fields (including usage details), without invented `id`, `object`,
  `created_at`, `status` or `safety_identifier` fields. Toolkit metadata remains
  namespaced under `x_decision_toolkit`.
- `variant` is explicitly documented as a local extension, not a TypeSafe or
  OpenAI field. Standard clients select a model and omit it.

## [0.1.7] — 2026-10-20

Variant consistency release. The selected quantization now agrees across the
library, CLI and REST API, including catalog defaults, persisted `dtk models
use` selections, per-call overrides, readiness/info metadata and response
payloads.

### Fixed

- `DecisionModel.variant` reports the explicit, persisted or catalog-default
  variant before and after resolution, without inheriting another model's
  selection.
- The REST registry resolves the catalog default for a configured model and
  honors the persisted variant for the current model. `/v1/info`, `/ready`,
  `/v1/models` and `/v1/decision-models` expose the effective default and
  `current_variant`.
- REST responses echo the selected variant under `x_decision_toolkit.variant`;
  a fixed injected model rejects a conflicting request variant.
- Added cross-surface tests for library resolution, CLI pass-through, registry
  defaults and System One/OpenAI REST requests, plus documentation for selection
  precedence.

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

- Versioned System One and OpenAI Decisions beta projections.
- Bearer auth, per-request model/variant, usage, standard error envelope.
- OpenAI-shaped `/v1/models`, rich `/v1/decision-models`, readiness `503`.
- Request ids, access logs, body limits, CORS and timeouts.

### Phase 1 — core

- Typed `DecisionModel` API with lifecycle, batching, async, usage, warnings,
  normalized confidence, capabilities, structured events and atomic model store.

[0.4.1]: https://pypi.org/project/decision-toolkit/0.4.1/
[0.4.0]: https://pypi.org/project/decision-toolkit/0.4.0/
[0.3.1]: https://pypi.org/project/decision-toolkit/0.3.1/
[0.3.0]: https://pypi.org/project/decision-toolkit/0.3.0/
[0.2.1]: https://pypi.org/project/decision-toolkit/0.2.1/
[0.2.0]: https://pypi.org/project/decision-toolkit/0.2.0/
[0.1.7]: https://pypi.org/project/decision-toolkit/0.1.7/
[0.1.6]: https://pypi.org/project/decision-toolkit/0.1.6/
[0.1.5]: https://pypi.org/project/decision-toolkit/0.1.5/
[0.1.4]: https://test.pypi.org/project/decision-toolkit/0.1.4/
[0.1.3]: https://pypi.org/project/decision-toolkit/0.1.3/
[0.1.2]: https://pypi.org/project/decision-toolkit/0.1.2/
[0.1.1]: https://pypi.org/project/decision-toolkit/0.1.1/
[0.1.0]: https://pypi.org/project/decision-toolkit/0.1.0/
