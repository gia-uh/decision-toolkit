# Changelog

All notable changes to this project are documented here. The project has not
published a distribution artifact yet; versions are repository milestones.

## [Unreleased]

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

## [0.1.0] — unreleased

The version is not tagged or packaged yet. Wheel/PyPI construction is deferred.
