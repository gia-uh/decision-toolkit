# Contributing

decision-toolkit is a curated runtime, not a mirror of model hubs. Changes that
add a model or an adapter must bring **real evidence**, not a promising README.

## Development setup

[uv](https://docs.astral.sh/uv/) is the only package manager used by this
repository. `uv.lock` is committed; change dependencies through `pyproject.toml`
and refresh the lock with `uv lock`.

```bash
uv sync --all-extras     # project + dev + onnx + server + tracing
uv run pytest -q         # fast suite (no model downloads)
```

The fast suite must not download models. Integration tests are opt-in:

```bash
uv run pytest -m integration -o addopts="" -s
```

The docs dependency group is separate:

```bash
uv run --group docs mkdocs serve
```

`tests/test_integration_gliner2.py` uses `DECISION_TOOLKIT_GLINER2_DIR` and
never downloads.

## Code rules

- Keep inference logic in adapters; the CLI and REST call the core.
- The core is the superset. Do not add wire-specific concepts to it.
- The REST layer may only project core capabilities; if a standard needs
  something, add it to the core first.
- Every new error gets a stable `code` and a test for its HTTP/CLI mapping.
- No silent truncation: mark `truncated`/`warnings` or fail in strict mode.
- Logs carry model/variant/request id, not state/question content by default.

## Model admission

A model enters `CATALOG` only when it has:

1. a clean commercial license;
2. a real public ONNX export;
3. an adapter contract implemented from that export;
4. a **pinned commit SHA** and SHA256 for every declared file;
5. measured parity against a named baseline;
6. CPU latency measured on a known machine;
7. a model card and reproducibility commands.

Models that fail any gate belong in `docs/ROADMAP.md`, not in the catalog.

## Bench evidence

Re-run the harness when a model, adapter or quantized build changes:

```bash
uv run python bench/run.py --suite bench/suites/smoke.jsonl \
  --repeats 3 --output bench/results/YYYY-MM-DD-smoke.json
```

Do not overwrite an old result with a different machine silently; add a new
dated file and record the environment. Latency is post-warmup and is not the
same metric as the historical admission values in the model cards.

## Tests and lint

```bash
uv run ruff check src tests bench
uv run pytest -q
uv run pytest -q -m fuzz        # malformed REST corpus
uv run dtk --version
uv run pip-audit                # dependency scan (CI runs this)
```

CI runs these on Python 3.12 and 3.13, plus `pip-audit`. `tests/test_help_examples.py`
parses every example printed by `dtk --help` and checks referenced catalog
models/variants; run the full examples against a populated cache before a
release. The tag-triggered release workflow builds and publishes the wheel with
uv; see [RELEASE.md](docs/RELEASE.md). For rate limits, metrics, tracing and the load
probe see [OPERATIONS.md](docs/OPERATIONS.md).

## Commits

Use a short imperative subject. Mention the layer (`core`, `rest`, `cli`,
`repro`, `docs`) and whether catalog evidence changed.
