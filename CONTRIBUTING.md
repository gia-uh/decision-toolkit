# Contributing

decision-toolkit is a curated runtime, not a mirror of model hubs. Changes that
add a model or an adapter must bring **real evidence**, not a promising README.

## Development setup (source only)

There is no wheel/PyPI release yet. Work from the source tree:

```bash
python -m venv .venv
. .venv/bin/activate
pip install pytest ruff PyYAML huggingface_hub fastapi httpx pydantic numpy
# for real inference:
pip install onnxruntime tokenizers
PYTHONPATH=src pytest -q
```

The fast suite must not download models. Integration tests are opt-in:

```bash
PYTHONPATH=src pytest -m integration -o addopts="" -s
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
PYTHONPATH=src python bench/run.py --suite bench/suites/smoke.jsonl \
  --repeats 3 --output bench/results/YYYY-MM-DD-smoke.json
```

Do not overwrite an old result with a different machine silently; add a new
dated file and record the environment. Latency is post-warmup and is not the
same metric as the historical admission values in the model cards.

## Tests and lint

```bash
ruff check src tests
PYTHONPATH=src pytest -q
python -m decision_toolkit --version
```

CI runs these on Python 3.12 and 3.13 from source. It does **not** build a
wheel; packaging is a later milestone.

## Commits

Use a short imperative subject. Mention the layer (`core`, `rest`, `cli`,
`repro`, `docs`) and whether catalog evidence changed.
