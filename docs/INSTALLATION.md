# Installation

## Requirements

- **Python 3.12 or 3.13**.
- A **CPU**; no GPU is required. The supported engine is ONNX Runtime on CPU.
- Disk space for the model cache. A default variant is usually 0.5–1.8 GB;
  the complete current catalog is about **13 GB**.
- Linux, macOS or Windows. CI covers Python 3.12 and 3.13.

The toolkit itself is a normal Python package. It does not run a daemon and
does not need a database.

## Extras

| Install | What you get |
| --- | --- |
| `decision-toolkit` | Library, CLI, catalog and model cache. Inference needs the runtime extra. |
| `decision-toolkit[onnx]` | Adds `onnxruntime`, `tokenizers` and `numpy`. Recommended for library and CLI use. |
| `decision-toolkit[server]` | Adds FastAPI and uvicorn. Install it together with the runtime extra for a working server. |
| `decision-toolkit[tracing]` | Adds the OpenTelemetry API for server tracing. |

## From PyPI

With uv:

```bash
uv pip install "decision-toolkit[onnx]"
# REST server:
uv pip install "decision-toolkit[onnx,server]"
```

Inside a uv-managed project, add it to `pyproject.toml` instead:

```bash
uv add "decision-toolkit[onnx]"
```

Published wheels remain compatible with pip for environments that do not use
uv:

```bash
python -m pip install "decision-toolkit[onnx]"
```

## From a source checkout

```bash
git clone https://github.com/gia-uh/decision-toolkit
cd decision-toolkit
uv sync --all-extras
uv run dtk --help
```

`uv sync` installs the locked development environment from `uv.lock`. It does
**not** download model checkpoints.

The project requires uv `>=0.11` for development, builds, CI and releases.

## Verify the install

```bash
dtk --version
dtk doctor
dtk models list
```

`dtk doctor` prints the toolkit home, cache directory, current model and which
runtime modules are available. `dtk models list` does not need a downloaded
model; it shows the catalog and the `cached` state of every variant.

To verify inference, download one model and run a decision:

```bash
dtk models download gliner2.5-multi
dtk models use gliner2.5-multi
dtk decide --model gliner2.5-multi \
  --state "Meeting with Acme." --q 'cat=choice:meeting|task|other'
```

## Where things live

The default toolkit home is:

```text
~/.local/share/decision-toolkit/
├── config.json
└── models/
    └── <id>@<variant>/
        ├── model.json          # cache metadata: origin, files, hashes
        └── ...                 # ONNX files, tokenizer and model config
```

| Thing | Default |
| --- | --- |
| Toolkit home | `~/.local/share/decision-toolkit/` |
| Model cache | `<home>/models/` |
| Configuration | `<home>/config.json` |
| Current model | `current` and `current_variant` in `config.json` |
| Device setting | `device` in `config.json` (the current engine is CPU) |

`config.json` is created on first use. It can point `cache_dir` at another disk:

```bash
dtk config list
dtk config get cache_dir
dtk config set cache_dir /data/dtk-models
```

The `home` key is read-only. For a non-default home, pass `--home DIR` **before**
the subcommand:

```bash
dtk --home /data/dtk-home models list
```

The library accepts the equivalent `home=` argument on `ModelStore` and
`DecisionModel` where applicable.

## No configuration environment variables

There are no environment variables for the toolkit home, cache or device. Use
`--home`/`home=` and `config.json`. The REST server optionally reads
`DECISION_TOOLKIT_API_KEY` for its bearer-token authentication; that is an auth
setting, not general configuration.

## Offline and air-gapped machines

After a model is in the cache, the library, CLI and REST server work offline.
Only `dtk models download` needs the network.

On a connected machine:

```bash
dtk models download gliner2.5-multi
dtk models verify gliner2.5-multi
```

Copy the **whole toolkit home** (or at least `models/` plus the relevant
selection in `config.json`) to the offline machine. On the offline machine:

```bash
dtk --home /data/dtk-home models current
dtk --home /data/dtk-home models verify gliner2.5-multi
```

For Python packages on an air-gapped machine, build a wheelhouse on a connected
machine and install from local wheels with the package manager or pip's
`--no-index --find-links` options.

A custom ONNX directory can be copied and imported locally; see
[Models](MODELS.md#import-a-local-model). Importing does not add a verified
catalog entry and does not download anything.

## Upgrading

```bash
uv pip install --upgrade "decision-toolkit[onnx]"
```

If uv does not see a new release, refresh its index metadata:

```bash
uv lock --refresh
uv pip install --refresh "decision-toolkit[onnx]"
```

See [Troubleshooting](TROUBLESHOOTING.md) for the common installation and cache
errors.

## Next steps

- [Quickstart](QUICKSTART.md) — first decision.
- [Models](MODELS.md) — cache, variants and offline imports.
- [Library guide](LIBRARY.md) — the Python API.
