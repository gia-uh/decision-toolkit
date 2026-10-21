# decision-toolkit

**Local-first decision models on CPU**: typed choice, yes/no (`noul`) and
score questions with a calibrated probability for every answer. One ONNX
engine, one adapter per model contract, and three surfaces over the same core:
library, CLI and REST.

Co-developed by [GIA-UH](https://github.com/gia-uh) and [Syalia S.R.L.](https://github.com/syalia-srl).

[Quickstart](QUICKSTART.md){ .md-button .md-button--primary }
[Installation](INSTALLATION.md){ .md-button }

[![PyPI](https://img.shields.io/pypi/v/decision-toolkit.svg)](https://pypi.org/project/decision-toolkit/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://pypi.org/project/decision-toolkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/gia-uh/decision-toolkit/blob/main/LICENSE)

## What it is

- A **library** that turns a state and typed questions into probabilities,
  without generating free text.
- A **CLI** (`dtk`) for decisions, classification, batches and model-cache
  management.
- An optional **REST server** implementing the TypeSafe System One schema and
  the OpenAI Decisions beta schema.
- A **model catalog** of open-weights ONNX builds, pinned by revision and
  SHA256, with selectable quantizations (variants) on CPU.
- A **local-first** toolkit: models are downloaded explicitly, decisions run in
  process, and there is no hosted service in the middle.

## What it is not

- It is **not a generative LLM**: it does not produce prose, explanations or
  chat turns.
- It is **not a rules engine**: the questions are typed, but the probabilities
  come from a model.
- It is **not a vector database** or a search service; it returns decisions.
- It does **not** download models implicitly. `dtk models download` is the only
  command that uses the network.
- It does **not** persist your states or decisions by default.

## Highlights

- **One ONNX engine** (`onnxruntime`), in process, on CPU.
- **Three surfaces**, one implementation: library, CLI and REST.
- **Pinned catalog**: revision, per-file SHA256 and ONNX-session verification.
- **Variants coexist**: download and compare quantizations without replacing
  the default.
- **Standard REST selection**: use the `id@variant` model alias; there is no
  non-standard `variant` field in request bodies.
- **Offline after download**: copy the cache or import a local model.

## Install

```bash
uv pip install "decision-toolkit[onnx]"          # library, CLI and runtime
uv pip install "decision-toolkit[onnx,server]"   # + REST server
```

See [Installation](INSTALLATION.md) for source checkouts, offline machines,
extras, requirements and where the cache lives.

## 30-second example

```bash
dtk models download gliner2.5-multi
dtk models use gliner2.5-multi
```

```python
from decision_toolkit import DecisionModel, choice, noul

with DecisionModel("gliner2.5-multi") as model:
    decision = model.decide(
        "Meeting with Acme: 10% discount, sign next week.",
        {
            "cat": choice("What kind of record is this?", {
                "meeting": "a meeting or call",
                "tasks": "things to do",
                "other": "anything else",
            }),
            "urgent": noul("Does it need a reply today?"),
        },
    )

print(decision["cat"].choice, decision["cat"].confidence)
print(decision["urgent"].noul)
```

The same decision from the CLI:

```bash
dtk decide --model gliner2.5-multi \
  --state "Meeting with Acme: 10% discount, sign next week." \
  --q 'cat=choice:meeting|tasks|other' \
  --q 'urgent=noul:Does it need a reply today?' --json
```

## Documentation map

| If you want to… | Read |
| --- | --- |
| Go from install to a first decision | [Quickstart](QUICKSTART.md) |
| Install, verify or work offline | [Installation](INSTALLATION.md) |
| Download, inspect or choose a model | [Models and the model cache](MODELS.md) |
| Compare every catalogued variant | [Variants and quantization](VARIANTS.md) |
| Use the Python API in depth | [Library guide](LIBRARY.md) |
| Read the full core contract | [Core API](CORE.md) |
| Run the examples | [Examples](EXAMPLES.md) |
| Copy a workflow | [Recipes](RECIPES.md) |
| Use the command line | [CLI](CLI.md) |
| Call the REST service | [REST API](REST.md) |
| Operate a server | [Operations](OPERATIONS.md) |
| Reproduce latency and parity | [Benchmarks](BENCH.md) |
| Fix a problem | [Troubleshooting](TROUBLESHOOTING.md) |
| Check a concept or term | [FAQ](FAQ.md) · [Glossary](GLOSSARY.md) |
| Build or publish a release | [Release](RELEASE.md) |

## Status

The current release line is **0.4.x**. The core API and the REST contracts are
usable, but the project is still evolving; pin the version you deploy and read
the [changelog](https://github.com/gia-uh/decision-toolkit/blob/main/CHANGELOG.md)
before upgrading.

REST compatibility is documented in [REST](REST.md). The TypeSafe System One
and OpenAI Decisions shapes are treated as pinned contracts; local extensions
are namespaced and documented separately.

## License

The toolkit is MIT licensed. Each model keeps its own license; see the
[model cards](models/README.md) and
[THIRD_PARTY_NOTICES](https://github.com/gia-uh/decision-toolkit/blob/main/THIRD_PARTY_NOTICES.md).
For bugs and feature requests, use the
[issue tracker](https://github.com/gia-uh/decision-toolkit/issues).
