# Third-party notices

decision-toolkit itself is MIT-licensed (see `LICENSE`). This file summarizes
third-party components referenced by the project. It is not a replacement for
the license text shipped by each upstream project.

## Curated model artifacts

The toolkit does not redistribute model weights; `dtk models download` fetches
them from HuggingFace at the pinned revisions recorded in `catalog.py`.

| Model | Upstream repository | Revision | License |
| --- | --- | --- | --- |
| GLiNER2.5 multi v1 | `jugaadsrl/gliner2.5-multi-v1-onnx` | `3de2c11f55853b7919f985a59bfc8df27b5ec991` | Apache-2.0 |
| GLiNER2.5-Decide | `onnx-community/GLiNER2.5-Decide-ONNX` | `18271032588167fe9a3fbb8c5a4900b1231f8119` | Apache-2.0 |
| Laya (English) | `nvkudva/laya-onnx-q8` | `cccc9a01e06189cbb1c153a51ea32642102e306d` | Apache-2.0 |
| Julia 1 | `SupersonicLabs/Julia-1-ONNX` | `82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8` | Apache-2.0 |
| OpenDecider nano | `manjunathshiva/opendecider-nano-ONNX` | `5137ae7b4f1cf8c55f1d05d3f5cdd3694d92da3e` | Apache-2.0 |

Each model card links the exact files and SHA256 digests. Review the upstream
license and model card before commercial use.

## Python dependencies

### Base

| Package | Purpose | License |
| --- | --- | --- |
| `huggingface_hub` | explicit model downloads | Apache-2.0 |
| `PyYAML` | questions/config files | MIT |

### `[onnx]` extra

| Package | Purpose | License |
| --- | --- | --- |
| `onnxruntime` | CPU inference | MIT |
| `tokenizers` | tokenizer runtime | Apache-2.0 |
| `numpy` | array input/output | BSD-3-Clause |

### `[server]` extra

| Package | Purpose | License |
| --- | --- | --- |
| `fastapi` | REST framework | MIT |
| `uvicorn` | ASGI server | BSD-3-Clause |
| `python-multipart` | form support | Apache-2.0 |
| `pydantic` | wire schemas (via FastAPI) | MIT |
| `starlette` | ASGI toolkit (via FastAPI) | BSD-3-Clause |

### `[tracing]` extra

| Package | Purpose | License |
| --- | --- | --- |
| `opentelemetry-api` | optional tracing spans | Apache-2.0 |

### Development / tests

| Package | Purpose | License |
| --- | --- | --- |
| `pytest` | tests | MIT |
| `ruff` | lint/format | MIT |
| `httpx` | HTTP test client | BSD-3-Clause |
| `openai` | contract tests only (optional; not a runtime dependency) | Apache-2.0 |
| `pip-audit` | CI dependency scan | Apache-2.0 |

`transformers` is an optional runtime preference used by `engine.load_tokenizer`
if installed; it is not required and not listed in `pyproject.toml`.

## Data and benchmarks

`bench/suites/calibration.jsonl` is a small synthetic Spanish routing set
created for this repository. It is not derived from a third-party dataset.
