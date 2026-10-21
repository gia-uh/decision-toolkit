# Quickstart

This page takes you from a fresh install to a first decision, then to the CLI
and the optional REST service. It uses `gliner2.5-multi`, the general-purpose
default in the catalog.

## 1. Install

`decision-toolkit` requires **Python 3.12+** and runs on CPU. Install the
inference runtime with the library and CLI:

```bash
uv pip install "decision-toolkit[onnx]"
```

If you manage a project with uv, use `uv add "decision-toolkit[onnx]"` instead.
The REST server is an optional extra; see step 5.

## 2. Get a model

The toolkit has its own cache. It never downloads a model implicitly, and
`dtk models download` is the only networked command.

```bash
dtk models download gliner2.5-multi
dtk models use gliner2.5-multi
dtk models current
```

The default `fp16` build is about **577 MB**. `dtk models list` shows every
catalogued variant, its size and whether it is already cached. See
[Models and the model cache](MODELS.md) for cache locations, offline installs
and other models.

## 3. First decision from the library

```python
from decision_toolkit import DecisionModel, choice, noul

with DecisionModel("gliner2.5-multi") as model:
    decision = model.decide(
        "Meeting with Acme: we agreed a 10% discount and will sign next week.",
        {
            "cat": choice("What kind of record is this?", {
                "meeting": "a meeting or call",
                "tasks": "things to do",
                "other": "anything else",
            }),
            "urgent": noul("Does it need a reply today?"),
        },
    )

print(decision["cat"].choice)        # "meeting"
print(decision["cat"].confidence)    # probability of the chosen option
print(decision["urgent"].noul)       # p(true)
```

No generated text is parsed: each answer is a typed probability. See the
[Library guide](LIBRARY.md) for score questions, batches and classification.

## 4. The same decision from the CLI

```bash
dtk decide --model gliner2.5-multi \
  --state "Meeting with Acme: 10% discount, sign next week." \
  --q 'cat=choice:meeting|tasks|other' \
  --q 'urgent=noul:Does it need a reply today?' --json
```

The CLI reads questions from a JSON/YAML file as well; see the [CLI guide](CLI.md).

## 5. Optional: start the REST server

Install the server extra (together with the inference runtime if you do not
already have it):

```bash
uv pip install "decision-toolkit[onnx,server]"
```

Start a local server:

```bash
dtk serve --model gliner2.5-multi --host 127.0.0.1 --port 8000 --warmup
```

Call the standard System One endpoint from another terminal:

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gliner2.5-multi",
    "state": "Meeting with Acme: 10% discount, sign next week.",
    "questions": {
      "cat": {
        "type": "choice",
        "instructions": "What kind of record is this?",
        "criteria": {"meeting": "a meeting or call", "tasks": "things to do", "other": null}
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does it need a reply today?"
      }
    }
  }' | python -m json.tool
```

Quantizations are selected with a **model alias**, for example
`"model": "gliner2.5-multi@fp32"`. There is no `variant` field in the request
body. See [REST](REST.md) for both public schemas, authentication and limits.

## 6. Try another variant

Variants are pinned ONNX builds of the same model. Downloading one does not
replace the default and does not change the current selection:

```bash
dtk models list
dtk models variants gliner2.5-multi
dtk models download gliner2.5-multi --variant fp32
dtk models use gliner2.5-multi --variant fp32
dtk models current                         # gliner2.5-multi@fp32
```

The same variant in a one-off command:

```bash
dtk decide --model gliner2.5-multi --variant fp16 \
  --state "Meeting with Acme." --q 'cat=choice:meeting|task|other' --json
```

In REST, use `gliner2.5-multi@fp32` instead of a body field. See
[Variants and quantization](VARIANTS.md) for the complete table.

## Next steps

- [Installation](INSTALLATION.md) — extras, source checkout, offline and
  where things live.
- [Models](MODELS.md) — cache, model selection and imports.
- [Library guide](LIBRARY.md) — the Python API.
- [Examples](EXAMPLES.md) — runnable library, CLI and REST examples.
- [Troubleshooting](TROUBLESHOOTING.md) · [FAQ](FAQ.md) — common errors and
  conceptual questions.
