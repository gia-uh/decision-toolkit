# Library guide

The programmatic API is the primary interface. It owns model loading, typed
questions, inference and result inspection; the CLI and REST server are thin
surfaces over it.

For every signature and validation rule, see the [Core API](CORE.md). This page
is the task-oriented path.

## Install and get a model

```bash
uv pip install "decision-toolkit[onnx]"
dtk models download gliner2.5-multi
```

The library never downloads a model implicitly. Use `dtk models use` to persist
a default, or pass `identifier=` and `variant=` explicitly.

## Load a model

`DecisionModel` is lazy: it loads the ONNX session on the first decision, or
when you call `warmup()`. Use it as a context manager so sessions are released
deterministically.

```python
from decision_toolkit import DecisionModel

with DecisionModel("gliner2.5-multi") as model:
    model.warmup()
    info = model.model_info()
    print(info.id, info.adapter, model.variant)
```

The same model at a specific quantization:

```python
model = DecisionModel("gliner2.5-multi", variant="fp32")
```

An explicit variant must already be cached. See [Models](MODELS.md) for
selection precedence and cache commands.

## Decide

A decision is a mapping of question names to typed answers:

```python
from decision_toolkit import DecisionModel, choice, noul, score

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
            "tone": score("What is the tone?", ["calm", "annoyed", "furious"]),
        },
    )

decision["cat"].choice          # "meeting"
decision["cat"].confidence      # probability of the chosen option
decision["urgent"].noul         # p(true)
decision["tone"].level          # most likely level
decision["tone"].score          # expected score
```

Every answer exposes:

- `kind`: `choice`, `noul` or `score`;
- `value` and the convenient accessors;
- `confidence` and `confidence_kind`;
- `truncated` and `warnings`;
- the raw adapter data when you need it.

`decision` also exposes `model`, `variant`, `adapter`, `usage`, `latency_ms`,
`warnings`, `request_id`, `id` and `truncated`. Use `decision.to_dict()` for the
serialized answer payload or `decision.to_payload()` for answers plus identity,
usage and latency.

## Classify one label

`classify` is a convenience over a choice question with a threshold:

```python
from decision_toolkit import DecisionModel

with DecisionModel("laya") as model:
    result = model.classify(
        "Dentist next Tuesday at 9",
        {
            "reminder": "an event with a date",
            "task": "a thing to do",
            "idea": "an idea",
        },
        threshold=0.6,
    )

print(result.label, result.confidence, result.below_threshold)
```

A result below the threshold is still returned; inspect `below_threshold`
instead of treating the label as certain.

## Batch and async

`decide_many` loads the model once and runs the states through the same
instance:

```python
with DecisionModel("gliner2.5-multi") as model:
    decisions = model.decide_many(
        [
            "Call Maria tomorrow.",
            "Ana is the marketing lead.",
            "What if we launch in September?",
        ],
        {
            "cat": choice("What kind of record is this?", {
                "meeting": "a meeting or call",
                "tasks": "things to do",
                "contact": "a person's details",
                "idea": "an idea or question",
                "other": "anything else",
            }),
        },
        request_id="batch-42",
    )
```

The current adapters execute batches sequentially; `Capabilities.batch` is
`False` until one declares native ONNX batching. The API is stable either way.

Async methods run the same core in a worker thread:

```python
decision = await model.adecide("Meeting with Acme.", {"urgent": noul("Today?")})
decisions = await model.adecide_many(states, questions)
classification = await model.aclassify("...", categories, threshold=0.6)
```

## Usage and latency

Every decision carries token accounting and timing:

```python
usage = decision.usage
print(usage.input_tokens, usage.output_tokens, usage.total_tokens)
print(usage.estimated)        # explicit: true when the count is estimated
print(decision.latency_ms)
```

`estimated` is always present. The core does not claim an exact model-internal
prompt count when the adapter cannot provide one.

## Strict mode and truncation

Adapters with a context window report truncation instead of hiding it:

```python
answer.truncated              # bool
answer.warnings               # ("input was truncated to fit max_len=1024",)
decision.truncated            # any answer was truncated
```

Use strict mode to raise `TruncationError` instead:

```python
model = DecisionModel("julia-1", strict=True)
```

`strict=True` can also be passed per `decide`/`classify` call. Shorten the state
or reduce the number of questions if the input does not fit.

## Request correlation

Use `request_context` when several calls belong to one application request:

```python
from decision_toolkit import request_context

with request_context("req-42"):
    decision = model.decide("...", {"urgent": noul("Today?")})

print(decision.request_id)      # "req-42"
```

The ID is propagated to structured events and to the returned decision.

## Capabilities and errors

Check what the selected model can do before relying on a feature:

```python
capabilities = model.capabilities
print(capabilities.question_types)
print(capabilities.languages)
print(capabilities.max_context_tokens)
```

The current catalog is text-only. `box`, assets and multi-label are typed in the
core but are not supported by the current adapters. Unsupported requests raise a
typed error instead of silently changing the question.

The exception taxonomy is documented in [Core API](CORE.md#errors). Catch the
base `DecisionToolkitError` for application-level handling, or the specific
subclass when the recovery path depends on the cause.

## Custom ONNX and adapters

For a local ONNX layout you control:

```python
from decision_toolkit import DecisionModel

model = DecisionModel.from_onnx(
    "./my-model",
    adapter="julia",
    files={
        "model": "model.onnx",
        "tokenizer": "tokenizer.json",
    },
)
```

`adapter` must match the actual layout, and `files` must provide every required
role. The toolkit does not guess either. Custom models are not catalogued and
do not receive hash verification or support.

Third-party adapters can be registered in-process:

```python
from decision_toolkit import register_adapter

register_adapter("my-layout", MyAdapter)
```

See [Core API](CORE.md#extensibility) for the adapter contract.

## Where to go next

- [Core API](CORE.md) — the full contract, errors and serialization.
- [Models](MODELS.md) — cache, variants and offline use.
- [CLI](CLI.md) — the same decisions from scripts.
- [REST](REST.md) — HTTP projection and public schemas.
- [Examples](EXAMPLES.md) — complete runnable examples.
