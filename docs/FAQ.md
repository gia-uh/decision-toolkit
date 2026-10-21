# FAQ

## Is this a generative LLM?

No. The models score typed questions and return probabilities. They do not
generate prose, explanations or chat turns, and there is no text parser between
the model and the answer.

## What does `noul` mean?

`noul` is a yes/no question. Its result is the probability of `true`, plus the
normalized confidence and metadata. The name avoids overloading “boolean” for a
model score that is not a hard logical value.

## What is the difference between `choice`, `noul` and `score`?

- `choice`: pick one option from a list. The core has a multi-label flag, but
  no model in the current catalog declares multi-label support.
- `noul`: estimate the probability of yes versus no.
- `score`: choose an ordered level such as `calm`, `annoyed`, `furious`; the
  expected level is available separately.

All three are validated against the selected model's capabilities.

## Why probabilities instead of labels?

A threshold can turn a probability into a label, but the probability preserves
uncertainty and lets each application choose its own operating point. Use
`classify(..., threshold=...)` when you want a label plus a `below_threshold`
flag.

Are the probabilities calibrated? Calibration evidence is measured per model
and variant in [Benchmarks](BENCH.md); do not assume that a probability from one
model is directly comparable with a probability from another.

## Does it run offline?

Yes, after the model is in the local cache. Only `dtk models download` needs the
network. The library, CLI and REST server otherwise run locally. See
[Models](MODELS.md#offline-workflow).

## Does it download models implicitly?

No. If a model or variant is missing, the call fails instead of downloading it.
Run `dtk models download <id>` explicitly. This keeps startup deterministic and
makes offline operation possible.

## Do I need a GPU?

No. The supported runtime is ONNX Runtime on CPU. GPU acceleration is not part
of the documented path. Larger models need more RAM and take longer.

## Which model should I start with?

For general choice, yes/no and score decisions, start with
`gliner2.5-multi` and its default variant. If your state is not English, start
with `laya-multilingual`. Read [Models](MODELS.md#choosing-a-model) and the
[model cards](models/README.md), then measure on a small labelled sample from
your domain.

## Which variant should I choose?

Start with the catalog default. Try `fp32` when quality matters more than
latency, and `q4`/`q4f16`/`q8` when memory or disk is tight. Verify quality on
your own data before switching. The default is a sensible starting point, not a
universal recommendation.

## Can I download several variants of the same model?

Yes. Variants coexist in the cache. Downloading one does not replace another;
`dtk models use` selects the persistent default, and `--variant` selects one
for a single command.

## Can I load two variants at the same time?

Yes, for example over REST with `gliner2.5-multi@fp16` and
`gliner2.5-multi@fp32`. They occupy separate registry slots. The library can
also hold separate `DecisionModel` instances. Do not assume their probability
outputs are interchangeable without evaluation.

## Can I mix models in one application?

You can load different models in separate instances, but a decision is made by
one model at a time. Different models have different adapters, calibration and
behavior; do not average or compare their probabilities without measuring what
that means for your task.

## Does REST use the TypeSafe and OpenAI schemas?

Yes. The server exposes System One at `/v1/systemone` and OpenAI Decisions at
`/v1/decisions`, with the pinned request and response shapes documented in
[REST](REST.md). Local extensions are namespaced.

## How do I select a quantization in REST?

Use a model alias in the standard `model` field:

```json
{"model": "gliner2.5-multi@fp32"}
```

There is no `variant` field in a System One or OpenAI request body. The base id
uses the server's `--variant` or the persisted default.

## Is the REST server authenticated?

It can be. `dtk serve --api-key ...` requires a bearer token, and
`DECISION_TOOLKIT_API_KEY` is an optional environment fallback. Without either,
the server is unauthenticated. It is local-first infrastructure: bind it to
localhost or put your own authenticated proxy/TLS in front of it when exposed.

## Does the server store my states or decisions?

No, not by default. It evaluates a request and returns a response; it does not
create a decision database. State and question text are not used as metric
labels. Application logs and your own proxy may still retain request metadata;
review [Operations](OPERATIONS.md#pii-and-logging).

## Can I use a custom ONNX model?

Yes, with an explicit adapter and file map, using `DecisionModel.from_onnx(...)`
or `dtk models import`. Custom models are not catalogued, do not receive hash
verification or support, and do not get a guessed layout. See
[Core API](CORE.md#custom-local-onnx) and [CLI](CLI.md#custom-models).

## What licenses do the models have?

The toolkit is MIT licensed. Each model has its own license, recorded in its
model card; check the card and the upstream model page before redistribution.
See [THIRD_PARTY_NOTICES](https://github.com/gia-uh/decision-toolkit/blob/main/THIRD_PARTY_NOTICES.md).

## What Python versions are supported?

Python 3.12 and 3.13. The package requires Python 3.12 or newer.

## Is this ready for production?

The 0.4.x line is usable, but still evolving. Pin the version you deploy, read
the [changelog](https://github.com/gia-uh/decision-toolkit/blob/main/CHANGELOG.md),
and use the operational limits and runbook in [Operations](OPERATIONS.md). The
REST contracts are pinned; the core API can still change between minor releases.

## Does it support images or assets?

The core has typed support for assets and `box`, but every model currently in
the catalog is text-only. The REST server rejects image/asset and `box` requests
for this catalog. See [REST](REST.md#not-implemented-by-this-server).

## Is it a vector database or search engine?

No. It returns decisions for typed questions. It does not store vectors, index
documents or provide a search UI.

## How do I report a bug?

Include the exact command or request, the output of `dtk doctor --json`, the
toolkit version, the model and variant, and any error code. Redact state text
that you cannot share. Use the repository issue tracker linked from the
[home page](index.md).
