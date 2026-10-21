# Troubleshooting

Start with the diagnostics:

```bash
dtk --version
dtk doctor
dtk models list --quant
```

`dtk doctor` shows the toolkit home, cache directory, current selection and
which runtime modules are installed. Add `--json` for a machine-readable report.

## “The model is not in the cache”

The toolkit never downloads a model implicitly. Download and select one:

```bash
dtk models
dtk models download gliner2.5-multi
dtk models use gliner2.5-multi
dtk models current
```

If the model is already on disk but not in the cache, copy the toolkit home from
the other machine or import a custom ONNX directory. See
[Models](MODELS.md#offline-workflow).

Check that you are using the expected home:

```bash
dtk doctor --json
```

A `--home DIR` flag must come **before** the subcommand:

```bash
dtk --home /data/dtk-home models list
```

## “Unknown variant” or “model not downloaded”

List the real variants and aliases for the model:

```bash
dtk models variants gliner2.5-multi
dtk models list --quant
```

Download the requested variant explicitly:

```bash
dtk models download gliner2.5-multi --variant fp32
```

A known variant can still be missing from the local cache; the `cached` column
in `dtk models list` shows the difference.

## `No module named onnxruntime` or `No module named tokenizers`

Install the inference runtime:

```bash
uv pip install "decision-toolkit[onnx]"
```

If you use a source checkout:

```bash
uv sync --all-extras
```

## The REST server says the server extra is missing

Install the server together with the runtime:

```bash
uv pip install "decision-toolkit[onnx,server]"
```

From a source checkout:

```bash
uv sync --all-extras
```

## Hash or session verification fails

```bash
dtk models verify gliner2.5-multi --variant fp16
dtk models verify --all --json
```

A verification failure exits with status `3`. If a catalog file is corrupted,
replace the variant:

```bash
dtk models download gliner2.5-multi --variant fp16 --force
```

For an offline copy, re-copy the cache entry and run verification again. The
catalog hashes are the source of truth; do not edit files inside a cached
catalog variant by hand.

## The server will not start because the port is in use

Choose another port:

```bash
dtk serve --model gliner2.5-multi --host 127.0.0.1 --port 8010
```

To find the process using port 8000 on Linux:

```bash
ss -ltnp | grep ':8000'
```

## REST returns 401

The server was started with `--api-key`, or `DECISION_TOOLKIT_API_KEY` is set,
and the request is missing the bearer token:

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $DECISION_TOOLKIT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gliner2.5-multi", "state": "...", "questions": {"urgent": {"type": "noul", "instructions": "Today?"}}}'
```

If the server is exposed beyond localhost, put authentication and TLS in front
of it. The built-in key is a local protection, not a full public deployment
story.

## REST returns 422

The request shape is wrong. Check the schema for the endpoint you are calling:
System One uses `{"detail": [...]}` validation errors; OpenAI Decisions uses
`{"error": {...}}`.

Common causes:

- `questions` is a list where the endpoint expects a mapping, or the reverse.
- A `choice` question is missing `criteria`.
- The model alias does not exist.
- A `variant` field was sent in the body. It is **not** part of either public
  schema; select the variant with `"model": "id@variant"`.

See [REST](REST.md) for complete request examples.

## REST returns 429, 503 or 504

These are operational limits:

- `429 rate_limit_exceeded`: wait for `Retry-After` or raise the configured
  rate limit.
- `503 server_busy`: the bounded concurrency limit is full; retry with
  backoff or increase `--max-concurrent` after measuring capacity.
- `504 timeout`: the request exceeded `--timeout`; shorten the input, reduce
  the batch size or raise the limit.

See [Operations](OPERATIONS.md#runbook) for the runbook and flags.

## The answer says it was truncated

The model hit its context window. The decision and each answer expose
`truncated` and `warnings`. In the library, use strict mode to fail instead of
returning a truncated answer:

```python
from decision_toolkit import DecisionModel

model = DecisionModel("julia-1", strict=True)
```

On the CLI, add `--strict` to `dtk decide`. Shorten the state or lower the
number of questions. See [Core API](CORE.md#truncation) for adapter limits.

## The model is slow or the process is killed

- Try a smaller or more quantized variant from `dtk models list`; verify its
  quality on your own samples.
- Use one model instance per process or worker instead of sharing across
  processes.
- For batches, tune the CLI batch size and avoid loading several large models
  at once.
- In the server, lower `--max-concurrent`, or run fewer workers.
- ONNX Runtime on CPU is the supported path; this project does not document a
  GPU acceleration path.

## A custom model fails to load

The toolkit does not guess a custom layout. Pass the adapter and every file
role explicitly:

```bash
dtk decide --model ./my-model --adapter julia \
  --files model=model.onnx \
  --files tokenizer=tokenizer.json \
  --state "..." --q 'urgent=noul:Today?'
```

For a cached custom model, use `models import` with the same `--adapter` and
`--files`. If the layout is not supported, use `DecisionModel.from_onnx(...)` for
the library or register an adapter; custom models are outside catalog support.

## A new release is not visible to uv

uv caches index metadata. Refresh it:

```bash
uv lock --refresh
uv pip install --refresh "decision-toolkit[onnx]"
```

Then check:

```bash
dtk --version
```

## A help example cannot find a model

Help examples and documentation examples do not download models. Download the
referenced model first:

```bash
dtk models download gliner2.5-multi
```

If the example uses a variant, download that variant as well. `dtk models
variants <id>` prints both the download and selection commands.

## Deleting a model fails because it is in use

Select another model first:

```bash
dtk models use laya
dtk models delete gliner2.5-multi --variant fp16
```

To remove every cached variant of a model, use `--all-variants --yes`; the
current variant is kept unless you switch away from it first.

## Getting more detail

```bash
dtk --verbose <command> ...
dtk --log-format json <command> ...
```

Expected failures print `dtk: error: ...` without a traceback. `--verbose`
adds tracebacks and debug logs to stderr.

If the problem is still unexplained, include the output of `dtk doctor --json`
and the exact command in the issue report (redact any state text you cannot
share).
