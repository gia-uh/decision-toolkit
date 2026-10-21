# Recipes

Short, copy-paste workflows built from the public surfaces. All of them assume
the model is already cached; nothing here downloads a model implicitly.

## Batch decisions from JSONL

Create `states.jsonl` with one state per line:

```json
{"state": "Customer wants a refund.", "id": "row-1"}
{"state": "Please change my seat to an aisle.", "id": "row-2"}
```

Create `questions.yaml`:

```yaml
questions:
  intent:
    type: choice
    instructions: What is the request?
    criteria:
      refund_request: asks for a refund
      seat_change: asks to change a seat
      other: anything else
  urgent:
    type: noul
    instructions: Does it need a reply today?
```

Run the batch once:

```bash
dtk decide --model gliner2.5-multi \
  --input-jsonl states.jsonl \
  --questions-file questions.yaml \
  --output-jsonl decisions.jsonl \
  --batch-size 32
```

The model is loaded once, and each output line includes `answers`, `usage`,
`latency_ms`, `warnings`, `model`, `variant` and `request_id`. A per-line
`questions` object overrides the global questions for that line. See
[CLI](CLI.md#batch-jsonl) for the full input contract.

## Select a variant per REST request

Start one server with the default model:

```bash
dtk serve --model gliner2.5-multi --host 127.0.0.1 --port 8000 --warmup
```

Send the base model id, or a catalogued alias:

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gliner2.5-multi@fp32",
    "state": "Meeting with Acme.",
    "questions": {
      "urgent": {"type": "noul", "instructions": "Does it need a reply today?"}
    }
  }'
```

Use only the standard `model` field. Do not send a `variant` field in the body.
Different aliases can be loaded concurrently; see
[REST](REST.md#variants-as-model-aliases).

## Run from a copied offline cache

On a connected machine, download and verify:

```bash
dtk models download gliner2.5-multi
dtk models verify gliner2.5-multi
```

Copy the toolkit home to the offline machine, then use it explicitly:

```bash
dtk --home /data/dtk-home models current
dtk --home /data/dtk-home models verify gliner2.5-multi
dtk --home /data/dtk-home decide --model gliner2.5-multi \
  --state "Meeting with Acme." --q 'urgent=noul:Today?'
```

See [Models](MODELS.md#offline-workflow) for the cache layout and import path.

## Use a custom ONNX model

A custom model needs an adapter and an explicit role map:

```bash
dtk models import ./my-model --name my-model \
  --adapter julia \
  --files model=model.onnx \
  --files tokenizer=tokenizer.json

dtk models use my-model
dtk decide --model my-model --state "..." --q 'urgent=noul:Today?'
```

The toolkit does not guess layouts or provide catalog hashes/support for custom
models. See [CLI](CLI.md#custom-models) and
[Core API](CORE.md#custom-local-onnx).

## Use a threshold in classification

```bash
dtk classify --model laya \
  --text "Dentist next Tuesday at 9" \
  --labels reminder,task,idea \
  --threshold 0.6 --top-k 3 --json
```

`threshold` only marks low-confidence results with `below_threshold`; it does
not turn the output into a hard rule. The library equivalent is
`DecisionModel.classify(..., threshold=...)`; see [Library guide](LIBRARY.md#classify-one-label).

## Serve an authenticated local endpoint

```bash
export DECISION_TOOLKIT_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
dtk serve --model gliner2.5-multi --host 127.0.0.1 --port 8000 --warmup
```

Clients must send:

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer $DECISION_TOOLKIT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gliner2.5-multi", "state": "...", "questions": {"urgent": {"type": "noul", "instructions": "Today?"}}}'
```

The built-in key is a local protection. Put TLS and an authenticated gateway in
front of the server before exposing it beyond localhost.
