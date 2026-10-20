# Examples

Runnable examples for the three surfaces. Models must already be cached:

```bash
dtk models download gliner2.5-multi
# or run from the source tree:
PYTHONPATH=src python -m decision_toolkit models download gliner2.5-multi
```

## Library

```python
from decision_toolkit import DecisionModel, choice, noul, score

with DecisionModel("gliner2.5-multi") as model:
    model.warmup()
    decision = model.decide(
        "Meeting with Acme: we agreed a 10% discount and will sign next week.",
        {
            "cat": choice("What kind of record is this?", {
                "meeting": "a meeting or call with topics and actions",
                "tasks": "things to do, with dates or owners",
                "contact": "a person's details",
                "idea": "an idea or question",
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
    decision.usage.to_dict()        # input/output tokens + estimated flag
    decision.latency_ms
```

### Batch and async

```python
with DecisionModel("gliner2.5-multi") as model:
    decisions = model.decide_many(
        ["Call Maria tomorrow.", "Ana is the marketing lead.", "What if we launch in September?"],
        {"cat": choice("What kind of record is this?", {
            "meeting": "a meeting or call",
            "tasks": "things to do",
            "contact": "a person's details",
            "idea": "an idea or question",
            "other": "anything else",
        })},
    )

# async wrapper (inside an async function)
decision = await model.adecide("Meeting with Acme.", {"urgent": noul("Today?")})
```

### One-label classification

```python
with DecisionModel("laya") as model:
    result = model.classify(
        "Dentist next Tuesday at 9",
        {"reminder": "an event with a date", "task": "a thing to do", "idea": "an idea"},
        threshold=0.6,
    )
    result.label, result.confidence, result.below_threshold
```

### Other model contracts

```python
DecisionModel("gliner2.5-decide").decide(
    "I want my money back.",
    {"intent": choice("What is the request?", {
        "refund_request": "asks for a refund",
        "seat_change": "asks to change a seat",
    })},
)

DecisionModel("julia-1").decide("I was charged twice.", {
    "team": choice("Which team?", {"billing": "refunds", "technical": "bugs"}),
    "urgent": noul("Today?"),
})

DecisionModel("opendecider-nano").decide("I was charged twice.", {
    "team": choice("Which team?", {"billing": "refunds", "technical": "bugs"}),
})
```

## REST

```bash
dtk serve --model gliner2.5-multi --warmup --port 8000
```

### System One

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Meeting with Acme: 10% discount, sign next week.",
    "questions": {
      "cat": {
        "type": "choice",
        "instructions": "What kind of record is this?",
        "criteria": {"meeting": "a meeting", "tasks": "things to do", "other": null}
      },
      "urgent": {"type": "noul", "instructions": "Does it need a reply today?"}
    }
  }' | jq
```

### OpenAI Decisions

```bash
curl -s http://127.0.0.1:8000/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Please change my seat to an aisle.",
    "questions": [{
      "type": "choice",
      "name": "intent",
      "instructions": "What is the request?",
      "choices": [
        {"value": "refund_request", "description": "asks for a refund"},
        {"value": "seat_change", "description": "asks to change a seat"}
      ]
    }]
  }' | jq
```

### Model per request and auth

```bash
dtk serve --model gliner2.5-multi --api-key secret --warmup

curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Authorization: Bearer secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gliner2.5-decide",
    "variant": "q4f16",
    "state": "I want my money back.",
    "questions": {
      "intent": {"type": "choice", "instructions": "What is the request?",
                 "criteria": {"refund_request": null, "seat_change": null}}
    }
  }' | jq
```

### Models, readiness and metrics

```bash
curl -s http://127.0.0.1:8000/v1/models | jq
curl -s http://127.0.0.1:8000/ready | jq
curl -s http://127.0.0.1:8000/metrics | head
```

## CLI

```bash
# catalog and cache
dtk models list --adapter --quant
dtk models list --json
dtk models current --json

# one decision
dtk decide --model gliner2.5-multi \
  --state "Meeting with Acme: 10% discount, sign next week." \
  --q 'cat=choice:meeting|tasks|contact|idea|other' \
  --q 'urgent=noul:Does it need a reply today?' --json

# questions from JSON/YAML
dtk decide --model gliner2.5-multi \
  --state-file state.txt --questions-file questions.yaml --json

# one-label classification
dtk classify --model laya \
  --text "Dentist next Tuesday at 9" \
  --labels reminder,task,idea --json

# batch JSONL
dtk decide --model gliner2.5-decide \
  --input-jsonl states.jsonl \
  --q 'intent=choice:refund_request|seat_change' \
  --output-jsonl decisions.jsonl

# variants and pinned evidence
dtk models variants gliner2.5-multi --json
dtk models verify --all --json
```

Custom model:

```bash
dtk decide \
  --model ./my-model \
  --adapter julia \
  --files model=model.onnx \
  --files tokenizer=tokenizer.json \
  --state "I was charged twice." \
  --q 'team=choice:billing|technical'
```

## Exit codes

`0` success · `1` runtime/model failure · `2` usage/invalid input ·
`3` verification failure · `130` interrupted.

Every result command accepts `--json` (except `decide`, which always emits JSON)
and `dtk --help`, `dtk decide --help`, etc. include runnable examples.
