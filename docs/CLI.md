# CLI (phase 3)

`dtk` is our own operational surface. Unlike REST, it has **no external
standard** to comply with: it is consistent with itself, script-friendly, and a
thin adapter over the programmatic core. Inference logic never lives in the
CLI.

## Command map

| Command | Purpose |
| --- | --- |
| `dtk models` | Inspect the local cache and the curated catalog. |
| `dtk models list` | Catalog, default variants and cache state (`--license`, `--quant`, `--adapter`). |
| `dtk models variants <id>` | Every real variant of a catalog model. |
| `dtk models download <id>` | Download the default or `--variant` build (`--force`, `--dry-run`). |
| `dtk models use <id>` | Set the persistent current model (`--variant`). |
| `dtk models current` | Show the current model (JSON includes the variant). |
| `dtk models delete <id>` | Delete a cached variant, or `--all-variants --yes`. |
| `dtk models verify` | Verify files/hashes/sessions for one model or `--all`. |
| `dtk models import` | Import a custom ONNX directory with `--adapter` and `--files`. |
| `dtk decide` | One decision or a JSONL batch. |
| `dtk classify` | One-label classification with a threshold. |
| `dtk config` | `list`, `get`, `set` the toolkit configuration. |
| `dtk serve` | REST server (System One + OpenAI Decisions). |
| `dtk doctor` | Environment and cache diagnostics. |
| `dtk version` | Version information. |
| `dtk completion` | Bash/zsh/fish completion snippet. |

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Runtime/model failure (not found, not downloaded, backend). |
| `2` | Usage or invalid input. |
| `3` | Verification failed. |
| `130` | Interrupted (`Ctrl-C`). |

Expected failures print `dtk: error: ...` to stderr, never a traceback.
`--verbose` adds logs and tracebacks for debugging.

## JSON and logging

Every inspection/result command accepts `--json`:

```bash
dtk models list --json
dtk models variants laya --json
dtk models current --json
dtk models verify --all --json
dtk decide ... --json
dtk classify ... --json
dtk config list --json
dtk doctor --json
dtk version --json
```

When `--json` is used, informational messages go to stderr and stdout contains
only the JSON document. `dtk decide` always emits JSON; `--pretty` indents it.

Global flags:

- `-q/--quiet`: suppress informational messages; results still print.
- `-v/--verbose`: debug logs and tracebacks on stderr.
- `--log-level DEBUG|INFO|WARNING|ERROR`
- `--log-format text|json`
- `--no-color` (accepted for scripts).

## Questions

The shorthand remains:

```bash
dtk decide --model gliner2.5-multi --state "..." \
  --q 'team=choice:billing=refunds|tech=bugs' \
  --q 'urgent=noul:Does it need a reply today?' \
  --q 'tone=score:calm|annoyed|furious' --json
```

For the full contract use a JSON/YAML file:

```yaml
questions:
  team:
    type: choice
    instructions: Which team?
    criteria:
      billing: refunds
      tech: null
    task: route
    multi_label: false
  urgent:
    type: noul
    instructions: Does it need a reply today?
  region:
    type: box
    instructions: Where is the object?
    criteria: [object]
```

```bash
dtk decide --model gliner2.5-multi --state "..." \
  --questions-file questions.yaml --json
```

`task`, `multi_label`, `box` and arbitrary adapter extensions are preserved
until the core validates them against the selected model's capabilities.

## Batch JSONL

One state per line. A line can be:

- a JSON string: `"Customer wants a refund."`
- a JSON object used as the state: `{"ticket": 42, "plan": "pro"}`
- an envelope with per-line questions and id:

```json
{"state": "Customer wants a refund.", "id": "row-1"}
{"state": "Please change my seat.", "questions": {"urgent": {"type": "noul", "instructions": "Today?"}}, "id": "row-2"}
```

```bash
dtk decide --model gliner2.5-multi \
  --input-jsonl states.jsonl \
  --questions-file questions.yaml \
  --output-jsonl decisions.jsonl \
  --batch-size 32
```

- The model is loaded **once** for the whole batch.
- `questions` on a line overrides the global questions for that line.
- Output is one JSON object per line with `answers`, `usage`, `latency_ms`,
  `warnings`, `model`, `variant` and `request_id`.
- A per-line `id`/`name` is copied to the output.
- Omit `--output-jsonl` to stream to stdout.

## Variants

```bash
dtk models use laya --variant q8
dtk models current --json
# {"current": "laya", "variant": "q8"}

dtk models download laya --variant q8 --dry-run --json
dtk models download laya --variant q8 --force

dtk models delete laya --all-variants --yes
dtk models verify --all --json
```

`models list --json` always includes license, quant, adapter, capabilities,
variants and cached variants; the text table adds those columns only on
request (`--license`, `--quant`, `--adapter`).

## Custom models

```bash
dtk decide \
  --model ./my-model \
  --adapter julia \
  --files model=model.onnx \
  --files tokenizer=tokenizer.json \
  --state "..." --q 'urgent=noul:Today?'
```

The same `--adapter`/`--files` pair works on `classify` and `serve`.
`models import` stores the role map and adapter in the cache metadata:

```bash
dtk models import ./my-model --name my-model --adapter julia \
  --files model=model.onnx --files tokenizer=tokenizer.json --json
```

## Config

```bash
dtk config list --json
dtk config get device
dtk config set device cpu
dtk config set current julia-1
dtk config set current_variant fp32
dtk config set cache_dir /data/dtk-models
```

Writable keys: `cache_dir`, `current`, `current_variant`, `device`. `home` is
read-only (use `--home`). `config set current` validates through the model
store and resets a variant that belonged to the old model; set
`current_variant` afterwards when needed.

## Serve

```bash
dtk serve --model gliner2.5-multi --warmup \
  --api-key "$DECISION_TOOLKIT_API_KEY" \
  --cors-origin http://localhost:3000 \
  --workers 2 --log-format json
```

- `--workers N` and `--reload` use an internal uvicorn factory; `--reload`
  is for development and cannot be combined with `--workers > 1`.
- `--max-models`, `--max-body-mb`, `--timeout`, `--warmup`, `--api-key`,
  `--cors-origin`, `--log-level` and `--log-format` are the server's phase 2
  controls.
- Custom `--adapter`/`--files` also apply to the server's default model.

## Completion

```bash
# bash
eval "$(dtk completion bash)"
# zsh
dtk completion zsh > ~/.zfunc/_dtk
# fish
dtk completion fish | source
```

## Examples

```bash
dtk models list --adapter --quant
dtk models download gliner2.5-multi
dtk decide --model gliner2.5-multi \
  --state "Acme agreed a 10% discount and will sign next week." \
  --q 'cat=choice:meeting|task|contact|idea|other' --json
dtk classify --model laya --text "Dentist next Tuesday at 9" \
  --labels reminder,task,idea --json
dtk models verify --all --json
dtk doctor --json
```

## Boundaries

- The CLI can use more of the core than REST (batch JSONL, custom file maps,
  config), but still does not reimplement inference.
- It does not have a compatibility obligation to System One/OpenAI; those are
  REST contracts.
- Metrics/rate limiting are server-side phase 5 features; see
  [OPERATIONS.md](OPERATIONS.md). Wheel/PyPI and distribution tagging are
  deferred to phase 6.
