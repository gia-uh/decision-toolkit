# CLI

`dtk` is our own operational surface. Unlike REST, it has **no external
standard** to comply with: it is consistent with itself, script-friendly, and a
thin adapter over the programmatic core. Inference logic never lives in the
CLI.

## Invocation

```text
dtk [global options] <command> [command options]
```

Global options must come **before** the command. This matters for `--home`:

```bash
dtk --home /data/dtk-home models list
```

| Global option | Meaning |
| --- | --- |
| `--home DIR` | Toolkit home (models cache + config). Default: `~/.local/share/decision-toolkit`. |
| `-q, --quiet` | Suppress informational output. |
| `-v, --verbose` | Debug logs and tracebacks on stderr. |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `--log-format FORMAT` | `text` or `json`. |
| `--no-color` | Disable colors (accepted for scripts). |
| `--version` | Print the toolkit version. |

Only `dtk models download` uses the network. The model must already be cached
for inference.

## Command map

| Command | Purpose |
| --- | --- |
| `dtk models` | Inspect the local cache and the model catalog. |
| `dtk models list` | Every catalog variant, default (`*`) and cached variants (`--license`, `--quant`, `--adapter`). |
| `dtk models variants <id>` | Every real variant of one model, with size, role and exact install command. |
| `dtk models download <id>` | Download the default or `--variant` quantization (`--force`, `--dry-run`). |
| `dtk models use <id>` | Set the persistent current model and variant (`--variant`). |
| `dtk models current` | Show the current model (text as `id@variant`; JSON separates them). |
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

## Command options

### Models

`dtk models` without a subcommand lists the cached entries. `dtk models list`
shows the catalog even when the cache is empty.

| Command | Important options |
| --- | --- |
| `dtk models list` | `--quant`, `--adapter`, `--license`, `--json` |
| `dtk models variants <id>` | `--json` |
| `dtk models download <id>` | `--variant NAME`, `--revision REV`, `--force`, `--dry-run`, `--json` |
| `dtk models use <id>` | `--variant NAME`, `--json` |
| `dtk models current` | `--json` |
| `dtk models delete <id>` | `--variant NAME`, `--all-variants`, `--yes`, `--json` |
| `dtk models verify [id]` | `--variant NAME`, `--all`, `--json` |
| `dtk models import <path>` | `--name NAME`, `--adapter NAME`, `--files ROLE=PATH`, `--json` |

`download` does not activate a variant; `use` persists the selection.
`--variant` on `decide`, `classify` or `serve` overrides it for one command.

### Decide

| Option | Meaning |
| --- | --- |
| `--model ID` | Catalog id or cached model path (default: current model). |
| `--variant NAME` | Variant/quantization for this command. |
| `-s, --state TEXT` | State as text. |
| `--state-file FILE` | Read the state from a file (`-` for stdin; JSON files are sent as JSON). |
| `--input-jsonl FILE` | Batch input, one state per line (`-` for stdin). |
| `--q NAME=SPEC` | Repeatable shorthand question. |
| `--questions-file FILE` | JSON/YAML question mapping. |
| `--output-jsonl FILE` | Batch output (`-` for stdout). |
| `--batch-size N` | JSONL chunk size (default 32). |
| `--strict` | Fail instead of returning truncated answers. |
| `--json` | Print JSON (the default output). |
| `--pretty` | Indent JSON output. |

### Classify

| Option | Meaning |
| --- | --- |
| `--model ID` | Catalog id or cached model path. |
| `--variant NAME` | Variant for this command. |
| `-t, --text TEXT` | Text to classify. |
| `--text-file FILE` | Read text from a file (`-` for stdin). |
| `--labels L1,L2,...` | Comma-separated labels. |
| `--labels-file FILE` | YAML `labels: {name: description}` mapping. |
| `--instructions TEXT` | Custom question text. |
| `--threshold FLOAT` | Flag results below it (default 0.5). |
| `--top-k N` | Number of ranked labels to show (default 3). |
| `--json` | Machine-readable output. |

### Serve

| Option | Meaning |
| --- | --- |
| `--model ID`, `--variant NAME` | Default model for the process. |
| `--host HOST`, `--port PORT` | Bind address and port (default `127.0.0.1:8000`). |
| `--workers N` | Uvicorn worker processes. |
| `--reload` | Development reload; cannot combine with `--workers > 1`. |
| `--api-key KEY` | Require `Authorization: Bearer <key>`. |
| `--cors-origin ORIGIN` | Allow a browser origin; repeatable. |
| `--max-body-mb N`, `--timeout SECONDS` | Request limits. |
| `--max-models N` | Idle models kept loaded (LRU). |
| `--max-concurrent N` | Bounded inference concurrency. |
| `--rate-limit N`, `--rate-window SECONDS`, `--rate-burst N` | Per-key rate limit. |
| `--no-metrics` | Disable `/metrics`. |
| `--tracing` | Enable optional OpenTelemetry spans. |
| `--warmup` | Warm the default model at startup. |
| `--log-level`, `--log-format` | Server logging overrides. |

See [Operations](OPERATIONS.md) for the semantics of these limits.

### Config, doctor and completion

```bash
dtk config list --json
dtk config get device
dtk config set device cpu
dtk config set cache_dir /data/dtk-models
dtk doctor
dtk doctor --json
dtk completion bash
```

Writable config keys are `cache_dir`, `current`, `current_variant` and `device`.
`home` is read-only; use `--home`.

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

## Variants and quantization

A quantization is a **variant**. `dtk models list` always shows every catalogued
variant and marks the default with `*`; the `cached` column shows which variants
are already downloaded.

| Model | Variants (default *) | Quantization | Size |
| --- | --- | --- | ---: |
| `gliner2.5-multi` | `fp16`*, `fp32` | fp16 / fp32 | 577.1 / 1131.9 MB |
| `gliner2.5-decide` | `q4f16`* | q4/fp16 | 523.5 MB |
| `laya` | `q8`* | int8 | 527.9 MB |
| `julia-1` | `fp32`* | fp32 | 618.2 MB |
| `opendecider-nano` | `q8f16`*, `fp16` | int8/fp16 / fp16 | 468.8 / 791.9 MB |
| `laya-multilingual` | `fp16`*, `fp32` | fp16 / fp32 | 681.8 / 1325.6 MB |
| `open-jev-deberta-v3-large` | `fp16`*, `fp32`, `q4`, `q4f16` | fp16 / fp32 / q4 / q4-fp16 | 883.5 / 1757.9 / 486.7 / 357.0 MB |
| `strands-decider-2b` | `q4f16`*, `q8` | q4-fp16 / int8 | 1113.6 / 1821.4 MB |

Each row is also addressable over REST as `model = "<id>@<variant>"`; the alias
is printed by `dtk models list` and by `dtk models variants <id>`.

```bash
# inspect, including the exact download command for each variant
dtk models list --quant
dtk models variants gliner2.5-multi
dtk models variants opendecider-nano --json

# download a different quantization into the cache (does not activate it)
dtk models download gliner2.5-multi --variant fp32
dtk models download gliner2.5-multi --variant fp32 --dry-run
dtk models download gliner2.5-multi --variant fp32 --force

# select it persistently, or only for one command
dtk models use gliner2.5-multi --variant fp32
dtk models current
# gliner2.5-multi@fp32
dtk models current --json
# {"current": "gliner2.5-multi", "variant": "fp32"}
dtk decide --model gliner2.5-multi --variant fp16 --state "..." --q 'cat=choice:a|b'

# verify or remove a specific variant
dtk models verify gliner2.5-multi --variant fp32
dtk models delete gliner2.5-multi --variant fp32
dtk models delete laya --all-variants --yes
dtk models verify --all --json
```

Downloading is the installation step for a model variant: there is no separate
`pip install` per quantization. Variants coexist in the cache. `download` does
not change the selection; `use` persists it, and `--variant` on a decision,
classification or serve command overrides it for that command. If `--variant`
is omitted, the catalog default is used unless `dtk models use` selected another
one.

`models list` text output has **one row per variant**, with `id`, `variant`,
`quant`, `role`, `size` and `cached`, followed by the exact download/select
commands. `--license` and `--adapter` add catalog columns; `--quant` is accepted
for compatibility because quantization is always shown. `--json` returns the
nested model tree: one object per model, each with its `variants` and
`cached_variants`.

See [Variants and quantization](VARIANTS.md) for the full table and cache
layout.

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
  `--cors-origin`, `--log-level` and `--log-format` are server controls.
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
dtk models variants gliner2.5-multi
dtk models download gliner2.5-multi --variant fp32
dtk models use gliner2.5-multi --variant fp32
dtk decide --model gliner2.5-multi --variant fp32 \
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
- Metrics and rate limiting are server-side features; see
  [OPERATIONS.md](OPERATIONS.md). Wheel/PyPI releases are documented in
  [RELEASE.md](RELEASE.md).
