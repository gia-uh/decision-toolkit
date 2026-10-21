# Models and the model cache

`decision-toolkit` uses its **own model cache**, independent from the
HuggingFace cache. The cache holds pinned ONNX builds, their tokenizer and
configuration files, and a small metadata file per variant.

The toolkit never downloads a model implicitly. `dtk models download` is the
only command that uses the network.

## The cache at a glance

```text
~/.local/share/decision-toolkit/
├── config.json
└── models/
    └── gliner2.5-multi@fp16/
        ├── model.json          # source, revision, file map, hashes
        └── ...                 # ONNX files and tokenizer/config
```

A cache entry is addressed as `<id>@<variant>`. For example, the same model can
have `gliner2.5-multi@fp16` and `gliner2.5-multi@fp32` downloaded side by side.

`config.json` stores the cache location, the current model and variant, and the
device setting. See [Installation](INSTALLATION.md#where-things-live) for the
full layout and how to move it.

## Commands

| Command | What it does |
| --- | --- |
| `dtk models` | List the models currently in the cache. |
| `dtk models list` | List every catalogued variant, its default, size and cache state. |
| `dtk models list --adapter --license` | Add adapter and license columns (`--quant` is accepted for compatibility). |
| `dtk models variants <id>` | Show every variant of one model, with size, role, alias and exact commands. |
| `dtk models download <id>` | Download the default or `--variant` build (network). |
| `dtk models use <id>` | Set the persistent current model and variant. |
| `dtk models current` | Show the current model as `id@variant`. |
| `dtk models verify <id>` | Verify catalog files, SHA256 hashes and ONNX sessions. |
| `dtk models delete <id>` | Delete a cached variant; refuses a variant that is in use. |
| `dtk models import <path>` | Import a local custom ONNX directory with an adapter and file map. |

Examples:

```bash
dtk models
dtk models list --quant
dtk models variants gliner2.5-multi
dtk models current --json
dtk models download gliner2.5-multi --variant fp32
dtk models use gliner2.5-multi --variant fp32
dtk models verify gliner2.5-multi --variant fp32
```

Downloading a variant does **not** change the selection. `dtk models use`
persists it; `--variant` on `dtk decide`, `dtk classify` or `dtk serve`
overrides it for one command.

## Current model and default variant

The catalog marks one variant per model with `*` as its default. The precedence
is:

1. an explicit `variant=` / `--variant` on a call;
2. the selection stored by `dtk models use`;
3. the catalog default.

Use `dtk models current` to see the persisted selection:

```bash
dtk models current
# gliner2.5-multi@fp32

dtk models current --json
# {"current": "gliner2.5-multi", "variant": "fp32"}
```

If no model has been selected yet, an explicit model id is required by the
CLI. The library can also be given `DecisionModel("...", variant="...")`.

## Choosing a model

There is no single best model for every task. Start with the smallest model
that passes your own evaluation, then move up only if the quality gain justifies
the latency.

| Model | Best first use | Notes |
| --- | --- | --- |
| `gliner2.5-multi` | General choice, yes/no and score decisions | Balanced CPU default; start here for routing and extraction-style decisions. |
| `gliner2.5-decide` | English support/refund intent decisions | Fused export; check its card for the measured example and language limit. |
| `laya` | Fast English one-label classification | Small and quick; good for thresholded classification. |
| `laya-multilingual` | Multilingual decisions and classification | Choose this when the state is not English. |
| `julia-1` | Decisions where upstream parity matters | Verified against the reference implementation. |
| `opendecider-nano` | Lightweight routing and classification | Small footprint; two variants. |
| `open-jev-deberta-v3-large` | Quality-first DeBERTa decisions | Four quantizations, from edge to fp32 quality. |
| `strands-decider-2b` | Larger 2B decision model | Expect more memory and latency than the small models. |

The [model cards](models/README.md) contain the measured evidence, licenses,
context notes and per-variant loading snippets. Do not choose a model from its
name alone; measure it on a small labelled sample from your domain.

## Choosing a variant

A **variant** is one real, pinned ONNX build of a model. A **quantization** is
the numeric format used by that build. They are related, but the variant name
is what you pass to `--variant`; it is not always the raw quantization label.

Rules of thumb:

- Start with the catalog default (`*`), usually a balanced `fp16` or `q4f16`.
- Try `fp32` when quality matters more than size and latency.
- Try `q4`, `q4f16` or `q8` when memory or disk is tight; verify quality on
  your data before switching.
- Do not assume a quantization that is smaller is better for your task, and do
  not compare probabilities from different models as if they were calibrated
  together.

The complete table, sizes, roles and download commands are in
[Variants and quantization](VARIANTS.md). `dtk models list` prints the same
information from the installed version.

## Variants and REST aliases

Every catalogued variant has a REST alias:

```text
gliner2.5-multi@fp16
gliner2.5-multi@fp32
laya-multilingual@fp16
```

Use the alias in the standard `model` field:

```json
{"model": "gliner2.5-multi@fp32", "state": "...", "questions": {"...": "..."}}
```

There is **no `variant` field** in System One or OpenAI request bodies. The base
id uses the server's `--variant` or persisted default. A server can load two
aliases of the same base model concurrently; see [REST](REST.md#variants-as-model-aliases).

## Offline workflow

Only `dtk models download` needs the network. After a model is cached, the
library, CLI and server work offline.

1. On a connected machine:
   ```bash
   dtk models download gliner2.5-multi
   dtk models verify gliner2.5-multi
   ```
2. Copy the toolkit home (or at least `models/` and the relevant
   `config.json`) to the offline machine.
3. On the offline machine:
   ```bash
   dtk --home /data/dtk-home models current
   dtk --home /data/dtk-home models verify gliner2.5-multi
   ```

You can also import a pre-existing ONNX directory into a cache. Catalog models
are pinned by the toolkit; a custom import is your responsibility.

## Import a local model

`dtk models import` copies a custom ONNX directory into the cache and records
its adapter and role map:

```bash
dtk models import ./my-model --name my-model \
  --adapter julia \
  --files model=model.onnx \
  --files tokenizer=tokenizer.json
```

Then use it like a catalog entry:

```bash
dtk models use my-model
```

Custom models do not appear in `dtk models list` as catalog variants and do not
receive catalog hashes or support. The adapter must match the model's actual
layout; the toolkit never guesses. The same custom model can be passed directly
with `DecisionModel.from_onnx(...)` for library use.

## Verify and remove

Verification checks declared files, SHA256 hashes and, when ONNX Runtime is
installed, that the ONNX session can be created:

```bash
dtk models verify gliner2.5-multi --variant fp16
dtk models verify --all --json
```

The command exits with status `3` if verification fails. Re-download a
corrupted catalog variant with:

```bash
dtk models download gliner2.5-multi --variant fp16 --force
```

To remove a cached variant:

```bash
dtk models delete gliner2.5-multi --variant fp16
dtk models delete laya --all-variants --yes
```

You cannot delete the variant that is currently in use; select another model
first with `dtk models use`.

## Next steps

- [Variants and quantization](VARIANTS.md) — complete table and selection model.
- [Model cards](models/README.md) — evidence and limitations per model.
- [Library guide](LIBRARY.md) — load a model and run a decision.
- [Troubleshooting](TROUBLESHOOTING.md) — cache, hash and runtime errors.
