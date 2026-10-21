# Variants and quantization

A model can ship one or more **variants**. A variant is one real ONNX build:
its quantization, exact files, download size and role. The variant name is the
value passed to `--variant`; it is not always the raw quantization label (for
example, Laya's `q8` variant is `int8`).

Installing a quantization is **not** a separate pip extra. `uv pip install`
installs the runtime; `dtk models download` installs a model variant into the
local model cache. Variants of the same model coexist, so you can compare them
without deleting the default.

Only catalogued and verified builds are selectable. Community conversions that
have not passed the catalog gates are not exposed as variants.

## Available variants

A `*` marks the default variant. It is used when `--variant` is omitted and no
other variant was selected with `dtk models use`.

| Model | Variant | Quantization | Size | Role |
| --- | --- | --- | ---: | --- |
| `gliner2.5-multi` | `fp16` * | fp16 encoder + fp32 classifier | 577.1 MB | default |
| `gliner2.5-multi` | `fp32` | fp32 encoder + fp32 classifier | 1131.9 MB | quality |
| `gliner2.5-decide` | `q4f16` * | q4/fp16 | 523.5 MB | default |
| `laya` | `q8` * | int8 | 527.9 MB | default |
| `julia-1` | `fp32` * | fp32 | 618.2 MB | default |
| `opendecider-nano` | `q8f16` * | int8/fp16 | 468.8 MB | default |
| `opendecider-nano` | `fp16` | fp16 | 791.9 MB | quality |

Models with a single row have only one catalogued, verified build. That does
not mean other conversions do not exist upstream; it means they have not passed
the parity, hash and license gates for this catalog.

## One selection model across surfaces

The same catalog variant is used by the library, CLI and REST API. Selection
precedence is:

1. an explicit per-call/per-request variant (`DecisionModel(..., variant=...)`,
   `--variant`, or `"variant"` in a REST body);
2. the CLI/server configuration (`dtk models use`, or `dtk serve --variant`);
3. the catalog default marked with `*`.

| Surface | Explicit selection | Default when omitted |
| --- | --- | --- |
| Library | `DecisionModel("gliner2.5-multi", variant="fp32")` | persisted variant for the current model, else catalog default |
| CLI | `--variant fp32` | variant selected by `dtk models use`, else catalog default |
| REST (local extension) | `"variant": "fp32"` in the request body | server `--variant`, else persisted variant for the current model, else catalog default |

`DecisionModel.variant` and the CLI report the selected variant (`id@variant` in
`dtk models current`). REST responses expose it under the namespaced
`x_decision_toolkit.variant` extension; `/v1/decision-models` includes
`current` and `current_variant`, plus all `variants` and `cached_variants`.
`variant` is **not** part of TypeSafe or OpenAI; standard clients should omit it
and rely on the server's `--variant`/persisted default. A server started with
`--model`/`--variant` reports that default in `/v1/info` and `/ready`.

The variant must have been downloaded before a library call or REST request
loads it. An unknown variant returns a model-not-found error; a known but
missing cache entry returns a model-not-downloaded error.

## Inspect the catalog

```bash
# every model, every variant, default (*) and what is already cached
dtk models list

# add the quantization label and other catalog columns
dtk models list --quant --adapter --license

# full details for one model, including the exact install command
dtk models variants gliner2.5-multi

# machine-readable form
dtk models variants opendecider-nano --json
```

Example:

```text
id                 variants        cached
gliner2.5-multi    fp16*, fp32     fp16
gliner2.5-decide   q4f16*          q4f16
opendecider-nano   q8f16*, fp16    q8f16

* default variant. Use `dtk models variants <id>` for details and install commands.
```

## Download a different quantization

`dtk models download` performs the network download, checks the pinned revision
and SHA256, and stores the variant in the cache:

```bash
# the default variant (fp16 here)
dtk models download gliner2.5-multi

# a different variant; downloaded alongside the default
dtk models download gliner2.5-multi --variant fp32

# inspect before downloading
dtk models download gliner2.5-multi --variant fp32 --dry-run

# replace an existing download of the same variant
dtk models download gliner2.5-multi --variant fp32 --force
```

Examples for the other multi-variant model:

```bash
dtk models download opendecider-nano --variant q8f16   # default
dtk models download opendecider-nano --variant fp16    # quality
```

Downloading does not change which variant is active. Use `dtk models list` to
see the `cached` column, or `dtk models verify <id> --variant <name>` to check
the files and ONNX sessions.

## Select a downloaded variant

Persist the selection:

```bash
dtk models use gliner2.5-multi --variant fp32
dtk models current
# gliner2.5-multi@fp32
dtk models current --json
# {"current": "gliner2.5-multi", "variant": "fp32"}
```

The selection is used by later commands that name the model without a variant:

```bash
dtk decide --model gliner2.5-multi --state "..." --q 'cat=choice:meeting|tasks'
```

Override it for a single command:

```bash
dtk decide --model gliner2.5-multi --variant fp16 --state "..." --q 'cat=choice:a|b'
dtk classify --model opendecider-nano --variant fp16 --text "..." --labels billing,other
dtk serve --model gliner2.5-multi --variant fp32 --warmup
```

From the library:

```python
from decision_toolkit import DecisionModel

model = DecisionModel("gliner2.5-multi", variant="fp32")
```

From REST, send the variant in the request body:

```json
{"model": "gliner2.5-multi", "variant": "fp32", "state": "...",
 "questions": {"cat": {"type": "choice", "instructions": "What kind?",
                       "criteria": {"meeting": null, "other": null}}}}
```

The variant must already be downloaded. If you name an unknown variant, the
error lists the catalogued choices, for example:

```text
dtk: error: 'gliner2.5-multi' has no variant 'q4'; available: fp16 (default), fp32
```

## Cache and removal

Variants are stored separately:

```text
~/.local/share/decision-toolkit/models/gliner2.5-multi@fp16/
~/.local/share/decision-toolkit/models/gliner2.5-multi@fp32/
```

Remove one variant or all cached variants:

```bash
dtk models delete gliner2.5-multi --variant fp32
dtk models delete gliner2.5-multi --all-variants --yes
```

The variant currently in use cannot be deleted.

## Why some models have only one variant

The catalog is curated, not a mirror. A variant is admitted only with a pinned
revision, a real ONNX export, SHA256 per file, a clean license and measured
parity/latency. An upstream repository may contain fp32, fp16, int8 or q4
builds that are not in the toolkit because they have not been validated.

For the per-model evidence and hashes, see the [model cards](models/README.md).
For the exact CLI surface, see [CLI](CLI.md).
