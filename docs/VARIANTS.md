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

| Model | Variant | REST alias | Quantization | Size | Role | Download |
| --- | --- | --- | --- | ---: | --- | --- |
| `gliner2.5-multi` | `fp16` * | `gliner2.5-multi@fp16` | fp16 | 577.1 MB | default | `dtk models download gliner2.5-multi --variant fp16` |
| `gliner2.5-multi` | `fp32` | `gliner2.5-multi@fp32` | fp32 | 1131.9 MB | quality | `dtk models download gliner2.5-multi --variant fp32` |
| `gliner2.5-decide` | `q4f16` * | `gliner2.5-decide@q4f16` | q4/fp16 | 523.5 MB | default | `dtk models download gliner2.5-decide --variant q4f16` |
| `laya` | `q8` * | `laya@q8` | int8 | 527.9 MB | default | `dtk models download laya --variant q8` |
| `julia-1` | `fp32` * | `julia-1@fp32` | fp32 | 618.2 MB | default | `dtk models download julia-1 --variant fp32` |
| `opendecider-nano` | `q8f16` * | `opendecider-nano@q8f16` | int8/fp16 | 468.8 MB | default | `dtk models download opendecider-nano --variant q8f16` |
| `opendecider-nano` | `fp16` | `opendecider-nano@fp16` | fp16 | 791.9 MB | quality | `dtk models download opendecider-nano --variant fp16` |
| `laya-multilingual` | `fp16` * | `laya-multilingual@fp16` | fp16 | 681.8 MB | default | `dtk models download laya-multilingual --variant fp16` |
| `laya-multilingual` | `fp32` | `laya-multilingual@fp32` | fp32 | 1325.6 MB | quality | `dtk models download laya-multilingual --variant fp32` |
| `open-jev-deberta-v3-large` | `q4f16` | `open-jev-deberta-v3-large@q4f16` | q4/fp16 | 357.0 MB | edge | `dtk models download open-jev-deberta-v3-large --variant q4f16` |
| `open-jev-deberta-v3-large` | `fp16` * | `open-jev-deberta-v3-large@fp16` | fp16 | 883.5 MB | default | `dtk models download open-jev-deberta-v3-large --variant fp16` |
| `open-jev-deberta-v3-large` | `q4` | `open-jev-deberta-v3-large@q4` | q4 | 486.7 MB | edge | `dtk models download open-jev-deberta-v3-large --variant q4` |
| `open-jev-deberta-v3-large` | `fp32` | `open-jev-deberta-v3-large@fp32` | fp32 | 1757.9 MB | quality | `dtk models download open-jev-deberta-v3-large --variant fp32` |
| `strands-decider-2b` | `q4f16` * | `strands-decider-2b@q4f16` | q4/fp16 | 1113.6 MB | default | `dtk models download strands-decider-2b --variant q4f16` |
| `strands-decider-2b` | `q8` | `strands-decider-2b@q8` | int8 | 1821.4 MB | quality | `dtk models download strands-decider-2b --variant q8` |

Models with a single row have only one catalogued, verified build. That does
not mean other conversions do not exist upstream; it means they have not passed
the parity, hash and license gates for this catalog.

## One selection model across surfaces

The same catalog variant is used by the library, CLI and REST API. Selection
precedence is:

1. an explicit per-call variant in the library or CLI
   (`DecisionModel(..., variant=...)`, `--variant`);
2. the CLI/server configuration (`dtk models use`, or `dtk serve --variant`);
3. the catalog default marked with `*`.

| Surface | Explicit selection | Default when omitted |
| --- | --- | --- |
| Library | `DecisionModel("gliner2.5-multi", variant="fp32")` | persisted variant for the current model, else catalog default |
| CLI | `--variant fp32` | variant selected by `dtk models use`, else catalog default |
| REST | `"model": "gliner2.5-multi@fp32"` (model alias) | base model id; server `--variant`/persisted default |

`DecisionModel.variant` and the CLI report the selected variant (`id@variant` in
`dtk models current`). REST responses expose it under the namespaced
`x_decision_toolkit.variant` extension; `/v1/decision-models` includes
`current` and `current_variant`, plus all `variants` and `cached_variants`.
`variant` is **not** part of TypeSafe or OpenAI, and standard REST clients must
not send it. They select only `model`; the local server chooses the quantization
from `--variant` or the persisted default. A server started with
`--model`/`--variant` reports that default in `/v1/info` and `/ready`.

The variant must have been downloaded before a library call or REST alias
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
id                variant  quant    role     size       cached
gliner2.5-multi   fp16*    fp16     default  577.1 MB   ✓
gliner2.5-multi   fp32     fp32     quality  1131.9 MB  ✓
gliner2.5-decide  q4f16*   q4/fp16  default  523.5 MB   ✓
laya              q8*      int8     default  527.9 MB   ✓

* default variant.
download: dtk models download <id> --variant <name>
select:   dtk models use <id> --variant <name>
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

For REST, select a quantization through the standard `model` field using the
alias `id@variant`:

```json
{"model": "gliner2.5-multi@fp32", "state": "...", "questions": {}}
```

The base id (`gliner2.5-multi`) uses the server `--variant`/persisted default.
`/v1/models` lists base models and every alias, and the registry can hold two
aliases of the same base model in separate slots. Do not add a `variant` field
to the request body.

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
