"""Command-line interface: ``dtk models|decide|classify|serve|config|doctor``.

The CLI is our own operational surface: it is a thin adapter over the public
library API and deliberately has no external wire standard to satisfy.  It is
consistent with itself: the same ``--json`` flag on every command, the same
exit-code vocabulary, no tracebacks for expected failures, and batch support
that loads a model once.

Exit codes:

* ``0`` success;
* ``1`` runtime/model failure;
* ``2`` usage or invalid input;
* ``3`` verification failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import yaml

from . import __version__, catalog
from .adapters import ADAPTERS
from .config import save_config
from .decision_model import DecisionModel
from .errors import (
    DecisionToolkitError,
    InvalidQuestionError,
    InvalidRequestError,
    ModelNotFoundError,
    UnsupportedCapabilityError,
    VerificationError,
)
from .models import ModelEntry, ModelStore
from .observability import configure_logging
from .schema import normalize_questions, parse_question_spec

PROG = "dtk"

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_VERIFICATION = 3


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "decision-toolkit: local typed decisions (choice / yes-no / score) with "
            "open-weights ONNX models, on CPU. One engine, one adapter per model contract."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--home",
        metavar="DIR",
        help="Toolkit home (models cache + config). Default: ~/.local/share/decision-toolkit.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Suppress informational output.")
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Show debug logs and tracebacks.")
    parser.add_argument(
        "--log-level",
        dest="cli_log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Toolkit log level (default WARNING; --verbose raises it to DEBUG).",
    )
    parser.add_argument(
        "--log-format",
        dest="cli_log_format",
        choices=("text", "json"),
        default="text",
        help="Log format for CLI diagnostics (default text).",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colors (accepted for scripts).")
    sub = parser.add_subparsers(dest="command", required=True)

    # ----------------------------------------------------------------- models
    models = sub.add_parser("models", help="Model cache and curated catalog.")
    _add_json(models)
    models_sub = models.add_subparsers(dest="models_command")

    models_list = models_sub.add_parser("list", help="The curated library (adapter, variants).")
    _add_json(models_list)
    models_list.add_argument("--license", action="store_true", dest="show_license")
    models_list.add_argument("--quant", action="store_true", dest="show_quant")
    models_list.add_argument("--adapter", action="store_true", dest="show_adapter")

    models_variants = models_sub.add_parser("variants", help="Variants of a catalog model.")
    models_variants.add_argument("model")
    _add_json(models_variants)

    models_download = models_sub.add_parser(
        "download", help="Download one variant (the only network command)."
    )
    models_download.add_argument("model", help="Catalog id.")
    models_download.add_argument("--variant", help="Variant name (default: the catalog default).")
    models_download.add_argument("--revision", help="Pin a revision.")
    models_download.add_argument("--force", action="store_true", help="Replace an existing download.")
    models_download.add_argument("--dry-run", action="store_true", help="Show what would be downloaded.")
    _add_json(models_download)

    models_use = models_sub.add_parser("use", help="Set the model (and optional variant) in use.")
    models_use.add_argument("model")
    models_use.add_argument("--variant", help="Variant name.")
    _add_json(models_use)

    models_current = models_sub.add_parser("current", help="Show the model in use.")
    _add_json(models_current)

    models_delete = models_sub.add_parser("delete", help="Delete cached variant(s).")
    models_delete.add_argument("model")
    models_delete.add_argument("--variant", help="Variant name (default: any cached variant).")
    models_delete.add_argument(
        "--all-variants", action="store_true", help="Delete every cached variant of the model."
    )
    models_delete.add_argument("--yes", action="store_true", help="Do not prompt for --all-variants.")
    _add_json(models_delete)

    models_verify = models_sub.add_parser("verify", help="Check files, hashes and ONNX sessions.")
    models_verify.add_argument("model", nargs="?", help="Cached model id (omit with --all).")
    models_verify.add_argument("--variant", help="Variant name.")
    models_verify.add_argument("--all", action="store_true", dest="verify_all", help="Verify every cached variant.")
    _add_json(models_verify)

    models_import = models_sub.add_parser("import", help="Copy a local model directory into the cache.")
    models_import.add_argument("path")
    models_import.add_argument("--name", help="Identifier to store it under.")
    _add_custom_args(models_import)
    _add_json(models_import)

    # ----------------------------------------------------------------- decide
    decide = sub.add_parser("decide", help="Answer typed questions about a state (System One).")
    _add_model_args(decide)
    state_group = decide.add_mutually_exclusive_group(required=True)
    state_group.add_argument("-s", "--state", help="The state text.")
    state_group.add_argument(
        "--state-file",
        help="Read the state from a file ('-' for stdin; .json files are sent as JSON).",
    )
    state_group.add_argument(
        "--input-jsonl",
        metavar="FILE",
        help="Batch: one state per JSONL line ('-' for stdin).",
    )
    questions_group = decide.add_mutually_exclusive_group()
    questions_group.add_argument(
        "--q",
        action="append",
        metavar="NAME=SPEC",
        help=(
            "A question: NAME='choice:a|b', NAME='choice:a=desc|b', "
            "NAME='noul:Does it need a reply today?' or NAME='score:calm|annoyed|furious'. Repeatable."
        ),
    )
    questions_group.add_argument(
        "--questions-file",
        metavar="FILE",
        help="JSON/YAML question mapping (supports task, multi_label, box and instructions).",
    )
    decide.add_argument("--output-jsonl", metavar="FILE", help="Batch output ('-' for stdout).")
    decide.add_argument("--batch-size", type=int, default=32, help="JSONL lines per chunk (default 32).")
    decide.add_argument("--strict", action="store_true", help="Fail instead of returning truncated answers.")
    decide.add_argument("--json", action="store_true", dest="as_json", help="Print JSON (default).")
    decide.add_argument("--pretty", action="store_true", help="Indent the JSON output.")

    # --------------------------------------------------------------- classify
    classify = sub.add_parser("classify", help="One-label classification with a threshold.")
    _add_model_args(classify)
    classify_group = classify.add_mutually_exclusive_group(required=True)
    classify_group.add_argument("-t", "--text", help="The text to classify.")
    classify_group.add_argument("--text-file", help="Read the text from a file ('-' for stdin).")
    classify.add_argument("--labels", help="Comma-separated labels: 'reminder,task,idea'.")
    classify.add_argument(
        "--labels-file",
        help="YAML file shaped 'labels: {name: description}' (descriptions help; recommended).",
    )
    classify.add_argument("--instructions", help="Custom question ('Which category fits best?').")
    classify.add_argument("--threshold", type=float, default=0.5, help="Flag answers below it (default 0.5).")
    classify.add_argument("--top-k", type=int, default=3, help="Show this many ranked labels (default 3).")
    _add_json(classify)

    # ------------------------------------------------------------------ serve
    serve = sub.add_parser("serve", help="Run the REST server (needs the [server] extra).")
    _add_model_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1, help="Uvicorn worker processes (default 1).")
    serve.add_argument("--reload", action="store_true", help="Reload on source changes (development).")
    serve.add_argument(
        "--api-key",
        help="Require `Authorization: Bearer <key>` (default: DECISION_TOOLKIT_API_KEY).",
    )
    serve.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="Allow a browser origin (repeatable; e.g. http://localhost:3000).",
    )
    serve.add_argument("--max-body-mb", type=float, default=16.0, help="Request body limit (default 16).")
    serve.add_argument("--timeout", type=float, default=120.0, help="Per-decision timeout in seconds.")
    serve.add_argument("--max-models", type=int, default=2, help="Idle models kept loaded (LRU).")
    serve.add_argument("--warmup", action="store_true", help="Warm the default model at startup.")
    serve.add_argument(
        "--log-level",
        default=argparse.SUPPRESS,
        help="Toolkit log level (default INFO, or the global --log-level).",
    )
    serve.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=argparse.SUPPRESS,
        help="Toolkit log format (default text, or the global --log-format).",
    )

    # ----------------------------------------------------------------- config
    config = sub.add_parser("config", help="Read and write toolkit configuration.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_list = config_sub.add_parser("list", help="Show all configuration values.")
    _add_json(config_list)
    config_get = config_sub.add_parser("get", help="Show one configuration value.")
    config_get.add_argument("key")
    _add_json(config_get)
    config_set = config_sub.add_parser("set", help="Set one configuration value.")
    config_set.add_argument("key")
    config_set.add_argument("value")
    _add_json(config_set)

    # ------------------------------------------------------------- misc
    doctor = sub.add_parser("doctor", help="Environment and cache diagnostics.")
    _add_json(doctor)

    version = sub.add_parser("version", help="Print the toolkit version.")
    _add_json(version)

    completion = sub.add_parser("completion", help="Print shell completion for bash/zsh/fish.")
    completion.add_argument("shell", choices=("bash", "zsh", "fish"))

    return parser


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        default=argparse.SUPPRESS,
        help="Print machine-readable JSON.",
    )


def _add_custom_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", help="Adapter name for a custom model (or an override).")
    parser.add_argument(
        "--files",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Custom model file role (repeatable): encoder=model.onnx ...",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="Catalog id, cached variant, local path or 'org/name' (default: the model in use).",
    )
    parser.add_argument("--variant", help="Variant name (default: the catalog default).")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime threads (default 4).")
    _add_custom_args(parser)


# --------------------------------------------------------------------- helpers
def _as_json(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "as_json", False))


def _json_out(payload: Any, *, pretty: bool = False) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False))


def _info(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "quiet", False):
        return
    print(message, file=sys.stderr if _as_json(args) else sys.stdout)


def _configure_cli_logging(args: argparse.Namespace) -> None:
    if not (args.verbose or args.quiet or args.cli_log_level or args.cli_log_format != "text"):
        return
    level = (
        "DEBUG"
        if args.verbose
        else "ERROR"
        if args.quiet
        else args.cli_log_level or "WARNING"
    )
    configure_logging(level, args.cli_log_format)


def _read_text(source: str | None) -> str:
    if source is None:
        raise InvalidRequestError("provide --state / --state-file")
    if source == "-":
        return sys.stdin.read()
    return Path(source).expanduser().read_text(encoding="utf-8")


def _read_state(args: argparse.Namespace) -> Any:
    raw = args.state if args.state is not None else _read_text(args.state_file)
    if args.state_file and str(args.state_file).endswith(".json"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise InvalidRequestError(f"{args.state_file}: invalid JSON: {error}") from error
    return raw


def _parse_questions(items: Sequence[str]) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for item in items:
        name, separator, spec = item.partition("=")
        if not separator:
            raise InvalidQuestionError(f"--q needs NAME=SPEC, got {item!r}")
        name = name.strip()
        if not name:
            raise InvalidQuestionError(f"question name cannot be empty in {item!r}")
        if name in questions:
            raise InvalidQuestionError(f"duplicate question name {name!r}")
        question = parse_question_spec(spec.strip())
        if question["type"] != "noul" and "instructions" not in question:
            readable = name.replace("_", " ").replace("-", " ")
            question["instructions"] = (
                f"Which {readable} fits the state best?"
                if question["type"] == "choice"
                else f"Rate the {readable} on the given levels."
            )
        questions[name] = question
    return questions


def _load_questions_file(path: str | Path) -> dict[str, dict[str, Any]]:
    target = Path(path).expanduser()
    text = target.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise InvalidQuestionError(f"{target}: invalid JSON/YAML: {error}") from error
    if not isinstance(data, Mapping):
        raise InvalidQuestionError(f"{target}: expected a question mapping")
    if isinstance(data.get("questions"), Mapping):
        data = data["questions"]
    return normalize_questions(data)


def _global_questions(args: argparse.Namespace) -> dict[str, dict[str, Any]] | None:
    if getattr(args, "questions_file", None):
        return _load_questions_file(args.questions_file)
    q = getattr(args, "q", None)
    if q:
        return _parse_questions(q)
    return None


def _parse_files(items: Sequence[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    files: dict[str, str] = {}
    for item in items:
        role, separator, path = item.partition("=")
        if not separator or not role.strip() or not path.strip():
            raise InvalidRequestError(f"--files needs ROLE=PATH, got {item!r}")
        files[role.strip()] = path.strip()
    return files


def _build_model(args: argparse.Namespace) -> DecisionModel:
    files = _parse_files(getattr(args, "files", None))
    kwargs: dict[str, Any] = {
        "variant": getattr(args, "variant", None),
        "home": args.home,
        "threads": getattr(args, "threads", 4),
        "strict": bool(getattr(args, "strict", False)),
    }
    adapter = getattr(args, "adapter", None)
    if adapter:
        kwargs["adapter"] = adapter
    if files:
        kwargs["files"] = files
    return DecisionModel(getattr(args, "model", None), **kwargs)


def _entry_payload(entry: ModelEntry) -> dict[str, Any]:
    return {
        "identifier": entry.identifier,
        "variant": entry.variant,
        "path": str(entry.path),
        "repo": entry.repo,
        "origin": entry.origin,
        "source": entry.source,
        "created_at": entry.created_at,
        "size_bytes": entry.size_bytes,
        "adapter": entry.adapter,
        "files": dict(entry.files),
        "extra_files": list(entry.extra_files),
        "sha256": dict(entry.sha256),
        "extra_sha256": dict(entry.extra_sha256),
        "revision": entry.revision,
    }


def _cached_set(store: ModelStore) -> set[tuple[str, str]]:
    return {(entry.identifier, entry.variant) for entry in store.entries()}


def _catalog_payload(model: catalog.CatalogModel, cached: set[tuple[str, str]], current: str | None) -> dict[str, Any]:
    payload = model.info.to_dict()
    payload["cached_variants"] = [
        variant.name for variant in model.variants if (model.id, variant.name) in cached
    ]
    payload["current"] = model.id == current
    return payload


def _variant_payload(variant: catalog.Variant, cached: set[tuple[str, str]], model_id: str) -> dict[str, Any]:
    return {
        "name": variant.name,
        "quant": variant.quant,
        "size_mb": variant.size_mb,
        "role": variant.role,
        "repo": variant.repo,
        "revision": variant.revision,
        "files": dict(variant.files),
        "extra_files": list(variant.extra_files),
        "notes": variant.notes,
        "verified_at": variant.verified_at,
        "latency_ms_p50": variant.latency_ms_p50,
        "parity": dict(variant.parity) if variant.parity else None,
        "sha256": dict(variant.sha256),
        "extra_sha256": dict(variant.extra_sha256),
        "cached": (model_id, variant.name) in cached,
    }


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    text_rows = [[str(cell) for cell in row] for row in rows]
    if sys.stdout.isatty():
        widths = [len(header) for header in headers]
        for row in text_rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
        for row in text_rows:
            print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    else:
        print("\t".join(headers))
        for row in text_rows:
            print("\t".join(row))


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover


def _chunks(lines: Iterator[str], size: int) -> Iterator[list[str]]:
    while True:
        chunk = list(islice(lines, size))
        if not chunk:
            return
        yield chunk


def _parse_jsonl_line(line: str) -> tuple[Any, Any, dict[str, Any]] | None:
    line = line.strip()
    if not line:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return line, None, {}
    if isinstance(value, Mapping) and "state" in value:
        meta = {key: value[key] for key in ("id", "name") if key in value}
        return value["state"], value.get("questions"), meta
    return value, None, {}


# ------------------------------------------------------------------ commands
def _cmd_models(args: argparse.Namespace) -> int:
    store = ModelStore(home=args.home)
    command = getattr(args, "models_command", None)
    as_json = _as_json(args)

    if command is None or command == "current":
        if command == "current":
            payload = {"current": store.current(), "variant": store.current_variant()}
            if as_json:
                _json_out(payload)
            else:
                print(store.current() or "(none)")
            return EXIT_OK
        entries = store.entries()
        if as_json:
            _json_out([_entry_payload(entry) for entry in entries])
            return EXIT_OK
        if not entries:
            _info(args, "The toolkit cache is empty. `dtk models list` shows the catalog.")
            return EXIT_OK
        current = store.current()
        current_variant = store.current_variant()
        rows = []
        for entry in entries:
            mark = "★" if (
                entry.identifier == current
                and (current_variant is None or entry.variant == current_variant)
            ) else " "
            rows.append(
                [
                    mark,
                    f"{entry.identifier}@{entry.variant}",
                    _human_bytes(entry.size_bytes),
                    entry.origin,
                ]
            )
        _print_table(["", "id", "size", "origin"], rows)
        return EXIT_OK

    if command == "list":
        cached = _cached_set(store)
        current = store.current()
        payloads = [_catalog_payload(model, cached, current) for model in catalog.CATALOG]
        if as_json:
            _json_out(payloads)
            return EXIT_OK
        headers = ["", "id", "default"]
        if args.show_adapter:
            headers.append("adapter")
        if args.show_license:
            headers.append("license")
        if args.show_quant:
            headers.append("quant")
        headers.append("cache")
        rows = []
        for model, payload in zip(catalog.CATALOG, payloads):
            default = model.default_variant
            label = default.name if default else "—"
            row: list[Any] = ["★" if model.id == current else " ", model.id, label]
            if args.show_adapter:
                row.append(model.adapter)
            if args.show_license:
                row.append(model.license)
            if args.show_quant:
                row.append(default.quant if default else "—")
            row.append("✓" if payload["cached_variants"] else "")
            rows.append(row)
        _print_table(headers, rows)
        return EXIT_OK

    if command == "variants":
        model = catalog.get(args.model)
        if model is None:
            raise ModelNotFoundError(f"unknown catalog model: {args.model}")
        cached = _cached_set(store)
        payload = {
            "id": model.id,
            "adapter": model.adapter,
            "status": model.status,
            "capabilities": model.capabilities.to_dict(),
            "variants": [
                _variant_payload(variant, cached, model.id) for variant in model.variants
            ],
        }
        if as_json:
            _json_out(payload)
            return EXIT_OK
        print(f"{model.id} · adapter={model.adapter} · status={model.status}")
        for variant in model.variants:
            cached_mark = " ✓" if (model.id, variant.name) in cached else ""
            print(
                f"  {variant.name:8s} {variant.quant:10s} {variant.size_mb:7.1f} MB  "
                f"role={variant.role:8s} repo={variant.repo}{cached_mark}"
            )
            print(f"           revision: {variant.revision or '(unpinned)'}")
            if variant.latency_ms_p50 is not None:
                verified = f" · verified {variant.verified_at}" if variant.verified_at else ""
                print(f"           latency p50: {variant.latency_ms_p50:.0f} ms{verified}")
            if variant.parity:
                summary = ", ".join(f"{key}={value}" for key, value in variant.parity.items())
                print(f"           parity: {summary}")
        if not model.variants:
            print("  (no downloadable ONNX variant yet)")
        return EXIT_OK

    if command == "download":
        model = catalog.get(args.model)
        if model is None:
            raise ModelNotFoundError(f"unknown catalog model: {args.model}")
        chosen = catalog.select_variant(model, args.variant)
        if args.dry_run:
            payload = {
                "identifier": model.id,
                "variant": chosen.name,
                "repo": chosen.repo,
                "revision": chosen.revision,
                "quant": chosen.quant,
                "size_mb": chosen.size_mb,
                "files": dict(chosen.files),
                "extra_files": list(chosen.extra_files),
                "sha256": dict(chosen.sha256),
                "extra_sha256": dict(chosen.extra_sha256),
                "cached": (model.id, chosen.name) in _cached_set(store),
            }
            if as_json:
                _json_out(payload)
            else:
                _info(
                    args,
                    f"would download {model.id}@{chosen.name} from {chosen.repo} "
                    f"({chosen.size_mb:.1f} MB)",
                )
            return EXIT_OK
        entry = store.download(
            args.model,
            variant=args.variant,
            revision=args.revision,
            force=args.force,
        )
        if as_json:
            _json_out(_entry_payload(entry))
        else:
            _info(args, f"downloaded {entry.identifier}@{entry.variant} -> {entry.path}")
        return EXIT_OK

    if command == "use":
        store.use(args.model, variant=args.variant)
        payload = {"current": store.current(), "variant": store.current_variant()}
        if as_json:
            _json_out(payload)
        else:
            suffix = f"@{args.variant}" if args.variant else ""
            _info(args, f"model in use: {args.model}{suffix}")
        return EXIT_OK

    if command == "delete":
        if args.all_variants and args.variant:
            raise InvalidRequestError("--variant and --all-variants are mutually exclusive")
        if args.all_variants:
            entries = [entry for entry in store.entries() if entry.identifier == args.model]
            if not entries:
                raise ModelNotFoundError(f"{args.model!r} is not in the toolkit cache")
            if not args.yes:
                if not sys.stdin.isatty():
                    raise InvalidRequestError("pass --yes to delete all variants non-interactively")
                answer = input(f"delete {len(entries)} variant(s) of {args.model}? [y/N] ")
                if answer.strip().lower() not in {"y", "yes"}:
                    _info(args, "aborted")
                    return EXIT_OK
            deleted: list[str] = []
            failed: list[str] = []
            for entry in entries:
                try:
                    store.delete(entry.identifier, entry.variant)
                    deleted.append(entry.variant)
                except ModelNotFoundError:
                    failed.append(entry.variant)
            payload = {"identifier": args.model, "deleted": deleted, "failed": failed}
            if as_json:
                _json_out(payload)
            else:
                for variant in deleted:
                    _info(args, f"deleted {args.model}@{variant}")
                for variant in failed:
                    _info(args, f"kept {args.model}@{variant} (in use)")
            return EXIT_RUNTIME if failed else EXIT_OK

        store.delete(args.model, variant=args.variant)
        payload = {"identifier": args.model, "variant": args.variant, "deleted": True}
        if as_json:
            _json_out(payload)
        else:
            _info(args, f"deleted {args.model}" + (f"@{args.variant}" if args.variant else ""))
        return EXIT_OK

    if command == "verify":
        if args.verify_all and (args.model or args.variant):
            raise InvalidRequestError("--all cannot be combined with a model or --variant")
        if args.verify_all:
            reports = [
                store.verify(entry.identifier, entry.variant)
                for entry in store.entries()
            ]
            payload = {"ok": all(report.get("ok") for report in reports), "reports": reports}
            if as_json:
                _json_out(payload)
            elif not reports:
                _info(args, "no cached models to verify")
            else:
                for report in reports:
                    print(json.dumps(report, indent=2, ensure_ascii=False))
            return EXIT_OK if payload["ok"] else EXIT_VERIFICATION
        if not args.model:
            raise InvalidRequestError("pass a model or --all")
        report = store.verify(args.model, variant=args.variant)
        if as_json:
            _json_out(report)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        return EXIT_OK if report.get("ok") else EXIT_VERIFICATION

    if command == "import":
        entry = store.import_model(
            args.path,
            name=args.name,
            adapter=args.adapter,
            files=_parse_files(args.files),
        )
        if as_json:
            _json_out(_entry_payload(entry))
        else:
            _info(args, f"imported {entry.identifier}@{entry.variant} -> {entry.path}")
        return EXIT_OK

    raise InvalidRequestError(f"unknown models command: {command}")  # pragma: no cover


def _decide_batch(args: argparse.Namespace) -> int:
    if args.output_jsonl and not args.input_jsonl:
        raise InvalidRequestError("--output-jsonl requires --input-jsonl")
    if (
        args.input_jsonl
        and args.input_jsonl != "-"
        and args.output_jsonl
        and args.output_jsonl != "-"
        and Path(args.input_jsonl).expanduser().resolve()
        == Path(args.output_jsonl).expanduser().resolve()
    ):
        raise InvalidRequestError("--input-jsonl and --output-jsonl must be different files")
    questions = _global_questions(args)
    model = _build_model(args)
    batch_size = max(1, int(args.batch_size))

    source = sys.stdin if args.input_jsonl == "-" else open(  # noqa: SIM115 - closed below
        Path(args.input_jsonl).expanduser(), encoding="utf-8"
    )
    target = sys.stdout if not args.output_jsonl or args.output_jsonl == "-" else open(  # noqa: SIM115
        Path(args.output_jsonl).expanduser(), "w", encoding="utf-8"
    )
    processed = 0
    try:
        for chunk in _chunks(iter(source), batch_size):
            for line in chunk:
                parsed = _parse_jsonl_line(line)
                if parsed is None:
                    continue
                state_value, line_questions, meta = parsed
                if line_questions is None:
                    selected_questions = questions
                elif isinstance(line_questions, Mapping):
                    selected_questions = normalize_questions(line_questions)
                else:
                    raise InvalidQuestionError("JSONL 'questions' must be an object")
                if not selected_questions:
                    raise InvalidQuestionError(
                        "no questions: pass --q/--questions-file or include 'questions' in each line"
                    )
                decision = model.decide(state_value, selected_questions, strict=args.strict)
                payload = decision.to_payload()
                if "id" in meta:
                    payload["id"] = meta["id"]
                if "name" in meta:
                    payload["name"] = meta["name"]
                target.write(json.dumps(payload, ensure_ascii=False) + "\n")
                processed += 1
    finally:
        if source is not sys.stdin:
            source.close()
        if target is not sys.stdout:
            target.close()
    if args.output_jsonl and args.output_jsonl != "-":
        _info(args, f"wrote {processed} decisions to {args.output_jsonl}")
    return EXIT_OK


def _cmd_decide(args: argparse.Namespace) -> int:
    if args.input_jsonl:
        return _decide_batch(args)
    if args.output_jsonl:
        raise InvalidRequestError("--output-jsonl requires --input-jsonl")
    if not args.q and not args.questions_file:
        raise InvalidQuestionError("provide --q or --questions-file")
    questions = _global_questions(args)
    if not questions:
        raise InvalidQuestionError("provide --q or --questions-file")
    model = _build_model(args)
    decision = model.decide(_read_state(args), questions, strict=args.strict)
    _json_out(decision.to_payload(), pretty=args.pretty)
    return EXIT_OK


def _load_categories(args: argparse.Namespace) -> Any:
    if args.labels_file:
        data = yaml.safe_load(Path(args.labels_file).expanduser().read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("labels"), dict):
            raise InvalidQuestionError(
                f"{args.labels_file}: expected a top-level mapping 'labels: {{name: description}}'"
            )
        return {str(k): str(v) for k, v in data["labels"].items()}
    if args.labels:
        return [label.strip() for label in args.labels.split(",") if label.strip()]
    raise InvalidQuestionError("provide --labels or --labels-file")


def _cmd_classify(args: argparse.Namespace) -> int:
    model = _build_model(args)
    text = args.text if args.text is not None else _read_text(args.text_file)
    result = model.classify(
        text,
        _load_categories(args),
        instructions=args.instructions,
        threshold=args.threshold,
    )
    if _as_json(args):
        _json_out(result.to_dict(args.top_k))
    else:
        print(
            f"{result.label}  {result.confidence:.3f}"
            + ("  (below threshold)" if result.below_threshold else "")
        )
        for label, score in result.ranked(args.top_k):
            print(f"  {label:24s} {score:.3f}")
    return EXIT_OK


def _serve_factory() -> Any:
    """Uvicorn factory used by ``--workers``/``--reload``."""
    raw = os.environ.get("DTK_SERVE_SETTINGS")
    if not raw:
        raise RuntimeError("DTK_SERVE_SETTINGS is missing")
    from .server import create_app

    return create_app(**json.loads(raw))


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        print(
            'the REST server needs the [server] extra: pip install "decision-toolkit[server]"',
            file=sys.stderr,
        )
        return EXIT_RUNTIME

    if args.workers < 1:
        raise InvalidRequestError("--workers must be at least 1")
    if args.reload and args.workers != 1:
        raise InvalidRequestError("--reload cannot be combined with --workers > 1")

    api_key = args.api_key if args.api_key is not None else os.environ.get("DECISION_TOOLKIT_API_KEY")
    files = _parse_files(args.files)
    log_level = getattr(args, "log_level", None) or args.cli_log_level or "INFO"
    log_format = getattr(args, "log_format", None) or args.cli_log_format
    settings = {
        "model_id": args.model,
        "variant": args.variant,
        "home": args.home,
        "threads": args.threads,
        "adapter": args.adapter,
        "files": files,
        "api_key": api_key,
        "cors_origins": args.cors_origin,
        "max_body_mb": args.max_body_mb,
        "timeout_seconds": args.timeout,
        "max_models": args.max_models,
        "warmup": args.warmup,
        "log_level": log_level,
        "log_format": log_format,
    }

    try:
        if args.workers == 1 and not args.reload:
            app = create_app(**settings)
            uvicorn.run(app, host=args.host, port=args.port)
        else:
            os.environ["DTK_SERVE_SETTINGS"] = json.dumps(settings)
            uvicorn.run(
                "decision_toolkit.cli:_serve_factory",
                factory=True,
                host=args.host,
                port=args.port,
                workers=args.workers,
                reload=args.reload,
                log_level=str(log_level).lower(),
            )
    except ImportError:
        print(
            'the REST server needs the [server] extra: pip install "decision-toolkit[server]"',
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    return EXIT_OK


def _config_payload(store: ModelStore) -> dict[str, Any]:
    return {
        "home": str(store.config.home),
        "cache_dir": str(store.config.cache_dir),
        "current": store.current(),
        "current_variant": store.current_variant(),
        "device": store.config.device,
    }


def _config_value(store: ModelStore, key: str) -> Any:
    if key == "home":
        return str(store.config.home)
    if key == "cache_dir":
        return str(store.config.cache_dir)
    if key == "current":
        return store.current()
    if key == "current_variant":
        return store.current_variant()
    if key == "device":
        return store.config.device
    raise InvalidRequestError(
        "unknown config key; expected home, cache_dir, current, current_variant or device"
    )


def _cmd_config(args: argparse.Namespace) -> int:
    store = ModelStore(home=args.home)
    command = args.config_command
    as_json = _as_json(args)

    if command == "list":
        payload = _config_payload(store)
        if as_json:
            _json_out(payload)
        else:
            for key, value in payload.items():
                print(f"{key}: {value if value is not None else '(none)'}")
        return EXIT_OK

    key = args.key
    if command == "get":
        value = _config_value(store, key)
        if as_json:
            _json_out({"key": key, "value": value})
        else:
            print(value if value is not None else "(none)")
        return EXIT_OK

    if command == "set":
        if key == "home":
            raise InvalidRequestError("home is read-only; pass --home to change it")
        value: Any = args.value
        if key in {"current", "current_variant"} and value == "":
            value = None
        if key == "current":
            if value is None:
                store.config.current = None
                store.config.current_variant = None
                save_config(store.config)
            else:
                # Changing the model resets a variant that belonged to the old one;
                # `config set current_variant` can select a new variant afterwards.
                store.use(str(value))
        elif key == "current_variant":
            if value is not None and store.current():
                model = catalog.get(store.current())
                if model is not None:
                    catalog.select_variant(model, str(value))
                elif store.get(store.current(), str(value)) is None:
                    raise ModelNotFoundError(
                        f"{store.current()!r} is not cached with variant {value!r}"
                    )
            store.config.current_variant = None if value is None else str(value)
            save_config(store.config)
        elif key == "cache_dir":
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (store.config.home / path).resolve()
            store.config.cache_dir = path
            path.mkdir(parents=True, exist_ok=True)
            save_config(store.config)
        elif key == "device":
            store.config.device = str(value)
            save_config(store.config)
        else:
            raise InvalidRequestError(
                "unknown or read-only config key; expected cache_dir, current, "
                "current_variant or device"
            )
        payload = {"key": key, "value": _config_value(store, key)}
        if as_json:
            _json_out(payload)
        else:
            print(f"{key} = {payload['value'] if payload['value'] is not None else '(none)'}")
        return EXIT_OK

    raise InvalidRequestError(f"unknown config command: {command}")  # pragma: no cover


def _cmd_doctor(args: argparse.Namespace) -> int:
    store = ModelStore(home=args.home)
    entries = store.entries()

    def probe(module: str) -> bool:
        try:
            __import__(module)
            return True
        except ImportError:
            return False

    report = {
        "toolkit": __version__,
        "python": sys.version.split()[0],
        "home": str(store.config.home),
        "cache_dir": str(store.models_dir),
        "current": store.current(),
        "current_variant": store.current_variant(),
        "device": store.config.device,
        "cached_models": [f"{entry.identifier}@{entry.variant}" for entry in entries],
        "catalog": {
            status: sum(1 for model in catalog.CATALOG if model.status == status)
            for status in ("m0", "m1", "planned", "phase2", "waitlist")
        },
        "adapters": sorted(ADAPTERS),
        "runtimes": {
            "onnxruntime": probe("onnxruntime"),
            "tokenizers": probe("tokenizers"),
            "fastapi (REST)": probe("fastapi"),
        },
    }
    if _as_json(args):
        _json_out(report)
    else:
        print(f"decision-toolkit {report['toolkit']} (python {report['python']})")
        print(f"home:       {report['home']}")
        print(f"cache_dir:  {report['cache_dir']}")
        print(f"current:    {report['current'] or '(none)'}", end="")
        if report["current_variant"]:
            print(f"@{report['current_variant']}")
        else:
            print()
        print(f"device:     {report['device']}")
        print(f"cached:     {', '.join(report['cached_models']) or '(empty)'}")
        print("catalog:    " + ", ".join(f"{k}={v}" for k, v in report["catalog"].items()))
        print(f"adapters:   {', '.join(report['adapters'])}")
        print("runtimes:")
        for label, ok in report["runtimes"].items():
            print(f"  {'✓' if ok else '·'} {label}: {'installed' if ok else 'not installed'}")
    return EXIT_OK


def _cmd_version(args: argparse.Namespace) -> int:
    payload = {
        "name": "decision-toolkit",
        "version": __version__,
        "python": sys.version.split()[0],
    }
    if _as_json(args):
        _json_out(payload)
    else:
        print(f"{PROG} {__version__}")
    return EXIT_OK


def _cmd_completion(args: argparse.Namespace) -> int:
    commands = "models decide classify serve config doctor version completion"
    if args.shell == "bash":
        print(f"complete -W '{commands}' {PROG}")
    elif args.shell == "zsh":
        print(f"#compdef {PROG}\n_arguments '1:command:({commands})'")
    else:
        print(f"complete -c {PROG} -f -a '{commands}'")
    return EXIT_OK


# ---------------------------------------------------------------------- main
def _exit_code(error: DecisionToolkitError) -> int:
    if isinstance(error, VerificationError):
        return EXIT_VERIFICATION
    if isinstance(
        error,
        (InvalidQuestionError, InvalidRequestError, UnsupportedCapabilityError),
    ):
        return EXIT_USAGE
    return EXIT_RUNTIME


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_cli_logging(args)
    try:
        if args.command == "models":
            return _cmd_models(args)
        if args.command == "decide":
            return _cmd_decide(args)
        if args.command == "classify":
            return _cmd_classify(args)
        if args.command == "serve":
            return _cmd_serve(args)
        if args.command == "config":
            return _cmd_config(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "version":
            return _cmd_version(args)
        if args.command == "completion":
            return _cmd_completion(args)
        parser.error(f"unknown command {args.command!r}")  # pragma: no cover
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except DecisionToolkitError as error:
        if args.verbose:
            traceback.print_exc()
        print(f"{PROG}: error: {error}", file=sys.stderr)
        return _exit_code(error)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        if args.verbose:
            traceback.print_exc()
        print(f"{PROG}: error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as error:
        if args.verbose:
            traceback.print_exc()
        print(f"{PROG}: error: {error}", file=sys.stderr)
        return EXIT_RUNTIME
    except Exception as error:  # noqa: BLE001 - CLI boundary must stay clean
        if args.verbose:
            traceback.print_exc()
        print(f"{PROG}: error: {error}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
