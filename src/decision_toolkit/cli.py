"""Command-line interface: ``dtk models|decide|classify|serve|doctor``.

The CLI is a thin adapter over the public library API: it contains no
inference logic of its own. Model references are resolved through the
toolkit's own model cache (:mod:`decision_toolkit.models`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__, catalog
from .adapters import ADAPTERS
from .decision_model import DecisionModel
from .errors import DecisionToolkitError, InvalidQuestionError
from .models import ModelStore
from .schema import parse_question_spec

PROG = "dtk"


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
    sub = parser.add_subparsers(dest="command", required=True)

    # ----------------------------------------------------------------- models
    models = sub.add_parser("models", help="Model cache and curated catalog.")
    models_sub = models.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="The curated library (adapter, variants).")
    models_variants = models_sub.add_parser("variants", help="Variants of a catalog model.")
    models_variants.add_argument("model")
    models_download = models_sub.add_parser("download", help="Download one variant (the only network command).")
    models_download.add_argument("model", help="Catalog id.")
    models_download.add_argument("--variant", help="Variant name (default: the catalog default).")
    models_download.add_argument("--revision", help="Pin a revision.")
    models_use = models_sub.add_parser("use", help="Set the model in use (persistent).")
    models_use.add_argument("model")
    models_sub.add_parser("current", help="Print the model in use.")
    models_delete = models_sub.add_parser("delete", help="Delete a cached variant.")
    models_delete.add_argument("model")
    models_delete.add_argument("--variant", help="Variant name (default: any cached variant).")
    models_verify = models_sub.add_parser("verify", help="Check a cached variant (files + ONNX sessions).")
    models_verify.add_argument("model")
    models_verify.add_argument("--variant", help="Variant name (default: any cached variant).")
    models_import = models_sub.add_parser("import", help="Copy a local model directory into the cache.")
    models_import.add_argument("path")
    models_import.add_argument("--name", help="Identifier to store it under.")

    # ----------------------------------------------------------------- decide
    decide = sub.add_parser("decide", help="Answer typed questions about a state (System One).")
    _add_model_args(decide)
    group = decide.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--state", help="The state text.")
    group.add_argument("--state-file", help="Read the state from a file ('-' for stdin; .json files are sent as JSON).")
    decide.add_argument(
        "--q",
        action="append",
        required=True,
        metavar="NAME=SPEC",
        help=(
            "A question: NAME='choice:a|b', NAME='choice:a=desc|b', "
            "NAME='noul:Does it need a reply today?' or NAME='score:calm|annoyed|furious'. Repeatable."
        ),
    )
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
    classify.add_argument("--json", action="store_true", dest="as_json", help="Print JSON.")

    # ------------------------------------------------------------------ serve
    serve = sub.add_parser("serve", help="Run the REST server (needs the [server] extra).")
    _add_model_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
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
    serve.add_argument("--log-level", default="INFO", help="Toolkit log level (default INFO).")
    serve.add_argument("--log-format", choices=("text", "json"), default="text")

    # ----------------------------------------------------------------- doctor
    doctor = sub.add_parser("doctor", help="Environment and cache diagnostics.")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="Catalog id, cached variant, local path or 'org/name' (default: the model in use).",
    )
    parser.add_argument("--variant", help="Variant name (default: the catalog default).")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime threads (default 4).")


# --------------------------------------------------------------------- helpers


def _read_text(source: str | None) -> str:
    if source is None:
        raise InvalidQuestionError("provide --state / --state-file")
    if source == "-":
        return sys.stdin.read()
    return Path(source).expanduser().read_text(encoding="utf-8")


def _read_state(args: argparse.Namespace) -> Any:
    raw = args.state if args.state is not None else _read_text(args.state_file)
    if args.state_file and str(args.state_file).endswith(".json"):
        return json.loads(raw)
    return raw


def _parse_questions(items: Sequence[str]) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for item in items:
        name, separator, spec = item.partition("=")
        if not separator:
            raise InvalidQuestionError(f"--q needs NAME=SPEC, got {item!r}")
        name = name.strip()
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


def _load_categories(args: argparse.Namespace) -> Any:
    if args.labels_file:
        import yaml

        data = yaml.safe_load(Path(args.labels_file).expanduser().read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("labels"), dict):
            raise InvalidQuestionError(
                f"{args.labels_file}: expected a top-level mapping 'labels: {{name: description}}'"
            )
        return {str(k): str(v) for k, v in data["labels"].items()}
    if args.labels:
        return [label.strip() for label in args.labels.split(",") if label.strip()]
    raise InvalidQuestionError("provide --labels or --labels-file")


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"  # pragma: no cover


# ------------------------------------------------------------------ commands


def _cmd_models(args: argparse.Namespace) -> int:
    store = ModelStore(home=args.home)
    command = getattr(args, "models_command", None)

    if command is None or command == "current":
        if command == "current":
            print(store.current() or "(none)")
            return 0
        entries = store.entries()
        if not entries:
            print("The toolkit cache is empty. `dtk models list` shows the catalog.")
            return 0
        current = store.current()
        for entry in entries:
            mark = "★" if entry.identifier == current else " "
            print(f"{mark} {entry.identifier + '@' + entry.variant:34s} "
                  f"{_human_bytes(entry.size_bytes):>10s}  {entry.origin}")
        return 0

    if command == "list":
        cached = {(entry.identifier, entry.variant) for entry in store.entries()}
        current = store.current()
        print(f"{'':2s} {'id':28s} {'adapter':15s} {'default':10s} cache")
        for model in catalog.CATALOG:
            mark = "★" if model.id == current else " "
            default = model.default_variant
            label = f"{default.name} ({default.quant})" if default else "—"
            have = "✓" if default and (model.id, default.name) in cached else ""
            print(f"{mark} {model.id:28s} {model.adapter:15s} {label:10s} {have}")
        return 0

    if command == "variants":
        model = catalog.get(args.model)
        if model is None:
            raise InvalidQuestionError(f"unknown catalog model: {args.model}")
        print(f"{model.id} · adapter={model.adapter} · status={model.status}")
        for variant in model.variants:
            print(f"  {variant.name:8s} {variant.quant:10s} {variant.size_mb:7.1f} MB  "
                  f"role={variant.role:8s} repo={variant.repo}")
            if variant.latency_ms_p50 is not None:
                verified = f" · verified {variant.verified_at}" if variant.verified_at else ""
                print(f"           latency p50: {variant.latency_ms_p50:.0f} ms{verified}")
            if variant.parity:
                summary = ", ".join(f"{key}={value}" for key, value in variant.parity.items())
                print(f"           parity: {summary}")
        if not model.variants:
            print("  (no downloadable ONNX variant yet)")
        return 0

    if command == "download":
        entry = store.download(args.model, variant=args.variant, revision=args.revision)
        print(f"downloaded {entry.identifier}@{entry.variant} -> {entry.path}")
        return 0

    if command == "use":
        store.use(args.model)
        print(f"model in use: {args.model}")
        return 0

    if command == "delete":
        store.delete(args.model, variant=args.variant)
        print(f"deleted {args.model}" + (f"@{args.variant}" if args.variant else ""))
        return 0

    if command == "verify":
        report = store.verify(args.model, variant=args.variant)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1

    if command == "import":
        entry = store.import_model(args.path, name=args.name)
        print(f"imported {entry.identifier}@{entry.variant} -> {entry.path}")
        return 0

    raise InvalidQuestionError(f"unknown models command: {command}")  # pragma: no cover


def _cmd_decide(args: argparse.Namespace) -> int:
    model = DecisionModel(
        args.model, variant=args.variant, home=args.home, threads=args.threads
    )
    decision = model.decide(_read_state(args), _parse_questions(args.q))
    print(json.dumps(decision.to_dict(), indent=2 if args.pretty else None, ensure_ascii=False))
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    model = DecisionModel(
        args.model, variant=args.variant, home=args.home, threads=args.threads
    )
    text = args.text if args.text is not None else _read_text(args.text_file)
    result = model.classify(
        text,
        _load_categories(args),
        instructions=args.instructions,
        threshold=args.threshold,
    )
    if getattr(args, "as_json", False):
        print(json.dumps(
            {
                "label": result.label,
                "confidence": result.confidence,
                "below_threshold": result.below_threshold,
                "ranked": result.ranked(args.top_k),
            },
            ensure_ascii=False,
        ))
    else:
        print(f"{result.label}  {result.confidence:.3f}" + ("  (below threshold)" if result.below_threshold else ""))
        for label, score in result.ranked(args.top_k):
            print(f"  {label:24s} {score:.3f}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .observability import configure_logging
        from .server import create_app
    except ImportError:
        print(
            'the REST server needs the [server] extra: pip install "decision-toolkit[server]"',
            file=sys.stderr,
        )
        return 2

    configure_logging(args.log_level, args.log_format)
    api_key = args.api_key if args.api_key is not None else os.environ.get("DECISION_TOOLKIT_API_KEY")
    try:
        app = create_app(
            model_id=args.model,
            variant=args.variant,
            home=args.home,
            threads=args.threads,
            api_key=api_key,
            cors_origins=args.cors_origin,
            max_body_mb=args.max_body_mb,
            timeout_seconds=args.timeout,
            max_models=args.max_models,
            warmup=args.warmup,
        )
    except ImportError:
        print(
            'the REST server needs the [server] extra: pip install "decision-toolkit[server]"',
            file=sys.stderr,
        )
        return 2
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


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
    if args.as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"decision-toolkit {report['toolkit']} (python {report['python']})")
        print(f"home:       {report['home']}")
        print(f"cache_dir:  {report['cache_dir']}")
        print(f"current:    {report['current'] or '(none)'}")
        print(f"device:     {report['device']}")
        print(f"cached:     {', '.join(report['cached_models']) or '(empty)'}")
        print("catalog:    " + ", ".join(f"{k}={v}" for k, v in report["catalog"].items()))
        print(f"adapters:   {', '.join(report['adapters'])}")
        print("runtimes:")
        for label, ok in report["runtimes"].items():
            print(f"  {'✓' if ok else '·'} {label}: {'installed' if ok else 'not installed'}")
    return 0


# ---------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            return _cmd_models(args)
        if args.command == "decide":
            return _cmd_decide(args)
        if args.command == "classify":
            return _cmd_classify(args)
        if args.command == "serve":
            return _cmd_serve(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        parser.error(f"unknown command {args.command!r}")  # pragma: no cover
        return 2
    except DecisionToolkitError as error:
        print(f"{PROG}: error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
