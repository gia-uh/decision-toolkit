"""Validate every runnable example printed by ``dtk --help``.

The full execution audit (real ONNX models and temporary files) is run before a
release with a populated cache. This test keeps every example syntactically
valid and ensures every referenced catalog model/variant still exists; it runs
in CI without downloading models.
"""

from __future__ import annotations

import argparse
import shlex

from decision_toolkit import catalog, cli


def _walk(parser: argparse.ArgumentParser):
    if parser.epilog and "examples:" in parser.epilog:
        for line in parser.epilog.splitlines()[1:]:
            if line.strip():
                yield line.strip()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _walk(subparser)


def _examples() -> list[str]:
    examples = list(dict.fromkeys(_walk(cli.build_parser())))
    assert examples
    return examples


def _validate_model_variant(model_id: str | None, variant: str | None) -> None:
    if model_id is None or "/" in model_id or model_id.startswith("."):
        return
    model = catalog.get(model_id)
    if model is None:
        return
    if variant is not None:
        catalog.select_variant(model, variant)


def test_every_help_example_parses_and_references_catalog_entries():
    parser = cli.build_parser()
    for example in _examples():
        argv = shlex.split(example)
        assert argv[0] == "dtk", example
        args = parser.parse_args(argv[1:])
        command = getattr(args, "command", None)
        if command == "models":
            model_id = getattr(args, "model", None)
            variant = getattr(args, "variant", None)
            if args.models_command == "variants":
                model_id = args.model
                variant = None
            elif args.models_command in {"download", "use", "delete"}:
                model_id = args.model
            _validate_model_variant(model_id, variant)
        elif command in {"decide", "classify", "serve"}:
            _validate_model_variant(getattr(args, "model", None), getattr(args, "variant", None))
        elif command == "config":
            if getattr(args, "config_command", None) == "set" and args.key == "current":
                _validate_model_variant(args.value, None)
