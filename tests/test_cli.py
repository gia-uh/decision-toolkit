"""The dtk CLI: catalog, variants, cache, JSON, batch, config and error paths."""

from __future__ import annotations

import json
import logging
import os

import pytest

from decision_toolkit import cli
from decision_toolkit.decision_model import DecisionModel as RealDecisionModel
from decision_toolkit.errors import InvalidQuestionError, InvalidRequestError
from decision_toolkit.models import ModelStore

from .conftest import FakeAdapter


@pytest.fixture(autouse=True)
def _restore_logging():
    logger = logging.getLogger("decision_toolkit")
    old_handlers = list(logger.handlers)
    old_propagate = logger.propagate
    old_level = logger.level
    yield
    logger.handlers = old_handlers
    logger.propagate = old_propagate
    logger.setLevel(old_level)


@pytest.fixture
def fake_cli_model(monkeypatch):
    calls: list[tuple[str | None, dict]] = []

    def factory(identifier=None, **kwargs):
        calls.append((identifier, dict(kwargs)))
        return RealDecisionModel(adapter=FakeAdapter())

    monkeypatch.setattr(cli, "DecisionModel", factory)
    return calls


def _out(capsys) -> str:
    return capsys.readouterr().out


# --------------------------------------------------------------------- parser
def test_parser_has_all_commands():
    parser = cli.build_parser()
    valid = {
        "models": ["models"],
        "decide": ["decide", "--state", "x", "--q", "a=noul:y"],
        "classify": ["classify", "--text", "x", "--labels", "a,b"],
        "serve": ["serve"],
        "config": ["config", "list"],
        "doctor": ["doctor"],
        "version": ["version"],
        "completion": ["completion", "bash"],
    }
    for command, argv in valid.items():
        assert parser.parse_args(argv).command == command

    serve = parser.parse_args(
        ["serve", "--workers", "2", "--reload", "--max-concurrent", "3",
         "--rate-limit", "10", "--no-metrics", "--tracing"]
    )
    assert serve.workers == 2
    assert serve.reload is True
    assert serve.max_concurrent == 3
    assert serve.rate_limit == 10
    assert serve.no_metrics is True
    assert serve.tracing is True


def test_help_contains_runnable_examples(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "examples:" in output
    assert "dtk models list --json" in output
    assert "dtk models variants gliner2.5-multi" in output
    assert "dtk decide --model" in output

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["models", "list", "--help"])
    assert excinfo.value.code == 0
    assert "examples:" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["models", "download", "--help"])
    assert excinfo.value.code == 0
    assert "dtk models variants" in capsys.readouterr().out


def test_parse_questions_adds_default_instructions():
    questions = cli._parse_questions(
        ["team=choice:billing=refunds|tech", "urgent=noul:Today?"]
    )
    assert questions["team"]["instructions"] == "Which team fits the state best?"
    assert questions["team"]["criteria"] == {"billing": "refunds", "tech": None}
    assert questions["urgent"] == {"type": "noul", "instructions": "Today?"}


def test_parse_questions_requires_name_and_rejects_duplicates():
    with pytest.raises(InvalidQuestionError):
        cli._parse_questions(["team"])
    with pytest.raises(InvalidQuestionError):
        cli._parse_questions(["team=noul:x", "team=noul:y"])


def test_parse_files():
    assert cli._parse_files(["model=model.onnx", "tokenizer=tok.json"]) == {
        "model": "model.onnx",
        "tokenizer": "tok.json",
    }
    assert cli._parse_files([]) is None
    with pytest.raises(InvalidRequestError):
        cli._parse_files(["model"])


def test_questions_file_json_and_yaml(tmp_path):
    json_file = tmp_path / "questions.json"
    json_file.write_text(
        json.dumps(
            {
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"billing": "refunds", "tech": None},
                        "task": "route",
                        "multi_label": True,
                    },
                    "region": {
                        "type": "box",
                        "instructions": "Where?",
                        "criteria": ["object"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    questions = cli._load_questions_file(json_file)
    assert questions["team"]["criteria"]["tech"] is None
    assert questions["team"]["task"] == "route"
    assert questions["team"]["multi_label"] is True
    assert questions["region"]["type"] == "box"

    yaml_file = tmp_path / "questions.yaml"
    yaml_file.write_text(
        "questions:\n"
        "  urgent:\n"
        "    type: noul\n"
        "    instructions: Today?\n",
        encoding="utf-8",
    )
    assert cli._load_questions_file(yaml_file)["urgent"]["type"] == "noul"

    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(InvalidQuestionError):
        cli._load_questions_file(bad)


# --------------------------------------------------------------------- models
def test_models_list_text(home, capsys):
    assert cli.main(["--home", str(home), "models", "list"]) == 0
    output = _out(capsys)
    for model_id in (
        "gliner2.5-multi",
        "gliner2.5-decide",
        "laya",
        "julia-1",
        "opendecider-nano",
    ):
        assert model_id in output
    assert "d3-lite" not in output
    assert "waitlist" not in output
    assert "fp16*" in output and "fp32" in output
    assert "q4f16*" in output
    assert "default variant" in output


def test_models_list_shows_all_cached_variants(home, fake_snapshot, capsys):
    store = ModelStore(home=home)
    store.download("gliner2.5-multi", variant="fp16")
    store.download("gliner2.5-multi", variant="fp32")
    assert cli.main(["--home", str(home), "models", "list"]) == 0
    output = _out(capsys)
    row = next(line for line in output.splitlines() if "gliner2.5-multi" in line)
    assert "fp16*, fp32" in row
    assert "fp16, fp32" in row


def test_models_list_json_and_columns(home, capsys):
    code = cli.main(
        [
            "--home",
            str(home),
            "models",
            "list",
            "--json",
            "--license",
            "--quant",
            "--adapter",
        ]
    )
    assert code == 0
    payload = json.loads(_out(capsys))
    entry = next(item for item in payload if item["id"] == "gliner2.5-multi")
    assert entry["adapter"] == "gliner2"
    assert entry["license"] == "apache-2.0"
    assert entry["default_variant"] == "fp16"
    assert entry["current_variant"] is None
    assert entry["variants"][0]["quant"] == "fp16"

    assert cli.main(
        ["--home", str(home), "models", "list", "--license", "--quant", "--adapter"]
    ) == 0
    text = _out(capsys)
    assert "license" in text
    assert "quant" in text
    assert "adapter" in text
    assert "fp16*, fp32" in text
    assert "fp16, fp32" in text


def test_models_variants_text_and_json(home, capsys):
    assert cli.main(["--home", str(home), "models", "variants", "gliner2.5-multi"]) == 0
    text = _out(capsys)
    assert "fp16" in text and "fp32" in text
    assert "577.1" in text and "1131.9" in text
    assert "latency p50" in text
    assert "parity:" in text and "baseline=" in text
    assert "download: dtk models download gliner2.5-multi --variant fp32" in text
    assert "selected variant: dtk models use gliner2.5-multi" in text

    assert cli.main(
        ["--home", str(home), "models", "variants", "gliner2.5-multi", "--json"]
    ) == 0
    payload = json.loads(_out(capsys))
    assert payload["adapter"] == "gliner2"
    assert payload["default_variant"] == "fp16"
    assert {variant["name"] for variant in payload["variants"]} == {"fp16", "fp32"}
    assert payload["variants"][0]["revision"]
    assert payload["variants"][0]["sha256"]


def test_models_empty_cache_message(home, capsys):
    assert cli.main(["--home", str(home), "models"]) == 0
    assert "empty" in _out(capsys)
    assert cli.main(["--home", str(home), "models", "--json"]) == 0
    assert json.loads(_out(capsys)) == []


def test_models_use_variant_and_current_json(home, capsys):
    assert cli.main(
        ["--home", str(home), "models", "use", "gliner2.5-multi", "--variant", "fp16"]
    ) == 0
    assert "gliner2.5-multi@fp16" in _out(capsys)
    assert cli.main(["--home", str(home), "models", "current", "--json"]) == 0
    assert json.loads(_out(capsys)) == {
        "current": "gliner2.5-multi",
        "variant": "fp16",
    }
    assert cli.main(["--home", str(home), "models", "current"]) == 0
    assert _out(capsys).strip() == "gliner2.5-multi@fp16"
    assert cli.main(["--home", str(home), "models", "list", "--json"]) == 0
    catalog_entry = next(
        item for item in json.loads(_out(capsys)) if item["id"] == "gliner2.5-multi"
    )
    assert catalog_entry["current"] is True
    assert catalog_entry["current_variant"] == "fp16"

    assert cli.main(["--home", str(home), "models", "use", "julia-1"]) == 0
    capsys.readouterr()
    assert cli.main(["--home", str(home), "models", "current"]) == 0
    assert _out(capsys).strip() == "julia-1"


def test_models_download_dry_run_and_force(home, fake_snapshot, capsys):
    assert cli.main(
        [
            "--home",
            str(home),
            "models",
            "download",
            "gliner2.5-multi",
            "--dry-run",
            "--json",
        ]
    ) == 0
    payload = json.loads(_out(capsys))
    assert payload["repo"] == "jugaadsrl/gliner2.5-multi-v1-onnx"
    assert payload["variant"] == "fp16"
    assert payload["default_variant"] == "fp16"
    assert {item["name"] for item in payload["available_variants"]} == {"fp16", "fp32"}
    assert fake_snapshot == []

    assert cli.main(
        ["--home", str(home), "models", "download", "gliner2.5-multi"]
    ) == 0
    capsys.readouterr()
    assert cli.main(
        ["--home", str(home), "models", "download", "gliner2.5-multi", "--force"]
    ) == 0
    assert len(fake_snapshot) == 2


def test_models_delete_all_variants(home, fake_snapshot, capsys):
    store = ModelStore(home=home)
    store.download("gliner2.5-multi", variant="fp16")
    store.download("gliner2.5-multi", variant="fp32")
    assert cli.main(
        [
            "--home",
            str(home),
            "models",
            "delete",
            "gliner2.5-multi",
            "--all-variants",
            "--yes",
            "--json",
        ]
    ) == 0
    payload = json.loads(_out(capsys))
    assert set(payload["deleted"]) == {"fp16", "fp32"}
    assert store.entries() == []


def test_models_verify_all_json_and_exit_code(home, fake_snapshot, monkeypatch, capsys):
    ModelStore(home=home).download("gliner2.5-multi")
    monkeypatch.setattr(
        ModelStore,
        "verify",
        lambda self, identifier, variant=None: {
            "identifier": identifier,
            "variant": variant,
            "ok": True,
        },
    )
    assert cli.main(["--home", str(home), "models", "verify", "--all", "--json"]) == 0
    payload = json.loads(_out(capsys))
    assert payload["ok"] is True
    assert payload["reports"][0]["identifier"] == "gliner2.5-multi"

    monkeypatch.setattr(
        ModelStore,
        "verify",
        lambda self, identifier, variant=None: {
            "identifier": identifier,
            "variant": variant,
            "ok": False,
        },
    )
    assert cli.main(["--home", str(home), "models", "verify", "--all", "--json"]) == 3


def test_models_import_with_metadata(home, tmp_path, capsys):
    source = tmp_path / "custom"
    source.mkdir()
    (source / "weights.onnx").write_bytes(b"onnx")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert cli.main(
        [
            "--home",
            str(home),
            "models",
            "import",
            str(source),
            "--name",
            "my-custom",
            "--adapter",
            "julia",
            "--files",
            "model=weights.onnx",
            "--files",
            "tokenizer=tokenizer.json",
            "--json",
        ]
    ) == 0
    payload = json.loads(_out(capsys))
    assert payload["identifier"] == "my-custom"
    assert payload["adapter"] == "julia"
    assert payload["files"]["model"] == "weights.onnx"


def test_models_delete_unknown_fails_cleanly(home, capsys):
    assert cli.main(["--home", str(home), "models", "delete", "nope"]) == 1
    assert "error" in capsys.readouterr().err


# --------------------------------------------------------------------- decide
def test_decide_single_json(fake_cli_model, home, capsys):
    code = cli.main(
        [
            "--home",
            str(home),
            "decide",
            "--state",
            "A state",
            "--q",
            "urgent=noul:Today?",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(_out(capsys))
    assert payload["answers"]["urgent"]["noul"] == 0.7
    assert payload["usage"]["input_tokens"] > 0
    assert len(fake_cli_model) == 1


def test_decide_questions_file(fake_cli_model, home, tmp_path, capsys):
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "questions:\n"
        "  team:\n"
        "    type: choice\n"
        "    instructions: Which team?\n"
        "    criteria: {billing: refunds, tech: bugs}\n",
        encoding="utf-8",
    )
    code = cli.main(
        [
            "--home",
            str(home),
            "decide",
            "--state",
            "state",
            "--questions-file",
            str(questions),
            "--json",
        ]
    )
    assert code == 0
    assert json.loads(_out(capsys))["answers"]["team"]["choice"] == "billing"


def test_decide_custom_adapter_and_files(fake_cli_model, home, capsys):
    code = cli.main(
        [
            "--home",
            str(home),
            "decide",
            "--state",
            "state",
            "--q",
            "urgent=noul:Today?",
            "--adapter",
            "julia",
            "--files",
            "model=model.onnx",
            "--files",
            "tokenizer=tokenizer.json",
            "--json",
        ]
    )
    assert code == 0
    _out(capsys)
    _, kwargs = fake_cli_model[0]
    assert kwargs["adapter"] == "julia"
    assert kwargs["files"] == {
        "model": "model.onnx",
        "tokenizer": "tokenizer.json",
    }


def test_decide_batch_jsonl(fake_cli_model, home, tmp_path, capsys):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "urgent": {"type": "noul", "instructions": "Today?"},
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "states.jsonl"
    source.write_text(
        json.dumps("first state") + "\n"
        + json.dumps(
            {
                "state": {"ticket": 2},
                "questions": {"ok": {"type": "noul", "instructions": "Fine?"}},
                "id": "row-2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "out.jsonl"
    code = cli.main(
        [
            "--home",
            str(home),
            "decide",
            "--input-jsonl",
            str(source),
            "--questions-file",
            str(questions),
            "--output-jsonl",
            str(target),
            "--batch-size",
            "1",
        ]
    )
    assert code == 0
    assert len(fake_cli_model) == 1
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert "urgent" in lines[0]["answers"]
    assert lines[1]["id"] == "row-2"
    assert "ok" in lines[1]["answers"]


def test_decide_batch_requires_questions(fake_cli_model, home, tmp_path, capsys):
    source = tmp_path / "states.jsonl"
    source.write_text(json.dumps("state") + "\n", encoding="utf-8")
    code = cli.main(
        ["--home", str(home), "decide", "--input-jsonl", str(source)]
    )
    assert code == 2
    assert "questions" in capsys.readouterr().err


def test_decide_without_model_fails_cleanly(home, capsys):
    code = cli.main(
        ["--home", str(home), "decide", "--state", "x", "--q", "urgent=noul:Today?"]
    )
    assert code == 1
    assert "error" in capsys.readouterr().err


# ------------------------------------------------------------------- classify
def test_classify_json(fake_cli_model, home, capsys):
    code = cli.main(
        [
            "--home",
            str(home),
            "classify",
            "--text",
            "Dentist next Tuesday at 9",
            "--labels",
            "reminder,task,idea",
            "--threshold",
            "0.5",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(_out(capsys))
    assert payload["label"] == "reminder"
    assert payload["confidence"] == pytest.approx(0.9)
    assert payload["usage"]["input_tokens"] > 0


def test_classify_requires_labels(home, capsys):
    code = cli.main(["--home", str(home), "classify", "--text", "x"])
    assert code == 2
    assert "labels" in capsys.readouterr().err


# --------------------------------------------------------------------- config
def test_config_list_get_set(home, capsys):
    assert cli.main(["--home", str(home), "config", "list", "--json"]) == 0
    payload = json.loads(_out(capsys))
    assert payload["current"] is None
    assert payload["current_variant"] is None

    assert cli.main(["--home", str(home), "config", "set", "device", "cpu", "--json"]) == 0
    assert json.loads(_out(capsys))["value"] == "cpu"

    assert cli.main(["--home", str(home), "config", "get", "device", "--json"]) == 0
    assert json.loads(_out(capsys))["value"] == "cpu"

    assert cli.main(
        ["--home", str(home), "config", "set", "current", "gliner2.5-multi", "--json"]
    ) == 0
    assert json.loads(_out(capsys))["value"] == "gliner2.5-multi"

    assert cli.main(
        ["--home", str(home), "config", "set", "current_variant", "fp32", "--json"]
    ) == 0
    assert json.loads(_out(capsys))["value"] == "fp32"

    assert cli.main(["--home", str(home), "config", "set", "home", "/tmp", "--json"]) == 2


# ------------------------------------------------------------------- misc
def test_doctor_json(home, capsys):
    assert cli.main(["--home", str(home), "doctor", "--json"]) == 0
    report = json.loads(_out(capsys))
    assert report["toolkit"]
    assert report["home"] == str(home.resolve())
    assert report["cached_models"] == []
    assert report["adapters"] == [
        "gliner2",
        "gliner2-decide",
        "julia",
        "laya",
        "opendecider",
    ]
    assert report["catalog"]["m0"] == 5
    assert "current_variant" in report


def test_doctor_text(home, capsys):
    assert cli.main(["--home", str(home), "doctor"]) == 0
    output = _out(capsys)
    assert "decision-toolkit" in output
    assert "adapters" in output


def test_version_and_completion(capsys):
    assert cli.main(["version", "--json"]) == 0
    assert json.loads(_out(capsys))["version"]
    assert cli.main(["completion", "bash"]) == 0
    assert "complete" in _out(capsys)


def test_quiet_suppresses_info(fake_cli_model, home, capsys):
    assert cli.main(["--home", str(home), "--quiet", "models", "use", "julia-1"]) == 0
    assert _out(capsys) == ""


def test_verbose_logs_to_stderr(fake_cli_model, home, capsys):
    code = cli.main(
        [
            "--home",
            str(home),
            "--verbose",
            "decide",
            "--state",
            "state",
            "--q",
            "urgent=noul:Today?",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["answers"]
    assert "decision.end" in captured.err


def test_serve_factory_builds_the_configured_app(monkeypatch, home):
    pytest.importorskip("fastapi")
    monkeypatch.setenv(
        "DTK_SERVE_SETTINGS",
        json.dumps(
            {
                "model_id": "gliner2.5-multi",
                "home": str(home),
                "log_level": None,
            }
        ),
    )
    app = cli._serve_factory()
    assert app.state.registry.default_identifier == "gliner2.5-multi"


def test_serve_workers_uses_the_uvicorn_factory(monkeypatch, home):
    pytest.importorskip("fastapi")
    import sys
    import types

    calls: dict = {}

    def run(app, **kwargs):
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=run))
    monkeypatch.setenv("DTK_SERVE_SETTINGS", "")
    code = cli.main(
        ["--home", str(home), "serve", "--workers", "2", "--log-level", "ERROR"]
    )
    assert code == 0
    assert calls["app"] == "decision_toolkit.cli:_serve_factory"
    assert calls["workers"] == 2
    assert "DTK_SERVE_SETTINGS" in os.environ


def test_serve_without_extra_fails_cleanly(home, capsys, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("no uvicorn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = cli.main(["--home", str(home), "serve"])
    assert code == 1
    assert "server" in capsys.readouterr().err
