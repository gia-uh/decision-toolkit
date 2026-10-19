"""The dtk CLI: catalog, variants, cache commands, doctor and error paths."""

from __future__ import annotations

import json

import pytest

from decision_toolkit.cli import _parse_questions, build_parser, main
from decision_toolkit.errors import InvalidQuestionError


def test_parser_has_all_commands():
    parser = build_parser()
    valid = {
        "models": ["models"],
        "decide": ["decide", "--state", "x", "--q", "a=noul:y"],
        "classify": ["classify", "--text", "x", "--labels", "a,b"],
        "serve": ["serve"],
        "doctor": ["doctor"],
    }
    for command, argv in valid.items():
        assert parser.parse_args(argv).command == command


def test_parse_questions_adds_default_instructions():
    questions = _parse_questions(["team=choice:billing=refunds|tech", "urgent=noul:Today?"])
    assert questions["team"]["instructions"] == "Which team fits the state best?"
    assert questions["team"]["criteria"] == {"billing": "refunds", "tech": None}
    assert questions["urgent"] == {"type": "noul", "instructions": "Today?"}


def test_parse_questions_requires_name_spec():
    with pytest.raises(InvalidQuestionError):
        _parse_questions(["team"])


def test_models_list_shows_catalog_and_hides_waitlist(home, capsys):
    assert main(["--home", str(home), "models", "list"]) == 0
    output = capsys.readouterr().out
    assert "gliner2.5-multi" in output
    assert "gliner2" in output
    assert "bosun" not in output
    assert "waitlist hidden" in output


def test_models_list_all_includes_waitlist(home, capsys):
    assert main(["--home", str(home), "models", "list", "--all"]) == 0
    assert "bosun-0.6b" in capsys.readouterr().out


def test_models_variants_lists_quantizations(home, capsys):
    assert main(["--home", str(home), "models", "variants", "gliner2.5-multi"]) == 0
    output = capsys.readouterr().out
    assert "fp16" in output and "fp32" in output
    assert "577.1" in output and "1131.9" in output


def test_models_empty_cache_message(home, capsys):
    assert main(["--home", str(home), "models"]) == 0
    assert "empty" in capsys.readouterr().out


def test_models_use_and_current(home, capsys):
    assert main(["--home", str(home), "models", "use", "julia-1"]) == 0
    capsys.readouterr()
    assert main(["--home", str(home), "models", "current"]) == 0
    assert capsys.readouterr().out.strip() == "julia-1"


def test_models_delete_unknown_fails_cleanly(home, capsys):
    assert main(["--home", str(home), "models", "delete", "nope"]) == 2
    assert "error" in capsys.readouterr().err


def test_decide_without_model_fails_cleanly(home, capsys):
    code = main(["--home", str(home), "decide", "--state", "x", "--q", "urgent=noul:Today?"])
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_classify_requires_labels(home, capsys):
    code = main(["--home", str(home), "classify", "--text", "x"])
    assert code == 2
    assert "labels" in capsys.readouterr().err


def test_doctor_json(home, capsys):
    assert main(["--home", str(home), "doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["toolkit"]
    assert report["home"] == str(home.resolve())
    assert report["cached_models"] == []
    assert report["adapters"] == ["gliner2"]
    assert report["catalog"]["m1"] == 1
    assert "onnxruntime" in report["runtimes"]


def test_doctor_text(home, capsys):
    assert main(["--home", str(home), "doctor"]) == 0
    output = capsys.readouterr().out
    assert "decision-toolkit" in output
    assert "adapters" in output


def test_serve_without_extra_fails_cleanly(home, capsys, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("no uvicorn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = main(["--home", str(home), "serve"])
    assert code == 2
    assert "server" in capsys.readouterr().err
