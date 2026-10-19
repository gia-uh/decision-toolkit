"""DecisionModel: decide/classify with an injected adapter, resolution and errors."""

from __future__ import annotations

import pytest

from decision_toolkit.decision_model import DecisionModel
from decision_toolkit.errors import (
    AdapterNotAvailableError,
    InvalidQuestionError,
    ModelNotFoundError,
)
from decision_toolkit.schema import choice, noul, score

from .conftest import FakeAdapter


def test_decide_returns_typed_answers(fake_model):
    decision = fake_model.decide(
        "A state",
        {
            "team": choice("Which team?", {"billing": "refunds", "tech": "bugs"}),
            "urgent": noul("Urgent?"),
            "tone": score("Tone?", ["calm", "annoyed", "furious"]),
        },
    )
    assert set(decision.keys()) == {"team", "urgent", "tone"}
    assert decision["team"].choice == "billing"
    assert decision["team"].confidence == 0.9
    assert decision["urgent"].noul == 0.7
    assert decision["tone"].score == 0.5
    assert decision["tone"].level == 1


def test_decide_sends_normalized_questions(fake_adapter, fake_model):
    fake_model.decide("state", {"urgent": noul("Urgent?")})
    _, state, questions = fake_adapter.calls[0]
    assert state == "state"
    assert questions == {"urgent": {"type": "noul", "instructions": "Urgent?"}}


def test_decide_rejects_bad_questions(fake_model):
    with pytest.raises(InvalidQuestionError):
        fake_model.decide("state", {})


def test_classify_mapping_with_threshold(fake_model):
    result = fake_model.classify(
        "Dentist next Tuesday at 9",
        {"reminder": "an event with a date", "task": "things to do", "idea": "an idea"},
        threshold=0.95,
    )
    assert result.label == "reminder"
    assert result.confidence == pytest.approx(0.9)
    assert result.below_threshold is True
    assert result.ranked(2)[0][0] == "reminder"


def test_classify_sequence_labels(fake_model):
    result = fake_model.classify("text", ["alpha", "beta"], threshold=0.5)
    assert result.label == "alpha"
    assert result.below_threshold is False


def test_classify_rejects_empty_inputs(fake_model):
    with pytest.raises(ValueError):
        fake_model.classify("   ", {"a": "b"})
    with pytest.raises(ValueError):
        fake_model.classify("text", [])


def test_unknown_model_raises(home):
    with pytest.raises(ModelNotFoundError):
        DecisionModel("nope", home=home).decide("state", {"q": noul("x")})


def test_catalog_model_not_downloaded_explains_the_command(home):
    model = DecisionModel("gliner2.5-multi", home=home)
    with pytest.raises(AdapterNotAvailableError) as excinfo:
        model.decide("state", {"q": noul("x")})
    assert "dtk models download gliner2.5-multi" in str(excinfo.value)


def test_wrong_variant_is_rejected(home):
    with pytest.raises(ModelNotFoundError):
        DecisionModel("gliner2.5-multi", variant="q4", home=home).decide("state", {"q": noul("x")})


def test_identifier_without_anything(home):
    model = DecisionModel(adapter=FakeAdapter(), home=home)
    assert model.identifier == "(none)"


def test_adapter_name_from_instance_and_catalog(fake_model, home):
    assert fake_model.adapter_name == "fake"
    assert DecisionModel("gliner2.5-multi", home=home).adapter_name == "gliner2"


def test_resolved_exposes_catalog_and_variant(home):
    model = DecisionModel("gliner2.5-multi", home=home)
    assert model.resolved.catalog.id == "gliner2.5-multi"
    assert model.resolved.variant.name == "fp16"


def test_custom_path_requires_files(home, tmp_path):
    model = DecisionModel(str(tmp_path), adapter="gliner2", home=home)
    with pytest.raises(AdapterNotAvailableError) as excinfo:
        model.decide("state", {"q": noul("x")})
    assert "files=" in str(excinfo.value)


def test_close_releases_backend(fake_model):
    fake_model.warmup()
    fake_model.close()
    assert fake_model._adapter is None  # noqa: SLF001 - white-box lifecycle check
