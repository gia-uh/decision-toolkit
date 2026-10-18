"""DecisionModel: decide/classify over an injected backend, resolution and errors."""

from __future__ import annotations

import pytest

from decision_toolkit.decision_model import DecisionModel
from decision_toolkit.errors import BackendNotAvailableError, ModelNotFoundError
from decision_toolkit.models import ResolvedModel
from decision_toolkit.schema import choice, noul, score

from .conftest import FakeBackend


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


def test_decide_sends_normalized_questions(fake_backend, fake_model):
    fake_model.decide("state", {"urgent": noul("Urgent?")})
    state, questions = fake_backend.calls[0]
    assert state == "state"
    assert questions == {"urgent": {"type": "noul", "instructions": "Urgent?"}}


def test_decide_rejects_bad_questions(fake_model):
    from decision_toolkit.errors import InvalidQuestionError

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


def test_planned_backend_is_catalogued_but_not_runnable(home):
    model = DecisionModel("thx-01", home=home)
    with pytest.raises(BackendNotAvailableError) as excinfo:
        model.decide("state", {"q": noul("x")})
    assert "M1" in str(excinfo.value)


def test_systemone_backend_requires_base_url(home):
    with pytest.raises(BackendNotAvailableError):
        DecisionModel(backend="systemone", home=home).decide("state", {"q": noul("x")})


def test_systemone_backend_posts_and_normalizes(home, monkeypatch):
    calls = []

    def fake_post(url, payload, headers=None, timeout=120):
        calls.append((url, payload, headers))
        return {"answers": {"q": {"type": "noul", "noul": 0.42}}}

    monkeypatch.setattr("decision_toolkit.backends.systemone._post_json", fake_post)
    model = DecisionModel(
        backend="systemone",
        backend_options={"base_url": "http://localhost:9999/", "api_key": "secret"},
        home=home,
    )
    decision = model.decide("state", {"q": noul("x")})
    assert decision["q"].noul == 0.42
    url, payload, headers = calls[0]
    assert url == "http://localhost:9999/v1/systemone"
    assert payload["state"] == "state"
    assert headers == {"Authorization": "Bearer secret"}


def test_systemone_backend_raises_on_missing_answers(home, monkeypatch):
    monkeypatch.setattr(
        "decision_toolkit.backends.systemone._post_json",
        lambda url, payload, headers=None, timeout=120: {"oops": True},
    )
    model = DecisionModel(backend="systemone", backend_options={"base_url": "http://x"}, home=home)
    from decision_toolkit.errors import BackendError

    with pytest.raises(BackendError):
        model.decide("state", {"q": noul("x")})


def test_default_uses_model_in_use(home, fake_snapshot_from_models):
    store = fake_snapshot_from_models
    store.download("laya")
    store.use("laya")
    model = DecisionModel(backend=FakeBackend(), home=home)
    assert model.identifier == "laya"


def test_identifier_without_anything(home):
    model = DecisionModel(backend=FakeBackend(), home=home)
    assert model.identifier == "(none)"


def test_resolved_exposes_catalog(home):
    model = DecisionModel("julia-1", home=home)
    assert model.resolved.catalog.id == "julia-1"
    assert model.backend_name == "julia"


def test_close_releases_backend(fake_model):
    fake_model.warmup()  # no-op for the fake backend, exercises the hook
    fake_model.close()
    assert fake_model._backend is None  # noqa: SLF001 - white-box lifecycle check


def test_resolved_model_dataclass_is_frozen():
    resolved = ResolvedModel("x", None, "org/name", None, None, "catalog")
    with pytest.raises(Exception):
        resolved.identifier = "y"  # type: ignore[misc]
