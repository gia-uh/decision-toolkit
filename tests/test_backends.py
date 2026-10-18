"""Backends: normalization with injected runtimes + missing-runtime errors."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from decision_toolkit.backends.julia import JuliaBackend
from decision_toolkit.backends.laya import LayaBackend
from decision_toolkit.backends.systemone import SystemOneBackend
from decision_toolkit.backends.von import VonBackend
from decision_toolkit.errors import BackendError, BackendNotAvailableError
from decision_toolkit.models import ResolvedModel

RESOLVED = ResolvedModel("x", None, "org/name", None, None, "catalog")
QUESTIONS = {
    "team": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}},
    "urgent": {"type": "noul", "instructions": "?"},
    "tone": {"type": "score", "instructions": "?", "criteria": ["low", "high"]},
}


# ------------------------------------------------------------------- Laya
class FakeLayaAgent:
    def __init__(self):
        self.seen: list[tuple[str, dict]] = []

    def predict(self, state, questions):
        self.seen.append((state, questions))
        return {"answers": {"urgent": {"type": "noul", "noul": 0.8}}}


def test_laya_backend_normalizes_and_serializes_state():
    agent = FakeLayaAgent()
    backend = LayaBackend(RESOLVED, agent=agent)
    answers = backend.system_one({"ticket": "x"}, QUESTIONS)
    assert answers["urgent"]["noul"] == 0.8
    state, questions = agent.seen[0]
    assert state == '{"ticket": "x"}'  # JSON, sorted keys
    assert questions == QUESTIONS


def test_laya_backend_bad_response_shape():
    backend = LayaBackend(RESOLVED, agent=SimpleNamespace(predict=lambda s, q: {"nope": 1}))
    with pytest.raises(BackendError):
        backend.system_one("state", QUESTIONS)


def test_laya_backend_missing_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)
    backend = LayaBackend(RESOLVED)
    with pytest.raises(BackendNotAvailableError) as excinfo:
        backend.system_one("state", QUESTIONS)
    assert "decision-toolkit[laya]" in str(excinfo.value)


# ------------------------------------------------------------------- Julia
class FakeJuliaEngine:
    def __init__(self):
        self.calls: list[dict] = []

    def predict(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        return {"answers": {"team": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}}}}


def test_julia_backend_normalizes():
    engine = FakeJuliaEngine()
    backend = JuliaBackend(RESOLVED, engine=engine)
    answers = backend.system_one("state", QUESTIONS)
    assert answers["team"]["choice"] == "a"
    assert engine.calls[0]["questions"] == QUESTIONS


def test_julia_backend_missing_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "julia", None)
    with pytest.raises(BackendNotAvailableError) as excinfo:
        JuliaBackend(RESOLVED).system_one("state", QUESTIONS)
    assert "Julia 1" in str(excinfo.value)


def test_julia_backend_requires_local_directory(monkeypatch):
    fake_runtime = SimpleNamespace(load_model=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "julia", fake_runtime)
    backend = JuliaBackend(RESOLVED)  # RESOLVED has no local path
    with pytest.raises(BackendNotAvailableError) as excinfo:
        backend.system_one("state", QUESTIONS)
    assert "local model directory" in str(excinfo.value)


# --------------------------------------------------------------------- Von
class FakeVonSdk:
    def system_one(self, state, questions):
        return SimpleNamespace(
            answers={
                "team": SimpleNamespace(choice="a", probabilities={"a": 0.9, "b": 0.1}, confidence=0.9),
                "urgent": SimpleNamespace(noul=0.61),
                "tone": SimpleNamespace(score=1.2, level=1, legend={"0": "low", "1": "high"},
                                        probabilities={"0": 0.2, "1": 0.8}),
            }
        )


def test_von_backend_normalizes_rich_objects():
    backend = VonBackend(RESOLVED, sdk=FakeVonSdk())
    answers = backend.system_one("state", QUESTIONS)
    assert answers["team"]["choice"] == "a"
    assert answers["team"]["probabilities"] == {"a": 0.9, "b": 0.1}
    assert answers["urgent"]["noul"] == 0.61
    assert answers["tone"]["score"] == 1.2
    assert answers["tone"]["level"] == 1
    assert answers["tone"]["legend"] == {"0": "low", "1": "high"}


def test_von_backend_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "von", None)
    with pytest.raises(BackendNotAvailableError):
        VonBackend(RESOLVED).system_one("state", QUESTIONS)


# ---------------------------------------------------------------- SystemOne
def test_systemone_backend_requires_base_url():
    with pytest.raises(BackendNotAvailableError):
        SystemOneBackend(RESOLVED)


def test_systemone_backend_url_normalization():
    backend = SystemOneBackend(RESOLVED, base_url="http://localhost:8000/")
    assert backend.url == "http://localhost:8000/v1/systemone"
