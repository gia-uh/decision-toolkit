"""Shared fixtures: a fake backend so the fast suite needs no models, network or GPU."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from decision_toolkit.decision_model import DecisionModel


class FakeBackend:
    """Deterministic System One answers built from the questions themselves."""

    name = "fake"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[Any, dict[str, dict[str, Any]]]] = []
        self._error = error

    def system_one(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        self.calls.append((state, {k: dict(v) for k, v in questions.items()}))
        if self._error is not None:
            raise self._error

        answers: dict[str, dict[str, Any]] = {}
        for key, question in questions.items():
            qtype = question["type"]
            if qtype == "choice":
                labels = list((question.get("criteria") or {"first": None, "second": None}))
                tail = 0.1 / max(1, len(labels) - 1)
                answers[key] = {
                    "type": "choice",
                    "choice": labels[0],
                    "probabilities": {label: (0.9 if index == 0 else tail) for index, label in enumerate(labels)},
                    "confidence": 0.9,
                }
            elif qtype == "noul":
                answers[key] = {"type": "noul", "noul": 0.7}
            else:
                answers[key] = {
                    "type": "score",
                    "score": 0.5,
                    "level": 1,
                    "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25},
                    "legend": {"0": "low", "1": "mid", "2": "high"},
                }
        return answers


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def fake_model(fake_backend: FakeBackend) -> DecisionModel:
    return DecisionModel(backend=fake_backend)


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def fake_snapshot_from_models(monkeypatch, home):
    """A ModelStore whose hub download is faked (no network)."""
    import decision_toolkit.models as models_module
    from decision_toolkit.models import ModelStore

    def _fake(repo_id, local_dir, revision=None, allow_patterns=None):
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(models_module, "_snapshot_download", _fake)
    return ModelStore(home=home)
