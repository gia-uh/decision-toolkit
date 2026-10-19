"""Shared fixtures: a fake adapter so the fast suite needs no models or runtime."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from decision_toolkit.decision_model import DecisionModel


class FakeAdapter:
    """Deterministic System One answers built from the questions themselves."""

    name = "fake"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[Any, Any, dict[str, dict[str, Any]]]] = []
        self._error = error

    def system_one(
        self, engine: Any, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        self.calls.append((engine, state, {k: dict(v) for k, v in questions.items()}))
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
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def fake_model(fake_adapter: FakeAdapter) -> DecisionModel:
    return DecisionModel(adapter=fake_adapter)


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def fake_snapshot(monkeypatch):
    """Replace the hub download with a local fake; record its arguments."""
    import decision_toolkit.models as models_module

    calls: list[dict] = []

    def _fake(repo_id, local_dir, revision=None, allow_patterns=None):
        calls.append(
            {"repo_id": repo_id, "local_dir": local_dir, "revision": revision,
             "allow_patterns": allow_patterns}
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        for pattern in allow_patterns or ["model.json"]:
            target = local_dir / pattern
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return str(local_dir)

    monkeypatch.setattr(models_module, "_snapshot_download", _fake)
    return calls
