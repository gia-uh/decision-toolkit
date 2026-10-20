"""Shared fixtures: a fake adapter so the fast suite needs no models or runtime."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

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
                labels = list(question.get("criteria") or {"first": None, "second": None})
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
                labels = list(question.get("criteria") or ["low", "mid", "high"])
                probabilities = [0.25, 0.5, 0.25] if len(labels) == 3 else [
                    1.0 / len(labels)
                ] * len(labels)
                answers[key] = {
                    "type": "score",
                    "score": sum(
                        index * probability for index, probability in enumerate(probabilities)
                    ),
                    "level": max(range(len(labels)), key=probabilities.__getitem__),
                    "probabilities": {
                        str(index): probability
                        for index, probability in enumerate(probabilities)
                    },
                    "legend": {str(index): label for index, label in enumerate(labels)},
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
    import dataclasses

    import decision_toolkit.models as models_module
    from decision_toolkit import catalog

    # Fake downloads write arbitrary bytes; remove the real pinned hashes for
    # the duration of this fixture so verification tests exercise the cache
    # metadata rather than real model artifacts.
    patched = tuple(
        dataclasses.replace(
            model,
            variants=tuple(
                dataclasses.replace(variant, sha256={}, extra_sha256={})
                for variant in model.variants
            ),
        )
        for model in catalog.CATALOG
    )
    monkeypatch.setattr(catalog, "CATALOG", patched)
    monkeypatch.setattr(catalog, "CATALOG_BY_ID", {model.id: model for model in patched})

    # Fake files are not valid ONNX graphs; make verify report the session
    # check as unavailable rather than failing on a synthetic artifact.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
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
