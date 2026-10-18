"""Integration: a real Laya checkpoint, on CPU (opt-in).

Run with an environment that has the Laya runtime installed:

    uv run pytest -m integration -o addopts=""

The model is read from the HuggingFace cache (downloaded once by ``laya``
itself). This test never uses the network if the checkpoint is already cached.
"""

from __future__ import annotations

import pytest

from decision_toolkit import DecisionModel, choice, noul, score

pytestmark = pytest.mark.integration


def test_laya_multilingual_roundtrip(tmp_path):
    pytest.importorskip("laya")

    model = DecisionModel("laya-multilingual", home=tmp_path / "home")
    answer = model.decide(
        state="I was charged twice for the March invoice and nobody has replied.",
        questions={
            "team": choice("Which team should handle this?", {
                "billing": "charges, invoices, refunds",
                "technical": "bugs and outages",
                "other": "everything else",
            }),
            "urgent": noul("Does the customer need a reply today?"),
            "tone": score("How upset is the customer?", ["calm", "annoyed", "furious"]),
        },
    )

    team = answer["team"]
    assert team.choice in {"billing", "technical", "other"}
    assert set(team.probabilities) == {"billing", "technical", "other"}
    assert abs(sum(team.probabilities.values()) - 1.0) < 1e-3
    assert 0.0 <= team.confidence <= 1.0

    urgent = answer["urgent"]
    assert urgent.noul is not None and 0.0 <= urgent.noul <= 1.0

    tone = answer["tone"]
    assert tone.score is not None and 0.0 <= tone.score <= 2.0
    print(f"\nteam={team.choice} ({team.confidence:.2f}) urgent={urgent.noul:.2f} tone={tone.score:.2f}")


def test_laya_classify_threshold(tmp_path):
    pytest.importorskip("laya")

    model = DecisionModel("laya-multilingual", home=tmp_path / "home")
    result = model.classify(
        "Remind me: dentist next Tuesday at 9 am.",
        {
            "reminder": "an event with a date or time",
            "task": "a list of things to do",
            "idea": "an idea or reflection",
        },
        threshold=0.5,
    )
    assert result.label in {"reminder", "task", "idea"}
    assert 0.0 <= result.confidence <= 1.0
    print(f"\nlabel={result.label} confidence={result.confidence:.2f}")
