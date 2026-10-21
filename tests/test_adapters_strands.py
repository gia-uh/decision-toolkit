"""Strands Decider adapter: prompt rendering, pointer positions and readout."""

from __future__ import annotations

import pytest

from decision_toolkit.adapters.strands_decider import StrandsDeciderAdapter

CHOICE_Q = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "refunds", "tech": None},
}
NOUL_Q = {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "today", "false": "later"}}
SCORE_Q = {"type": "score", "instructions": "Tone?", "criteria": ["calm", "annoyed", "furious"]}


class Encoded:
    def __init__(self, text):
        self.ids = list(range(len(text)))
        self.offsets = [(index, index + 1) for index in range(len(text))]


class FakeTokenizer:
    def __init__(self):
        self.calls: list[str] = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        return Encoded(text)


class FakeModelSession:
    def __init__(self, scores):
        self.scores = list(scores)
        self.feeds = None

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        self.feeds = feeds
        count = feeds["option_pos"].shape[1]
        return [numpy.asarray([self.scores[:count]], dtype=numpy.float32)]


class FakeEngine:
    def __init__(self, scores):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModelSession(scores)
        self._config = {"decider": {"max_length": 4096, "temperature": 1.0, "temperature_by_kind": {}}}

    def read_json(self, role):
        assert role == "config"
        return self._config

    def session(self, role):
        assert role == "model"
        return self.model


def test_choice_prompt_and_pointer_positions():
    pytest.importorskip("numpy")
    engine = FakeEngine([2.0, 0.0])
    answers = StrandsDeciderAdapter().system_one(engine, "the state", {"team": CHOICE_Q})
    assert answers["team"]["choice"] == "billing"
    assert answers["team"]["probabilities"]["billing"] == pytest.approx(0.880797, abs=1e-5)
    assert "<state>\nthe state\n</state>" in engine.tokenizer.calls[0]
    assert "<question type=\"choice\">" in engine.tokenizer.calls[0]
    assert engine.model.feeds["option_pos"].shape == (1, 2)
    assert engine.model.feeds["answer_pos"].shape == (1,)


def test_noul_and_score_options():
    pytest.importorskip("numpy")
    noul = StrandsDeciderAdapter().system_one(FakeEngine([0.0, 2.0]), "state", {"urgent": NOUL_Q})
    assert noul["urgent"]["noul"] == pytest.approx(0.880797, abs=1e-5)
    score = StrandsDeciderAdapter().system_one(FakeEngine([0.0, 0.0, 0.0]), "state", {"tone": SCORE_Q})
    assert score["tone"]["score"] == pytest.approx(1.0)
    assert score["tone"]["legend"]["2"] == "furious"
