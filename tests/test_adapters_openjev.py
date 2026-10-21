"""Open-Jev adapter: marker layout, option grouping and readout."""

from __future__ import annotations

import pytest

from decision_toolkit.adapters.openjev import OpenJevAdapter

CHOICE_Q = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "refunds", "tech": None},
}
NOUL_Q = {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "today", "false": "later"}}
SCORE_Q = {"type": "score", "instructions": "Tone?", "criteria": ["calm", "annoyed", "furious"]}


class FakeTokenizer:
    def __init__(self):
        self.calls: list[str] = []
        self._vocab = {"[CLS]": 1, "[SEP]": 2, "[STATE]": 3, "[Q]": 4, "[OPT]": 5}
        self._next = 10

    def token_to_id(self, token):
        return self._vocab.get(token)

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        if text not in self._vocab:
            self._vocab[text] = self._next
            self._next += 1
        return [self._vocab[text]]


class FakeModelSession:
    def __init__(self, logits):
        self.logits = list(logits)
        self.feeds = None

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        self.feeds = feeds
        return [numpy.asarray([self.logits], dtype=numpy.float32)]


class FakeEngine:
    def __init__(self, logits):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModelSession(logits)
        self._config = {"open_jev": {"temperature": 1.0, "max_len": 512, "max_state_tokens": 256}}

    def read_json(self, role):
        assert role == "config"
        return self._config

    def session(self, role):
        assert role == "model"
        return self.model


def test_choice_builds_markers_and_grouped_probabilities():
    pytest.importorskip("numpy")
    engine = FakeEngine([2.0, 0.0])
    answers = OpenJevAdapter().system_one(engine, "the state", {"team": CHOICE_Q})
    assert answers["team"]["choice"] == "billing"
    assert answers["team"]["probabilities"]["billing"] == pytest.approx(0.880797, abs=1e-5)
    assert engine.tokenizer.calls[:3] == ["the state", "Which team?", "billing"]
    assert engine.model.feeds["pair_q"].shape == (1, 2)
    assert engine.model.feeds["seg"].shape == engine.model.feeds["input_ids"].shape


def test_noul_uses_no_yes_order():
    pytest.importorskip("numpy")
    answers = OpenJevAdapter().system_one(FakeEngine([0.0, 2.0]), "state", {"urgent": NOUL_Q})
    assert answers["urgent"]["noul"] == pytest.approx(0.880797, abs=1e-5)
    assert answers["urgent"]["probabilities"]["false"] == pytest.approx(0.119203, abs=1e-5)


def test_score_uses_ordered_levels():
    pytest.importorskip("numpy")
    answers = OpenJevAdapter().system_one(FakeEngine([0.0, 0.0, 0.0]), "state", {"tone": SCORE_Q})
    assert answers["tone"]["score"] == pytest.approx(1.0)
    assert answers["tone"]["legend"]["2"] == "furious"
