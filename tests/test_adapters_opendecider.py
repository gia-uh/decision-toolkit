"""OpenDecider adapter: marker prompt, readout and answers (fake engine)."""

from __future__ import annotations

import pytest

from decision_toolkit.adapters.opendecider import OpendeciderAdapter
from decision_toolkit.errors import AdapterError

CHOICE_Q = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "refunds", "tech": None},
}
NOUL_Q = {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "today", "false": "later"}}
SCORE_Q = {"type": "score", "instructions": "Tone?", "criteria": ["calm", "annoyed", "furious"]}

CONFIG = {"max_len": 128, "special_ids": {"cls": 2, "sep": 1, "mask": 4}}


class FakeTokenizer:
    def __init__(self):
        self.calls: list[str] = []
        self._vocab: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        if text not in self._vocab:
            self._vocab[text] = len(self._vocab) + 10
        return [self._vocab[text]]


class FakeModelSession:
    def __init__(self, scores, mask_id=4):
        self.scores = list(scores)
        self.mask_id = mask_id
        self.feeds = None

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        self.feeds = feeds
        ids = feeds["input_ids"][0]
        logits = numpy.zeros((1, len(ids)), dtype=numpy.float32)
        positions = [index for index, token in enumerate(ids) if token == self.mask_id]
        for score, position in zip(self.scores, positions):
            logits[0, position] = score
        return [logits]


class FakeEngine:
    def __init__(self, scores, config=None):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModelSession(scores)
        self._config = dict(config or CONFIG)

    def read_json(self, role):
        assert role == "config"
        return self._config

    def session(self, role):
        assert role == "model"
        return self.model


def test_choice_sequence_and_probabilities():
    pytest.importorskip("numpy")
    engine = FakeEngine([2.0, 0.0])
    answers = OpendeciderAdapter().system_one(engine, "the state", {"team": CHOICE_Q})

    team = answers["team"]
    assert team["choice"] == "billing"
    assert team["probabilities"]["billing"] == pytest.approx(0.880797, abs=1e-5)
    calls = engine.tokenizer.calls
    assert calls[0] == "question: Which team?"
    assert calls[1] == " billing: refunds" and calls[2] == " tech"
    assert calls[-1] == "input: the state"
    assert len(engine.model.feeds["input_ids"][0]) > 0


def test_noul_false_then_true():
    pytest.importorskip("numpy")
    answers = OpendeciderAdapter().system_one(FakeEngine([0.0, 2.0]), "state", {"urgent": NOUL_Q})
    urgent = answers["urgent"]
    assert urgent["noul"] == pytest.approx(0.880797, abs=1e-5)
    assert urgent["probabilities"]["false"] == pytest.approx(0.119203, abs=1e-5)


def test_score_levels():
    pytest.importorskip("numpy")
    answers = OpendeciderAdapter().system_one(FakeEngine([0.0, 0.0, 0.0]), "state", {"tone": SCORE_Q})
    assert answers["tone"]["score"] == pytest.approx(1.0)
    assert answers["tone"]["legend"]["2"] == "furious"


def test_json_state_is_serialized():
    pytest.importorskip("numpy")
    engine = FakeEngine([1.0, 0.0])
    OpendeciderAdapter().system_one(engine, {"ticket": 42}, {"team": CHOICE_Q})
    assert 'input: {"ticket": 42}' in engine.tokenizer.calls


def test_missing_special_ids_is_explained():
    with pytest.raises(AdapterError):
        OpendeciderAdapter()._special_ids({})
