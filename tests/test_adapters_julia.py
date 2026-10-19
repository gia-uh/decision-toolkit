"""Julia-1 adapter: one graph, sequence construction and answers (fake engine)."""

from __future__ import annotations

import pytest

from decision_toolkit.adapters.julia import JuliaAdapter
from decision_toolkit.errors import AdapterError

CHOICE_Q = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "refunds", "tech": None},
}
NOUL_Q = {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "today", "false": "later"}}
SCORE_Q = {"type": "score", "instructions": "Tone?", "criteria": ["calm", "annoyed", "furious"]}


class FakeTokenizer:
    def __init__(self, special=True):
        self.calls: list[str] = []
        self._vocab: dict[str, int] = {"<eos>": 1, "<bos>": 2, "<mask>": 4}
        self._special = special

    def token_to_id(self, token):
        return self._vocab.get(token) if self._special else None

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        if text not in self._vocab:
            self._vocab[text] = len(self._vocab) + 10
        return [self._vocab[text]]


class FakeModelSession:
    def __init__(self, logits):
        self.logits = list(logits)
        self.feeds = None

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        self.feeds = feeds
        markers = len(feeds["marker_pos"][0])
        return [numpy.asarray([self.logits[:markers]], dtype=numpy.float32)]


class FakeEngine:
    def __init__(self, logits, special=True):
        self.tokenizer = FakeTokenizer(special=special)
        self.model = FakeModelSession(logits)

    def session(self, role):
        assert role == "model"
        return self.model


def test_choice_sequence_and_probabilities():
    pytest.importorskip("numpy")
    engine = FakeEngine([2.0, 0.0])
    answers = JuliaAdapter().system_one(engine, "the state", {"team": CHOICE_Q})

    team = answers["team"]
    assert team["choice"] == "billing"
    assert team["probabilities"]["billing"] == pytest.approx(0.880797, abs=1e-5)
    assert team["confidence"] == pytest.approx(0.880797, abs=1e-5)

    calls = engine.tokenizer.calls
    assert calls[0] == "choice question: Which team?"
    assert calls[1] == " billing: refunds" and calls[2] == " tech"
    assert engine.model.feeds["qtype"].shape == (1,)
    assert engine.model.feeds["qtype"][0] == 0
    assert len(engine.model.feeds["marker_pos"][0]) == 2


def test_noul_false_then_true():
    pytest.importorskip("numpy")
    answers = JuliaAdapter().system_one(FakeEngine([0.0, 2.0]), "state", {"urgent": NOUL_Q})
    urgent = answers["urgent"]
    assert urgent["noul"] == pytest.approx(0.880797, abs=1e-5)
    assert urgent["probabilities"]["false"] == pytest.approx(0.119203, abs=1e-5)


def test_score_levels_are_rendered_as_given():
    pytest.importorskip("numpy")
    engine = FakeEngine([0.0, 0.0, 0.0])
    answers = JuliaAdapter().system_one(engine, "state", {"tone": SCORE_Q})
    assert answers["tone"]["score"] == pytest.approx(1.0)
    assert answers["tone"]["legend"]["2"] == "furious"
    assert engine.model.feeds["qtype"][0] == 1
    assert " calm" in engine.tokenizer.calls


def test_json_state_is_serialized():
    pytest.importorskip("numpy")
    engine = FakeEngine([1.0, 0.0])
    JuliaAdapter().system_one(engine, {"ticket": 42}, {"team": CHOICE_Q})
    assert '{"ticket": 42}' in engine.tokenizer.calls


def test_missing_special_tokens_is_explained():
    pytest.importorskip("numpy")
    with pytest.raises(AdapterError) as excinfo:
        JuliaAdapter().system_one(FakeEngine([0.0, 1.0], special=False), "state", {"team": CHOICE_Q})
    assert "tokenizer" in str(excinfo.value)


def test_max_length_and_head_length_are_configurable():
    adapter = JuliaAdapter(max_length=128, head_length=64)
    assert adapter.max_length == 128
    assert adapter.head_length == 64
