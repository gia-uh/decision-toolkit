"""Fused GLiNER2.5-Decide adapter: [L] schema and single-graph readout."""

from __future__ import annotations

import pytest

from decision_toolkit.adapters.gliner2_decide import Gliner2DecideAdapter
from decision_toolkit.errors import AdapterError

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
        self._vocab: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        if text not in self._vocab:
            self._vocab[text] = len(self._vocab) + 10
        return [self._vocab[text]]


class FakeFusedSession:
    def __init__(self, logits):
        self.logits = list(logits)
        self.feeds = None

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        self.feeds = feeds
        markers = len(feeds["marker_positions"][0])
        return [numpy.asarray([self.logits[:markers]], dtype=numpy.float32)]


class FakeEngine:
    def __init__(self, logits):
        self.tokenizer = FakeTokenizer()
        self.model = FakeFusedSession(logits)
        self.files = {"model": "model.onnx", "tokenizer": "tokenizer.json"}

    def session(self, role):
        assert role == "model"
        return self.model


def test_choice_schema_uses_L_markers_and_sep_struct():
    pytest.importorskip("numpy")
    engine = FakeEngine([2.0, 0.0])
    answers = Gliner2DecideAdapter().system_one(engine, "the state", {"team": CHOICE_Q})

    team = answers["team"]
    assert team["choice"] == "billing"
    assert team["probabilities"]["billing"] == pytest.approx(0.880797, abs=1e-5)
    calls = engine.tokenizer.calls
    assert calls[:5] == ["(", "[P]", "Which team? [DESCRIPTION] billing: refunds", "[L]", "billing"]
    assert "[SEP_STRUCT]" in calls and "[SEP_TEXT]" in calls
    assert len(engine.model.feeds["marker_positions"][0]) == 2


def test_noul_maps_yes_no():
    pytest.importorskip("numpy")
    answers = Gliner2DecideAdapter().system_one(FakeEngine([2.0, 0.0]), "state", {"urgent": NOUL_Q})
    urgent = answers["urgent"]
    assert urgent["noul"] == pytest.approx(0.880797, abs=1e-5)
    assert urgent["probabilities"]["false"] == pytest.approx(0.119203, abs=1e-5)


def test_score_levels():
    pytest.importorskip("numpy")
    answers = Gliner2DecideAdapter().system_one(FakeEngine([0.0, 0.0, 0.0]), "state", {"tone": SCORE_Q})
    assert answers["tone"]["score"] == pytest.approx(1.0)
    assert answers["tone"]["legend"]["2"] == "furious"


def test_mismatch_is_explained():
    pytest.importorskip("numpy")
    with pytest.raises(AdapterError):
        Gliner2DecideAdapter().system_one(FakeEngine([1.0]), "state", {"team": CHOICE_Q})
