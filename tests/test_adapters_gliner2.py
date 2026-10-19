"""GLiNER2 adapter: schema construction, readout and answers (fake engine)."""

from __future__ import annotations

import pytest

from decision_toolkit.adapters.gliner2 import Gliner2Adapter
from decision_toolkit.errors import InvalidQuestionError

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
            self._vocab[text] = len(self._vocab) + 1
        return [self._vocab[text]]


class FakeEncoderSession:
    def __init__(self):
        self.feeds = None

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        ids = numpy.asarray(feeds["input_ids"])
        self.feeds = ids
        return [numpy.zeros((1, ids.shape[1], 4), dtype=numpy.float32)]


class FakeClassifierSession:
    def __init__(self, logits, input_name="hidden_state"):
        self.logits = list(logits)
        self.input_name = input_name
        self.queries = None

    def get_inputs(self):
        return [type("Input", (), {"name": self.input_name})()]

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        self.queries = feeds[self.input_name]
        return [numpy.asarray(self.logits, dtype=numpy.float32).reshape(-1)]


class FakeEngine:
    def __init__(self, logits, tokenizer=None, classifier_input="hidden_state"):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.encoder = FakeEncoderSession()
        self.classifier = FakeClassifierSession(logits, input_name=classifier_input)
        self.files = {"encoder": "encoder.onnx", "classifier": "classifier.onnx",
                      "tokenizer": "tokenizer.json"}

    def session(self, role):
        assert role in {"encoder", "classifier"}
        return self.encoder if role == "encoder" else self.classifier


class FakeFusedSession:
    def __init__(self, logits):
        self.logits = list(logits)
        self.feeds = None

    def get_inputs(self):
        def item(name, shape):
            return type("Input", (), {"name": name, "shape": shape})()

        return [
            item("input_ids", [1, "L"]),
            item("attention_mask", [1, "L"]),
            item("marker_positions", [1, "K"]),
        ]

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        self.feeds = feeds
        markers = len(feeds["marker_positions"][0])
        return [numpy.asarray([self.logits[:markers]], dtype=numpy.float32)]


class FakeFusedEngine:
    def __init__(self, logits):
        self.tokenizer = FakeTokenizer()
        self.model = FakeFusedSession(logits)
        self.files = {"model": "model.onnx", "tokenizer": "tokenizer.json"}

    def session(self, role):
        assert role == "model"
        return self.model


def test_choice_schema_and_probabilities():
    pytest.importorskip("numpy")
    engine = FakeEngine([3.0, 1.0])
    adapter = Gliner2Adapter()
    answers = adapter.system_one(engine, "the state", {"team": CHOICE_Q})

    team = answers["team"]
    assert team["choice"] == "billing"
    assert team["probabilities"]["billing"] == pytest.approx(0.880797, abs=1e-5)
    assert team["confidence"] == pytest.approx(0.880797, abs=1e-5)

    calls = engine.tokenizer.calls
    # schema (cached repeats omitted): ( [P] parent [C] billing [C] tech ) [SEP_TEXT] words
    assert calls[:5] == ["(", "[P]", "classify [DESCRIPTION] billing: refunds", "[C]", "billing"]
    assert "[SEP_TEXT]" in calls
    assert "the" in calls and "state" in calls


def test_noul_and_score_mapping():
    pytest.importorskip("numpy")
    adapter = Gliner2Adapter()
    answers = adapter.system_one(FakeEngine([0.2, 0.8]), "state", {"urgent": NOUL_Q})
    assert answers["urgent"]["noul"] == pytest.approx(0.354344, abs=1e-5)

    answers = adapter.system_one(FakeEngine([0.0, 0.0, 0.0]), "state", {"tone": SCORE_Q})
    assert answers["tone"]["score"] == pytest.approx(1.0)
    assert answers["tone"]["legend"]["2"] == "furious"


def test_task_override_from_question():
    pytest.importorskip("numpy")
    engine = FakeEngine([1.0, 0.0])
    adapter = Gliner2Adapter()
    adapter.system_one(engine, "state", {"team": {**CHOICE_Q, "task": "categoria"}})
    assert "categoria [DESCRIPTION] billing: refunds" in engine.tokenizer.calls


def test_descriptions_can_be_disabled():
    pytest.importorskip("numpy")
    engine = FakeEngine([1.0])
    adapter = Gliner2Adapter(use_descriptions=False)
    adapter.system_one(engine, "state", {"one": {"type": "choice", "instructions": "?",
                                                 "criteria": {"billing": "refunds"}}})
    assert "classify" in engine.tokenizer.calls
    assert not any("[DESCRIPTION]" in call for call in engine.tokenizer.calls)


def test_choice_without_criteria_is_rejected():
    adapter = Gliner2Adapter()
    with pytest.raises(InvalidQuestionError):
        adapter.system_one(FakeEngine([]), "state", {"free": {"type": "choice", "instructions": "?"}})


def test_option_count_mismatch_is_explained():
    pytest.importorskip("numpy")
    adapter = Gliner2Adapter()
    with pytest.raises(Exception) as excinfo:
        adapter.system_one(FakeEngine([1.0]), "state", {"team": CHOICE_Q})
    assert "1 scores" in str(excinfo.value)


def test_classifier_input_name_is_read_from_the_graph():
    pytest.importorskip("numpy")
    engine = FakeEngine([2.0, 0.0], classifier_input="choice_states")
    adapter = Gliner2Adapter()
    answers = adapter.system_one(engine, "state", {"team": CHOICE_Q})
    assert answers["team"]["choice"] == "billing"
    assert engine.classifier.queries.shape[0] == 2


def test_fused_graph_uses_marker_positions():
    pytest.importorskip("numpy")
    engine = FakeFusedEngine([2.0, 0.0])
    adapter = Gliner2Adapter()
    answers = adapter.system_one(engine, "the state", {"team": CHOICE_Q})
    assert answers["team"]["choice"] == "billing"
    assert answers["team"]["probabilities"]["billing"] == pytest.approx(0.880797, abs=1e-5)
    assert engine.model.feeds["marker_positions"].shape[1] == 2
    assert engine.model.feeds["input_ids"].shape[0] == 1
