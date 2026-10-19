"""Laya adapter: sequence construction, readout and answers (fake engine)."""

from __future__ import annotations

import pytest

from decision_toolkit.adapters.laya import LayaAdapter
from decision_toolkit.errors import AdapterError

CHOICE_Q = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "refunds", "tech": None},
}
NOUL_Q = {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "today", "false": "later"}}
SCORE_Q = {"type": "score", "instructions": "Tone?", "criteria": ["calm", "annoyed", "furious"]}

CONFIG = {
    "max_len": 1024,
    "head_max_len": 256,
    "temperature": [1.0, 1.0, 1.0],
    "temperature_by_options": {},
}


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


class FakeEncoderSession:
    def __init__(self):
        self.feeds = None

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        ids = numpy.asarray(feeds["input_ids"])
        self.feeds = ids
        return [numpy.zeros((1, ids.shape[1], 8), dtype=numpy.float32)]


class FakeHeadSession:
    def __init__(self, logits, hidden_name="hidden_states", qtype_1d=False):
        self.logits = list(logits)
        self.hidden_name = hidden_name
        self.qtype_1d = qtype_1d
        self.feeds = None

    def get_inputs(self):
        def item(name, shape):
            return type("Input", (), {"name": name, "shape": shape})()

        return [
            item(self.hidden_name, [1, "L", 8]),
            item("attention_mask", [1, "L"]),
            item("marker_pos", [1, "K"]),
            item("marker_mask", [1, "K"]),
            item("qtype", [1] if self.qtype_1d else [1, 1]),
        ]

    def run(self, output_names, feeds):
        numpy = pytest.importorskip("numpy")
        self.feeds = feeds
        markers = len(feeds["marker_pos"][0])
        return [
            numpy.asarray([self.logits[:markers]], dtype=numpy.float32),
            numpy.asarray([[1.0, 0.0]], dtype=numpy.float32),
        ]


class FakeEngine:
    def __init__(self, logits, special=True, config=None, hidden_name="hidden_states",
                 qtype_1d=False):
        self.tokenizer = FakeTokenizer(special=special)
        self.encoder = FakeEncoderSession()
        self.head = FakeHeadSession(logits, hidden_name=hidden_name, qtype_1d=qtype_1d)
        self._config = dict(config or CONFIG)

    def read_json(self, role):
        assert role == "config"
        return self._config

    def session(self, role):
        assert role in {"encoder", "head"}
        return self.encoder if role == "encoder" else self.head


def test_choice_sequence_and_probabilities():
    pytest.importorskip("numpy")
    engine = FakeEngine([2.0, 0.0])
    answers = LayaAdapter().system_one(engine, "the state", {"team": CHOICE_Q})

    team = answers["team"]
    assert team["choice"] == "billing"
    assert team["probabilities"]["billing"] == pytest.approx(0.880797, abs=1e-5)
    assert team["answer_confidence"] == pytest.approx(0.880797, abs=1e-5)
    assert 0.0 <= team["confidence"] <= 1.0
    assert team["action"]["act_probability"] == pytest.approx(0.731059, abs=1e-5)

    calls = engine.tokenizer.calls
    assert calls[0] == "choice question: Which team?"
    assert calls[1] == " billing: refunds" and calls[2] == " tech"
    assert "[SEP]" not in calls  # separators are ids, not text tokens
    assert engine.head.feeds["qtype"][0][0] == 0
    markers = engine.head.feeds["marker_pos"][0]
    assert len(markers) == 2


def test_noul_uses_false_then_true_order():
    pytest.importorskip("numpy")
    answers = LayaAdapter().system_one(FakeEngine([0.0, 2.0]), "state", {"urgent": NOUL_Q})
    urgent = answers["urgent"]
    assert urgent["noul"] == pytest.approx(0.880797, abs=1e-5)
    assert urgent["probabilities"]["true"] == pytest.approx(0.880797, abs=1e-5)
    assert urgent["probabilities"]["false"] == pytest.approx(0.119203, abs=1e-5)


def test_score_expected_and_qtype():
    pytest.importorskip("numpy")
    engine = FakeEngine([0.0, 0.0, 0.0])
    answers = LayaAdapter().system_one(engine, "state", {"tone": SCORE_Q})
    assert answers["tone"]["score"] == pytest.approx(1.0)
    assert answers["tone"]["legend"]["2"] == "furious"
    assert engine.head.feeds["qtype"][0][0] == 1


def test_noul_qtype_is_two():
    pytest.importorskip("numpy")
    engine = FakeEngine([0.0, 1.0])
    LayaAdapter().system_one(engine, "state", {"urgent": NOUL_Q})
    assert engine.head.feeds["qtype"][0][0] == 2


def test_temperature_is_clamped_like_the_torch_runtime():
    pytest.importorskip("numpy")
    config = dict(CONFIG)
    config["temperature_by_options"] = {"choice:2": 100.0}
    engine = FakeEngine([2.0, 0.0], config=config)
    answers = LayaAdapter().system_one(engine, "state", {"team": CHOICE_Q})
    # clamped to 5.0 -> softmax([0.4, 0]) ≈ 0.599, not the ~0.5 of an unclamped 100.0
    assert answers["team"]["probabilities"]["billing"] == pytest.approx(0.599, abs=0.005)


def test_missing_special_tokens_is_explained():
    pytest.importorskip("numpy")
    with pytest.raises(AdapterError) as excinfo:
        LayaAdapter().system_one(FakeEngine([0.0, 1.0], special=False), "state", {"team": CHOICE_Q})
    assert "tokenizer" in str(excinfo.value)


def test_english_checkpoint_head_names_and_qtype_shape():
    pytest.importorskip("numpy")
    engine = FakeEngine([2.0, 0.0], hidden_name="hidden", qtype_1d=True)
    answers = LayaAdapter().system_one(engine, "state", {"team": CHOICE_Q})
    assert answers["team"]["choice"] == "billing"
    assert engine.head.feeds["hidden"].shape[0] == 1
    assert engine.head.feeds["qtype"].shape == (1,)


def test_english_tokenizer_convention_is_supported():
    pytest.importorskip("numpy")

    class EnglishTokenizer(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self._vocab.update({"[MASK]": 50284, "[CLS]": 50281, "[SEP]": 50282})
            self._vocab.pop("<mask>", None)
            self._vocab.pop("<bos>", None)
            self._vocab.pop("<eos>", None)

    engine = FakeEngine([2.0, 0.0])
    engine.tokenizer = EnglishTokenizer()
    answers = LayaAdapter().system_one(engine, "state", {"team": CHOICE_Q})
    assert answers["team"]["choice"] == "billing"
    assert engine.head.feeds["qtype"][0][0] == 0
