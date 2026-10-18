"""Typed views over raw System One answers."""

from __future__ import annotations

from decision_toolkit.results import Answer, Classification, wrap_answers


def test_answer_choice_view():
    answer = Answer("team", "choice", {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.8, "tech": 0.2},
        "confidence": 0.8,
    })
    assert answer.choice == "billing"
    assert answer.probabilities == {"billing": 0.8, "tech": 0.2}
    assert answer.confidence == 0.8
    assert answer.noul is None


def test_answer_noul_from_probabilities_fallback():
    answer = Answer("urgent", "noul", {"type": "noul", "probabilities": {"true": 0.91, "false": 0.09}})
    assert answer.noul == 0.91


def test_answer_score_levels():
    answer = Answer("tone", "score", {
        "type": "score",
        "score": 1.4,
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
        "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
    })
    assert answer.score == 1.4
    assert answer.level == "2"
    assert answer.legend["2"] == "furious"
    assert answer.confidence == 0.5  # fallback to the max probability


def test_wrap_answers_fills_type_from_question():
    wrapped = wrap_answers(
        {"team": {"choice": "billing"}},
        {"team": {"type": "choice", "instructions": "?"}},
    )
    assert wrapped["team"].type == "choice"


def test_classification_ranked():
    classification = Classification(
        label="a", confidence=0.6, below_threshold=False, scores={"a": 0.6, "b": 0.3, "c": 0.1}
    )
    assert classification.ranked(2) == [("a", 0.6), ("b", 0.3)]
    assert classification.ranked()[0] == ("a", 0.6)
