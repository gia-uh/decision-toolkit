"""The System One contract: builders, validation and the OpenAI dialect."""

from __future__ import annotations

import pytest

from decision_toolkit.errors import InvalidQuestionError
from decision_toolkit.schema import (
    Choice,
    Noul,
    Score,
    choice,
    normalize_questions,
    noul,
    openai_to_systemone,
    parse_question_spec,
    score,
    systemone_answers_to_openai,
)


# --------------------------------------------------------------------- builders
def test_choice_builder():
    question = choice("Which team?", {"billing": "refunds", "tech": None}).to_question()
    assert question == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "refunds", "tech": None},
    }


def test_noul_builder_with_criteria():
    question = noul("Urgent?", {"true": "today", "false": "can wait"}).to_question()
    assert question["type"] == "noul"
    assert question["criteria"] == {"true": "today", "false": "can wait"}


def test_score_builder():
    question = score("How upset?", ["calm", "annoyed", "furious"]).to_question()
    assert question == {
        "type": "score",
        "instructions": "How upset?",
        "criteria": ["calm", "annoyed", "furious"],
    }


@pytest.mark.parametrize(
    "builder",
    [
        lambda: Choice(instructions=" "),
        lambda: Noul(instructions=""),
        lambda: Score(instructions="x", criteria=[f"l{i}" for i in range(11)]),
        lambda: Noul(instructions="x", criteria={"maybe": "y"}),
        lambda: Score(instructions="x", criteria=["ok", " "]),
    ],
)
def test_invalid_builders_raise(builder):
    with pytest.raises(InvalidQuestionError):
        builder().to_question()


def test_choice_without_criteria_is_allowed():
    assert choice("Free choice?").to_question() == {"type": "choice", "instructions": "Free choice?"}


# ------------------------------------------------------------------- normalize
def test_normalize_mixed_objects_and_raw_dicts():
    normalized = normalize_questions(
        {
            "team": choice("Which team?", {"a": "x", "b": "y"}),
            "urgent": noul("Urgent?"),
            "tone": score("Tone?", ["calm", "angry"]),
            "raw": {"type": "choice", "instructions": "Pick", "criteria": {"z": None}},
        }
    )
    assert set(normalized) == {"team", "urgent", "tone", "raw"}
    assert normalized["raw"]["type"] == "choice"


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"q": 3},
        {"q": {"type": "other", "instructions": "x"}},
        {" ": noul("x")},
    ],
)
def test_normalize_rejects_bad_shapes(questions):
    with pytest.raises(InvalidQuestionError):
        normalize_questions(questions)


# ------------------------------------------------------------------ cli specs
def test_parse_question_spec_choice_with_descriptions():
    question = parse_question_spec("choice:yes=meets it|no=does not")
    assert question == {
        "type": "choice",
        "criteria": {"yes": "meets it", "no": "does not"},
    }


def test_parse_question_spec_score_and_noul():
    assert parse_question_spec("score:low|mid|high") == {"type": "score", "criteria": ["low", "mid", "high"]}
    assert parse_question_spec("noul:Is it urgent?") == {"type": "noul", "instructions": "Is it urgent?"}


@pytest.mark.parametrize(
    "spec",
    ["noul", "weird:a|b", "choice:", "score:", "choice:a||b", ""],
)
def test_parse_question_spec_errors(spec):
    with pytest.raises(InvalidQuestionError):
        parse_question_spec(spec)


# --------------------------------------------------------- OpenAI translation
def test_openai_to_systemone_full_request():
    state, questions, names = openai_to_systemone(
        "Customer wants a refund.",
        [
            {"type": "predicate", "name": "refund", "instructions": "Is it a refund request?"},
            {
                "type": "choice",
                "name": "team",
                "instructions": "Which team?",
                "choices": [{"value": "billing"}, {"value": "tech", "description": "bugs"}],
            },
            {
                "type": "score",
                "instructions": "How urgent?",
                "levels": [{"label": "low"}, {"label": "high"}],
            },
        ],
    )
    assert state == "Customer wants a refund."
    assert names == ["refund", "team", "q2"]
    assert questions["refund"]["type"] == "noul"
    assert questions["team"]["criteria"] == {"billing": None, "tech": "bugs"}
    assert questions["team"]["_values"] == ["billing", "tech"]
    assert questions["q2"]["criteria"] == ["low", "high"]


def test_openai_to_systemone_accepts_message_lists():
    state, _, _ = openai_to_systemone(
        [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        [{"type": "predicate", "name": "ok", "instructions": "Fine?"}],
    )
    assert state == "Hello"


def test_openai_to_systemone_rejects_images_and_bad_questions():
    with pytest.raises(InvalidQuestionError):
        openai_to_systemone(
            [{"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "x"}]}],
            [{"type": "predicate", "name": "ok", "instructions": "Fine?"}],
        )
    with pytest.raises(InvalidQuestionError):
        openai_to_systemone("x", [{"type": "unknown"}])
    with pytest.raises(InvalidQuestionError):
        openai_to_systemone("x", [{"type": "choice", "choices": [{"value": "one"}]}])


def test_systemone_answers_to_openai_roundtrip():
    answers = {
        "refund": {"type": "noul", "noul": 0.91},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.8, "tech": 0.2},
            "confidence": 0.8,
        },
        "q2": {
            "type": "score",
            "score": 1.2,
            "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
            "legend": {"0": "low", "1": "mid", "2": "high"},
        },
    }
    translated = systemone_answers_to_openai(answers, ["refund", "team", "q2", "missing"])
    assert translated[0] == {"type": "predicate", "name": "refund", "probability": 0.91}
    assert translated[1]["choice"] == "billing"
    assert translated[1]["probabilities"][0] == {"value": "billing", "probability": 0.8}
    assert translated[2]["probabilities"][2]["label"] == "high"
    assert translated[3] == {"type": "refusal", "name": "missing"}
