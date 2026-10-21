"""`decision_toolkit.schema` — the System One contract, as typed Python.

TypeSafe's System One request is a ``state`` plus a mapping of named, typed
``questions``. Three primitives cover every decision the toolkit runs:

=============  ==================================  ==============================
primitive      you give                            you get
=============  ==================================  ==============================
``choice``     options with optional descriptions  the chosen option + probabilities
``noul``       a yes/no question                   the probability of "true"
``score``      ordered levels, lowest first       the expected level + probabilities
=============  ==================================  ==============================

This module builds and validates those questions and translates between System
One and OpenAI's Decisions dialect (``predicate`` / ``choice`` / ``score``),
which the REST server exposes at ``/v1/decisions``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Union

from .errors import InvalidQuestionError, InvalidRequestError

#: Wire limits (System One as served by TypeSafe Jev; LLM gateway limits are stricter).
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MIN_SCORE_LEVELS = 1

QuestionValue = Union["Choice", "Noul", "Score", "Box", Mapping[str, Any]]


def _require_instructions(instructions: str) -> str:
    if not isinstance(instructions, str) or not instructions.strip():
        raise InvalidQuestionError("every question needs non-empty `instructions`")
    return instructions


@dataclass(frozen=True)
class State:
    """A System One state: text, optional assets, or both.

    Text-only states are accepted as plain strings for convenience.  ``State``
    exists so callers can be explicit about assets and so the core can validate
    multimodal input against the selected model's capabilities before loading
    the model.
    """

    text: str = ""
    assets: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InvalidRequestError("State.text must be a string")
        assets = tuple(self.assets or ())
        for index, asset in enumerate(assets):
            if not isinstance(asset, Mapping):
                raise InvalidRequestError(f"State.assets[{index}] must be a mapping")
        object.__setattr__(self, "assets", assets)

    def to_systemone(self) -> dict[str, Any]:
        """The JSON-friendly System One state payload."""
        payload: dict[str, Any] = {"text": self.text}
        if self.assets:
            payload["assets"] = [dict(asset) for asset in self.assets]
        return payload

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Choice:
    """Pick one of the described options; probabilities over all of them."""

    instructions: str
    criteria: Mapping[str, str | None] | None = None

    def to_question(self) -> dict[str, Any]:
        _require_instructions(self.instructions)
        question: dict[str, Any] = {"type": "choice", "instructions": self.instructions}
        if self.criteria is not None:
            if not 1 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
                raise InvalidQuestionError(
                    f"a choice needs 1 to {MAX_CHOICE_OPTIONS} options, got {len(self.criteria)}"
                )
            question["criteria"] = {
                str(label): (None if description is None else str(description))
                for label, description in self.criteria.items()
            }
        return question


@dataclass(frozen=True)
class Noul:
    """A yes/no judgment; the answer is the probability that it is true.

    ``criteria`` optionally describes what a ``false`` and a ``true`` mean:
    ``{"false": "...", "true": "..."}``. Without it, the toolkit passes the
    question at face value — the weakest path, as documented by every vendor.
    """

    instructions: str
    criteria: Mapping[str, str] | None = None

    def to_question(self) -> dict[str, Any]:
        _require_instructions(self.instructions)
        question: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            unknown = set(self.criteria) - {"false", "true"}
            if unknown:
                raise InvalidQuestionError(
                    f"noul criteria keys must be 'false' and/or 'true', got {sorted(unknown)}"
                )
            question["criteria"] = {str(k): str(v) for k, v in self.criteria.items()}
        return question


@dataclass(frozen=True)
class Score:
    """A position on an ordered, described scale (lowest first)."""

    instructions: str
    criteria: Sequence[str]

    def to_question(self) -> dict[str, Any]:
        _require_instructions(self.instructions)
        levels = list(self.criteria)
        if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
            raise InvalidQuestionError(
                f"a score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {len(levels)}"
            )
        if any(not isinstance(level, str) or not level.strip() for level in levels):
            raise InvalidQuestionError("score levels must be non-empty strings")
        return {"type": "score", "instructions": self.instructions, "criteria": levels}


@dataclass(frozen=True)
class Box:
    """A region/box question over an asset (System One ``box`` primitive).

    The current catalog has no multimodal/box model, so the core will validate
    this question and then raise ``UnsupportedCapabilityError`` for those
    models.  The typed builder is here so a future adapter can implement the
    same contract without changing the public vocabulary.
    """

    instructions: str
    criteria: Sequence[str] | None = None

    def to_question(self) -> dict[str, Any]:
        _require_instructions(self.instructions)
        question: dict[str, Any] = {"type": "box", "instructions": self.instructions}
        if self.criteria is not None:
            labels = [str(label).strip() for label in self.criteria]
            if not labels or any(not label for label in labels):
                raise InvalidQuestionError("box criteria must be non-empty labels")
            question["criteria"] = labels
        return question


def choice(instructions: str, criteria: Mapping[str, str | None] | None = None) -> Choice:
    """Build a ``choice`` question (constructor shorthand)."""
    return Choice(instructions=instructions, criteria=criteria)


def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> Noul:
    """Build a ``noul`` (yes/no) question (constructor shorthand)."""
    return Noul(instructions=instructions, criteria=criteria)


def score(instructions: str, criteria: Sequence[str]) -> Score:
    """Build a ``score`` question from ordered levels, lowest first."""
    return Score(instructions=instructions, criteria=criteria)


def box(instructions: str, criteria: Sequence[str] | None = None) -> Box:
    """Build a ``box`` question (region/asset primitive)."""
    return Box(instructions=instructions, criteria=criteria)


def normalize_state(state: Any) -> Any:
    """Validate a System One state without converting its text form.

    Plain strings and JSON states are returned unchanged; :class:`State` stays
    typed so the caller's assets survive until capability validation.
    """
    if state is None:
        raise InvalidRequestError("state cannot be None")
    if isinstance(state, State):
        return state
    if isinstance(state, (str, Mapping, list, tuple)):
        return state
    # Scalars are stringified by the adapters; keep the historical permissive
    # behaviour for custom adapters.
    return state


def normalize_questions(questions: Mapping[str, QuestionValue]) -> dict[str, dict[str, Any]]:
    """Validate a question mapping and return the canonical System One payload.

    Accepts :class:`Choice` / :class:`Noul` / :class:`Score` / :class:`Box`
    objects or raw question dictionaries.  Known primitives are validated here
    once (type, instructions, option/level counts); unknown vendor extensions
    are rejected rather than passed through silently.

    Extra keys used by adapters/extensions (``task``, ``multi_label``,
    ``_values``...) are preserved.
    """
    if not isinstance(questions, Mapping) or not questions:
        raise InvalidQuestionError("`questions` must be a non-empty mapping of name -> question")

    normalized: dict[str, dict[str, Any]] = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise InvalidQuestionError(f"question names must be non-empty strings, got {name!r}")
        if isinstance(question, (Choice, Noul, Score, Box)):
            normalized[name] = question.to_question()
            continue
        if isinstance(question, Mapping):
            qtype = question.get("type")
            if qtype not in {"choice", "noul", "score", "box"}:
                raise InvalidQuestionError(
                    f"question {name!r} has type {qtype!r}; "
                    "expected choice, noul, score or box"
                )
            normalized[name] = _validate_raw_question(name, question)
            continue
        raise InvalidQuestionError(
            f"question {name!r} must be a Choice/Noul/Score/Box or a mapping, "
            f"got {type(question).__name__}"
        )
    return normalized


def _validate_raw_question(name: str, question: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one raw canonical question, preserving extension keys."""
    payload = dict(question)
    qtype = payload["type"]
    instructions = payload.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise InvalidQuestionError(f"question {name!r} needs non-empty `instructions`")

    if qtype == "choice":
        criteria = payload.get("criteria")
        if criteria is not None:
            if not isinstance(criteria, Mapping):
                raise InvalidQuestionError(f"question {name!r}: choice criteria must be a mapping")
            if not 1 <= len(criteria) <= MAX_CHOICE_OPTIONS:
                raise InvalidQuestionError(
                    f"question {name!r}: a choice needs 1 to {MAX_CHOICE_OPTIONS} "
                    f"options, got {len(criteria)}"
                )
            for label, description in criteria.items():
                if not isinstance(label, str) or not label.strip():
                    raise InvalidQuestionError(
                        f"question {name!r}: choice labels must be non-empty strings"
                    )
                if description is not None and not isinstance(description, str):
                    raise InvalidQuestionError(
                        f"question {name!r}: choice descriptions must be strings or null"
                    )
    elif qtype == "noul":
        criteria = payload.get("criteria")
        if criteria is not None:
            if not isinstance(criteria, Mapping):
                raise InvalidQuestionError(f"question {name!r}: noul criteria must be a mapping")
            unknown = set(criteria) - {"false", "true"}
            if unknown:
                raise InvalidQuestionError(
                    f"question {name!r}: noul criteria keys must be 'false' and/or "
                    f"'true', got {sorted(unknown)}"
                )
    elif qtype == "score":
        levels = payload.get("criteria")
        if (
            not isinstance(levels, Sequence)
            or isinstance(levels, (str, bytes))
            or not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS
        ):
            raise InvalidQuestionError(
                f"question {name!r}: a score needs {MIN_SCORE_LEVELS} to "
                f"{MAX_SCORE_LEVELS} ordered levels"
            )
        if any(not isinstance(level, str) or not level.strip() for level in levels):
            raise InvalidQuestionError(f"question {name!r}: score levels must be non-empty strings")
    elif qtype == "box":
        criteria = payload.get("criteria")
        if criteria is not None:
            if (
                not isinstance(criteria, Sequence)
                or isinstance(criteria, (str, bytes))
                or not criteria
            ):
                raise InvalidQuestionError(
                    f"question {name!r}: box criteria must be a non-empty sequence of labels"
                )
            if any(not isinstance(label, str) or not label.strip() for label in criteria):
                raise InvalidQuestionError(
                    f"question {name!r}: box labels must be non-empty strings"
                )
    return payload


def parse_question_spec(spec: str) -> dict[str, Any]:
    """Parse a CLI question spec into a raw System One question.

    ``choice:yes|no``, ``choice:yes=meets it|no=does not``, ``noul`` (with an
    optional ``noul:instructions in the key`` handled by the CLI), ``score:a|b|c``.
    """
    if not isinstance(spec, str) or ":" not in spec and spec not in {"noul"}:
        raise InvalidQuestionError(
            f"cannot parse question spec {spec!r}; use choice:a|b, choice:a=desc|b, noul or score:a|b|c"
        )
    if spec == "noul":
        raise InvalidQuestionError("a noul spec needs an instruction: noul:<text>")
    kind, _, body = spec.partition(":")
    kind = kind.strip()
    body = body.strip()
    if kind == "noul":
        return {"type": "noul", "instructions": body}
    if kind not in {"choice", "score"}:
        raise InvalidQuestionError(f"unknown question kind {kind!r}; expected choice, noul or score")
    parts = [part.strip() for part in body.split("|")]
    if kind == "choice":
        criteria: dict[str, str | None] = {}
        for part in parts:
            if not part:
                raise InvalidQuestionError("empty option in choice spec")
            label, sep, description = part.partition("=")
            criteria[label.strip()] = description.strip() if sep else None
        return {"type": "choice", "criteria": criteria}
    levels = [part for part in parts if part]
    if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
        raise InvalidQuestionError(f"a score spec needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")
    return {"type": "score", "criteria": levels}


# --------------------------------------------------------------------------- #
# OpenAI Decisions dialect (https://developers.openai.com/api/docs/guides/decisions)
# --------------------------------------------------------------------------- #

_OPENAI_TYPES = {"predicate", "choice", "score"}


def openai_to_systemone(
    input_value: Any,
    questions: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]], list[str]]:
    """Translate an OpenAI Decisions request into System One.

    Returns ``(state, questions, names)``: the state as text, the System One
    question mapping, and the answer names in request order (used to re-map the
    response). A ``predicate`` becomes a ``noul``; choice values keep their
    string/boolean form; score levels map to their labels.
    """
    state = _openai_input_to_state(input_value)
    if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)) or not questions:
        raise InvalidQuestionError("`questions` must be a non-empty list")

    systemone: dict[str, dict[str, Any]] = {}
    names: list[str] = []
    for index, question in enumerate(questions):
        if not isinstance(question, Mapping):
            raise InvalidQuestionError(f"question {index} must be an object")
        qtype = question.get("type")
        if qtype not in _OPENAI_TYPES:
            raise InvalidQuestionError(f"question {index}: unsupported type {qtype!r}")
        name = str(question.get("name") or f"q{index}")
        names.append(name)
        instructions = str(question.get("instructions") or "")
        if qtype == "predicate":
            systemone[name] = Noul(instructions=instructions or "Is this true?").to_question()
        elif qtype == "choice":
            choices = question.get("choices")
            if not isinstance(choices, Sequence) or len(choices) < 2:
                raise InvalidQuestionError(f"question {name!r}: a choice needs at least 2 entries")
            criteria = {}
            for entry in choices:
                if not isinstance(entry, Mapping) or "value" not in entry:
                    raise InvalidQuestionError(
                        f"question {name!r}: every choice entry needs a 'value'"
                    )
                criteria[str(entry["value"])] = entry.get("description")
            question_obj = Choice(instructions=instructions or "Which choice best fits the input?",
                                  criteria=criteria).to_question()
            # keep the original values so the response echoes them faithfully
            question_obj["_values"] = [entry["value"] for entry in choices]
            systemone[name] = question_obj
        else:  # score
            levels = question.get("levels")
            if not isinstance(levels, Sequence) or len(levels) < MIN_SCORE_LEVELS:
                raise InvalidQuestionError(f"question {name!r}: a score needs at least 2 levels")
            labels = [str(entry.get("label")) if isinstance(entry, Mapping) else str(entry)
                      for entry in levels]
            systemone[name] = Score(
                instructions=instructions or "Score the input on the given scale.", criteria=labels
            ).to_question()
    return state, systemone, names


def systemone_answers_to_openai(
    answers: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    *,
    values_by_name: Mapping[str, Sequence[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Translate System One answers into the OpenAI Decisions answer list.

    ``values_by_name`` maps a choice question to the original (possibly
    non-string) values, so the response echoes ``true``/``1`` instead of the
    string labels the System One projection uses internally.
    """
    result: list[dict[str, Any]] = []
    values_by_name = values_by_name or {}
    for name in names:
        answer = answers.get(name)
        if answer is None:
            result.append({"type": "refusal", "name": name})
            continue
        qtype = answer.get("type")
        if qtype == "noul":
            result.append({"type": "predicate", "name": name,
                           "probability": float(answer.get("noul", answer.get("p_true", 0.0)))})
        elif qtype == "choice":
            probabilities = answer.get("probabilities") or {}
            original = values_by_name.get(name) or ()
            lookup = {str(value): value for value in original}
            if isinstance(probabilities, Mapping):
                entries = [{"value": lookup.get(label, label), "probability": float(p)}
                           for label, p in probabilities.items()]
            else:  # pragma: no cover - defensive for odd adapters
                entries = [{"value": lookup.get(label, label), "probability": float(p)}
                           for label, p in dict(probabilities or {}).items()]
            chosen = answer.get("choice")
            result.append({"type": "choice", "name": name,
                           "choice": lookup.get(chosen, chosen),
                           "probabilities": entries,
                           "confidence": answer.get("confidence", _max_probability(entries))})
        elif qtype == "score":
            probabilities = answer.get("probabilities") or {}
            legend = answer.get("legend") or {}
            entries = [
                {"value": index, "label": legend.get(str(index), legend.get(index)),
                 "probability": float(p)}
                for index, p in sorted(
                    ((int(k), v) for k, v in probabilities.items()), key=lambda item: item[0]
                )
            ]
            result.append({"type": "score", "name": name,
                           "score": answer.get("score"),
                           "probabilities": entries,
                           "confidence": answer.get("confidence", _max_probability(entries))})
        else:
            result.append({"type": "refusal", "name": name})
    return result


def _openai_input_to_state(input_value: Any) -> str:
    """Flatten an OpenAI ``input`` (string or message list) into System One state text."""
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, Sequence):
        chunks: list[str] = []
        for message in input_value:
            if isinstance(message, str):
                chunks.append(message)
                continue
            if not isinstance(message, Mapping):
                raise InvalidQuestionError("input messages must be objects or strings")
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            if isinstance(content, Sequence):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, Mapping):
                        if part.get("type") in {"input_text", "text"}:
                            parts.append(str(part.get("text", "")))
                        elif part.get("type") == "input_image":
                            raise InvalidQuestionError(
                                "image input is not supported by this toolkit yet"
                            )
                    elif isinstance(part, str):
                        parts.append(part)
                chunks.append("\n".join(parts))
        return "\n".join(chunks)
    if isinstance(input_value, (Mapping, list)):  # JSON state passthrough
        return json.dumps(input_value, ensure_ascii=False, sort_keys=True)
    raise InvalidQuestionError("`input` must be a string or a list of messages")


def _max_probability(entries: Sequence[Mapping[str, Any]]) -> float:
    return max((float(entry.get("probability", 0.0)) for entry in entries), default=0.0)


__all__ = [
    "Box",
    "Choice",
    "Noul",
    "Score",
    "State",
    "box",
    "choice",
    "normalize_questions",
    "normalize_state",
    "noul",
    "openai_to_systemone",
    "parse_question_spec",
    "score",
    "systemone_answers_to_openai",
]
