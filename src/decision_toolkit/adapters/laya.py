"""`decision_toolkit.adapters.laya` — Laya's split encoder+head layout.

Contract (ported from the reference implementation and validated against the
mininvox export, where it reproduced the torch runtime's answers):

    [CLS] <type> question: <instructions> [SEP] ([MASK] opt0) ([MASK] opt1) … [SEP] <state> [SEP]
    encoder(input_ids, attention_mask) -> last_hidden_state
    head(hidden_states, marker_pos, marker_mask, qtype, attention_mask)
        -> logits per marker + act_logits
    choice / score: softmax(logits / temperature)
    noul: options are [false, true], so p(true) = p[1]
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from ..engine import OnnxEngine, encode
from ..errors import AdapterError
from .base import (
    build_choice_answer,
    build_noul_answer,
    build_score_answer,
    question_options,
    softmax,
    state_to_text,
)

logger = logging.getLogger(__name__)

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = ["choice", "score", "noul"]


class LayaAdapter:
    """System One over a Laya split export (encoder + head + rl_agent_config)."""

    name = "laya"

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _special_ids(tokenizer: Any) -> tuple[int, int, int]:
        # Laya checkpoints ship two tokenizer serializations of the same contract:
        #   multilingual: <mask>/<bos>/<eos>   ·   English: [MASK]/[CLS]/[SEP]
        for mask_name, bos_name, eos_name in (
            ("<mask>", "<bos>", "<eos>"),
            ("[MASK]", "[CLS]", "[SEP]"),
        ):
            ids = (
                tokenizer.token_to_id(mask_name),
                tokenizer.token_to_id(bos_name),
                tokenizer.token_to_id(eos_name),
            )
            if all(token is not None for token in ids):
                return ids  # type: ignore[return-value]
        raise AdapterError(
            "the tokenizer exposes neither <mask>/<bos>/<eos> nor [MASK]/[CLS]/[SEP]; "
            "this is not a Laya tokenizer"
        )

    @staticmethod
    def _render_options(question: Mapping[str, Any]) -> list[str]:
        criteria = question.get("criteria")
        kind = question.get("type")
        if kind == "choice":
            return [f"{key}: {value}" if value else key for key, value in criteria.items()]
        if kind == "score":
            return [f"level {index}: {level}" for index, level in enumerate(criteria)]
        criteria = criteria or {}
        return [
            "false: " + (criteria.get("false") or "no, the statement does not hold"),
            "true: " + (criteria.get("true") or "yes, the statement holds"),
        ]

    def _sequence(
        self,
        tokenizer: Any,
        state_text: str,
        question: Mapping[str, Any],
        options: list[str],
        config: Mapping[str, Any],
    ) -> tuple[list[int], list[int], bool]:
        max_len = int(config.get("max_len", 1024))
        head_max_len = int(config.get("head_max_len", 256))
        mask, bos, eos = self._special_ids(tokenizer)
        truncated = False

        instructions = str(question.get("instructions") or "")
        head_ids = encode(tokenizer, f"{question['type']} question: {instructions}")
        option_ids: list[list[int]] = []
        for option in options:
            encoded = encode(tokenizer, " " + option)
            if len(encoded) > 48:
                truncated = True
            option_ids.append([mask, *encoded[:48]])

        budget = head_max_len - sum(len(option) for option in option_ids)
        if budget < 16:
            truncated = True
            per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
            option_ids = [option[:per] for option in option_ids]
            budget = head_max_len - sum(len(option) for option in option_ids)
        if len(head_ids) > max(8, budget):
            truncated = True
        head_ids = head_ids[: max(8, budget)]

        ids = [bos, *head_ids, eos]
        markers: list[int] = []
        for option in option_ids:
            markers.append(len(ids))
            ids.extend(option)
        ids.append(eos)

        room = max(0, max_len - len(ids) - 1)
        state_ids = encode(tokenizer, state_text)
        if len(state_ids) > room:
            truncated = True
        state_ids = state_ids[:room]
        all_ids = (ids + state_ids + [eos])[:max_len]
        if len(markers) > sum(1 for marker in markers if marker < max_len):
            truncated = True
        return all_ids, [marker for marker in markers if marker < max_len], truncated

    @staticmethod
    def _temperature(config: Mapping[str, Any], qtype: int, count: int) -> float:
        size = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
        bucket = f"{QTYPE_NAMES[qtype]}:{size}"
        by_options = config.get("temperature_by_options") or {}
        temperatures = config.get("temperature") or [1.0, 1.0, 1.0]
        value = float(by_options.get(bucket, temperatures[qtype]))
        # the torch runtime clamps temperatures outside [0.5, 5] before applying them
        return min(5.0, max(0.5, value))

    @staticmethod
    def _confidence(probabilities: list[float]) -> float:
        count = len(probabilities)
        if count < 2:
            return 1.0
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probabilities)
        return 1 - entropy / math.log(count)

    # -------------------------------------------------------------- inference
    def system_one(
        self, engine: OnnxEngine, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        import numpy as np

        config = engine.read_json("config")
        tokenizer = engine.tokenizer
        state_text = state_to_text(state)
        answers: dict[str, dict[str, Any]] = {}

        for name, question in questions.items():
            kind, labels, _ = question_options(question)
            options = self._render_options(question)
            ids, markers, truncated = self._sequence(
                tokenizer, state_text, question, options, config
            )
            if len(markers) != len(options):
                raise AdapterError(
                    f"{name!r}: the options do not fit in {config.get('head_max_len')} tokens"
                )
            qtype = QTYPES[kind]

            input_ids = np.asarray([ids], dtype=np.int64)
            attention = np.ones_like(input_ids)
            hidden = engine.session("encoder").run(
                None, {"input_ids": input_ids, "attention_mask": attention}
            )[0]

            head = engine.session("head")
            inputs = {item.name: item for item in head.get_inputs()}
            hidden_name = next(
                (name for name in ("hidden_states", "hidden", "last_hidden_state") if name in inputs),
                None,
            )
            if hidden_name is None:
                raise AdapterError(
                    f"the head exposes no hidden-state input; inputs: {sorted(inputs)}"
                )
            qtype_value = (
                np.asarray([qtype], dtype=np.int64)
                if len(inputs["qtype"].shape) == 1
                else np.asarray([[qtype]], dtype=np.int64)
            )
            logits, act_logits = head.run(None, {
                hidden_name: hidden,
                "attention_mask": attention,
                "marker_pos": np.asarray([markers], dtype=np.int64),
                "marker_mask": np.ones((1, len(markers)), dtype=bool),
                "qtype": qtype_value,
            })

            raw = np.asarray(logits)[0][: len(markers)]
            scaled = (raw / self._temperature(config, qtype, len(markers))).tolist()
            probabilities = softmax(scaled)
            act_probability = softmax(np.asarray(act_logits)[0].tolist())[0]

            if kind == "choice":
                answer = build_choice_answer(labels, probabilities)
            elif kind == "noul":
                # Laya's options are [false, true]: p[1] is the probability of true
                answer = build_noul_answer(probabilities[1], probabilities[0])
            else:
                answer = build_score_answer(labels, probabilities)

            if kind in {"choice", "score"}:
                answer["answer_confidence"] = answer["confidence"]
                answer["distribution_confidence"] = self._confidence(probabilities)
            answer["confidence_kind"] = "probability"
            answer["action"] = {"act_probability": float(act_probability)}
            if truncated:
                answer["truncated"] = True
                answer["warnings"] = [
                    f"input was truncated to fit max_len={config.get('max_len', 1024)}"
                ]
            answers[name] = answer

        return answers


__all__ = ["LayaAdapter"]
