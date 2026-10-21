"""Quickstart: one decision, on CPU, from the library.

Run with any runnable model, for example::

    python examples/quickstart.py laya-multilingual

It needs the model runtime (``pip install "decision-toolkit[onnx]"``) and, the
first time, its checkpoint download (``dtk models download laya-multilingual``).
See ``dtk models list`` for what is runnable today.
"""

from __future__ import annotations

import sys

from decision_toolkit import DecisionModel, choice, noul, score


def main(argv: list[str]) -> int:
    model_id = argv[0] if argv else "laya-multilingual"
    model = DecisionModel(model_id)

    answer = model.decide(
        state="I was charged twice for the March invoice and nobody has replied.",
        questions={
            "team": choice("Which team should handle this?", {
                "billing": "charges, invoices, refunds",
                "technical": "bugs and outages",
                "cancellation": "closing the account",
            }),
            "urgent": noul("Does the customer need a reply today?"),
            "tone": score("How upset is the customer?", ["calm", "annoyed", "furious"]),
        },
    )

    keys = answer.keys()
    for key in keys:
        item = answer[key]
        if item.type == "choice":
            print(f"{key:10s} choice={item.choice!r} confidence={item.confidence:.2f}")
            for label, probability in sorted(item.probabilities.items(), key=lambda kv: -kv[1]):
                print(f"           {label:14s} {probability:.3f}")
        elif item.type == "noul":
            print(f"{key:10s} p(true)={item.noul:.3f}")
        else:
            print(f"{key:10s} score={item.score:.2f} level={item.level} legend={dict(item.legend)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
