# decision-toolkit

Local-first toolkit for **open-weights decision models** on CPU: typed choice /
yes-no / score questions with calibrated probabilities, one ONNX engine and one
adapter per model contract.

```bash
pip install decision-toolkit
pip install "decision-toolkit[onnx]"     # model runtime
pip install "decision-toolkit[server]"   # REST server
```

PyPI: <https://pypi.org/project/decision-toolkit/>

## Start here

- [Examples](EXAMPLES.md) — library, REST and CLI.
- [Core API](CORE.md) — the superset used by every surface.
- [REST](REST.md) — System One 1.14 + OpenAI Decisions beta.
- [CLI](CLI.md) — operational commands and batch workflows.
- [Operations](OPERATIONS.md) — rate limits, metrics, tracing and runbook.
- [Benchmarks](BENCH.md) — reproducible latency/parity/calibration.
- [Models](models/README.md) — the curated catalog.
- [Release](RELEASE.md) — build, TestPyPI/PyPI and Trusted Publishing.

## One example

```python
from decision_toolkit import DecisionModel, choice, noul

with DecisionModel("gliner2.5-multi") as model:
    decision = model.decide(
        "Meeting with Acme: 10% discount, sign next week.",
        {
            "cat": choice("What kind of record is this?", {
                "meeting": "a meeting or call",
                "tasks": "things to do",
                "other": "anything else",
            }),
            "urgent": noul("Does it need a reply today?"),
        },
    )

decision["cat"].choice
decision["cat"].confidence
decision["urgent"].noul
decision.usage.to_dict()
```

## Trusted publishing

Every release from `0.1.2` onward is published through GitHub OIDC to TestPyPI
and PyPI. See [RELEASE.md](RELEASE.md).
