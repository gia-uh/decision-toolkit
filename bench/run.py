#!/usr/bin/env python3
"""Reproducible smoke/parity/calibration harness.

Usage (from the repository root)::

    PYTHONPATH=src python bench/run.py --suite bench/suites/smoke.jsonl
    PYTHONPATH=src python bench/run.py --suite bench/suites/calibration.jsonl \
        --output bench/results/2026-10-20-calibration.json

The suite is JSONL. Each line has:

* ``model`` — catalog id;
* ``state`` — text/JSON state;
* ``questions`` — raw System One questions;
* ``expected`` — optional ``{question_name: expected_value}`` for regression;
* ``labels`` — optional ``{question_name: gold_label}`` for calibration.

The harness never downloads models: they must already be in the toolkit cache.
It reports per-model latency, parity against ``expected`` and, when labels are
present, Brier/ECE/threshold calibration.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from decision_toolkit import DecisionModel
from decision_toolkit.calibration import (
    brier_score,
    expected_calibration_error,
    pairs_from_rows,
    threshold_sweep,
)


def environment() -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for module in ("onnxruntime", "tokenizers", "numpy"):
        try:
            values[module] = __import__(module).__version__
        except ImportError:
            values[module] = None
    return values


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
        if not isinstance(case, dict):
            raise ValueError(  # noqa: TRY004 - a malformed suite is input validation
                f"{path}:{number}: each line must be a JSON object"
            )
        for key in ("model", "state", "questions"):
            if key not in case:
                raise ValueError(f"{path}:{number}: missing {key!r}")
        case.setdefault("id", f"case-{number}")
        cases.append(case)
    return cases


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def match_value(answer: Any, expected: Any) -> bool:
    actual = answer.value
    if isinstance(expected, bool):
        if answer.type == "noul":
            return (float(actual) >= 0.5) is expected
        return bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) <= 1e-6
        except (TypeError, ValueError):
            return False
    return actual == expected


def run_case(
    model: DecisionModel,
    case: dict[str, Any],
    *,
    repeats: int,
) -> dict[str, Any]:
    last = None
    latencies: list[float] = []
    for _ in range(max(1, repeats)):
        last = model.decide(case["state"], case["questions"])
        if last.latency_ms is not None:
            latencies.append(last.latency_ms)
    assert last is not None
    expected = case.get("expected") or {}
    matched = all(match_value(last[name], value) for name, value in expected.items())
    return {
        "id": case["id"],
        "matched": matched if expected else None,
        "expected": expected or None,
        "actual": {
            name: answer.value for name, answer in last.answers.items()
        },
        "types": {
            name: answer.type for name, answer in last.answers.items()
        },
        "confidence": {
            name: answer.confidence for name, answer in last.answers.items()
        },
        "latency_ms": latencies,
        "usage": last.usage.to_dict(),
        "warnings": list(last.warnings),
        "truncated": last.truncated,
    }


def run_suite(
    path: Path,
    *,
    home: str | None,
    repeats: int,
    models_filter: set[str] | None,
) -> dict[str, Any]:
    cases = load_cases(path)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if models_filter and case["model"] not in models_filter:
            continue
        by_model[case["model"]].append(case)

    results: dict[str, Any] = {}
    overall_ok = True
    for model_id, model_cases in by_model.items():
        model: DecisionModel | None = None
        try:
            model = DecisionModel(model_id, home=home)
            model.warmup()
            # Load sessions/tokenizer before measuring: the first decision is
            # dominated by lazy runtime initialization, not steady-state cost.
            if model_cases:
                model.decide(model_cases[0]["state"], model_cases[0]["questions"])
            model_results: list[dict[str, Any]] = []
            latencies: list[float] = []
            pairs: list[tuple[float, bool]] = []
            matched = 0
            expected_count = 0
            for case in model_cases:
                result = run_case(model, case, repeats=repeats)
                model_results.append(result)
                latencies.extend(result["latency_ms"])
                if result["matched"] is not None:
                    expected_count += 1
                    matched += int(result["matched"])
                for name, gold in (case.get("labels") or {}).items():
                    actual = result["actual"].get(name)
                    types = result["types"]
                    confidence = result["confidence"].get(name)
                    if actual is None:
                        continue
                    if types.get(name) == "choice" and confidence is not None:
                        pairs.append((float(confidence), actual == gold))
                    elif types.get(name) == "noul":
                        probability = float(actual)
                        pairs.append(
                            (
                                max(probability, 1.0 - probability),
                                (probability >= 0.5) == bool(gold),
                            )
                        )
            calibration = None
            if pairs:
                normalized = pairs_from_rows(pairs)
                calibration = {
                    "n": len(normalized),
                    "accuracy": (
                        sum(1 for _, correct in normalized if correct) / len(normalized)
                    ),
                    "brier": brier_score(normalized),
                    "ece": expected_calibration_error(normalized),
                    "thresholds": {
                        str(threshold): values
                        for threshold, values in threshold_sweep(normalized).items()
                    },
                }
            parity = None
            if expected_count:
                parity = {
                    "cases": expected_count,
                    "matched": matched,
                    "accuracy": matched / expected_count,
                }
                overall_ok = overall_ok and matched == expected_count
            results[model_id] = {
                "variant": model.variant,
                "cases": len(model_cases),
                "parity": parity,
                "latency_ms": {
                    "runs": len(latencies),
                    "p50": statistics.median(latencies) if latencies else None,
                    "p95": percentile(latencies, 0.95),
                },
                "calibration": calibration,
                "results": model_results,
            }
        except Exception as error:  # noqa: BLE001 - a bench must report, not crash
            overall_ok = False
            results[model_id] = {
                "error": str(error),
                "error_type": type(error).__name__,
                "cases": len(model_cases),
                "ok": False,
            }
        finally:
            if model is not None:
                model.close()

    return {
        "suite": str(path),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "repeats": repeats,
        "environment": environment(),
        "models": results,
        "ok": overall_ok,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--home", help="Toolkit home (default: standard cache).")
    parser.add_argument("--repeats", type=int, default=2, help="Runs per case (default 2).")
    parser.add_argument("--models", help="Comma-separated model ids.")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout.")
    args = parser.parse_args(argv)

    models_filter = (
        {item.strip() for item in args.models.split(",") if item.strip()}
        if args.models
        else None
    )
    report = run_suite(
        args.suite,
        home=args.home,
        repeats=max(1, args.repeats),
        models_filter=models_filter,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
