"""The reproducible bench suite files stay valid and pinned to the catalog."""

from __future__ import annotations

from pathlib import Path

from bench.run import load_cases, percentile
from decision_toolkit import catalog

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_suite_covers_every_model():
    cases = load_cases(ROOT / "bench/suites/smoke.jsonl")
    assert {case["model"] for case in cases} == {model.id for model in catalog.CATALOG}
    assert all(case.get("expected") for case in cases)


def test_calibration_suite_is_labeled():
    cases = load_cases(ROOT / "bench/suites/calibration.jsonl")
    assert cases
    assert all(case.get("labels") for case in cases)
    assert {case["model"] for case in cases} == {"gliner2.5-multi"}


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == 2.9
    assert percentile([], 0.95) is None
