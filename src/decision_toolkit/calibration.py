"""`decision_toolkit.calibration` — do the probabilities mean what they say?

Decision models answer with probabilities; whether those probabilities are
*calibrated* (a 0.8 answer is right ~80% of the time) varies by model, by
quantization and by domain. The toolkit keeps this small but first-class:
measure before you threshold.

A *pair* is ``(probability_given_to_the_chosen_answer, was_it_right)`` — build
it from your own labelled rows with :func:`pairs_from_rows`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

Pair = tuple[float, bool]


def pairs_from_rows(rows: Iterable[tuple[float, bool]]) -> list[Pair]:
    """Normalize/validate ``(probability, correct)`` rows."""
    pairs: list[Pair] = []
    for probability, correct in rows:
        p = float(probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability out of [0, 1]: {p}")
        pairs.append((p, bool(correct)))
    return pairs


def brier_score(pairs: Sequence[Pair]) -> float:
    """Mean squared error of the probabilities (lower is better; 0 is perfect)."""
    if not pairs:
        raise ValueError("brier_score needs at least one pair")
    return sum((probability - float(correct)) ** 2 for probability, correct in pairs) / len(pairs)


def expected_calibration_error(pairs: Sequence[Pair], n_bins: int = 10) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over bins."""
    if not pairs:
        raise ValueError("expected_calibration_error needs at least one pair")
    bins: list[list[Pair]] = [[] for _ in range(n_bins)]
    for probability, correct in pairs:
        index = min(int(probability * n_bins), n_bins - 1)
        bins[index].append((probability, correct))

    total = len(pairs)
    error = 0.0
    for bucket in bins:
        if not bucket:
            continue
        confidence = sum(p for p, _ in bucket) / len(bucket)
        accuracy = sum(1.0 for _, correct in bucket if correct) / len(bucket)
        error += (len(bucket) / total) * abs(accuracy - confidence)
    return error


def threshold_sweep(pairs: Sequence[Pair], thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9)
                    ) -> dict[float, dict[str, float]]:
    """Coverage and kept accuracy at each confidence threshold.

    ``coverage`` is the share of items the model answers at or above the
    threshold; ``accuracy_kept`` is how often those answers are right. An item
    counts as right when its probability side (>= 0.5) matches its label —
    pass ``correct=True/False`` directly to decide a different rule.
    """
    if not pairs:
        raise ValueError("threshold_sweep needs at least one pair")
    report: dict[float, dict[str, float]] = {}
    for threshold in thresholds:
        kept = [(p, c) for p, c in pairs if p >= threshold]
        report[float(threshold)] = {
            "coverage": len(kept) / len(pairs),
            "accuracy_kept": (sum(1.0 for _, c in kept) / len(kept)) if kept else 0.0,
            "n_kept": float(len(kept)),
        }
    return report


__all__ = ["Pair", "brier_score", "expected_calibration_error", "pairs_from_rows", "threshold_sweep"]
