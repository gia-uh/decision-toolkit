"""decision-toolkit — local-first toolkit for open-weights decision models.

One ONNX engine, one adapter per model contract, three surfaces (library, CLI,
REST). Decisions, not text: typed ``choice`` / yes-no (``noul``) / ``score``
questions with a calibrated probability for every option, on CPU.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .calibration import brier_score, expected_calibration_error, threshold_sweep
from .decision_model import DecisionModel
from .errors import (
    AdapterError,
    AdapterNotAvailableError,
    DecisionToolkitError,
    InvalidQuestionError,
    ModelNotFoundError,
)
from .models import ModelStore
from .results import Answer, Classification, Decision
from .schema import Choice, Noul, Score, choice, noul, score

__all__ = [
    "__version__",
    # core
    "DecisionModel",
    "ModelStore",
    # results
    "Answer",
    "Classification",
    "Decision",
    # schema
    "Choice",
    "Noul",
    "Score",
    "choice",
    "noul",
    "score",
    # calibration
    "brier_score",
    "expected_calibration_error",
    "threshold_sweep",
    # errors
    "DecisionToolkitError",
    "ModelNotFoundError",
    "AdapterNotAvailableError",
    "AdapterError",
    "InvalidQuestionError",
]
