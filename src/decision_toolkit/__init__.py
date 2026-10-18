"""decision-toolkit — local-first toolkit for open-weights decision models.

Decisions, not text: typed ``choice`` / yes-no (``noul``) / ``score`` questions
over a state, answered with a calibrated probability for every option, on CPU.
Library-first, with CLI (``dtk``) and REST interfaces as thin adapters.

>>> from decision_toolkit import DecisionModel, choice, noul
>>> model = DecisionModel("laya")                       # doctest: +SKIP
>>> answer = model.decide("Refund the duplicate charge.", {   # doctest: +SKIP
...     "team": choice("Which team?", {"billing": "refunds", "tech": "bugs"}),
...     "urgent": noul("Does it need a reply today?"),
... })
>>> answer["team"].choice, answer["urgent"].noul       # doctest: +SKIP
('billing', 0.91)
"""

from __future__ import annotations

__version__ = "0.0.1"

from .calibration import brier_score, expected_calibration_error, threshold_sweep
from .decision_model import DecisionModel
from .errors import (
    BackendError,
    BackendNotAvailableError,
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
    "BackendNotAvailableError",
    "InvalidQuestionError",
    "BackendError",
]
