"""decision-toolkit — local-first toolkit for open-weights decision models.

One ONNX engine, one adapter per model contract, three surfaces (library, CLI,
REST). Decisions, not text: typed ``choice`` / yes-no (``noul``) / ``score``
questions with a calibrated probability for every option, on CPU.
"""

from __future__ import annotations

__version__ = "0.1.6"

from .adapters import register_adapter
from .calibration import brier_score, expected_calibration_error, threshold_sweep
from .catalog import Capabilities, ModelInfo
from .decision_model import DecisionModel
from .errors import (
    AdapterError,
    AdapterNotAvailableError,
    AdapterNotImplementedError,
    BackendFailureError,
    DecisionToolkitError,
    InvalidQuestionError,
    InvalidRequestError,
    ModelNotDownloadedError,
    ModelNotFoundError,
    MultimodalNotSupportedError,
    ServerBusyError,
    TruncationError,
    UnsupportedCapabilityError,
    VerificationError,
)
from .models import ModelStore
from .observability import (
    configure_logging,
    ensure_request_id,
    get_request_id,
    request_context,
)
from .results import Answer, Classification, Decision, Usage
from .schema import (
    Box,
    Choice,
    Noul,
    Score,
    State,
    box,
    choice,
    normalize_state,
    noul,
    score,
)

__all__ = [
    # errors
    "AdapterError",
    "AdapterNotAvailableError",
    "AdapterNotImplementedError",
    # results
    "Answer",
    "BackendFailureError",
    # schema
    "Box",
    # catalog
    "Capabilities",
    "Choice",
    "Classification",
    "Decision",
    # core
    "DecisionModel",
    "DecisionToolkitError",
    "InvalidQuestionError",
    "InvalidRequestError",
    "ModelInfo",
    "ModelNotDownloadedError",
    "ModelNotFoundError",
    "ModelStore",
    "MultimodalNotSupportedError",
    "Noul",
    "Score",
    "ServerBusyError",
    "State",
    "TruncationError",
    "UnsupportedCapabilityError",
    "Usage",
    "VerificationError",
    "__version__",
    "box",
    # calibration
    "brier_score",
    "choice",
    # observability
    "configure_logging",
    "ensure_request_id",
    "expected_calibration_error",
    "get_request_id",
    "normalize_state",
    "noul",
    "register_adapter",
    "request_context",
    "score",
    "threshold_sweep",
]
