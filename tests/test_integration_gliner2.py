"""Integration: the GLiNER2 adapter over a real local ONNX export, on CPU.

Opt-in. Point it at a directory with the split export (encoder + classifier +
tokenizer.json), e.g. the mininvox build::

    DECISION_TOOLKIT_GLINER2_DIR=/path/to/gliner25 \
        uv run pytest -m integration -o addopts="" -s

Never downloads anything: the files must already be on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from decision_toolkit import DecisionModel, choice

pytestmark = pytest.mark.integration

ENCODERS = ("encoder_int8.onnx", "encoder_q8.onnx", "encoder_fp16.onnx", "encoder.onnx")

CATEGORIES = {
    "reunion": "una reunión o llamada con temas, decisiones y acciones",
    "tareas": "cosas pendientes por hacer, con fechas o responsables",
    "contacto": "datos de una persona: nombre, rol, empresa, teléfono, correo",
    "idea": "una idea, reflexión, observación o pregunta",
    "otras": "saludo, charla suelta o cualquier cosa que no sea ninguna",
}


def _files() -> tuple[Path, dict[str, str]]:
    value = os.environ.get("DECISION_TOOLKIT_GLINER2_DIR")
    if not value:
        pytest.skip("set DECISION_TOOLKIT_GLINER2_DIR to a local GLiNER2 ONNX export")
    directory = Path(value)
    encoder = next((directory / name for name in ENCODERS if (directory / name).is_file()), None)
    classifier = directory / "classifier.onnx"
    tokenizer = directory / "tokenizer.json"
    if encoder is None or not classifier.is_file() or not tokenizer.is_file():
        pytest.skip(f"{directory} does not hold an encoder + classifier + tokenizer.json")
    return directory, {
        "encoder": encoder.name,
        "classifier": classifier.name,
        "tokenizer": tokenizer.name,
    }


def test_gliner2_real_onnx_choice():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    directory, files = _files()

    model = DecisionModel(
        str(directory),
        adapter="gliner2",
        files=files,
        adapter_options={"task": "categoria", "use_descriptions": True},
    )
    result = model.decide(
        "Reunión con Acme: acordamos un descuento del 10% y firmar la semana que viene.",
        {"cat": choice("¿Qué tipo de registro describe el texto?", CATEGORIES)},
    )
    answer = result["cat"]
    assert answer.choice in CATEGORIES
    probabilities = answer.probabilities
    assert set(probabilities) == set(CATEGORIES)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-3
    assert 0.0 <= answer.confidence <= 1.0
    print(f"\ncat={answer.choice} conf={answer.confidence:.2f} probs={probabilities}")
