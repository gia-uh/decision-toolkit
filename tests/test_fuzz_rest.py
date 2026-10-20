"""Deterministic fuzz/negative corpus for the REST boundary.

The goal is not coverage for its own sake: malformed input must never turn into
a 500 or a traceback. Every failure must use the standard envelope.
"""

from __future__ import annotations

import random

import pytest

from decision_toolkit.decision_model import DecisionModel
from decision_toolkit.server import create_app

from .conftest import FakeAdapter

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

pytestmark = pytest.mark.fuzz


def _random_value(rng: random.Random, depth: int = 0):
    choices = [
        None,
        True,
        False,
        0,
        1,
        -1,
        3.14,
        "",
        "SECRET_FUZZ_STATE",
        "x" * 300,
        [],
        {},
    ]
    if depth < 3:
        choices.extend(
            [
                [rng.choice(choices) for _ in range(rng.randint(0, 3))],
                {
                    f"k{index}": rng.choice(choices)
                    for index in range(rng.randint(0, 3))
                },
            ]
        )
    return rng.choice(choices)


def _client(api_key: str | None = None) -> TestClient:
    return TestClient(
        create_app(
            model=DecisionModel(adapter=FakeAdapter()),
            api_key=api_key,
            max_concurrent=2,
        )
    )


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/decisions"])
def test_fuzz_requests_never_return_500(path):
    rng = random.Random(20261020)
    client = _client()
    for _ in range(120):
        payload = {
            "state": _random_value(rng),
            "input": _random_value(rng),
            "questions": _random_value(rng),
            "model": _random_value(rng),
            "variant": _random_value(rng),
        }
        response = client.post(path, json=payload)
        assert response.status_code != 500, (path, payload, response.text)
        if response.status_code >= 400:
            body = response.json()
            assert "error" in body
            assert body["error"]["message"]


def test_fuzz_auth_headers_never_return_500():
    rng = random.Random(20261021)
    client = _client(api_key="correct")
    values = ["", "Bearer", "Basic abc", "Bearer wrong", "x" * 2000, "Bearer " + "y" * 2000]
    for _ in range(50):
        header = rng.choice(values)
        response = client.get("/v1/models", headers={"Authorization": header})
        assert response.status_code in {200, 401}
        if response.status_code == 401:
            assert "error" in response.json()


def test_fuzz_systemone_question_shapes_never_return_500():
    rng = random.Random(20261022)
    client = _client()
    for _ in range(120):
        questions = {
            f"q{index}": {
                "type": rng.choice(["choice", "noul", "score", "box", "nope", None]),
                "instructions": _random_value(rng),
                "criteria": _random_value(rng),
            }
            for index in range(rng.randint(0, 4))
        }
        response = client.post(
            "/v1/systemone",
            json={"state": "SECRET_FUZZ_STATE", "questions": questions},
        )
        assert response.status_code != 500, (questions, response.text)
        if response.status_code >= 400:
            assert "error" in response.json()
