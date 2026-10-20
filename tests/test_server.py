"""The REST surface: System One and OpenAI Decisions over the same engine."""

from __future__ import annotations

import pytest

from decision_toolkit.decision_model import DecisionModel
from decision_toolkit.server import create_app

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from .conftest import FakeAdapter


@pytest.fixture
def client():
    app = create_app(model=DecisionModel(adapter=FakeAdapter()))
    return TestClient(app)


def test_health_and_ready(client):
    assert client.get("/v1/health").json() == {"status": "ok"}
    ready = client.get("/v1/ready").json()
    assert ready["ready"] is True
    assert ready["adapter"] == "fake"


def test_info(client):
    info = client.get("/v1/info").json()
    assert info["name"] == "decision-toolkit"
    assert info["model"] == "(none)"
    assert info["adapter"] == "fake"


def test_systemone_endpoint(client):
    response = client.post(
        "/v1/systemone",
        json={
            "state": "Customer wants a refund.",
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": "refunds", "tech": "bugs"}},
                "urgent": {"type": "noul", "instructions": "Today?"},
            },
        },
    )
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["team"]["choice"] == "billing"
    assert answers["urgent"]["noul"] == 0.7


def test_systemone_rejects_bad_body(client):
    assert client.post("/v1/systemone", json={"questions": {}}).status_code == 400
    assert client.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 400
    assert client.post(
        "/v1/systemone", json={"state": "x", "questions": {"q": {"type": "bogus"}}}
    ).status_code == 400


def test_decisions_endpoint_openai_dialect(client):
    response = client.post(
        "/v1/decisions",
        json={
            "input": "Customer wants a refund.",
            "questions": [
                {"type": "predicate", "name": "refund", "instructions": "Is it a refund?"},
                {"type": "choice", "name": "team", "instructions": "Which team?",
                 "choices": [{"value": "billing"}, {"value": "tech"}]},
                {"type": "score", "name": "urgency", "instructions": "How urgent?",
                 "levels": [{"label": "low"}, {"label": "mid"}, {"label": "high"}]},
            ],
        },
    )
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers[0]["type"] == "predicate"
    assert answers[0]["probability"] == 0.7
    assert answers[1]["choice"] == "billing"
    assert answers[2]["probabilities"][1]["label"] == "mid"


def test_decisions_rejects_bad_body(client):
    assert client.post("/v1/decisions", json={"questions": []}).status_code == 400
    assert client.post("/v1/decisions", json={"input": "x", "questions": "nope"}).status_code == 400


def test_models_endpoint_lists_catalog_with_variants(client):
    payload = client.get("/v1/models").json()
    entry = next(item for item in payload["catalog"] if item["id"] == "gliner2.5-multi")
    assert entry["adapter"] == "gliner2"
    assert {variant["name"] for variant in entry["variants"]} == {"fp16", "fp32"}
    default = next(variant for variant in entry["variants"] if variant["role"] == "default")
    assert default["latency_ms_p50"] == 353.0
    assert default["parity"]["baseline"] == "mininvox int8 build"
    assert default["verified_at"] == "2026-10-19"
