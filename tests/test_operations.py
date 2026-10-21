"""Phase 5 operations: rate limits, backpressure, metrics and PII."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from decision_toolkit.decision_model import DecisionModel
from decision_toolkit.limits import ConcurrencyGate, RateLimiter
from decision_toolkit.metrics import Metrics, metric_path
from decision_toolkit.server import create_app

from .conftest import FakeAdapter

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def _body(state="state", question="SECRET_QUESTION"):
    return {
        "state": state,
        "questions": {"q": {"type": "noul", "instructions": question}},
    }


# ----------------------------------------------------------------- primitives
def test_rate_limiter_token_bucket():
    limiter = RateLimiter(rate=2, window=60, burst=2)
    assert limiter.allow("k", now=0.0)[0] is True
    assert limiter.allow("k", now=0.0)[0] is True
    allowed, retry = limiter.allow("k", now=0.0)
    assert allowed is False
    assert retry > 0
    # one token refills after 30 seconds at 2/60 per second
    assert limiter.allow("k", now=30.0)[0] is True


def test_rate_limiter_disabled_and_isolated_keys():
    disabled = RateLimiter(rate=0)
    assert disabled.allow("anything")[0] is True

    limiter = RateLimiter(rate=1, window=60, burst=1)
    assert limiter.allow("a", now=0)[0] is True
    assert limiter.allow("a", now=0)[0] is False
    assert limiter.allow("b", now=0)[0] is True


def test_concurrency_gate_rejects_when_full():
    async def run():
        gate = ConcurrencyGate(1)
        assert await gate.try_acquire() is True
        assert gate.active == 1
        assert await gate.try_acquire() is False
        await gate.release()
        assert gate.active == 0
        assert await gate.try_acquire() is True
        await gate.release()
        disabled = ConcurrencyGate(0)
        assert await disabled.try_acquire() is True
        await disabled.release()

    asyncio.run(run())


def test_metrics_render_and_path_normalization():
    metrics = Metrics()
    metrics.inc("dtk_http_requests_total", {"method": "GET", "path": "/v1/models", "status": "200"})
    metrics.observe("dtk_http_request_duration_seconds", {"method": "GET", "path": "/v1/models"}, 0.01)
    metrics.set_gauge("dtk_inflight_requests", value=2)
    text = metrics.render()
    assert "dtk_http_requests_total{method=\"GET\",path=\"/v1/models\",status=\"200\"} 1" in text
    assert "dtk_http_request_duration_seconds_bucket" in text
    assert "dtk_inflight_requests 2" in text
    assert metric_path("/v1/models/laya") == "/v1/models/{id}"
    assert metric_path("/nope") == "other"
    assert "SECRET_QUESTION" not in text


# -------------------------------------------------------------------- server
def test_metrics_endpoint_reports_decisions_without_content():
    client = TestClient(create_app(model=DecisionModel(adapter=FakeAdapter())))
    response = client.post("/v1/systemone", json=_body(state="SECRET_STATE_123"))
    assert response.status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    assert "dtk_http_requests_total" in text
    assert "dtk_decisions_total" in text
    assert "dtk_decision_duration_seconds" in text
    assert "SECRET_STATE_123" not in text
    assert "SECRET_QUESTION" not in text


def test_metrics_endpoint_can_be_disabled():
    client = TestClient(
        create_app(model=DecisionModel(adapter=FakeAdapter()), metrics_enabled=False)
    )
    assert client.get("/metrics").status_code == 404


def test_rate_limit_returns_429_and_exempts_health():
    client = TestClient(
        create_app(model=DecisionModel(adapter=FakeAdapter()), rate_limit=1, rate_window=60)
    )
    assert client.post("/v1/systemone", json=_body()).status_code == 200
    limited = client.post("/v1/systemone", json=_body())
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(limited.headers["retry-after"]) >= 1
    assert client.get("/health").status_code == 200
    assert "dtk_rate_limited_total" in client.get("/metrics").text


def test_backpressure_returns_503_instead_of_queueing():
    class SlowAdapter(FakeAdapter):
        name = "slow"

        def system_one(self, engine, state, questions):
            time.sleep(0.25)
            return super().system_one(engine, state, questions)

    app = create_app(
        model=DecisionModel(adapter=SlowAdapter()),
        max_concurrent=1,
        timeout_seconds=10,
    )
    httpx = pytest.importorskip("httpx")

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await asyncio.gather(
                client.post("/v1/systemone", json=_body()),
                client.post("/v1/systemone", json=_body()),
            )

    responses = asyncio.run(run())
    statuses = {response.status_code for response in responses}
    assert statuses == {200, 503}
    busy = next(response for response in responses if response.status_code == 503)
    assert busy.json()["error"]["code"] == "server_busy"
    assert int(busy.headers["retry-after"]) >= 1


def test_logs_and_errors_do_not_leak_state_question_or_token(caplog):
    app = create_app(model=DecisionModel(adapter=FakeAdapter()), api_key="SECRET_TOKEN_789")
    client = TestClient(app)

    with caplog.at_level(logging.DEBUG, logger="decision_toolkit"):
        response = client.post(
            "/v1/systemone",
            json=_body(state="SECRET_STATE_123", question="SECRET_QUESTION"),
            headers={"Authorization": "Bearer SECRET_TOKEN_789"},
        )
        bad = client.post(
            "/v1/systemone",
            json={
                "state": "SECRET_STATE_123",
                "questions": {"q": {"type": "nope", "instructions": "SECRET_QUESTION"}},
            },
            headers={"Authorization": "Bearer SECRET_TOKEN_789"},
        )

    assert response.status_code == 200
    assert bad.status_code == 422
    text = caplog.text
    assert "SECRET_STATE_123" not in text
    assert "SECRET_QUESTION" not in text
    assert "SECRET_TOKEN_789" not in text
    assert "SECRET_STATE_123" not in response.text
    assert "SECRET_QUESTION" not in response.text
    assert "SECRET_STATE_123" not in bad.text
    assert "SECRET_QUESTION" not in bad.text
    assert "SECRET_TOKEN_789" not in bad.text
