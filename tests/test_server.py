"""The REST projection: System One, OpenAI Decisions, contracts and operations."""

from __future__ import annotations

import logging

import pytest

from decision_toolkit.decision_model import DecisionModel
from decision_toolkit.models import ModelStore
from decision_toolkit.registry import ModelRegistry

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from decision_toolkit.server import access_logger, create_app

from .conftest import FakeAdapter


def _app(**kwargs):
    return create_app(model=DecisionModel(adapter=FakeAdapter()), **kwargs)


class _VariantModel(DecisionModel):
    """Fake inference model that reports a selected catalog variant."""

    def __init__(self, variant: str) -> None:
        super().__init__(adapter=FakeAdapter())
        self._selected_variant = variant

    @property
    def identifier(self) -> str:
        return "gliner2.5-multi"

    @property
    def variant(self) -> str | None:
        return self._selected_variant


@pytest.fixture
def client():
    return TestClient(_app())


def _systemone_body(**extra):
    return {
        "state": "Customer wants a refund.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "refunds", "tech": "bugs"},
            },
            "urgent": {"type": "noul", "instructions": "Today?"},
        },
        **extra,
    }


# ---------------------------------------------------------------- meta/health
def test_health_and_ready(client):
    assert client.get("/v1/health").json() == {"status": "ok"}
    assert client.get("/health").status_code == 200
    ready = client.get("/v1/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert ready.json()["adapter"] == "fake"


def test_ready_is_503_until_the_default_model_is_warm(home):
    app = create_app(model_id="gliner2.5-multi", home=str(home))
    response = TestClient(app).get("/v1/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert "not warm" in response.json()["detail"]


def test_info_exposes_pinned_contract_versions(client):
    info = client.get("/v1/info").json()
    assert info["name"] == "decision-toolkit"
    assert info["contracts"]["systemone"] == "0.2.0"
    assert info["contracts"]["openai_decisions"] == "beta"
    assert info["registry"]["default_model"] == "(none)"


def test_registry_default_variant_uses_the_catalog_default(home):
    registry = ModelRegistry(model_id="laya", home=str(home))
    assert registry.default_identifier == "laya"
    assert registry.default_variant == "q8"
    ready, detail = registry.is_ready()
    assert ready is False
    assert detail["variant"] == "q8"


def test_server_reports_catalog_default_variant(home):
    app = create_app(model_id="laya", home=str(home))
    client = TestClient(app)
    info = client.get("/v1/info").json()
    assert info["registry"]["default_model"] == "laya"
    assert info["registry"]["default_variant"] == "q8"
    ready = client.get("/v1/ready")
    assert ready.status_code == 503
    assert ready.json()["variant"] == "q8"


def test_registry_default_variant_uses_persisted_selection(home, fake_snapshot):
    store = ModelStore(home=home)
    store.download("gliner2.5-multi", variant="fp16")
    store.download("gliner2.5-multi", variant="fp32")
    store.use("gliner2.5-multi", variant="fp32")
    registry = ModelRegistry(model_id="gliner2.5-multi", home=str(home))
    assert registry.default_variant == "fp32"
    implicit = ModelRegistry(home=str(home))
    assert implicit.default_identifier == "gliner2.5-multi"
    assert implicit.default_variant == "fp32"


def test_rest_default_request_uses_persisted_variant(
    monkeypatch, home, fake_snapshot
):
    from decision_toolkit import registry as registry_module

    store = ModelStore(home=home)
    store.download("gliner2.5-multi", variant="fp16")
    store.download("gliner2.5-multi", variant="fp32")
    store.use("gliner2.5-multi", variant="fp32")

    seen: list[tuple[str | None, str | None]] = []

    def fake_model(identifier=None, **kwargs):
        seen.append((identifier, kwargs.get("variant")))
        return _VariantModel(kwargs.get("variant") or "fp16")

    monkeypatch.setattr(registry_module, "DecisionModel", fake_model)
    app = create_app(model_id="gliner2.5-multi", home=str(home))
    response = TestClient(app).post("/v1/systemone", json=_systemone_body())
    assert response.status_code == 200
    assert response.json()["x_decision_toolkit"]["variant"] == "fp32"
    assert seen == [("gliner2.5-multi", "fp32")]


def test_rest_explicit_variant_is_passed_to_the_model(monkeypatch, home):
    from decision_toolkit import registry as registry_module

    seen: list[tuple[str | None, str | None]] = []

    def fake_model(identifier=None, **kwargs):
        seen.append((identifier, kwargs.get("variant")))
        return _VariantModel(kwargs.get("variant") or "fp16")

    monkeypatch.setattr(registry_module, "DecisionModel", fake_model)
    app = create_app(model_id="gliner2.5-multi", home=str(home))
    response = TestClient(app).post(
        "/v1/systemone", json=_systemone_body(variant="fp32")
    )
    assert response.status_code == 200
    assert response.json()["x_decision_toolkit"]["variant"] == "fp32"
    assert seen == [("gliner2.5-multi", "fp32")]


def test_rest_openai_request_passes_variant(monkeypatch, home):
    from decision_toolkit import registry as registry_module

    seen: list[tuple[str | None, str | None]] = []

    def fake_model(identifier=None, **kwargs):
        seen.append((identifier, kwargs.get("variant")))
        return _VariantModel(kwargs.get("variant") or "fp16")

    monkeypatch.setattr(registry_module, "DecisionModel", fake_model)
    app = create_app(model_id="gliner2.5-multi", home=str(home))
    response = TestClient(app).post(
        "/v1/decisions",
        json={
            "input": "Customer wants a refund.",
            "variant": "fp32",
            "questions": [
                {
                    "type": "choice",
                    "name": "team",
                    "instructions": "Which team?",
                    "choices": [
                        {"value": "billing", "description": "refunds"},
                        {"value": "technical", "description": "bugs"},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["x_decision_toolkit"]["variant"] == "fp32"
    assert seen == [("gliner2.5-multi", "fp32")]


def test_rest_catalog_exposes_current_variant(home, fake_snapshot):
    store = ModelStore(home=home)
    store.download("gliner2.5-multi", variant="fp16")
    store.download("gliner2.5-multi", variant="fp32")
    store.use("gliner2.5-multi", variant="fp32")

    app = create_app(model_id="gliner2.5-multi", home=str(home))
    client = TestClient(app)
    rich = client.get("/v1/decision-models").json()
    entry = next(item for item in rich["data"] if item["id"] == "gliner2.5-multi")
    assert entry["current"] is True
    assert entry["current_variant"] == "fp32"
    assert set(entry["cached_variants"]) == {"fp16", "fp32"}

    info = client.get("/v1/info").json()
    assert info["registry"]["default_model"] == "gliner2.5-multi"
    assert info["registry"]["default_variant"] == "fp32"
    ready = client.get("/v1/ready")
    assert ready.status_code == 503
    assert ready.json()["variant"] == "fp32"


def test_fixed_model_rejects_a_different_variant(client):
    response = client.post(
        "/v1/systemone", json=_systemone_body(variant="fp16")
    )
    assert response.status_code == 404
    assert "bound" in response.json()["error"]["message"]


# ----------------------------------------------------------------- system one
def test_systemone_exact_success_shape(client):
    response = client.post(
        "/v1/systemone",
        json={
            "state": "I was charged twice and nobody replied.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {"billing": "payments", "shipping": None, "technical": None},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["can wait", "this week", "today"],
                },
                "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "(none)"
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] > 0
    assert list(body["answers"]) == ["route", "urgency", "angry"]

    route = body["answers"]["route"]
    assert route["type"] == "choice"
    assert route["choice"] == "billing"
    assert set(route["probabilities"]) == {"billing", "shipping", "technical"}
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-6
    assert route["choice"] == max(route["probabilities"], key=route["probabilities"].get)
    assert 0.0 <= route["confidence"] <= 1.0

    urgency = body["answers"]["urgency"]
    assert urgency["type"] == "score"
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today"}
    assert abs(urgency["score"] - 1.0) < 1e-6
    assert 0.0 <= urgency["confidence"] <= 1.0

    angry = body["answers"]["angry"]
    assert angry["type"] == "noul"
    assert angry["noul"] == 0.7

    assert body["x_decision_toolkit"]["spec"] == "0.2.0"
    assert body["x_decision_toolkit"]["usage"]["estimated"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"questions": {"q": {"type": "noul", "instructions": "x"}}},
        {"state": "x"},
        {"state": "x", "questions": {}},
        {"state": "x", "questions": {"q": {"type": "unknown", "instructions": "x"}}},
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": 123}}},
        {"state": "x", "questions": {"q": {"type": "choice", "instructions": "x"}}},
        {
            "state": "x",
            "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}},
        },
    ],
)
def test_systemone_invalid_requests_use_the_validation_detail(client, payload):
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail and detail[0]["msg"]


def test_systemone_accepts_standard_rich_instructions_and_criteria(client):
    response = client.post(
        "/v1/systemone",
        json={
            "state": {"message": "I was charged twice."},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": {"task": "route the message"},
                    "criteria": {
                        "billing": {"meaning": "payments and refunds"},
                        "other": ["anything else"],
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": [{"label": "low"}, ["high"]],
                },
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answers"]["route"]["type"] == "choice"
    assert body["answers"]["severity"]["type"] == "score"


def test_systemone_accepts_single_level_score(client):
    response = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["only level"],
                }
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["answers"]["severity"]["type"] == "score"


def test_systemone_accepts_assets_field_but_rejects_unsupported_vision(client):
    response = client.post(
        "/v1/systemone",
        json={
            "model": "gliner2.5-multi",
            "state": "photo",
            "assets": {
                "photo": {
                    "type": "image",
                    "format": "png",
                    "data_base64": "AAAA",
                }
            },
            "questions": {
                "q": {"type": "noul", "instructions": "Is it valid?"}
            },
        },
    )
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "multimodal_not_supported"


def test_embedded_images_are_501_in_both_contracts(client):
    systemone = client.post(
        "/v1/systemone",
        json={
            "state": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
            "questions": {"q": {"type": "noul", "instructions": "Is it valid?"}},
        },
    )
    assert systemone.status_code == 501
    assert systemone.json()["error"]["code"] == "multimodal_not_supported"

    decisions = client.post(
        "/v1/decisions",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,AAA"}
                    ],
                }
            ],
            "questions": [{"type": "predicate", "name": "q", "instructions": "Is it valid?"}],
        },
    )
    assert decisions.status_code == 501
    assert decisions.json()["error"]["code"] == "multimodal_not_supported"


def test_systemone_images_are_501_and_bad_urls_are_400(client):
    data_url = "data:image/png;base64,AAAA"
    response = client.post("/v1/systemone", json=_systemone_body(images=[data_url]))
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "multimodal_not_supported"

    response = client.post(
        "/v1/systemone", json=_systemone_body(images=["https://example.test/image.png"])
    )
    assert response.status_code == 422
    assert response.json()["detail"]

    response = client.post(
        "/v1/systemone", json=_systemone_body(images=[data_url] * 9)
    )
    assert response.status_code == 422
    assert response.json()["detail"]


# -------------------------------------------------------------- openai dialect
def test_decisions_exact_success_shape(client):
    response = client.post(
        "/v1/decisions",
        json={
            "input": "Customer wants a refund.",
            "questions": [
                {"type": "predicate", "name": "refund", "instructions": "Is it a refund?"},
                {
                    "type": "choice",
                    "name": "team",
                    "instructions": "Which team?",
                    "choices": [{"value": "billing"}, {"value": "tech"}],
                },
                {
                    "type": "score",
                    "name": "urgency",
                    "instructions": "How urgent?",
                    "levels": [{"label": "low"}, {"label": "high"}],
                },
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "answers", "usage", "x_decision_toolkit"}
    assert body["model"] == "(none)"
    assert body["usage"]["input_tokens"] >= 0
    assert body["usage"]["output_tokens"] > 0
    assert body["usage"]["total_tokens"] == (
        body["usage"]["input_tokens"] + body["usage"]["output_tokens"]
    )
    assert "input_tokens_details" in body["usage"]
    assert "output_tokens_details" in body["usage"]
    assert body["answers"][0] == {
        "type": "predicate",
        "name": "refund",
        "probability": 0.7,
    }
    assert body["answers"][1]["choice"] == "billing"
    assert body["answers"][2]["probabilities"][1]["label"] == "high"
    assert body["x_decision_toolkit"]["spec"] == "beta"


def test_decisions_choice_values_roundtrip(client):
    response = client.post(
        "/v1/decisions",
        json={
            "input": "x",
            "questions": [
                {
                    "type": "choice",
                    "name": "answer",
                    "instructions": "Pick",
                    "choices": [{"value": 1}, {"value": 2}],
                }
            ],
        },
    )
    assert response.status_code == 200
    answer = response.json()["answers"][0]
    assert answer["choice"] == 1
    assert [entry["value"] for entry in answer["probabilities"]] == [1, 2]


def test_decisions_rejects_duplicate_names_and_bad_bodies(client):
    response = client.post(
        "/v1/decisions",
        json={
            "input": "x",
            "questions": [
                {"type": "predicate", "name": "same"},
                {"type": "predicate", "name": "same"},
            ],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"

    assert client.post("/v1/decisions", json={"questions": []}).status_code == 400
    assert client.post(
        "/v1/decisions", json={"input": "x", "questions": "nope"}
    ).status_code == 400


def test_http_is_a_projection_of_the_core_decision():
    from decision_toolkit.contracts import systemone_response

    model = DecisionModel(adapter=FakeAdapter())
    questions = {
        "urgent": {"type": "noul", "instructions": "Today?"},
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "refunds", "tech": None},
        },
    }
    core = model.decide("state", questions, request_id="core-1")
    expected = systemone_response(core)
    client = TestClient(create_app(model=model))
    body = client.post(
        "/v1/systemone", json={"state": "state", "questions": questions}
    ).json()
    assert body["answers"] == expected.answers
    assert body["usage"] == expected.usage.model_dump()


# ------------------------------------------------------------------ operations
def test_bearer_auth_is_optional_but_enforced_when_configured():
    app = _app(api_key="secret")
    client = TestClient(app)
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/ready").status_code == 200

    unauthorized = client.get("/v1/models")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"
    assert unauthorized.headers["www-authenticate"] == "Bearer"

    assert client.get(
        "/v1/models", headers={"Authorization": "Bearer secret"}
    ).status_code == 200
    assert client.post(
        "/v1/systemone",
        json=_systemone_body(),
        headers={"Authorization": "Bearer secret"},
    ).status_code == 200


def test_json_log_format_emits_one_object_per_line(capsys):
    import json

    from decision_toolkit.observability import configure_logging, log_event

    logger = configure_logging("INFO", "json", logger_name="decision_toolkit.test.json")
    try:
        log_event(logger, "test.event", request_id="req-json", status=200)
        line = capsys.readouterr().err.strip()
        payload = json.loads(line)
        assert payload["event"] == "test.event"
        assert payload["request_id"] == "req-json"
        assert payload["status"] == 200
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.propagate = True


def test_request_id_is_accepted_generated_and_logged(caplog):
    client = TestClient(_app())
    echoed = client.get("/v1/health", headers={"x-request-id": "req-123"})
    assert echoed.headers["x-request-id"] == "req-123"

    with caplog.at_level(logging.INFO, logger=access_logger.name):
        generated = client.get("/v1/health")
    assert generated.headers["x-request-id"]
    assert generated.headers["x-request-id"] != "req-123"
    assert "http.request" in caplog.text


def test_timeout_returns_504_with_a_stable_code():
    import time

    class SlowAdapter(FakeAdapter):
        name = "slow"

        def system_one(self, engine, state, questions):
            time.sleep(0.2)
            return super().system_one(engine, state, questions)

    app = create_app(model=DecisionModel(adapter=SlowAdapter()), timeout_seconds=0.01)
    response = TestClient(app).post("/v1/systemone", json=_systemone_body())
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "timeout"


def test_body_limit_returns_413_error_envelope():
    client = TestClient(_app(max_body_mb=0.0001))
    response = client.post("/v1/systemone", json=_systemone_body(state="x" * 500))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_cors_is_configurable_and_preflight_skips_auth():
    client = TestClient(
        _app(api_key="secret", cors_origins=["https://example.test"])
    )
    response = client.options(
        "/v1/systemone",
        headers={
            "Origin": "https://example.test",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://example.test"
    assert response.headers["x-request-id"]


def test_unknown_model_is_404_with_envelope():
    client = TestClient(_app())
    response = client.post(
        "/v1/systemone", json=_systemone_body(model="does-not-exist")
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


# -------------------------------------------------------------------- models
def test_models_endpoint_uses_the_openai_shape(client):
    payload = client.get("/v1/models").json()
    assert payload["object"] == "list"
    assert all(item["object"] == "model" for item in payload["data"])
    assert {"id", "object", "created", "owned_by"} <= set(payload["data"][0])
    assert "gliner2.5-multi" in {item["id"] for item in payload["data"]}


def test_models_detail_and_rich_catalog(client):
    detail = client.get("/v1/models/gliner2.5-multi").json()
    assert detail["id"] == "gliner2.5-multi"
    assert detail["x_decision_toolkit"]["adapter"] == "gliner2"
    assert detail["x_decision_toolkit"]["capabilities"]["question_types"] == [
        "choice",
        "noul",
        "score",
    ]

    rich = client.get("/v1/decision-models").json()
    entry = next(item for item in rich["data"] if item["id"] == "gliner2.5-multi")
    assert {variant["name"] for variant in entry["variants"]} == {"fp16", "fp32"}
    default = next(v for v in entry["variants"] if v["role"] == "default")
    assert default["latency_ms_p50"] == 353.0
    assert default["parity"]["baseline"] == "mininvox int8 build"
    assert entry["capabilities"]["max_options"] == 255


def test_openai_sdk_can_list_models(client):
    openai = pytest.importorskip("openai")

    sdk = openai.OpenAI(
        api_key="test", base_url="http://testserver/v1", http_client=client
    )
    ids = {model.id for model in sdk.models.list()}
    assert "gliner2.5-multi" in ids
    assert "julia-1" in ids
    detail = sdk.models.retrieve("gliner2.5-multi")
    assert detail.id == "gliner2.5-multi"
    assert detail.owned_by == "decision-toolkit"


def test_openai_sdk_parses_the_error_envelope(client):
    openai = pytest.importorskip("openai")

    sdk = openai.OpenAI(
        api_key="test", base_url="http://testserver/v1", http_client=client
    )
    with pytest.raises(openai.NotFoundError) as excinfo:
        sdk.models.retrieve("does-not-exist")
    assert excinfo.value.status_code == 404
    assert excinfo.value.body["code"] == "model_not_found"


def test_openai_sdk_can_post_a_decision(client):
    openai = pytest.importorskip("openai")

    sdk = openai.OpenAI(
        api_key="test", base_url="http://testserver/v1", http_client=client
    )
    payload = sdk.post(
        "/decisions",
        cast_to=dict,
        body={
            "input": "Customer wants a refund.",
            "questions": [
                {"type": "predicate", "name": "refund", "instructions": "Is it a refund?"}
            ],
        },
    )
    assert payload["model"]
    assert payload["answers"][0]["type"] == "predicate"


def test_openapi_exposes_real_contract_schemas(client):
    schema = client.get("/openapi.json").json()
    assert "/v1/systemone" in schema["paths"]
    assert "/v1/decisions" in schema["paths"]
    assert "SystemOneRequest" in schema["components"]["schemas"]
    assert "OpenAIDecisionsRequest" in schema["components"]["schemas"]
    assert "ErrorResponse" in schema["components"]["schemas"]


def test_per_request_model_selection_uses_the_registry(monkeypatch, home):
    from decision_toolkit import registry as registry_module

    def fake_model(identifier, **kwargs):
        return DecisionModel(adapter=FakeAdapter())

    monkeypatch.setattr(registry_module, "DecisionModel", fake_model)
    app = create_app(model_id="alpha", home=str(home))
    client = TestClient(app)
    assert client.post(
        "/v1/systemone", json=_systemone_body(model="alpha")
    ).status_code == 200
    assert client.post(
        "/v1/systemone", json=_systemone_body(model="beta", variant="q8")
    ).status_code == 200
    loaded = app.state.registry.describe()["loaded"]
    keys = {item["key"] for item in loaded}
    assert "alpha@" in keys
    assert "beta@q8" in keys
