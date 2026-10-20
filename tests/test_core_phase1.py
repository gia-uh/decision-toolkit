"""Phase 1 core: types, lifecycle, batching, usage, capabilities, logging, store."""

from __future__ import annotations

import asyncio
import logging

import pytest

from decision_toolkit.adapters import (
    create_adapter,
    register_adapter,
    unregister_adapter,
    validate_adapter_files,
)
from decision_toolkit.catalog import Capabilities
from decision_toolkit.decision_model import DecisionModel
from decision_toolkit.errors import (
    AdapterNotAvailableError,
    InvalidRequestError,
    ModelNotFoundError,
    TruncationError,
    UnsupportedCapabilityError,
)
from decision_toolkit.models import ModelStore
from decision_toolkit.observability import request_context
from decision_toolkit.results import Usage
from decision_toolkit.schema import State, box, noul

from .conftest import FakeAdapter


@pytest.fixture
def store(home) -> ModelStore:
    return ModelStore(home=home)


class TruncatingAdapter(FakeAdapter):
    name = "truncating"

    def system_one(self, engine, state, questions):
        answers = dict(super().system_one(engine, state, questions))
        for answer in answers.values():
            answer["truncated"] = True
            answer["warnings"] = ["input was truncated"]
        return answers


# --------------------------------------------------------------- basic surface
def test_decision_has_usage_latency_and_identity(fake_model):
    decision = fake_model.decide("state", {"urgent": noul("Urgent?")})
    assert isinstance(decision.usage, Usage)
    assert decision.usage.input_tokens > 0
    assert decision.usage.output_tokens > 0
    assert decision.usage.total_tokens == (
        decision.usage.input_tokens + decision.usage.output_tokens
    )
    assert decision.usage.estimated is True
    assert decision.latency_ms is not None and decision.latency_ms >= 0
    assert decision.model == "(none)"
    assert decision.adapter == "fake"
    assert decision.request_id
    payload = decision.to_payload()
    assert payload["usage"]["estimated"] is True
    assert payload["answers"]["urgent"]["noul"] == 0.7


def test_request_context_is_attached_to_decision(fake_model):
    with request_context("req-phase1") as request_id:
        assert request_id == "req-phase1"
        decision = fake_model.decide("state", {"urgent": noul("Urgent?")})
    assert decision.request_id == "req-phase1"
    assert decision.id == "req-phase1"


def test_confidence_is_normalized_once(fake_model):
    decision = fake_model.decide(
        "state",
        {
            "team": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": None, "b": None},
            },
            "urgent": noul("Urgent?"),
        },
    )
    assert decision["team"].confidence == pytest.approx(0.9)
    assert decision["team"].confidence_kind == "probability"
    assert decision["urgent"].confidence == pytest.approx(0.7)
    assert decision["urgent"].confidence_kind == "probability"


# ------------------------------------------------------------------ lifecycle
def test_context_manager_closes_and_close_is_idempotent():
    model = DecisionModel(adapter=FakeAdapter())
    with model as entered:
        assert entered is model
        model.warmup()
        assert model._adapter is not None
    assert model._adapter is None
    model.close()
    model.close()
    assert model._adapter is None


def test_injected_adapter_can_be_reused_after_close():
    model = DecisionModel(adapter=FakeAdapter())
    model.warmup()
    model.close()
    decision = model.decide("state", {"urgent": noul("Urgent?")})
    assert decision["urgent"].noul == 0.7


# --------------------------------------------------------------- batching/async
def test_decide_many_loads_once_and_keeps_request_ids(fake_model, fake_adapter):
    decisions = fake_model.decide_many(
        ["one", "two", "three"], {"urgent": noul("Urgent?")}, request_id="batch-1"
    )
    assert len(decisions) == 3
    assert [decision.request_id for decision in decisions] == [
        "batch-1:0",
        "batch-1:1",
        "batch-1:2",
    ]
    assert len(fake_adapter.calls) == 3


def test_decide_many_accepts_per_state_questions(fake_model):
    decisions = fake_model.decide_many(
        ["one", "two"],
        [{"a": noul("A?")}, {"b": noul("B?")}],
    )
    assert set(decisions[0].keys()) == {"a"}
    assert set(decisions[1].keys()) == {"b"}


def test_decide_many_rejects_mismatched_question_count(fake_model):
    with pytest.raises(InvalidRequestError):
        fake_model.decide_many(["one", "two"], [{"a": noul("A?")}])


def test_classify_many(fake_model):
    results = fake_model.classify_many(
        ["first", "second"], {"a": "first", "b": "second"}, threshold=0.5
    )
    assert [result.label for result in results] == ["a", "a"]
    assert all(result.usage.total_tokens > 0 for result in results)


def test_async_wrappers(fake_model):
    async def run():
        decision = await fake_model.adecide("state", {"urgent": noul("Urgent?")})
        results = await fake_model.aclassify_many(
            ["first", "second"], {"a": "first", "b": "second"}
        )
        many = await fake_model.adecide_many(
            ["one", "two"], {"urgent": noul("Urgent?")}
        )
        return decision, results, many

    decision, results, many = asyncio.run(run())
    assert decision["urgent"].noul == 0.7
    assert len(results) == 2
    assert len(many) == 2


# ------------------------------------------------------------------ truncation
def test_truncated_answers_are_visible_and_strict_raises():
    adapter = TruncatingAdapter()
    model = DecisionModel(adapter=adapter)
    decision = model.decide("state", {"urgent": noul("Urgent?")})
    assert decision.truncated is True
    assert decision["urgent"].truncated is True
    assert "input was truncated" in decision.warnings

    strict = DecisionModel(adapter=TruncatingAdapter(), strict=True)
    with pytest.raises(TruncationError):
        strict.decide("state", {"urgent": noul("Urgent?")})


# ---------------------------------------------------------------- capabilities
def test_capabilities_come_from_the_catalog(home):
    model = DecisionModel("gliner2.5-multi", home=home)
    assert isinstance(model.capabilities, Capabilities)
    assert set(model.capabilities.question_types) == {"choice", "noul", "score"}
    assert model.capabilities.multimodal is False
    assert model.capabilities.languages == ("multi",)
    info = model.model_info()
    assert info.id == "gliner2.5-multi"
    assert info.capabilities.to_dict()["max_options"] == 255
    assert info.to_dict()["default_variant"] == "fp16"


def test_box_question_is_rejected_before_inference(fake_model, fake_adapter):
    with pytest.raises(UnsupportedCapabilityError):
        fake_model.decide("state", {"region": box("Where is it?", ["object"])})
    assert fake_adapter.calls == []


def test_assets_are_rejected_by_a_text_only_model(fake_model):
    state = State(text="state", assets=[{"url": "https://example.test/image.png"}])
    with pytest.raises(UnsupportedCapabilityError):
        fake_model.decide(state, {"urgent": noul("Urgent?")})


def test_state_text_is_passed_to_adapters(fake_model, fake_adapter):
    fake_model.decide(State(text="typed state"), {"urgent": noul("Urgent?")})
    assert fake_adapter.calls[0][1].text == "typed state"


# ------------------------------------------------------------- custom adapters
def test_register_adapter_factory_and_entry_point_shape():
    unregister_adapter("fake-registered")
    try:
        register_adapter("fake-registered", lambda **options: FakeAdapter())
        adapter = create_adapter("fake-registered")
        assert adapter.name == "fake"
    finally:
        unregister_adapter("fake-registered")


def test_from_onnx_builds_an_engine_with_an_explicit_file_map(tmp_path):
    unregister_adapter("fake-from-onnx")
    try:
        register_adapter("fake-from-onnx", lambda **options: FakeAdapter())
        model = DecisionModel.from_onnx(
            tmp_path,
            adapter="fake-from-onnx",
            files={"model": "model.onnx", "tokenizer": "tokenizer.json"},
        )
        decision = model.decide("state", {"urgent": noul("Urgent?")})
        assert decision["urgent"].noul == 0.7
        assert model._engine is not None
    finally:
        unregister_adapter("fake-from-onnx")


def test_from_onnx_requires_files(tmp_path):
    with pytest.raises(AdapterNotAvailableError):
        DecisionModel.from_onnx(tmp_path, adapter="julia")


def test_adapter_file_roles_are_validated(tmp_path):
    model = DecisionModel(str(tmp_path), adapter="julia", files={"model": "model.onnx"})
    with pytest.raises(AdapterNotAvailableError) as excinfo:
        model.decide("state", {"urgent": noul("Urgent?")})
    assert "tokenizer" in str(excinfo.value)

    with pytest.raises(AdapterNotAvailableError):
        validate_adapter_files("gliner2", {"tokenizer": "tokenizer.json"})
    validate_adapter_files(
        "gliner2", {"tokenizer": "tokenizer.json", "encoder": "e.onnx", "classifier": "c.onnx"}
    )


# ------------------------------------------------------------------- logging
def test_decisions_emit_structured_events(fake_model, caplog):
    with caplog.at_level(logging.INFO, logger="decision_toolkit.decision_model"):
        fake_model.decide("state", {"urgent": noul("Urgent?")}, request_id="log-1")
    assert "decision.start" in caplog.text
    assert "decision.end" in caplog.text
    assert '"request_id": "log-1"' in caplog.text
    assert "input_tokens" in caplog.text


# --------------------------------------------------------------- model store
def test_model_store_ignores_in_flight_partial_directories(store):
    partial = store.models_dir / ".laya@q8.partial-123"
    partial.mkdir(parents=True)
    (partial / "model.json").write_text('{"identifier": "laya", "variant": "q8"}', encoding="utf-8")
    assert store.entries() == []


def test_model_store_variant_use_and_resolve(store, fake_snapshot):
    store.download("gliner2.5-multi", variant="fp16")
    store.use("gliner2.5-multi", variant="fp16")
    assert store.current_variant() == "fp16"
    resolved = store.resolve()
    assert resolved.variant.name == "fp16"
    assert resolved.adapter == "gliner2"

    # The selection survives a fresh store (config.json roundtrip).
    assert ModelStore(home=store.config.home).current_variant() == "fp16"


def test_model_store_import_records_adapter_and_files(store, tmp_path):
    source = tmp_path / "custom"
    source.mkdir()
    (source / "weights.onnx").write_bytes(b"onnx")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")

    entry = store.import_model(
        source,
        name="my-custom",
        adapter="julia",
        files={"model": "weights.onnx", "tokenizer": "tokenizer.json"},
    )
    assert entry.adapter == "julia"
    assert entry.files == {"model": "weights.onnx", "tokenizer": "tokenizer.json"}
    assert entry.sha256["model"]

    resolved = store.resolve("my-custom")
    assert resolved.adapter == "julia"
    assert resolved.path is not None


def test_download_records_hashes_and_verify_detects_corruption(store, fake_snapshot):
    entry = store.download("gliner2.5-multi")
    assert entry.sha256
    report = store.verify("gliner2.5-multi")
    assert report["ok"] is True
    assert all(item["ok"] for item in report["hashes"].values())

    first_role = next(iter(entry.files))
    (entry.path / entry.files[first_role]).write_bytes(b"corrupted")
    report = store.verify("gliner2.5-multi")
    assert report["ok"] is False
    assert report["hashes"][first_role]["ok"] is False


def test_use_rejects_unknown_variant(store):
    with pytest.raises(ModelNotFoundError):
        store.use("gliner2.5-multi", variant="q4")
