from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from langslice_harness.api.models import (
    ENGINE_METHODS,
    EngineError,
    EngineEventEnvelope,
    EngineProgressEvent,
    EngineRequest,
    EstimateRequest,
    export_schema_bundle,
)


def test_engine_request_defaults_are_not_shared() -> None:
    first = EngineRequest(id="a", method="version")
    second = EngineRequest(id="b", method="version")
    first.params["x"] = 1
    assert second.params == {}


def test_estimate_request_validates_required_fields() -> None:
    with pytest.raises(ValidationError):
        EstimateRequest.model_validate({"atlas": "allen_mouse_25um"})


def test_export_schema_bundle_contains_engine_shapes() -> None:
    bundle = cast(dict[str, Any], export_schema_bundle())
    assert "schemas" in bundle
    assert "EngineRequest" in bundle["schemas"]
    assert "EngineResultEnvelope" in bundle["schemas"]
    assert "EstimateRequest" in bundle["schemas"]


def test_engine_request_method_schema_matches_engine_methods() -> None:
    bundle = cast(dict[str, Any], export_schema_bundle())
    method_schema = bundle["schemas"]["EngineRequest"]["properties"]["method"]
    assert method_schema["enum"] == list(ENGINE_METHODS)


def test_engine_request_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        EngineRequest.model_validate({"id": "a", "method": "unknown.method", "params": {}})


def test_event_envelope_requires_event_kind() -> None:
    event = EngineProgressEvent(kind="progress", message="loading", stage="estimate")
    envelope = EngineEventEnvelope(id="req-1", type="event", event=event)
    assert envelope.event.kind == "progress"
    err = EngineError(code="runtime_error", message="boom")
    assert err.details == {}
