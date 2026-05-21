from __future__ import annotations

import io
import json
from typing import Any, cast

import pytest

from langslice_harness.api import runtime
from langslice_harness.api.models import EngineLogEvent, EngineRequest, EstimateResult
from langslice_harness.api.service import handle_request, run_stdio


def _run_lines(*lines: str) -> list[dict[str, object]]:
    input_stream = io.StringIO("\n".join(lines) + "\n")
    output_stream = io.StringIO()
    exit_code = run_stdio(input_stream=input_stream, output_stream=output_stream)
    assert exit_code == 0
    payload_lines = [line for line in output_stream.getvalue().splitlines() if line.strip()]
    return [json.loads(line) for line in payload_lines]


def test_version_request_success() -> None:
    messages = _run_lines(json.dumps({"id": "1", "method": "version", "params": {}}))
    result = cast(dict[str, Any], messages[0]["result"])
    assert len(messages) == 1
    assert messages[0]["id"] == "1"
    assert messages[0]["type"] == "result"
    assert "version" in result


def test_unknown_method_returns_validation_error() -> None:
    messages = _run_lines(json.dumps({"id": "1", "method": "nope", "params": {}}))
    error = cast(dict[str, Any], messages[0]["error"])
    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert error["code"] == "validation_error"


def test_handle_request_unknown_method_still_raises_for_bypassed_validation() -> None:
    request = EngineRequest.model_construct(id="1", method="nope", params={})
    with pytest.raises(KeyError):
        handle_request(request, emit=lambda _event: None)


def test_bad_json_returns_error_with_null_id() -> None:
    messages = _run_lines("{")
    error = cast(dict[str, Any], messages[0]["error"])
    assert len(messages) == 1
    assert messages[0]["id"] is None
    assert messages[0]["type"] == "error"
    assert error["code"] == "bad_json"


def test_validation_error_returns_error() -> None:
    messages = _run_lines(
        json.dumps(
            {
                "id": "2",
                "method": "estimate.run",
                "params": {"atlas": "allen_mouse_25um"},
            }
        )
    )
    error = cast(dict[str, Any], messages[0]["error"])
    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert error["code"] == "validation_error"


def test_progress_events_are_emitted_before_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_estimate(request, emit=None):  # noqa: ANN001
        assert request.image_path == "slice.png"
        if emit is not None:
            emit(EngineLogEvent(kind="log", message="starting"))
        return EstimateResult(position_mm=1.5, reasoning="ok", debug_dir=None)

    monkeypatch.setattr(runtime, "run_estimate", fake_run_estimate)
    messages = _run_lines(
        json.dumps(
            {
                "id": "3",
                "method": "estimate.run",
                "params": {
                    "image_path": "slice.png",
                    "atlas": "allen_mouse_25um",
                },
            }
        )
    )
    event = cast(dict[str, Any], messages[0]["event"])
    assert len(messages) == 2
    assert messages[0]["type"] == "event"
    assert event["kind"] == "log"
    assert messages[1]["type"] == "result"


def test_validation_error_for_unknown_param_key() -> None:
    messages = _run_lines(
        json.dumps(
            {
                "id": "4",
                "method": "estimate.run",
                "params": {
                    "image_path": "slice.png",
                    "atlas": "allen_mouse_25um",
                    "plaen": "coronal",
                },
            }
        )
    )
    error = cast(dict[str, Any], messages[0]["error"])
    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert error["code"] == "validation_error"
