"""The engine's wire: the parser's ``serve`` command, the request/event schema
(protocol 3: ``setup.status``, ``abba.prepare``, ``abba.follow``), and the
JSON-lines service (errors, a long request ended by the input closing,
native stdout kept off the wire)."""

from __future__ import annotations

import io
import json
import threading
from typing import Any, cast

import pytest
from pydantic import ValidationError

from langslice.doors.abba.contract import (
    ENGINE_METHODS,
    EngineRequest,
    export_schema_bundle,
)
from langslice.doors.commands.parser import build_parser, main
from langslice.hosts.service import handle_request, run_stdio

# --- parser -----------------------------------------------------------------


def test_serve_requires_stdio() -> None:
    assert build_parser().parse_args(["serve", "--stdio"]).stdio is True
    with pytest.raises(SystemExit):
        main(["serve"])


def test_cli_protocol_is_optional_and_explicit_values_survive():
    parser = build_parser()
    unknown = parser.parse_args(["linear", "run", "/tmp/sections"])
    assert unknown.position_thickness_um is None and unknown.position_interval_um is None
    known = parser.parse_args(["linear", "run", "/tmp/sections", "--position-thickness-um",
                               "50", "--position-interval-um", "200"])
    assert known.position_thickness_um == 50 and known.position_interval_um == 200


# --- schema -----------------------------------------------------------------


def test_schema_bundle_and_request_methods_match() -> None:
    bundle = cast(dict[str, Any], export_schema_bundle())
    assert "EngineRequest" in bundle["schemas"]
    assert "AbbaPrepareRequest" in bundle["schemas"]
    method_schema = bundle["schemas"]["EngineRequest"]["properties"]["method"]
    assert method_schema["enum"] == list(ENGINE_METHODS)
    assert ENGINE_METHODS == ("setup.status", "abba.prepare", "abba.follow")
    for gone in ("setup.login", "setup.api_key", "linear.run", "linear.estimate",
                 "preprocess.preview", "mcp.prepare"):
        with pytest.raises(ValidationError):
            EngineRequest.model_validate({"id": "a", "method": gone, "params": {}})


# --- service ----------------------------------------------------------------


def _run_lines(*lines: str) -> list[dict[str, object]]:
    input_stream = io.StringIO("\n".join(lines) + "\n")
    output_stream = io.StringIO()
    exit_code = run_stdio(input_stream=input_stream, output_stream=output_stream)
    assert exit_code == 0
    payload_lines = [line for line in output_stream.getvalue().splitlines() if line.strip()]
    return [json.loads(line) for line in payload_lines]


@pytest.mark.parametrize("line, request_id, code", [
    (json.dumps({"id": "1", "method": "nope", "params": {}}), "1", "validation_error"),
    ("{", None, "bad_json"),
    (json.dumps({"id": "2", "method": "abba.follow", "params": {}}),
     "2", "validation_error"),
    (json.dumps({"id": "3", "method": "abba.follow",
                 "params": {"job_folder": "/x", "job_fodler": "/y"}}),
     "3", "validation_error"),
    (json.dumps({"id": "4", "method": "setup.status", "params": {"x": 1}}),
     "4", "runtime_error"),
])
def test_a_bad_request_gets_one_error_message(line, request_id, code) -> None:
    messages = _run_lines(line)
    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert messages[0]["id"] == request_id
    assert cast(dict[str, Any], messages[0]["error"])["code"] == code


def test_handle_request_unknown_method_still_raises_for_bypassed_validation() -> None:
    request = EngineRequest.model_construct(id="1", method="nope", params={})
    with pytest.raises(AssertionError, match="unhandled engine method nope"):
        handle_request(request, emit=lambda _event: None)


def test_setup_status_is_the_installation_and_protocol_3() -> None:
    messages = _run_lines(json.dumps({"id": "s", "method": "setup.status"}))
    result = cast(dict[str, Any], messages[0]["result"])
    assert set(result) == {"version", "protocol_version", "python_executable",
                           "environment_prefix"}
    assert result["protocol_version"] == 3


def test_a_follow_streams_its_events_and_ends_when_the_input_closes(monkeypatch):
    """The follower's events go out as data events of its request; the input
    closing (the connector's end) stops it and its result follows."""
    from langslice.hosts import service

    def follow(request, emit, stop: threading.Event):
        emit({"kind": "waiting"})
        assert stop.wait(10), "the closed input must stop the follower"
        emit({"kind": "checkpoint", "initial": False, "host_updates": []})

    monkeypatch.setattr(service, "follow", follow)
    messages = _run_lines(json.dumps({"id": "f", "method": "abba.follow",
                                      "params": {"job_folder": "/nowhere"}}))
    assert [m["type"] for m in messages] == ["event", "event", "result"]
    assert messages[0]["event"] == {"kind": "data", "payload": {"kind": "waiting"}}
    assert messages[2] == {"id": "f", "type": "result", "result": {}}


def test_native_stdout_is_separate_from_wire_in_real_process(tmp_path):
    import os
    import subprocess
    import sys

    script = """
import os
from langslice.doors import credentials as setup
from langslice.hosts.service import run_stdio
def status():
    print('Python diagnostic')
    os.write(1, b'native diagnostic\\n')
    return {'protocol_version': 1}
setup.setup_status = status
run_stdio()
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps({"id": "wire", "method": "setup.status"}) + "\n",
        text=True, capture_output=True, timeout=60,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["result"] == {"protocol_version": 1}
    assert "Python diagnostic" in process.stderr
    assert "native diagnostic" in process.stderr
