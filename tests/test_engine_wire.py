"""The engine's wire: parser commands of ``serve`` and ``abba``, the request/event
schema, and the JSON-lines service (errors, secrets never echoed, checkpoint before
result, native stdout kept off the wire)."""

from __future__ import annotations

import io
import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from langslice.doors.api.models import (
    ENGINE_METHODS,
    EngineRequest,
    export_schema_bundle,
)
from langslice.doors.cli import build_parser, main
from langslice.hosts.api.service import handle_request, run_stdio
from langslice.hosts.cli import run_abba

# --- parser -----------------------------------------------------------------


def test_serve_requires_stdio() -> None:
    assert build_parser().parse_args(["serve", "--stdio"]).stdio is True
    with pytest.raises(SystemExit):
        main(["serve"])


def test_run_abba_without_a_connector_jar_says_how_to_get_one(monkeypatch, tmp_path) -> None:
    from langslice.hosts.integrations import abba_launch

    monkeypatch.delenv(abba_launch.CONNECTOR_JAR_ENV, raising=False)
    monkeypatch.setattr(abba_launch, "repository_root", lambda: tmp_path)
    with pytest.raises(SystemExit, match="mvn package"):
        run_abba(build_parser().parse_args(["abba"]))


def test_cli_protocol_is_optional_and_explicit_values_survive():
    parser = build_parser()
    unknown = parser.parse_args(["linear", "run", "/tmp/sections"])
    assert unknown.thickness is None and unknown.interval is None
    known = parser.parse_args(["linear", "run", "/tmp/sections", "--thickness", "50",
                               "--interval", "200"])
    assert known.thickness == 50 and known.interval == 200


def test_saved_job_prompt_handles_unknown_protocol():
    from langslice.core.spec import JobSpec, PositionSpec
    from langslice.doors.api.saved_jobs import copy_prompt

    spec = JobSpec(image_folder=".", tasks=["position"])
    assert "section thickness not supplied, interval not supplied" in copy_prompt("job", spec)
    spec.position = PositionSpec(interval_um=200)
    assert "section thickness not supplied, interval 200 µm" in copy_prompt("job", spec)


# --- schema -----------------------------------------------------------------


def test_schema_bundle_and_request_methods_match() -> None:
    bundle = cast(dict[str, Any], export_schema_bundle())
    assert "EngineRequest" in bundle["schemas"]
    assert "EngineResultEnvelope" in bundle["schemas"]
    method_schema = bundle["schemas"]["EngineRequest"]["properties"]["method"]
    assert method_schema["enum"] == list(ENGINE_METHODS)
    with pytest.raises(ValidationError):
        EngineRequest.model_validate({"id": "a", "method": "unknown.method", "params": {}})


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
    (json.dumps({"id": "2", "method": "linear.estimate", "params": {"spec": {}}}),
     "2", "validation_error"),
    (json.dumps({"id": "4", "method": "linear.estimate",
                 "params": {"spec": {}, "n_slices": 3, "plaen": "coronal"}}),
     "4", "validation_error"),
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


def test_login_url_is_a_structured_event(monkeypatch):
    from langslice.doors.api import setup

    def login(on_url, timeout_s):
        assert timeout_s == 30
        on_url("https://example.test/login")
        print("Provider library diagnostic must not corrupt JSON")
        return {"configured": True}

    monkeypatch.setattr(setup, "login_oauth", login)
    messages = _run_lines(json.dumps({
        "id": "login", "method": "setup.login", "params": {"timeout_s": 30},
    }))
    assert messages[0]["event"] == {
        "kind": "data", "payload": {"kind": "login_url", "url": "https://example.test/login"},
    }
    assert messages[1]["result"] == {"configured": True}


def test_api_key_validation_and_exception_never_echo_secret(monkeypatch):
    from langslice.doors.api import setup

    secret = "test-private-api-key"
    invalid = _run_lines(json.dumps({
        "id": "key", "method": "setup.api_key",
        "params": {"provider": "unsupported", "api_key": secret},
    }))
    assert secret not in json.dumps(invalid)

    def fail(*args):
        raise RuntimeError(f"Request failed with key {secret}")

    monkeypatch.setattr(setup, "save_api_key", fail)
    failed = _run_lines(json.dumps({
        "id": "key", "method": "setup.api_key",
        "params": {"provider": "openai-api", "api_key": secret},
    }))
    assert failed[0]["type"] == "error"
    assert secret not in json.dumps(failed)


def test_linear_service_emits_checkpoint_before_result(monkeypatch):
    from langslice.doors.api import abba_worker, setup

    monkeypatch.setattr(setup, "apply_saved_credentials", lambda: None)

    def run(params, emit):
        emit({"kind": "checkpoint", "initial": True, "host_updates": []})
        return {"state": {"submitted": True}}

    monkeypatch.setattr(abba_worker, "run_linear", run)
    messages = _run_lines(json.dumps({"id": "agent", "method": "linear.run"}))
    assert messages[0]["event"]["payload"]["initial"] is True
    assert messages[1]["result"]["state"]["submitted"] is True


def test_native_stdout_is_separate_from_wire_in_real_process(tmp_path):
    import os
    import subprocess
    import sys

    script = """
import os
from langslice.doors.api import setup
from langslice.hosts.api.service import run_stdio
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
        text=True, capture_output=True, timeout=20,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["result"] == {"protocol_version": 1}
    assert "Python diagnostic" in process.stderr
    assert "native diagnostic" in process.stderr
