from __future__ import annotations

import io
import json
from typing import Any, cast

import pytest

from langslice.doors.api.models import EngineRequest
from langslice.hosts.api.service import handle_request, run_stdio


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
    assert messages[0]["id"] == "1"


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
                "method": "linear.estimate",
                "params": {"spec": {}},
            }
        )
    )
    error = cast(dict[str, Any], messages[0]["error"])
    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert error["code"] == "validation_error"


def test_validation_error_for_unknown_param_key() -> None:
    messages = _run_lines(
        json.dumps(
            {
                "id": "4",
                "method": "linear.estimate",
                "params": {"spec": {}, "n_slices": 3, "plaen": "coronal"},
            }
        )
    )
    error = cast(dict[str, Any], messages[0]["error"])
    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert error["code"] == "validation_error"


def test_setup_status_does_not_import_registration_runtime(monkeypatch, tmp_path):
    from langslice.doors.api import setup

    monkeypatch.setattr(setup, "setup_status", lambda: {"protocol_version": 1})
    messages = _run_lines(json.dumps({"id": "setup", "method": "setup.status"}))
    assert messages[0]["result"] == {"protocol_version": 1}


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
