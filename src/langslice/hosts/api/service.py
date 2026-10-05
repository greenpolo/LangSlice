"""Stdio service for LangSlice engine requests."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager, redirect_stdout
from typing import TextIO, cast

from pydantic import ValidationError

from langslice.doors.api import runtime
from langslice.doors.api.models import (
    EngineDataEvent,
    EngineError,
    EngineErrorEnvelope,
    EngineEventEnvelope,
    EngineRequest,
    EngineResultEnvelope,
    LinearEstimateRequest,
    LinearEstimateResult,
    PreprocessPreviewRequest,
    SetupApiKeyRequest,
    SetupLoginRequest,
)

EmitEventEnvelope = Callable[[EngineEventEnvelope], None]


def _write_json_line(output_stream: TextIO, payload: dict[str, object]) -> None:
    output_stream.write(json.dumps(payload, ensure_ascii=True) + "\n")
    output_stream.flush()


def _emit_error(
    *,
    output_stream: TextIO,
    request_id: str | None,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> None:
    envelope = EngineErrorEnvelope(
        id=request_id,
        type="error",
        error=EngineError(code=code, message=message, details=details or {}),
    )
    _write_json_line(output_stream, envelope.model_dump(mode="json"))


def handle_request(request: EngineRequest, emit: EmitEventEnvelope) -> EngineResultEnvelope:
    def data_emit(payload: dict[str, object]) -> None:
        emit(EngineEventEnvelope(
            id=request.id, type="event", event=EngineDataEvent(payload=payload),
        ))

    if request.method.startswith("setup."):
        from langslice.doors.api import setup

        if request.method == "setup.status":
            if request.params:
                raise ValueError("setup.status takes no parameters")
            data = setup.setup_status()
        elif request.method == "setup.login":
            login = SetupLoginRequest.model_validate(request.params)
            data = setup.login_oauth(
                on_url=lambda url: data_emit({"kind": "login_url", "url": url}),
                timeout_s=login.timeout_s,
            )
        else:
            credentials = SetupApiKeyRequest.model_validate(request.params)
            data = setup.save_api_key(credentials.provider, credentials.api_key)
        return EngineResultEnvelope(id=request.id, type="result", result=data)

    if request.method == "linear.estimate":
        # A table lookup: no model, no credentials, no engine import.
        from langslice.agent.cost import estimate

        wanted = LinearEstimateRequest.model_validate(request.params)
        result = LinearEstimateResult.model_validate(
            estimate(wanted.spec, wanted.n_slices, wanted.locked),
        )
        return EngineResultEnvelope(
            id=request.id, type="result", result=result.model_dump(mode="json"),
        )

    if request.method == "preprocess.preview":
        # Local image work only: no model, no credentials, no engine import.
        from langslice.doors.api.abba_worker import preview_preprocess

        preview = preview_preprocess(PreprocessPreviewRequest.model_validate(request.params))
        return EngineResultEnvelope(
            id=request.id, type="result", result=preview.model_dump(mode="json"),
        )

    if request.method == "claude.prepare":
        from langslice.doors.api.claude_jobs import prepare_claude

        return EngineResultEnvelope(
            id=request.id, type="result", result=prepare_claude(request.params),
        )

    if request.method != "version":
        from langslice.doors.api.setup import apply_saved_credentials

        apply_saved_credentials()

    if request.method == "linear.run":
        from langslice.doors.api import abba_worker

        data = abba_worker.run_linear(request.params, data_emit)
        return EngineResultEnvelope(id=request.id, type="result", result=data)

    if request.method == "version":
        result = runtime.get_version()
    else:
        raise KeyError(request.method)

    return EngineResultEnvelope(
        id=request.id,
        type="result",
        result=result.model_dump(mode="json"),
    )


def run_stdio(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> int:
    # Scientific libraries may print through Python OR native stdout. Keep both
    # away from the protocol; a dedicated duplicate of stdout carries JSON only.
    with _protocol_output(output_stream) as out_stream:
        return _run_requests(input_stream, out_stream)


@contextmanager
def _protocol_output(output_stream: TextIO | None):
    if output_stream is not None:
        with redirect_stdout(sys.stderr):
            yield output_stream
        return
    sys.stdout.flush()
    saved = os.dup(sys.stdout.fileno())
    wire = os.fdopen(os.dup(saved), "w", encoding="utf-8", buffering=1)
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with redirect_stdout(sys.stderr):
            yield wire
    finally:
        wire.close()
        os.dup2(saved, 1)
        os.close(saved)


def _run_requests(input_stream: TextIO | None, out_stream: TextIO) -> int:
    in_stream = input_stream if input_stream is not None else cast(TextIO, sys.stdin)

    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue
        parsed_id: str | None = None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit_error(
                output_stream=out_stream,
                request_id=None,
                code="bad_json",
                message="Unable to parse JSON request",
                details={"error": str(exc)},
            )
            continue

        try:
            if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                parsed_id = payload["id"]
            request = EngineRequest.model_validate(payload)
            parsed_id = request.id
            envelope = handle_request(
                request,
                emit=lambda event_envelope: _write_json_line(
                    out_stream,
                    event_envelope.model_dump(mode="json"),
                ),
            )
        except ValidationError as exc:
            _emit_error(
                output_stream=out_stream,
                request_id=parsed_id,
                code="validation_error",
                message="Request validation failed",
                # Validation inputs can contain an API key: never echo them.
                details={"errors": [
                    {"type": error["type"], "loc": error["loc"]}
                    for error in exc.errors(include_input=False, include_context=False)
                ]},
            )
            continue
        except KeyError:
            _emit_error(
                output_stream=out_stream,
                request_id=parsed_id,
                code="invalid_request",
                message="The request is missing a required field or setting.",
            )
            continue
        except Exception as exc:  # noqa: BLE001
            sensitive = isinstance(payload, dict) and str(payload.get("method", "")).startswith(
                "setup."
            )
            _emit_error(
                output_stream=out_stream,
                request_id=parsed_id,
                code="runtime_error",
                message=("Setup failed; check your installation or try signing in again."
                         if sensitive else "Runtime request handling failed"),
                details={} if sensitive else {"error": str(exc)},
            )
            continue

        _write_json_line(out_stream, envelope.model_dump(mode="json"))

    return 0
