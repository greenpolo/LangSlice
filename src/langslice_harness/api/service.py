"""Stdio service for LangSlice engine requests."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import TextIO, cast

from pydantic import ValidationError

from langslice_harness.api import runtime
from langslice_harness.api.models import (
    EngineError,
    EngineErrorEnvelope,
    EngineEventEnvelope,
    EngineLogEvent,
    EngineProgressEvent,
    EngineRequest,
    EngineResultEnvelope,
    EstimateRequest,
    ExportRequest,
    QuickAffineRequest,
    RegisterRequest,
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
    def runtime_emit(event: EngineProgressEvent | EngineLogEvent) -> None:
        emit(EngineEventEnvelope(id=request.id, type="event", event=event))

    if request.method == "version":
        result = runtime.get_version()
    elif request.method == "estimate.run":
        params = EstimateRequest.model_validate(request.params)
        result = runtime.run_estimate(params, emit=runtime_emit)
    elif request.method == "register.run":
        params = RegisterRequest.model_validate(request.params)
        result = runtime.run_register(params, emit=runtime_emit)
    elif request.method == "quick_affine.run":
        params = QuickAffineRequest.model_validate(request.params)
        result = runtime.run_quick_affine(params, emit=runtime_emit)
    elif request.method == "export.run":
        params = ExportRequest.model_validate(request.params)
        result = runtime.run_export(params, emit=runtime_emit)
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
    in_stream = input_stream if input_stream is not None else cast(TextIO, sys.stdin)
    out_stream = output_stream if output_stream is not None else cast(TextIO, sys.stdout)

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
                details={"errors": exc.errors()},
            )
            continue
        except KeyError as exc:
            _emit_error(
                output_stream=out_stream,
                request_id=parsed_id,
                code="unknown_method",
                message=f"Unknown method: {exc.args[0]}",
            )
            continue
        except Exception as exc:  # noqa: BLE001
            _emit_error(
                output_stream=out_stream,
                request_id=parsed_id,
                code="runtime_error",
                message="Runtime request handling failed",
                details={"error": str(exc)},
            )
            continue

        _write_json_line(out_stream, envelope.model_dump(mode="json"))

    return 0
