"""Stdio service for LangSlice engine requests (``langslice serve --stdio``).

One JSON request per input line, answered on stdout by its events and then
one result or error envelope (:mod:`langslice.doors.abba.contract`). A
long request (``abba.follow``) runs until the input closes: the rest of the
input is read by a thread of its own, and its end stops the request.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager, redirect_stdout
from typing import TextIO, cast

from pydantic import ValidationError

from langslice.doors import credentials as setup
from langslice.doors.abba.contract import (
    AbbaFollowRequest,
    AbbaPrepareRequest,
    EngineDataEvent,
    EngineError,
    EngineErrorEnvelope,
    EngineEventEnvelope,
    EngineRequest,
    EngineResultEnvelope,
)
from langslice.doors.abba.follow import follow
from langslice.doors.abba.prepare import prepare_job

logger = logging.getLogger(__name__)

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


def handle_request(
    request: EngineRequest,
    emit: EmitEventEnvelope,
    *,
    stop: threading.Event | None = None,
) -> EngineResultEnvelope:
    """Answer one request: its events through *emit*, then its result.

    Parameters
    ----------
    request
        The validated request.
    emit
        Where each event envelope of the request goes.
    stop
        Set when the request's input closed; ``abba.follow`` runs until
        then (None: until the process ends).

    Returns
    -------
    EngineResultEnvelope
        The request's result.
    """
    def data_emit(payload: Mapping[str, object]) -> None:
        emit(EngineEventEnvelope(
            id=request.id, type="event", event=EngineDataEvent(payload=dict(payload)),
        ))

    if request.method == "setup.status":
        if request.params:
            raise ValueError("setup.status takes no parameters")
        result: dict[str, object] = dict(setup.setup_status())
    elif request.method == "abba.prepare":
        folder = prepare_job(AbbaPrepareRequest.model_validate(request.params))
        result = {"job_folder": str(folder)}
    elif request.method == "abba.follow":
        follow(AbbaFollowRequest.model_validate(request.params), data_emit,
               stop or threading.Event())
        result = {}
    else:
        raise AssertionError(f"unhandled engine method {request.method}")
    return EngineResultEnvelope(id=request.id, type="result", result=result)


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


def _input_closed(in_stream: TextIO) -> threading.Event:
    """Return an event set once *in_stream* ends, read by a thread of its own.

    A line that arrives meanwhile is no request: a long request owns the
    service until its input closes. The line is logged and dropped.
    """
    closed = threading.Event()

    def drain() -> None:
        for line in in_stream:
            if line.strip():
                logger.warning("Ignoring a request sent while another runs")
        closed.set()

    threading.Thread(target=drain, name="langslice-stdin", daemon=True).start()
    return closed


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
            stop = _input_closed(in_stream) if request.method == "abba.follow" else None
            envelope = handle_request(
                request,
                emit=lambda event_envelope: _write_json_line(
                    out_stream,
                    event_envelope.model_dump(mode="json"),
                ),
                stop=stop,
            )
        except ValidationError as exc:
            _emit_error(
                output_stream=out_stream,
                request_id=parsed_id,
                code="validation_error",
                message="Request validation failed",
                details={"errors": [
                    {"type": error["type"], "loc": error["loc"]}
                    for error in exc.errors(include_input=False, include_context=False)
                ]},
            )
            continue
        except Exception as exc:  # noqa: BLE001 - any failure is the request's error reply
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
