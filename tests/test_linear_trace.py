"""The full-content session trace: what a model turn writes, and when it is off."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.genai import types

from langslice.agent.trace import SessionTrace, open_trace


class _Agent:
    name = "linear_stack"
    model = "fake-model"
    instruction = "the job statement"


class _Event:
    """The shape ``SessionTrace.event`` reads off an ADK event."""

    partial = False
    author = "linear_stack"

    def __init__(self, content: Any) -> None:
        self.content = content


def _model_event(text: str) -> _Event:
    return _Event(types.Content(role="model", parts=[types.Part.from_text(text=text)]))


def _records(trace: SessionTrace) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace.path.read_text().splitlines()]


def test_model_turns_are_recorded_with_a_step_counter(tmp_path: Path):
    trace = SessionTrace(tmp_path, "linear_stack", agent=_Agent())
    trace.event(_model_event("placing the stack"), turn=1)
    trace.nudge("debrief question", turn=2)
    # The debrief answer is one more model turn in the same session, after
    # submit: it has to land as a `model` record like any other.
    trace.event(_model_event("Debrief: I wanted labelled images."), turn=2)

    records = _records(trace)
    assert records[0]["kind"] == "session"
    assert records[0]["system_instruction"] == "the job statement"
    assert [r["kind"] for r in records[1:]] == ["model", "nudge", "model"]

    turns = [r for r in records if r["kind"] == "model"]
    assert [r["step"] for r in turns] == [1, 2]
    assert [r["turn"] for r in turns] == [1, 2]
    assert turns[1]["text"] == "Debrief: I wanted labelled images."


def test_tracing_is_off_unless_the_env_var_is_set(monkeypatch):
    monkeypatch.delenv("LANGSLICE_TRACE_DIR", raising=False)
    assert open_trace("linear_stack", agent=_Agent()) is None
