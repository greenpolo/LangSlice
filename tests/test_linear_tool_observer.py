"""Actual execution events remain ordered, sanitized, and observational."""

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from langslice.core.state import SliceState, StackState
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import _serialized, build_tools


def test_execution_events_run_inside_lock_and_name_their_targets():
    state = StackState(slices=[SliceState("a", 0, 0), SliceState("b", 1, 1)])
    lock = threading.Lock()
    events = []

    def mark_damage(section: str):
        record = state.resolve(section)
        assert record is not None
        events.append({"body": record.id})
        state.slices.reverse()
        for index, item in enumerate(state.slices):
            item.index_corrected = index
        return {"status": "ok", "id": record.id}

    def observer(event):
        assert lock.locked()
        events.append(event)

    wrapped = _serialized(mark_damage, lock, state=state, on_event=observer)
    assert inspect.signature(wrapped) == inspect.signature(mark_damage)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(wrapped, "a") for _ in range(2)]
        for future in futures:
            future.result()
    assert [event.get("kind", "body") for event in events] == [
        "tool_start", "body", "tool_end", "tool_start", "body", "tool_end",
    ]
    # A section is named by its filename, whatever the tool does to the order.
    assert events[0]["target_ids"] == events[2]["target_ids"] == ["a"]
    assert events[3]["target_ids"] == events[5]["target_ids"] == ["a"]
    assert events[0]["execution_id"] == events[2]["execution_id"]
    assert events[0]["execution_id"] != events[3]["execution_id"]
    assert events[2]["views"] == []  # no picture was saved


def test_observer_sanitizes_metadata_without_mutating_result():
    events = []
    result = {"status": "ok", TOOL_MEDIA_PARTS_KEY: [b"image"],
              "nested": {"thought_signature": b"secret", "encrypted_content": "secret"}}

    def look(mode, tool_context=None):
        return result

    wrapped = _serialized(look, threading.Lock(), on_event=events.append)
    assert wrapped("section", tool_context=SimpleNamespace(function_call_id="call-7")) is result
    assert events[0]["id"] == "call-7"
    assert events[0]["args"] == {"mode": "section"}
    assert events[1]["response"] == {"status": "ok", "nested": {}}
    assert result[TOOL_MEDIA_PARTS_KEY] == [b"image"]


def test_callback_failures_never_change_tool_results_or_exceptions():
    def broken_observer(event):
        raise RuntimeError("display gone")

    def task(fail=False):
        if fail:
            raise ValueError("invalid slice")
        return 42

    wrapped = _serialized(task, threading.Lock(), on_event=broken_observer)
    assert wrapped() == 42
    with pytest.raises(ValueError, match="invalid slice"):
        wrapped(True)
    events = []
    wrapped = _serialized(task, threading.Lock(), on_event=events.append)
    with pytest.raises(ValueError):
        wrapped(True)
    assert events[-1]["kind"] == "tool_end"
    assert events[-1]["response"]["status"] == "error"


def test_build_tools_forwards_execution_observer(tmp_path):
    from langslice.core.spec import JobSpec
    from langslice.job.layout import JobLayout

    events = []
    state = StackState(slices=[SliceState("a", 0, 0)])
    ctx: Any = SimpleNamespace(position_range=(0, 10),
                               layout=JobLayout.for_images(tmp_path),
                               results_path=str(tmp_path / "linear_results.json"))
    box = build_tools(state, ctx, JobSpec(str(tmp_path), tasks=[]), on_event=events.append)
    result = next(tool for tool in box.tools if tool.__name__ == "status")()
    assert result["status"] == "ok"
    assert [event["kind"] for event in events] == ["tool_start", "tool_end"]
    assert events[0]["name"] == "status" and events[0]["target_ids"] == ["a"]
