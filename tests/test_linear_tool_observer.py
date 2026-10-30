"""Actual execution events remain ordered, sanitized, and observational."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.tools.pipeline import ToolCall, run_call


def _box(folder: Path, state: Stack, on_event=None):
    """A toolbox over *state* with no tasks, its workspace a stand-in."""
    from langslice.job.layout import JobLayout, job_folder_for
    from langslice.job.settings.spec import JobSpec

    folder.mkdir(exist_ok=True)
    spec = JobSpec(str(folder), tasks=[])
    ctx: Any = SimpleNamespace(position_range=(0, 10), brain_scale=1.0,
                               settings=spec.image_settings(),
                               section_channels=lambda _id: (("gray",), []),
                               layout=JobLayout(job_folder_for(folder), folder),
                               results_path=str(folder / "linear_results.json"))
    return build_tools(state, ctx, spec, on_event=on_event)


def _through(box, name: str, function):
    """*function* run as the tool *name* of *box*, through every step."""
    return lambda *args, **kwargs: run_call(ToolCall(name, function, box, args, kwargs))


def test_execution_events_run_inside_lock_and_name_their_targets(tmp_path):
    state = Stack(slices=[Section("a", 0, 0), Section("b", 1, 1)])
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
        assert box.lock.locked()
        events.append(event)

    box = _box(tmp_path, state, on_event=observer)
    wrapped = _through(box, "mark_damage", mark_damage)
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


def test_observer_sanitizes_metadata_without_mutating_result(tmp_path):
    events = []
    result = {"status": "ok", TOOL_MEDIA_PARTS_KEY: [b"image"],
              "nested": {"thought_signature": b"secret", "encrypted_content": "secret"}}

    def look(mode, tool_context=None):
        return result

    box = _box(tmp_path, Stack(slices=[Section("a", 0, 0)]), on_event=events.append)
    wrapped = _through(box, "look", look)
    assert wrapped("section", tool_context=SimpleNamespace(function_call_id="call-7")) is result
    assert events[0]["id"] == "call-7"
    assert events[0]["args"] == {"mode": "section"}
    assert events[1]["response"] == {"status": "ok", "nested": {}}
    assert result[TOOL_MEDIA_PARTS_KEY] == [b"image"]


def test_callback_failures_never_change_tool_results_or_exceptions(tmp_path):
    def broken_observer(event):
        raise RuntimeError("display gone")

    def task(fail=False):
        if fail:
            raise ValueError("invalid slice")
        return 42

    state = Stack(slices=[Section("a", 0, 0)])
    wrapped = _through(_box(tmp_path / "broken", state, on_event=broken_observer), "status", task)
    assert wrapped() == 42
    with pytest.raises(ValueError, match="invalid slice"):
        wrapped(True)
    events = []
    wrapped = _through(_box(tmp_path / "listened", state, on_event=events.append), "status",
                       task)
    with pytest.raises(ValueError):
        wrapped(True)
    assert events[-1]["kind"] == "tool_end"
    assert events[-1]["response"]["status"] == "error"


def test_build_tools_forwards_execution_observer(tmp_path):
    events = []
    box = _box(tmp_path, Stack(slices=[Section("a", 0, 0)]), on_event=events.append)
    result = next(tool for tool in box.tools if tool.__name__ == "status")()
    assert result["status"] == "ok"
    assert [event["kind"] for event in events] == ["tool_start", "tool_end"]
    assert events[0]["name"] == "status" and events[0]["target_ids"] == ["a"]
