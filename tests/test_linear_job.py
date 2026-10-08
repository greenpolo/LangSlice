"""The job layer: persisted undo, versioned files, live shared editing, and
the tool doors' gates kept off the job."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec, PositionSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.checkpoint import (
    STATE_FORMAT_VERSION,
    load_checkpoint,
    read_checkpoint,
    write_checkpoint,
)
from langslice.job.history import HISTORY_FORMAT_VERSION
from langslice.job.job import UNDO_DEPTH, Job
from tests.fakes import SlabAtlas
from tests.linear_tool_helpers import tool_named as _tool

_ATLAS = SlabAtlas()


def _folder(tmp_path: Path, n: int = 3) -> Path:
    for index in range(n):
        Image.fromarray(np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)).save(
            tmp_path / f"s{index}.png")
    return tmp_path


def _open(folder: Path, **spec_kwargs: Any) -> tuple[Job, Any]:
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    return job, ctx


def _edit_on_disk(path: str, edit: Any) -> None:
    """What a script does: read the file, change it, write it in place."""
    data = json.loads(Path(path).read_text())
    edit(data)
    Path(path).write_text(json.dumps(data))


# --- undo, persisted -----------------------------------------------------------


def test_undo_history_survives_a_reopen(tmp_path: Path):
    folder = _folder(tmp_path)
    job, _ = _open(folder)
    start = job.state.by_id("s0.png").position_mm  # the starting position
    before = job.snapshot()
    job.state.by_id("s0.png").position_mm = 1.5
    job.commit(before)
    before = job.snapshot()
    job.state.notes.append("second step")
    job.commit(before)

    history = json.loads(Path(job.undo_path).read_text())
    assert history["format_version"] == HISTORY_FORMAT_VERSION
    assert history["state_format_version"] == STATE_FORMAT_VERSION
    assert len(history["undo"]) == 2 and history["redo"] == []

    again, _ = _open(folder)  # resume is the default
    assert again.state.by_id("s0.png").position_mm == 1.5
    assert again.undo() and "second step" not in again.state.notes
    assert again.undo() and again.state.by_id("s0.png").position_source == "default"
    assert again.state.by_id("s0.png").position_mm == start != 1.5
    assert not again.undo()
    third, _ = _open(folder)
    assert third.state.by_id("s0.png").position_source == "default"
    assert third.state.by_id("s0.png").position_mm == start
    assert third.redo() and third.state.by_id("s0.png").position_mm == 1.5


def test_a_fresh_job_empties_an_old_history_and_the_depth_is_kept(tmp_path: Path):
    folder = _folder(tmp_path)
    job, _ = _open(folder)
    for index in range(UNDO_DEPTH + 5):
        before = job.snapshot()
        job.state.notes.append(f"note {index}")
        job.commit(before)
    assert len(job.undo_stack) == UNDO_DEPTH
    assert len(json.loads(Path(job.undo_path).read_text())["undo"]) == UNDO_DEPTH

    fresh, _ = _open(folder, resume=False)
    assert fresh.undo_stack == [] and not fresh.undo()
    assert json.loads(Path(fresh.undo_path).read_text())["undo"] == []


def test_a_toolbox_undo_reaches_back_across_a_reopen(tmp_path: Path):
    folder = _folder(tmp_path)
    job, ctx = _open(folder, tasks=["position"])
    box = build_tools(job.state, ctx, job.spec, job=job)
    start = job.state.by_id("s1.png").position_mm
    written = _tool(box, "position_sections")([{"id": "s1.png", "position_mm": 2.0}],
                                              view=False)
    assert written["status"] == "ok"

    again, ctx = _open(folder, tasks=["position"])
    box = build_tools(again.state, ctx, again.spec, job=again)
    undone = _tool(box, "undo")()
    assert undone["status"] == "ok" and undone["undo_depth"] == 0
    assert again.state.by_id("s1.png").position_source == "default"
    assert again.state.by_id("s1.png").position_mm == start != 2.0


# --- versioned files ---------------------------------------------------------------


def test_the_checkpoint_is_versioned(tmp_path: Path):
    folder = _folder(tmp_path)
    _job, ctx = _open(folder)
    saved = json.loads(Path(ctx.checkpoint_path).read_text())
    assert saved["format_version"] == STATE_FORMAT_VERSION
    assert load_checkpoint(ctx.checkpoint_path) is not None


def test_a_checkpoint_from_a_newer_langslice_is_refused(tmp_path: Path):
    folder = _folder(tmp_path)
    _job, ctx = _open(folder)
    _edit_on_disk(ctx.checkpoint_path,
                  lambda data: data.update(format_version=STATE_FORMAT_VERSION + 1))
    with pytest.raises(ValueError, match="newer LangSlice"):
        read_checkpoint(ctx.checkpoint_path)


def test_an_unreadable_history_starts_empty(tmp_path: Path):
    folder = _folder(tmp_path)
    job, _ = _open(folder)
    Path(job.undo_path).write_text("{not json")
    again, _ = _open(folder)
    assert again.undo_stack == []


# --- live shared editing ------------------------------------------------------------


def test_a_script_edit_is_picked_up_before_the_next_tool_and_is_undoable(tmp_path: Path):
    folder = _folder(tmp_path)
    job, ctx = _open(folder, tasks=["position"])
    box = build_tools(job.state, ctx, job.spec, job=job)
    _tool(box, "position_sections")([{"id": "s0.png", "position_mm": 1.0}], view=False)
    start = job.state.by_id("s1.png").position_mm

    def script(data: dict[str, Any]) -> None:
        data["slices"][1]["position_mm"] = 4.5
        data["notes"].append("script: placed s1")

    _edit_on_disk(ctx.checkpoint_path, script)
    rows = {row["id"]: row for row in _tool(box, "status")()["rows"]}
    assert rows["s1.png"]["position_mm"] == 4.5
    assert job.state.by_id("s0.png").position_mm == 1.0

    # The next write builds on the script's edit instead of overwriting it.
    _tool(box, "note")("after the script")
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved.by_id("s1.png").position_mm == 4.5
    assert saved.notes[-2:] == ["script: placed s1", "after the script"]

    # One undo step per write: the note, then the script's edit, then ours.
    assert _tool(box, "undo")()["status"] == "ok"
    assert job.state.notes[-1] == "script: placed s1"
    assert _tool(box, "undo")()["status"] == "ok"
    assert job.state.by_id("s1.png").position_source == "default"
    assert job.state.by_id("s1.png").position_mm == start != 4.5
    assert job.state.by_id("s0.png").position_mm == 1.0
    saved = load_checkpoint(ctx.checkpoint_path).by_id("s1.png")
    assert saved.position_source == "default" and saved.position_mm == start


def test_an_unchanged_or_half_written_file_is_not_a_step(tmp_path: Path):
    folder = _folder(tmp_path)
    job, ctx = _open(folder)
    write_checkpoint(job.state, ctx.checkpoint_path)  # rewritten, same content
    assert job.sync() is None and job.undo_stack == []

    Path(ctx.checkpoint_path).write_text('{"slices": [')  # a script mid-write
    assert job.sync() is None and job.undo_stack == []
    Path(ctx.checkpoint_path).write_text(json.dumps(
        {"format_version": STATE_FORMAT_VERSION, **job.state.to_dict(), "notes": ["finished"]}))
    assert job.sync() is not None
    assert job.state.notes == ["finished"] and len(job.undo_stack) == 1


def test_another_job_writing_both_files_hands_over_its_history(tmp_path: Path):
    folder = _folder(tmp_path)
    first, _ = _open(folder)
    second, _ = _open(folder)
    start = second.state.by_id("s2.png").position_mm
    before = second.snapshot()
    second.state.by_id("s2.png").position_mm = 6.0
    second.commit(before)

    assert first.sync() is not None
    assert first.state.by_id("s2.png").position_mm == 6.0
    assert len(first.undo_stack) == 1  # second's step, not an extra one
    assert first.undo() and first.state.by_id("s2.png").position_source == "default"
    assert first.state.by_id("s2.png").position_mm == start != 6.0


def test_a_job_made_around_a_state_takes_the_files_as_they_stand(tmp_path: Path):
    folder = _folder(tmp_path)
    job, ctx = _open(folder)
    before = job.snapshot()
    job.state.notes.append("on disk")
    job.commit(before)
    # A toolbox built without a job makes one; the file it finds is not news.
    fresh_state = load_checkpoint(ctx.checkpoint_path)
    fresh_state.notes.append("in memory only")
    box = build_tools(fresh_state, ctx, job.spec)
    assert box.job.undo_stack == []
    _tool(box, "status")()
    assert fresh_state.notes[-1] == "in memory only"


# --- gates belong to the tool door ----------------------------------------------------


def test_the_job_is_never_gated_and_undo_resets_the_door_gates(tmp_path: Path):
    folder = _folder(tmp_path)
    job, ctx = _open(folder, tasks=["position"], position=PositionSpec(gated=True))
    # A library write needs no view first.
    before = job.snapshot()
    for index, record in enumerate(job.state.in_order()):
        record.position_mm = 1.0 + index
    job.commit(before)
    assert job.submit_errors([]) is None

    box = build_tools(job.state, ctx, job.spec, job=job)
    look = _tool(box, "look")
    position = _tool(box, "position_sections")
    # The tool door's gate: a section not looked at since its last write is refused.
    refused = position([{"id": "s0.png", "position_mm": 1.5}], view=False)
    assert refused["error"] == "NOT_COMPARED"
    assert job.state.by_id("s0.png").position_mm == 1.0
    look("overlay", sections=["s0.png"])
    assert position([{"id": "s0.png", "position_mm": 1.5}], view=False)["status"] == "ok"
    look("overlay", sections=["s0.png"])
    look("positioning")
    assert box.reviewed and box.compared.get("s0.png")
    # Undo moves s0 back to 1.0: that is a write to its position.
    assert _tool(box, "undo")()["status"] == "ok"
    assert not box.reviewed and "s0.png" not in box.compared
    refused = _tool(box, "submit")("summary", [], [])
    assert refused["error"] == "NOT_REVIEWED"
    # An undo that moves no position leaves the looks alone.
    look("positioning")
    _tool(box, "note")("a note")
    assert _tool(box, "undo")()["status"] == "ok"
    assert box.reviewed
