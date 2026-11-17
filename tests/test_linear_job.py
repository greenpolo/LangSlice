"""The job layer: undo across a reopen and live shared editing (undo files and
format versions are in test_job_folder.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from langslice.doors.agent.engine import build_context
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import Job
from langslice.job.settings.spec import JobSpec
from langslice.job.state_file import (
    STATE_FORMAT_VERSION,
    stack_to_json,
    write_checkpoint,
)
from tests.fakes import SlabAtlas
from tests.linear_tool_helpers import load_checkpoint
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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    edit(data)
    Path(path).write_text(json.dumps(data), encoding="utf-8")



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
    assert again.state.by_id("s1.png").position_mm == start != 2.0



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

    _tool(box, "note")("after the script")
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved.by_id("s1.png").position_mm == 4.5
    assert saved.notes[-2:] == ["script: placed s1", "after the script"]

    assert _tool(box, "undo")()["status"] == "ok"
    assert job.state.notes[-1] == "script: placed s1"
    assert _tool(box, "undo")()["status"] == "ok"
    assert job.state.by_id("s1.png").position_mm == start != 4.5
    assert job.state.by_id("s0.png").position_mm == 1.0
    saved = load_checkpoint(ctx.checkpoint_path).by_id("s1.png")
    assert saved.position_mm == start


def test_an_unchanged_or_half_written_file_is_not_a_step(tmp_path: Path):
    folder = _folder(tmp_path)
    job, ctx = _open(folder)
    write_checkpoint(job.state, ctx.checkpoint_path)
    assert job.sync() is None and job.undo_stack == []

    Path(ctx.checkpoint_path).write_text('{"slices": [',
        encoding="utf-8")
    assert job.sync() is None and job.undo_stack == []
    Path(ctx.checkpoint_path).write_text(json.dumps(
        {"format_version": STATE_FORMAT_VERSION, **stack_to_json(
            job.state), "notes": ["finished"]}), encoding="utf-8")
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
    assert len(first.undo_stack) == 1
    assert first.undo()
    assert first.state.by_id("s2.png").position_mm == start != 6.0


def test_a_job_made_around_a_state_takes_the_files_as_they_stand(tmp_path: Path):
    folder = _folder(tmp_path)
    job, ctx = _open(folder)
    before = job.snapshot()
    job.state.notes.append("on disk")
    job.commit(before)
    fresh_state = load_checkpoint(ctx.checkpoint_path)
    fresh_state.notes.append("in memory only")
    box = build_tools(fresh_state, ctx, job.spec)
    assert box.job.undo_stack == []
    _tool(box, "status")()
    assert fresh_state.notes[-1] == "in memory only"
