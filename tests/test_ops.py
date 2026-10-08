"""The operations (``langslice.ops``) as a script calls them: plain arguments,
one undo step per call, a plain record back, no gates and no pictures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec
from langslice.core.state import SliceState
from langslice.job.job import Job
from langslice.ops import appearance, damage, notes, positions, transforms
from langslice.ops.refusal import Refused
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _open(tmp_path: Path, n: int = 3, **spec_kwargs: Any) -> tuple[Job, Any]:
    for index in range(n):
        Image.fromarray(np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)).save(
            tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    return job, ctx


def _record(job: Job, name: str) -> SliceState:
    record = job.state.by_id(name)
    assert record is not None
    return record


def test_positions_are_clamped_written_and_one_undo_step(tmp_path: Path):
    job, ctx = _open(tmp_path)
    low, high = ctx.position_range
    start = _record(job, "s1.png").position_mm
    done = positions.position_sections(job, ctx, [
        {"id": "s0.png", "position_mm": low + 0.1}, {"id": 1, "position_mm": high + 5.0},
        {"id": "nope.png", "position_mm": 0.2}])
    assert done.written == [("s0.png", low + 0.1), ("s1.png", high)]
    assert done.clamped == [("s1.png", high + 5.0, high)]
    assert done.unknown == ["nope.png"] and done.touched == ["s0.png", "s1.png"]
    assert len(job.undo_stack) == 1
    assert job.undo() and _record(job, "s1.png").position_source == "default"
    assert _record(job, "s1.png").position_mm == start != high
    # Nothing to write: no undo step.
    assert positions.position_sections(
        job, ctx, [{"id": "nope.png", "position_mm": 0.0}]).written == []
    assert job.undo_stack == []


def test_cutting_angles_drop_the_render_cache(tmp_path: Path):
    job, ctx = _open(tmp_path)
    ctx.render_cache["stale"] = object()
    positions.position_sections(job, ctx, [], {"pitch_deg": 2.0, "yaw_deg": -1.0})
    assert job.state.cutting_angles_deg == {"pitch": 2.0, "yaw": -1.0}
    # Dropped (the checkpoint's registration.json may render the placed
    # sections again at the new angles).
    assert "stale" not in ctx.render_cache and len(job.undo_stack) == 1


def test_a_host_damage_note_is_a_note_alone(tmp_path: Path):
    """``inputs.damaged`` is the section's ``damage_note``, not a damage mark:
    the section is damaged only once it has marked regions."""
    job, ctx = _open(tmp_path, inputs={"damaged": {"s1.png": "host note"}})
    record = _record(job, "s1.png")
    assert record.damage_note == "host note" and not record.damaged
    assert record.damaged_regions == []
    # Clearing marks the section does not have writes nothing; the note stays.
    done = damage.mark_damage(job, ctx, "s1.png", [], "ignored")
    assert not done.written and done.by_user and not done.damaged
    assert done.note == "host note" and job.undo_stack == []
    with pytest.raises(Refused) as unknown:
        damage.mark_damage(job, ctx, "x.png", [])
    assert unknown.value.code == "UNKNOWN_SLICE_IDS"


def test_notes_append_and_refuse_empty(tmp_path: Path):
    job, _ = _open(tmp_path)
    assert notes.add_note(job, "  first  ")[-1] == "first"
    with pytest.raises(Refused):
        notes.add_note(job, "   ")
    assert len(job.undo_stack) == 1


def test_appearance_plan_matches_the_write(tmp_path: Path):
    job, _ = _open(tmp_path)
    settings = {"channel_weights": None, "clahe_clip": 2.0, "clahe_tiles": 8,
                "n4": False, "denoise": False}
    planned = appearance.planned_settings(job.state, "preprocessed", ["s1.png"], settings,
                                          "s1.png")
    assert job.state.appearance == {}  # planning writes nothing
    done = appearance.set_appearance(job, ["preprocessed"], ["s1.png"], settings)
    assert done.in_force["preprocessed"] == {"sections": {"s1.png": planned}}
    assert planned == settings and len(job.undo_stack) == 1


def test_transform_records_and_one_undo_step_for_a_batch(tmp_path: Path):
    job, _ = _open(tmp_path, inputs={"locked": ["s2.png"]})
    knobs = {"rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0,
             "translate_x_mm": 0.0, "translate_y_mm": 0.0}
    record = transforms.transform_record(
        size=(40, 30), um_per_px=10.0, calibration={"section_um_per_px": 10.0, "source": "host"},
        pivot=None, pivot_frac=[0.5, 0.5], knobs=knobs, note=" why ")
    assert record["params"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert record["physical"] == {**knobs, "shear": 0.0, "pivot": [0.5, 0.5]}
    assert record["note"] == "why"
    assert not transforms.same_transform(None, record)
    assert transforms.same_transform({**record, "note": "other"}, record)

    fitted = transforms.fit_transform(
        {"params": [1, 0, 0, 0, 1, 0], "physical": {}, "iou": 0.5,
                    "calibration": {}, "mirrored": False},
        exclude=["HY"], fit_atlas="nissl")
    assert fitted["regions"] == {"include": [], "exclude": ["HY"]}
    assert fitted["fit_atlas"] == "nissl"

    assert transforms.set_transforms(job, {"s0.png": record, "s1.png": fitted}) == [
        "s0.png", "s1.png"]
    assert len(job.undo_stack) == 1
    assert job.undo() and _record(job, "s1.png").transform is None
    with pytest.raises(Refused) as locked:
        transforms.set_transforms(job, {"s0.png": record, "s2.png": record})
    assert locked.value.code == "LOCKED" and _record(job, "s0.png").transform is None
    assert transforms.set_transforms(job, {}) == []
