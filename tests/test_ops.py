"""The operations (``langslice.ops``) as a script calls them: plain arguments,
one undo step per call, a plain record back, no gates and no pictures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec, TransformSpec
from langslice.core.state import SliceState
from langslice.job.job import Job
from langslice.ops import appearance, damage, notes, order, orientation, positions, transforms
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


def test_set_positions_clamps_writes_and_takes_one_undo_step(tmp_path: Path):
    job, ctx = _open(tmp_path)
    low, high = ctx.position_range
    done = positions.set_positions(job, ctx, [("s0.png", low + 0.1), (1, high + 5.0),
                                              ("nope.png", 0.2)])
    assert done.written == [("s0.png", low + 0.1), ("s1.png", high)]
    assert done.clamped == [("s1.png", high + 5.0, high)]
    assert done.unknown == ["nope.png"] and done.touched == ["s0.png", "s1.png"]
    assert len(job.undo_stack) == 1
    assert job.undo() and _record(job, "s1.png").position_mm is None
    # Nothing to write: no undo step.
    assert positions.set_positions(job, ctx, [("nope.png", 0.0)]).written == []
    assert job.undo_stack == []


def test_cutting_angles_drop_the_render_cache(tmp_path: Path):
    job, ctx = _open(tmp_path)
    ctx.render_cache["stale"] = object()
    positions.set_cutting_angles(job, ctx, 2.0, -1.0)
    assert job.state.cutting_angles_deg == {"pitch": 2.0, "yaw": -1.0}
    assert ctx.render_cache == {} and len(job.undo_stack) == 1


def test_reorder_moves_a_block_and_refuses_bad_names(tmp_path: Path):
    job, _ = _open(tmp_path)
    done = order.reorder(job, ["s2.png"], after="start")
    assert [record.id for record in job.state.in_order()] == ["s2.png", "s0.png", "s1.png"]
    assert sorted(done.moved) == ["s0.png", "s1.png", "s2.png"]
    with pytest.raises(Refused) as unknown:
        order.reorder(job, ["x.png"])
    assert unknown.value.payload() == {"status": "error", "error": "UNKNOWN_SLICE_IDS",
                                       "unknown": ["x.png"]}
    with pytest.raises(Refused) as anchor:
        order.reorder(job, ["s0.png"], after="s0.png")
    assert anchor.value.code == "BAD_ARGS"
    assert len(job.undo_stack) == 1


def test_orientation_clears_a_stale_transform_and_honours_the_spec(tmp_path: Path):
    job, _ = _open(tmp_path, transform=TransformSpec(flip=False))
    _record(job, "s0.png").transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    done = orientation.orient_sections(job, [{"id": "s0.png", "rotate_deg": 90},
                                             {"id": "s1.png", "flip": True},
                                             {"id": "s2.png", "rotate_deg": 45}])
    assert done.applied == ["s0.png", "s1.png", "s2.png"]
    assert done.cleared_transforms == ["s0.png"]
    assert _record(job, "s0.png").transform is None
    assert {row["error"] for row in done.rejected} == {"FLIP_DISABLED", "BAD_ROTATION"}
    assert _record(job, "s1.png").flip is False


def test_damage_flags_and_host_flags(tmp_path: Path):
    job, _ = _open(tmp_path, inputs={"damaged": {"s1.png": "host note"}})
    done = damage.mark_damaged(job, [{"id": "s0.png", "note": " torn "},
                                     {"id": "s1.png", "damaged": False}])
    assert done.marked == ["s0.png"] and done.unmarked == []
    assert done.rejected == [{"id": "s1.png", "error": "DAMAGE_SET_BY_USER"}]
    assert _record(job, "s0.png").damage_note == "torn"


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
    planned = appearance.planned_settings(job.state, "view", ["s1.png"], settings, "s1.png")
    assert job.state.appearance == {}  # planning writes nothing
    done = appearance.set_appearance(job, ["view", "fit"], ["s1.png"], settings)
    assert done.in_force["view"] == {"sections": {"s1.png": planned}}
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
        "elastix", {"params": [1, 0, 0, 0, 1, 0], "physical": {}, "iou": 0.5,
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


def test_view_slices_lists_channels_only_for_the_sections_pictured(monkeypatch):
    """With keep_going, a section whose picture failed is reported, and its
    file is not read again for its channel names."""
    from types import SimpleNamespace

    from langslice.ops import views

    def picture(_workspace: Any, _state: Any, record: Any, _options: Any) -> str:
        if record.id == "bad.png":
            raise OSError("unreadable")
        return "picture"

    asked: list[str] = []

    def channels(section_id: str) -> tuple[list[str], None]:
        asked.append(section_id)
        return ["DAPI"], None

    monkeypatch.setattr(views, "section_picture", picture)
    done = views.view_slices(
        SimpleNamespace(state=None), SimpleNamespace(section_channels=channels),  # type: ignore[arg-type]
        [SimpleNamespace(id="good.png"), SimpleNamespace(id="bad.png")],  # type: ignore[list-item]
        SimpleNamespace(mode="channels"), keep_going=True)  # type: ignore[arg-type]
    assert done.channels == {"good.png": ["DAPI"]} and asked == ["good.png"]
    assert done.failed == [{"id": "bad.png", "message": "unreadable"}]
