"""`ops.transforms.interactive_transform` (orientation and knobs, absolute values, one
undo step) and `elastix_affine` (regions, automatic damage exclusion, region box)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec, TransformSpec
from langslice.job.job import Job
from langslice.ops import transforms
from langslice.ops.refusal import Refused
from tests.test_linear_fit_affine_regions import HalvesAtlas, _left_half_section

_ATLAS = HalvesAtlas()


def _open(tmp_path: Path, n: int = 2, position_mm: float | None = 1.0,
          **spec_kwargs: Any) -> tuple[Job, Any]:
    image: Image.Image = _left_half_section()
    for index in range(n):
        image.save(tmp_path / f"s{index}.png")
    inputs = {"pixel_size_um": 50.0, **spec_kwargs.pop("inputs", {})}
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   inputs=inputs, **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    job = Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)
    for record in job.state.slices:
        record.position_mm = position_mm
    return job, ctx


def _record(job: Job, name: str) -> Any:
    record = job.state.by_id(name)
    assert record is not None
    return record


def test_one_undo_step_for_orientation_and_knobs_on_several_sections(tmp_path: Path):
    job, ctx = _open(tmp_path)
    done = transforms.interactive_transform(job, ctx, [
        {"id": "s0.png", "flip": True, "rotation_deg": 3.0, "scale_x": 1.1},
        {"id": "s1.png", "rotate_quarter": 90, "translate_x_mm": 0.1},
    ])
    assert done.written == ["s0.png", "s1.png"]
    assert [entry.error for entry in done.entries] == [None, None]
    assert (_record(job, "s0.png").flip, _record(job, "s1.png").rotation_deg) == (True, 90)
    physical = _record(job, "s0.png").transform["physical"]
    assert physical["rotation_deg"] == 3.0 and physical["scale_x"] == 1.1
    assert physical["scale_y"] == 1.0 and physical["translate_x_mm"] == 0.0
    assert len(job.undo_stack) == 1
    assert job.undo()
    for name in ("s0.png", "s1.png"):
        record = _record(job, name)
        assert (record.flip, record.rotation_deg, record.transform) == (False, 0, None)


def test_a_knob_left_out_keeps_its_value_and_the_same_numbers_write_nothing(tmp_path: Path):
    job, ctx = _open(tmp_path)
    transforms.interactive_transform(job, ctx, [
        {"id": "s0.png", "rotation_deg": 3.0, "scale_x": 1.1, "shear": 0.02,
         "pivot": [0.4, 0.6]}])
    done = transforms.interactive_transform(job, ctx, [
        {"id": "s0.png", "translate_x_mm": 0.2}])
    assert done.written == ["s0.png"]
    physical = _record(job, "s0.png").transform["physical"]
    assert physical["translate_x_mm"] == 0.2
    assert physical["rotation_deg"] == 3.0 and physical["scale_x"] == 1.1
    assert physical["shear"] == 0.02 and physical["pivot"] == [0.4, 0.6]
    assert len(job.undo_stack) == 2
    again = transforms.interactive_transform(job, ctx, [
        {"id": "s0.png", "translate_x_mm": 0.2, "rotation_deg": 3.0}])
    assert again.written == [] and again.entries[0].written is False
    assert len(job.undo_stack) == 2


def test_an_orientation_change_keeps_the_knob_values_and_says_so(tmp_path: Path):
    job, ctx = _open(tmp_path)
    transforms.interactive_transform(job, ctx, [
        {"id": "s0.png", "rotation_deg": 2.0, "scale_y": 0.9, "translate_y_mm": -0.1}])
    held = dict(_record(job, "s0.png").transform["physical"])
    done = transforms.interactive_transform(job, ctx, [{"id": "s0.png", "rotate_quarter": 90}])
    entry = done.entries[0]
    record = _record(job, "s0.png")
    assert record.rotation_deg == 90 and record.transform is not None
    assert record.transform["physical"] == held
    assert entry.orientation == {"flip": False, "rotate_quarter": 90, "changed": True}
    assert entry.kept is not None and entry.kept["scale_y"] == 0.9
    assert "knob values were kept" in entry.message
    assert len(job.undo_stack) == 2
    # No transform and no knob given: the orientation alone, no transform made.
    alone = transforms.interactive_transform(job, ctx, [{"id": "s1.png", "flip": True}])
    assert alone.entries[0].message == "" and _record(job, "s1.png").flip is True
    assert _record(job, "s1.png").transform is None


def test_refusals_write_nothing(tmp_path: Path):
    job, ctx = _open(tmp_path, n=3, inputs={"locked": ["s2.png"]},
                     transform=TransformSpec(flip=False))
    _record(job, "s1.png").position_mm = None
    done = transforms.interactive_transform(job, ctx, [
        {"id": "s2.png", "rotation_deg": 1.0},
        {"id": "s0.png", "flip": True},
        {"id": "s0.png", "rotate_quarter": 45},
        {"id": "s0.png", "scale_x": float("inf")},
        {"id": "s0.png", "scale_x": "wide"},
        {"id": "s1.png", "rotation_deg": 1.0},
        {"id": "nope.png", "rotation_deg": 1.0},
        "s0.png",
    ])
    assert [entry.error["error"] for entry in done.entries if entry.error] == [
        "LOCKED", "FLIP_DISABLED", "BAD_ROTATION", "BAD_ARGS", "BAD_ARGS", "NO_POSITION",
        "UNKNOWN_SLICE_IDS", "BAD_ARGS"]
    assert done.written == [] and job.undo_stack == []
    assert _record(job, "s0.png").transform is None


# --- elastix_affine ----------------------------------------------------------------------


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the Elastix fitter with one that records its regions and atlas image."""
    seen: list[dict[str, Any]] = []

    def fake(_state: Any, _ws: Any, record: Any, *, include: Any = (), exclude: Any = (),
             atlas_image: str = "template") -> dict[str, Any]:
        seen.append({"id": record.id, "include": tuple(include), "exclude": tuple(exclude),
                     "atlas_image": atlas_image})
        return {"status": "ok", "id": record.id, "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "physical": {"rotation_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0,
                             "translate_x_mm": 0.0, "translate_y_mm": 0.0, "shear": 0.0,
                             "pivot": [0.5, 0.5]},
                "iou": 0.9, "calibration": {}, "mirrored": False}

    monkeypatch.setattr("langslice.core.transform.fit_elastix", fake)
    return seen


def test_elastix_affine_restricts_and_carries_the_region_box(
        tmp_path: Path, calls: list[dict[str, Any]]):
    job, ctx = _open(tmp_path)
    done = transforms.elastix_affine(job, ctx, ["s0.png"], restrict_to=["L"],
                                     atlas_image="nissl")
    assert calls == [{"id": "s0.png", "include": ("L",), "exclude": (),
                      "atlas_image": "nissl"}]
    assert done.fitted == ["s0.png"] and len(job.undo_stack) == 1
    record = _record(job, "s0.png")
    assert record.transform["kind"] == "elastix" and record.transform["fit_atlas"] == "nissl"
    assert record.transform["regions"] == {"include": ["L"], "exclude": []}
    x0, y0, x1, y1 = done.rows[0]["restrict_box"]
    assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    # The left half of the brain: the box leaves the right side of the canvas out.
    assert x1 < 0.8 and (x0 + x1) / 2 < 0.5
    # Every section when none is named; "ara" still reads as the template; no regions, no box.
    calls.clear()
    everything = transforms.elastix_affine(job, ctx, atlas_image="ara")
    assert [call["id"] for call in calls] == ["s0.png", "s1.png"]
    assert {call["atlas_image"] for call in calls} == {"template"}
    assert all("restrict_box" not in row for row in everything.rows)


def test_elastix_affine_leaves_marked_damage_out_on_its_own(
        tmp_path: Path, calls: list[dict[str, Any]]):
    job, ctx = _open(tmp_path)
    _record(job, "s0.png").damaged_regions = ["R"]
    transforms.elastix_affine(job, ctx, ["s0.png", "s1.png"])
    assert {call["id"]: call["exclude"] for call in calls} == {"s0.png": ("R",),
                                                               "s1.png": ()}
    # restrict_to another region keeps the mark out; naming the marked region wins.
    calls.clear()
    transforms.elastix_affine(job, ctx, ["s0.png"], restrict_to=["L"])
    transforms.elastix_affine(job, ctx, ["s0.png"], restrict_to=["R"])
    assert [(call["include"], call["exclude"]) for call in calls] == [
        (("L",), ("R",)), (("R",), ())]


def test_elastix_affine_refuses_bad_arguments_and_unfittable_sections(
        tmp_path: Path, calls: list[dict[str, Any]]):
    job, ctx = _open(tmp_path)
    with pytest.raises(Refused) as unknown:
        transforms.elastix_affine(job, ctx, ["nope.png"])
    assert unknown.value.code == "UNKNOWN_SLICE_IDS"
    with pytest.raises(Refused) as image:
        transforms.elastix_affine(job, ctx, atlas_image="colour")
    assert image.value.code == "BAD_ARGS"
    with pytest.raises(Refused) as regions:
        transforms.elastix_affine(job, ctx, ["s0.png"], restrict_to=[3])  # type: ignore[list-item]
    assert regions.value.code == "BAD_ARGS"
    assert calls == [] and job.undo_stack == []


def test_elastix_affine_refuses_a_locked_section_and_fits_a_noted_one(
        tmp_path: Path, calls: list[dict[str, Any]]):
    """A section the user locked is refused; a damage note alone (no marked
    regions) is no damage, so that section is fitted whole."""
    job, ctx = _open(tmp_path, inputs={"locked": ["s1.png"],
                                       "damaged": {"s0.png": "torn"}})
    assert [record.id for record in transforms.fit_targets(job)] == ["s0.png"]
    done = transforms.elastix_affine(job, ctx, ["s0.png", "s1.png"])
    assert [row.get("error") for row in done.rows] == [None, "LOCKED"]
    assert done.fitted == ["s0.png"]
    assert calls == [{"id": "s0.png", "include": (), "exclude": (), "atlas_image": "template"}]
