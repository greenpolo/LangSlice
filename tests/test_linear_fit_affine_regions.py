"""Fits by atlas regions.

The silhouette fit on the kept atlas regions only (``ops.transforms.fit_affine``
with include/exclude: an operation no tool offers now, kept for scripts), and
``elastix_affine``'s ``restrict_to`` checked at the tool door.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from langslice.ops import transforms
from tests.fakes import EllipseAtlas, ellipse_section

LEFT, RIGHT, CORE = 2, 3, 4

#: The whole-outline fit on the ellipse fixture, as computed by the code before
#: regions existed: regions must not move it.
PINNED_IOU = 0.985
PINNED_PHYSICAL = {"rotation_deg": -0.004, "scale_x": 0.9998, "scale_y": 1.1398,
                   "translate_x_mm": -0.2817, "translate_y_mm": -0.041, "shear": 0.0001,
                   "pivot": [0.5, 0.5]}
PINNED_PARAMS = [0.999848764400981, -1.8221567469621935e-05, -0.002050823046910042,
                 8.817667081387976e-05, 1.1397842132941491, -0.07076133781644516]


class HalvesAtlas:
    """One ellipse of brain split into a left and a right half, 100 um voxels.

    A small ``CORE`` region sits deep inside the left half, away from the
    outline.
    """

    atlas_name = "fake_halves_100um"
    orientation = "asr"
    resolution = (100.0, 100.0, 100.0)
    metadata = {"species": "mouse"}

    def __init__(self) -> None:
        plane = np.zeros((96, 128), dtype=np.int32)
        cv2.ellipse(plane, (64, 48), (46, 30), 0, 0, 360, LEFT, -1)
        plane[:, 64:][plane[:, 64:] == LEFT] = RIGHT
        cv2.circle(plane, (40, 48), 3, CORE, -1)
        self.annotation = np.repeat(plane[None], 20, axis=0)
        self.template = (self.annotation > 0).astype(np.uint8) * 200
        paths = {1: [1], LEFT: [1, LEFT], RIGHT: [1, RIGHT], CORE: [1, LEFT, CORE]}
        names = {1: "root", LEFT: "L", RIGHT: "R", CORE: "C"}
        self.structures = {
            uid: {"id": uid, "acronym": names[uid], "name": names[uid],
                  "structure_id_path": paths[uid], "rgb_triplet": [200, 200, 200]}
            for uid in paths
        }


def _left_half_section() -> Image.Image:
    """The atlas's LEFT half only, at 50 um/px (twice the atlas's pixels)."""
    canvas = np.full((160, 200, 3), 240, dtype=np.uint8)
    cv2.ellipse(canvas, (100, 80), (92, 60), 0, 0, 360, (30, 30, 30), -1)
    canvas[:, 100:] = 240
    return Image.fromarray(canvas, mode="RGB")


def _box(folder: Path, atlas: Any, image: Image.Image, *, pixel_size_um: float | None = None,
         position_mm: float = 10.0):
    image.save(folder / "s0.png")
    inputs = {"pixel_size_um": pixel_size_um} if pixel_size_um else {}
    spec = JobSpec(image_folder=str(folder), model="fake-model", preprocess="none",
                   inputs=inputs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    state.slices[0].position_mm = position_mm
    return state, build_tools(state, ctx, spec)


def _tool(box: Any, name: str) -> Any:
    return next(t for t in box.tools if t.__name__ == name)


def _fit(box: Any, *, include: tuple[str, ...] = (),
         exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """The silhouette fit of the one section (the operation): its row."""
    job = box.job
    done = transforms.fit_affine(job, job.workspace, [job.state.slices[0]],
                                 method="silhouette", include=include, exclude=exclude)
    return done.rows[0]


def test_the_default_fit_is_unchanged(tmp_path: Path):
    """No regions: the whole-outline fit, numbers pinned from before regions existed."""
    state, box = _box(tmp_path, EllipseAtlas(), ellipse_section())
    row = _fit(box)
    assert "regions" not in row
    assert row["iou"] == PINNED_IOU
    assert row["physical"] == PINNED_PHYSICAL
    assert state.slices[0].transform["params"] == pytest.approx(PINNED_PARAMS, abs=1e-12)
    assert "regions" not in state.slices[0].transform


def test_exclude_fits_the_kept_half_at_true_scale(tmp_path: Path):
    state, box = _box(tmp_path, HalvesAtlas(), _left_half_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    whole = _fit(box)
    _tool(box, "undo")()
    # Against the whole outline the half section is turned onto the long axis.
    assert abs(whole["physical"]["rotation_deg"]) > 45 and whole["iou"] < 0.9
    row = _fit(box, exclude=("R",))
    assert row["status"] == "ok", row
    physical = row["physical"]
    assert physical["scale_x"] == pytest.approx(1.0, abs=0.08)
    assert physical["scale_y"] == pytest.approx(1.0, abs=0.08)
    assert abs(physical["rotation_deg"]) < 3.0
    assert abs(physical["translate_x_mm"]) < 0.15 and abs(physical["translate_y_mm"]) < 0.15
    assert row["iou"] > 0.9
    assert row["regions"]["exclude"] == ["R"]
    assert row["regions"]["atlas_kept_fraction"] == pytest.approx(0.5, abs=0.05)
    assert row["regions"]["tissue_used_fraction"] > 0.95
    assert state.slices[0].transform["regions"] == {"include": [], "exclude": ["R"]}

    # The section's marked damage regions are left out the same way.
    _tool(box, "undo")()
    assert _tool(box, "mark_damage")("s0.png", ["R"])["status"] == "ok"
    marked = _fit(box)
    assert marked["status"] == "ok" and marked["regions"]["exclude"] == ["R"]
    assert marked["physical"] == physical


def test_include_restricts_and_says_what_it_cannot_measure(tmp_path: Path):
    state, box = _box(tmp_path, HalvesAtlas(), _left_half_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    row = _fit(box, include=("L",))
    assert row["status"] == "ok", row
    assert 0 < row["regions"]["outline_share"] < 1
    assert "Only the outline counts" in row["regions"]["note"]
    assert row["physical"]["scale_x"] == pytest.approx(1.0, abs=0.1)
    inside = _fit(box, include=("C",))
    assert inside["error"] == "REGIONS_INSIDE_OUTLINE"


def test_the_fit_tool_checks_its_regions_first(tmp_path: Path):
    """elastix_affine's restrict_to is checked against the atlas before any fit."""
    state, box = _box(tmp_path, HalvesAtlas(), _left_half_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    fit = _tool(box, "elastix_affine")
    assert fit(["s0.png"], restrict_to=["nope"])["error"] == "UNKNOWN_REGIONS"
    assert fit(["s0.png"], restrict_to=["L:middle"])["error"] == "BAD_ARGS"
    assert state.slices[0].transform is None


class WholeAtlas(HalvesAtlas):
    """The same ellipse as ONE region (``L``) across both hemispheres."""

    def __init__(self) -> None:
        super().__init__()
        self.annotation = np.where(self.annotation == RIGHT, LEFT, self.annotation)


def test_a_one_sided_exclusion_keeps_the_other_hemisphere(tmp_path: Path):
    """Excluding ``L:right`` drops only the right half of a region spanning both."""
    state, box = _box(tmp_path, WholeAtlas(), _left_half_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    row = _fit(box, exclude=("L:right",))
    assert row["status"] == "ok", row
    assert row["regions"]["atlas_kept_fraction"] == pytest.approx(0.5, abs=0.05)
    assert abs(row["physical"]["rotation_deg"]) < 3.0
    assert row["physical"]["scale_x"] == pytest.approx(1.0, abs=0.08)
    assert row["iou"] > 0.9
    assert state.slices[0].transform["regions"] == {"include": [], "exclude": ["L:right"]}
    # Excluding the side the tissue IS on leaves the wrong half to fit against.
    _tool(box, "undo")()
    wrong = _fit(box, exclude=("L:left",))
    assert wrong["status"] == "ok" and wrong["regions"]["tissue_used_fraction"] < 0.2


def _tall_section() -> Image.Image:
    canvas = np.full((200, 160, 3), 240, dtype=np.uint8)
    cv2.ellipse(canvas, (80, 100), (60, 92), 0, 0, 360, (30, 30, 30), -1)
    return Image.fromarray(canvas, mode="RGB")


def test_a_side_on_a_turned_placement_is_a_refusal_not_an_exception(tmp_path: Path):
    """A one-sided region on a placement turned past 45 degrees has no side."""
    state, box = _box(tmp_path, HalvesAtlas(), _tall_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    turned = _tool(box, "interactive_transform")(
        [{"id": "s0.png", "rotation_deg": 90, "scale_x": 1, "scale_y": 1,
          "translate_x_mm": 0, "translate_y_mm": 0}], view=False)
    assert turned["status"] == "ok"
    before = dict(state.slices[0].transform or {})
    result = _tool(box, "elastix_affine")(["s0.png"], restrict_to=["L:left"])
    assert result["error"] == "NOTHING_FITTED"
    assert result["results"][0]["error"] == "SIDES_AMBIGUOUS"
    assert "pictures" not in result
    assert state.slices[0].transform == before
