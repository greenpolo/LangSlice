"""``elastix_affine``'s ``restrict_to``: checked at the tool door, and a side
that a turned placement cannot resolve refused."""

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

LEFT, RIGHT, CORE = 2, 3, 4

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


def test_the_fit_tool_checks_its_regions_first(tmp_path: Path):
    """elastix_affine's restrict_to is checked against the atlas before any fit."""
    state, box = _box(tmp_path, HalvesAtlas(), _left_half_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    fit = _tool(box, "elastix_affine")
    assert fit(["s0.png"], restrict_to=["nope"])["error"] == "UNKNOWN_REGIONS"
    assert fit(["s0.png"], restrict_to=["L:middle"])["error"] == "BAD_ARGS"
    assert state.slices[0].transform is None


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


def test_a_restricted_fit_highlights_its_regions_and_zooms_only_when_it_gains(tmp_path: Path):
    """The picture of a fit by restrict_to draws those regions thick and the
    others faint, says so, and zooms to their box only when that magnifies."""
    from langslice.ops.transforms import MIN_ZOOM_GAIN, zoom_window

    _state, box = _box(tmp_path, HalvesAtlas(), _left_half_section(), pixel_size_um=50.0,
                       position_mm=1.0)
    half = _tool(box, "elastix_affine")(["s0.png"], restrict_to=["L"])
    assert half["status"] == "ok"
    (picture,) = half["pictures"]
    assert "regions L drawn thick, the other borders faint" in picture["caption"]
    zoomed = zoom_window(half["results"][0]["restrict_box"]) is not None
    assert ("from the image file" in picture["caption"]) == zoomed  # a zoom reads the file
    assert zoom_window([0.0, 0.1, 0.9, 0.8]) is None  # a 1.1x zoom: drawn whole
    assert zoom_window([0.2, 0.2, 0.2 + 1 / MIN_ZOOM_GAIN, 0.5]) is not None


def test_tiny_region_is_refused_before_elastix_and_does_not_write(tmp_path: Path, monkeypatch):
    from langslice.core.deformable import engines

    atlas = HalvesAtlas()
    # Three atlas pixels across: the case that can crash the native engine.
    atlas.annotation[atlas.annotation == CORE] = LEFT
    atlas.annotation[:, 47:50, 39:42] = CORE
    state, box = _box(tmp_path, atlas, _left_half_section(),
                      pixel_size_um=50.0, position_mm=0.5)
    before = box.job.snapshot()

    def must_not_run(*args, **kwargs):
        pytest.fail("Tiny region reached the native Elastix engine")

    monkeypatch.setattr(engines, "run_elastix_affine", must_not_run)
    result = _tool(box, "elastix_affine")(["s0.png"], restrict_to=["C"], view=False)
    row = result["results"][0]
    assert row["error"] == "REGION_TOO_SMALL"
    assert "Region is too small for elastix" in row["message"]
    assert box.job.snapshot() == before
    assert state.slices[0].transform is None
