"""``elastix_affine``'s ``restrict_to``: checked at the tool door, and a side
that a turned placement cannot resolve refused."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.doors.agent.engine import build_context
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from langslice.job.settings.spec import JobSpec

LEFT, RIGHT, CORE = 2, 3, 4


# HalvesAtlas and _left_half_section are used by tests/test_damage_regions.py,
# tests/test_ops_interactive_transform.py and tests/test_tool_plugins.py
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
    before = state.slices[0].transform
    result = _tool(box, "elastix_affine")(["s0.png"], restrict_to=["L:left"])
    assert result["error"] == "WRONG_SIDE_PAIR"
    assert result["results"][0]["error"] == "WRONG_SIDE_PAIR"
    assert "pictures" not in result
    assert state.slices[0].transform == before


def test_tiny_region_is_refused_before_elastix_and_does_not_write(tmp_path: Path, monkeypatch):
    from langslice.core.fits.deformable import engines

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


def test_a_script_fit_answers_as_the_tool_row_does(tmp_path: Path, monkeypatch):
    """A script's Elastix fit refuses with the code the tool's row carries: a
    side the placement lacks is ``WRONG_SIDE_PAIR`` (not ``FIT_FAILED``), and a
    region too small is ``REGION_TOO_SMALL`` with its text written for the job."""
    from langslice.core.fits.deformable import engines
    from langslice.core.refusal import Refused
    from langslice.tools.transforms import elastix_fit

    state, box = _box(tmp_path, HalvesAtlas(), _tall_section(), pixel_size_um=50.0,
                      position_mm=1.0)
    _tool(box, "interactive_transform")(
        [{"id": "s0.png", "rotation_deg": 90, "scale_x": 1, "scale_y": 1,
          "translate_x_mm": 0, "translate_y_mm": 0}], view=False)
    with pytest.raises(Refused) as caught:
        elastix_fit(box.job, box.run.ctx, "s0.png", restrict_to=["L:left"])
    assert caught.value.code == "WRONG_SIDE_PAIR"

    atlas = HalvesAtlas()
    atlas.annotation[atlas.annotation == CORE] = LEFT
    atlas.annotation[:, 47:50, 39:42] = CORE
    folder = tmp_path / "tiny"
    folder.mkdir()
    state, box = _box(folder, atlas, _left_half_section(), pixel_size_um=50.0, position_mm=0.5)
    monkeypatch.setattr(engines, "run_elastix_affine",
                        lambda *a, **k: pytest.fail("Tiny region reached Elastix"))
    with pytest.raises(Refused) as caught:
        elastix_fit(box.job, box.run.ctx, "s0.png", restrict_to=["C"])
    assert caught.value.code == "REGION_TOO_SMALL"
    assert "[[" not in caught.value.message
    assert "interactive_transform" in caught.value.message


def test_a_fit_refusal_without_a_message_is_passed_on_without_one(
    tmp_path: Path, monkeypatch
):
    from langslice.core.fits import affine
    from langslice.core.refusal import Refused
    from langslice.tools import transforms

    state, box = _box(tmp_path, HalvesAtlas(), _left_half_section(),
                      pixel_size_um=50.0, position_mm=0.5)

    def refuse(*args, **kwargs):
        raise Refused("NO_STRUCTURES", regions=["C"])

    monkeypatch.setattr(affine, "elastix_affine", refuse)
    with pytest.raises(Refused) as caught:
        transforms.elastix_fit(box.job, box.run.ctx, "s0.png")
    assert caught.value.code == "NO_STRUCTURES"
    assert caught.value.facts == {"regions": ["C"]}
