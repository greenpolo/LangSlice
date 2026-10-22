"""The looking tools through the tool door: look, zoom and status
(``make_image``: ``tests/test_make_image_tool.py``)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.job.background import Landed
from langslice.job.job import ingest
from tests.deformable_synthetic import SyntheticAtlas, bump_field, render_section
from tests.fakes import SlabAtlas
from tests.linear_tool_helpers import tool_named as _tool

POSITIONS = (0.05, 0.15, 0.10, 0.20, 0.12)


@pytest.fixture(scope="module")
def atlas() -> SyntheticAtlas:
    return SyntheticAtlas()


def _synthetic(tmp_path: Path, atlas: SyntheticAtlas, n: int = 3, **spec_kwargs: Any):
    """``(state, ctx, box)``: *n* sections rendered from the synthetic atlas at
    25 um/px, each at its position."""
    for seed in range(n):
        image, _labels = render_section(atlas, bump_field([]), seed=seed)
        image.save(tmp_path / f"s{seed}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   inputs={"pixel_size_um": 25.0}, **spec_kwargs)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    for record, position in zip(state.in_order(), POSITIONS, strict=False):
        record.position_mm = position
    return state, ctx, build_tools(state, ctx, spec)


def _rgb_section(seed: int) -> np.ndarray:
    """Three distinct channels: a broad red, a sparse green, a striped blue."""
    yy, xx = np.mgrid[:120, :160]
    body = ((yy - 60) / 45.0) ** 2 + ((xx - 80) / 65.0) ** 2 < 1
    red = np.where(body, 90 + (xx % 7) * 10, 0)
    green = np.zeros_like(red)
    green[40 + seed:55 + seed, 60:75] = 250
    blue = np.where(body & (yy % 10 < 3), 160, 0)
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def _rgb(tmp_path: Path):
    """``(state, ctx, box)``: two three-channel sections on a slab atlas."""
    for index in range(2):
        Image.fromarray(_rgb_section(index)).save(tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model")
    slab = SlabAtlas(height=48, width=64)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: slab)
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 5.0 + index
    return state, ctx, build_tools(state, ctx, spec)


def _numbers(result: dict[str, Any]) -> list[int]:
    return [entry["id"] for entry in result["pictures"]]


# --- look -------------------------------------------------------------------------


def test_look_numbers_each_picture_and_saves_it_once(tmp_path: Path, atlas):
    state, _, box = _synthetic(tmp_path, atlas, n=5)
    before = state.to_dict()
    result = _tool(box, "look")("section")
    assert result["status"] == "ok"
    # Four pictures at most; the rest is named.
    assert _numbers(result) == [1, 2, 3, 4]
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 4
    assert [entry["caption"].split(" section")[0] for entry in result["pictures"]] == [
        "s0.png", "s1.png", "s2.png", "s3.png"]
    (rest,) = result["not_shown"]
    assert rest["sections"] == ["s4.png"]
    # Saved by its operation, not again by the door.
    records = box.job.views.records()
    assert [(r.seq, r.tool) for r in records] == [(i, "look") for i in (1, 2, 3, 4)]
    # A look writes nothing.
    assert state.to_dict() == before and box.job.undo_stack == []


def test_look_modes_draw_their_pictures(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    look = _tool(box, "look")
    (atlas_plane,) = look("atlas", positions_mm=[0.1])["pictures"]
    assert atlas_plane["caption"].startswith("atlas at 0.10 mm")
    (overlay,) = look("overlay", sections=["s1.png"])["pictures"]
    assert overlay["caption"].startswith("s1.png overlay, at 0.15 mm")
    assert "identity transform" in overlay["caption"] and "atlas borders" in overlay["caption"]
    linear = look("overlay", sections=["s1.png"], warp="none")
    assert linear["status"] == "ok"
    (positioning,) = look("positioning", positions_mm=[0.1])["pictures"]
    assert positioning["caption"].startswith("positioning: 3 sections")
    (image,) = look("section", sections=["s0"], images=["preprocessed"])["pictures"]
    assert "section image preprocessed" in image["caption"]
    (raw,) = look("section", sections=["s0"], images=["gray"])["pictures"]
    assert "section image gray (raw channel, as the file holds it)" in raw["caption"]


def test_look_refuses_bad_requests_and_draws_nothing(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    look = _tool(box, "look")
    assert look("atlas") == {"status": "error", "error": "NO_POSITIONS",
                             "message": "atlas mode needs positions_mm or one section"}
    assert look("sideways")["error"] == "UNKNOWN_MODE"
    nope = look("section", sections=["nope.png", "0"])
    assert nope["error"] == "UNKNOWN_SLICE_IDS" and nope["unknown"] == ["nope.png", "0"]
    assert nope["filenames"] == ["s0.png", "s1.png", "s2.png"]
    assert nope["message"].startswith("No section is named 'nope.png', '0'.")
    assert "Sections have no numbers" in nope["message"]
    assert look("overlay", atlas_opacity=2.0)["error"] == "BAD_ARGS"
    assert look("section", images=["DAPI"])["error"] == "UNKNOWN_IMAGE"
    assert look("overlay", warp="bent")["error"] == "BAD_WARP"
    assert box.job.views.records() == []


# --- zoom -------------------------------------------------------------------------


def test_zoom_redraws_the_newest_picture_or_the_one_named(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    zoom = _tool(box, "zoom")
    assert zoom([0, 0, 50, 50])["error"] == "NO_PICTURE"
    _tool(box, "look")("atlas", positions_mm=[0.1])
    first = zoom([0, 0, 50, 50])
    assert first["status"] == "ok" and first["picture"] == 1 and first["redrawn"] is True
    assert _numbers(first) == [2] and len(first[TOOL_MEDIA_PARTS_KEY]) == 1
    # 0 is the newest picture that is not itself a zoom.
    assert zoom([0, 0, 50, 50], 0)["picture"] == 1
    # A zoom can be zoomed by its number.
    assert zoom([0, 0, 20, 20], 2)["picture"] == 2
    assert zoom([0, 0, 50, 50], 99)["error"] == "UNKNOWN_PICTURE"
    assert [r.tool for r in box.job.views.records()] == ["look", "zoom", "zoom", "zoom"]
    # One region on the latest overlay.
    _tool(box, "look")("overlay", sections=["s0.png"])
    region = zoom(region="STR")
    assert region["status"] == "ok" and region["redrawn"]


def test_zoom_after_a_change_draws_the_picture_as_it_was(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    (shown,) = _tool(box, "look")("overlay", sections=["s0.png"])["pictures"]
    _tool(box, "interactive_transform")([{"id": "s0.png", "rotation_deg": 10.0}],
                                        view=False)
    result = _tool(box, "zoom")([0, 0, 60, 60], shown["id"])
    assert result["status"] == "ok" and result["stale"] is True


# --- status -------------------------------------------------------------------------------


def test_status_is_the_table_and_what_the_run_lets_change(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    result = _tool(box, "status")()
    assert result["status"] == "ok"
    assert [row["id"] for row in result["rows"]] == ["s0.png", "s1.png", "s2.png"]
    assert result["rows"][0]["position_mm"] == 0.05
    assert result["cutting_angles_deg"] == {"pitch": 0.0, "yaw": 0.0}
    assert result["interval_breaks"] == []
    assert result["raw_channels"] == ["gray"]
    assert result["section_images"] == {"preprocessed": {
        "opening": "the file as read, in gray, unenhanced", "made": "when the job opened"}}
    assert "background_running" not in result
    assert result["can_change"] == [
        "positions (the order follows them)", "cutting angles",
        "orientation and mirroring and in-plane transforms by hand",
        "in-plane transforms by elastix_affine", "damaged regions",
        "the section images the pictures and fits read (make_image)", "masks for the fits"]


def test_status_lists_the_background_work_still_running(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    release = threading.Event()

    def land() -> Landed:
        assert release.wait(10)
        return Landed(status="done", text="nothing to write.")

    work = box.job.background.start("demo", ["s0.png"], land, wait_for_images=False)
    try:
        (running,) = _tool(box, "status")()["background_running"]
        assert running["id"] == work.id and running["kind"] == "demo"
        assert running["sections"] == ["s0.png"] and running["status"] == "running"
    finally:
        release.set()
        box.job.background.wait_all()
    after = _tool(box, "status")()
    assert "background_running" not in after
    assert after["background"][0].startswith(f"{work.id} demo of s0.png finished")


def test_an_overlay_shows_the_stored_transform(tmp_path: Path):
    _, _, box = _rgb(tmp_path)
    look = _tool(box, "look")
    before = look("overlay", sections=["s0.png"])
    assert "identity transform" in before["pictures"][0]["caption"]
    _tool(box, "interactive_transform")([{"id": "s0.png", "rotation_deg": 20,
                                          "translate_x_mm": 3}], view=False)
    after = look("overlay", sections=["s0.png"])
    assert "interactive transform" in after["pictures"][0]["caption"]
    assert after[TOOL_MEDIA_PARTS_KEY][0].tobytes() != before[TOOL_MEDIA_PARTS_KEY][0].tobytes()


