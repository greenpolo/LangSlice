"""The looking tools through the tool door: look, zoom, the two channel
tools, grep_atlas_view, status and the job-folder tools."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core import appearance as looks
from langslice.core.sections import render_slice
from langslice.core.spec import JobSpec, NonlinearSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.job.background import Landed
from langslice.job.checkpoint import load_checkpoint
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
    state, _, box = _synthetic(tmp_path, atlas)
    before = state.to_dict()
    result = _tool(box, "look")("section")
    assert result["status"] == "ok"
    assert _numbers(result) == [1, 2, 3]
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 3
    assert [entry["caption"].split(" section")[0] for entry in result["pictures"]] == [
        "s0.png", "s1.png", "s2.png"]
    assert "25.0 um/px" in result["pictures"][0]["caption"]
    assert "not_shown" not in result
    # Saved by its operation, not again by the door.
    records = box.job.views.records()
    assert [(r.seq, r.tool) for r in records] == [(1, "look"), (2, "look"), (3, "look")]
    # A look writes nothing.
    assert state.to_dict() == before and box.job.undo_stack == []


def test_look_shows_four_pictures_and_names_the_rest(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas, n=5)
    result = _tool(box, "look")("section")
    assert len(result["pictures"]) == 4 and len(result[TOOL_MEDIA_PARTS_KEY]) == 4
    (rest,) = result["not_shown"]
    assert rest["sections"] == ["s4.png"]
    assert rest["how"].startswith("call again with mode='section', sections=['s4.png']")


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
    (channel,) = look("section", sections=["s0"], channels=["preprocessed"])["pictures"]
    assert "preprocessed channel" in channel["caption"]


def test_look_refuses_bad_requests_and_draws_nothing(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    look = _tool(box, "look")
    assert look("atlas") == {"status": "error", "error": "NO_POSITIONS",
                             "message": "atlas mode needs positions_mm"}
    assert look("sideways")["error"] == "UNKNOWN_MODE"
    nope = look("section", sections=["nope.png", "0"])
    assert nope["error"] == "UNKNOWN_SLICE_IDS" and nope["unknown"] == ["nope.png", "0"]
    assert nope["filenames"] == ["s0.png", "s1.png", "s2.png"]
    assert nope["message"].startswith("No section is named 'nope.png', '0'.")
    assert "Sections have no numbers" in nope["message"]
    assert look("overlay", atlas_opacity=2.0)["error"] == "BAD_ARGS"
    assert look("section", channels=["DAPI"])["error"] == "UNKNOWN_CHANNEL"
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


def test_zoom_after_a_change_draws_the_picture_as_it_was(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    (shown,) = _tool(box, "look")("overlay", sections=["s0.png"])["pictures"]
    _tool(box, "interactive_transform")([{"id": "s0.png", "rotation_deg": 10.0}],
                                        view=False)
    result = _tool(box, "zoom")([0, 0, 60, 60], shown["id"])
    assert result["status"] == "ok" and result["stale"] is True


# --- set_channel_properties ----------------------------------------------------------


def test_channel_properties_persist_are_undoable_and_captioned(tmp_path: Path, atlas):
    state, _, box = _synthetic(tmp_path, atlas)
    setter = _tool(box, "set_channel_properties")
    result = setter("gray", gamma=1.5, contrast_limits=[20, 200])
    assert result["status"] == "ok" and result["changed"] is True
    assert result["channel"] == "gray"
    assert result["properties"]["gamma"] == 1.5
    assert result["sections"] == ["s0.png", "s1.png", "s2.png"]
    assert result["intensities"]["dtype"] == "uint8"
    assert len(box.job.undo_stack) == 1
    (picture,) = _tool(box, "look")("section", sections=["s0.png"])["pictures"]
    assert "gray 20-200 gamma 1.5" in picture["caption"]
    assert _tool(box, "status")()["channel_display"] == "gray 20-200 gamma 1.5"
    # The same settings again change nothing and take no undo step.
    assert setter("gray", gamma=1.5, contrast_limits=[20, 200])["changed"] is False
    assert len(box.job.undo_stack) == 1
    assert setter("dapi", gamma=2.0) == {"status": "error", "error": "UNKNOWN_CHANNEL",
                                         "channel": "dapi", "channels": ["gray"]}
    _tool(box, "undo")()
    assert _tool(box, "status")()["channel_display"] == "gray auto"
    setter("gray", gamma=2.0)
    assert setter("gray", reset=True)["properties"] is None


# --- set_preprocessed_channel_properties -------------------------------------------------


def test_preprocessed_recipe_returns_before_and_after_pictures(tmp_path: Path):
    state, _, box = _rgb(tmp_path)
    result = _tool(box, "set_preprocessed_channel_properties")(
        sections=["s0.png"], channel_weights=[0, 0, 1])
    assert result["status"] == "ok", result
    assert result["scope"] == ["s0.png"] and result["shown"] == ["s0.png"]
    assert result["channels"] == ["red", "green", "blue"]
    before, after = result["pictures"]
    assert "before this call" in before["caption"] and "after this call" in after["caption"]
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 2
    assert np.asarray(result[TOOL_MEDIA_PARTS_KEY][0]).tobytes() != np.asarray(
        result[TOOL_MEDIA_PARTS_KEY][1]).tobytes()
    # The door saved the two plain pictures, with the numbers it listed.
    records = box.job.views.records()
    assert [r.seq for r in records] == [before["id"], after["id"]]
    assert {r.tool for r in records} == {"set_preprocessed_channel_properties"}
    assert looks.preprocessed_settings(state, "s0.png")["channel_weights"] == [0, 0, 1]
    assert looks.preprocessed_settings(state, "s1.png") is None


def test_a_section_recipe_wins_and_a_reset_returns_to_the_stack(tmp_path: Path):
    state, _, box = _rgb(tmp_path)
    tool = _tool(box, "set_preprocessed_channel_properties")
    whole = tool(channel_weights=[1, 0, 0])
    assert whole["scope"] == "stack" and whole["shown"] == ["s0.png", "s1.png"]
    assert tool(sections=["s1.png"], channel_weights=[0, 1, 0], clahe_tiles=4)["scope"] == [
        "s1.png"]
    assert looks.preprocessed_settings(state, "s0.png")["channel_weights"] == [1, 0, 0]
    assert looks.preprocessed_settings(state, "s1.png")["clahe_tiles"] == 4
    assert _tool(box, "status")()["preprocessed_recipe"]["stack"]["channel_weights"] == [
        1, 0, 0]
    tool(sections=["s1.png"], reset=True)
    assert looks.preprocessed_settings(state, "s1.png")["channel_weights"] == [1, 0, 0]
    tool(reset=True)
    assert looks.preprocessed_settings(state, "s0.png") is None
    assert _tool(box, "status")()["preprocessed_recipe"] == "default"


def test_the_recipe_is_one_undoable_checkpointed_write(tmp_path: Path):
    state, ctx, box = _rgb(tmp_path)
    _tool(box, "set_preprocessed_channel_properties")(channel_weights=[0, 1, 0])
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved is not None
    assert saved.appearance["preprocessed"]["stack"]["channel_weights"] == [0, 1, 0]
    assert len(box.job.undo_stack) == 1
    assert _tool(box, "undo")()["status"] == "ok"
    assert state.appearance == {}
    assert _tool(box, "redo")()["status"] == "ok"
    assert state.appearance["preprocessed"]["stack"]["channel_weights"] == [0, 1, 0]


def test_the_recipe_refuses_bad_settings_before_writing(tmp_path: Path):
    state, _, box = _rgb(tmp_path)
    tool = _tool(box, "set_preprocessed_channel_properties")
    assert tool(channel_weights=[1, 1])["error"] == "CHANNEL_COUNT_MISMATCH"
    assert tool(clahe_clip=-1)["error"] == "BAD_ARGS"
    assert tool(sections=["nope.png"])["error"] == "UNKNOWN_SLICE_IDS"
    assert state.appearance == {} and not box.job.undo_stack


def test_the_preprocessed_channel_is_what_look_draws_as_preprocessed(tmp_path: Path):
    state, ctx, box = _rgb(tmp_path)
    record = state.by_id("s0.png")
    raw = np.asarray(render_slice(ctx, record, long_edge=2048, frame=False)).copy()
    default = np.asarray(looks.preprocessed_image(ctx, state, record, long_edge=512)).copy()
    _tool(box, "set_preprocessed_channel_properties")(channel_weights=[0, 0, 1])
    ctx.render_cache.clear()
    # Geometry is measured on the default render, whatever the recipe.
    assert np.array_equal(raw, np.asarray(render_slice(ctx, record, long_edge=2048,
                                                       frame=False)))
    changed = np.asarray(looks.preprocessed_image(ctx, state, record, long_edge=512))
    assert not np.array_equal(default, changed)


def test_ants_steps_run_when_the_extra_is_installed(tmp_path: Path):
    pytest.importorskip("ants")
    state, _, box = _rgb(tmp_path)
    result = _tool(box, "set_preprocessed_channel_properties")(
        sections=["s0.png"], n4=True, denoise=True)
    assert result["status"] == "ok", result
    assert looks.preprocessed_settings(state, "s0.png")["n4"] is True


# --- grep_atlas_view ----------------------------------------------------------------------


def test_grep_atlas_view_highlights_regions_at_clamped_positions(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    view = _tool(box, "grep_atlas_view")
    result = view(["TH"], [0.1, 9.0])
    assert result["status"] == "ok" and result["regions"] == ["TH"]
    assert result["clamped"] == [{"asked_mm": 9.0, "used_mm": 0.25}]
    assert len(result["pictures"]) == 2 and len(result[TOOL_MEDIA_PARTS_KEY]) == 2
    assert "regions highlighted: TH" in result["pictures"][0]["caption"]
    assert [r.tool for r in box.job.views.records()] == ["grep_atlas_view"] * 2
    assert view(["XYZ"], [0.1])["error"] == "UNKNOWN_REGIONS"
    assert view(["TH"], [])["error"] == "BAD_ARGS"


# --- status -------------------------------------------------------------------------------


def test_status_is_the_table_and_what_the_run_lets_change(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    result = _tool(box, "status")()
    assert result["status"] == "ok"
    assert [row["id"] for row in result["rows"]] == ["s0.png", "s1.png", "s2.png"]
    assert result["rows"][0]["position_mm"] == 0.05
    assert result["cutting_angles_deg"] == {"pitch": 0.0, "yaw": 0.0}
    assert result["interval_breaks"] == []
    assert result["preprocessed_recipe"] == "default"
    assert result["channel_display"] == "gray auto"
    assert "background_running" not in result
    assert result["can_change"] == [
        "positions (the order follows them)", "cutting angles",
        "orientation and mirroring and in-plane transforms by hand",
        "in-plane transforms by elastix_affine", "damaged regions",
        "channel display and the preprocessed channel"]


def test_status_names_the_deformation_tools_of_a_nonlinear_run(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas, tasks=["nonlinear"],
                           nonlinear=NonlinearSpec(provider="none"))
    assert _tool(box, "status")()["can_change"] == [
        "damaged regions", "deformations (ants_syn)",
        "channel display and the preprocessed channel"]


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


# --- the job-folder tools -------------------------------------------------------------------


def test_the_job_folder_tools_read_the_pictures_index(tmp_path: Path, atlas):
    _, _, box = _synthetic(tmp_path, atlas)
    _tool(box, "look")("section")
    box.job.views.flush()  # the pictures are written in the background
    listed = _tool(box, "list_files")()
    assert listed["status"] == "ok" and "views.jsonl" in listed["text"]
    assert set(listed) == {"status", "text"}
    found = _tool(box, "search_files")("s1.png section")
    assert "views.jsonl:2:" in found["text"]
    read = _tool(box, "read_file")
    lines = read("views.jsonl", 0, 2)
    assert lines["text"].startswith("views.jsonl: lines 1-2 of 3")
    assert "continue with offset=2" in lines["text"]
    record = box.job.views.records()[0]
    picture = read(f"{record.path}/view.jpg")
    assert picture["picture"] == record.seq and "zoom with picture 1" in picture["text"]
    assert read("../outside.txt")["error"] == "BAD_PATH"
    assert _tool(box, "search_files")("")["error"] == "BAD_ARGS"


# --- what look draws -------------------------------------------------------------------------


def _pixels(result: dict[str, Any]) -> np.ndarray:
    image = np.asarray(result[TOOL_MEDIA_PARTS_KEY][0].convert("RGB")).astype(int)
    return image[image.shape[0] // 4: image.shape[0] // 2]  # tissue rows, above the caption


def test_one_raw_channel_in_gray_several_overlaid_in_colour(tmp_path: Path):
    state, _, box = _rgb(tmp_path)
    look = _tool(box, "look")
    green = look("section", sections=["s0.png"], channels=["green"])
    assert "raw green in gray auto" in green["pictures"][0]["caption"]
    pixels = _pixels(green)
    assert np.abs(pixels[..., 0] - pixels[..., 1]).max() <= 12  # grayscale
    both = look("section", sections=["s0.png"], channels=["red", "blue"])
    assert "raw red, blue auto" in both["pictures"][0]["caption"]
    mixed = _pixels(both)
    assert mixed[..., 0].max() > 200 and mixed[..., 2].max() > 40
    assert mixed[..., 1].max() < 40  # no green channel in a red + blue overlay
    assert "raw red, green, blue auto" in look("section", sections=["s0.png"])[
        "pictures"][0]["caption"]
    assert state.appearance == {}


def test_a_flat_channel_is_dimmed_in_an_overlay():
    from langslice.core.channels import fine_detail

    rng = np.random.default_rng(0)
    textured = rng.uniform(0.2, 1.0, (64, 64)).astype(np.float32)
    flat = np.full((64, 64), 0.9, dtype=np.float32)
    assert fine_detail(flat) < 0.01 < fine_detail(textured)


def test_atlas_layers_and_the_host_without_nissl(tmp_path: Path):
    _, _, box = _rgb(tmp_path)
    look = _tool(box, "look")

    def drawn(layers: list[str]) -> tuple[str, bytes]:
        result = look("atlas", positions_mm=[5.0], atlas_layers=layers)
        return result["pictures"][0]["caption"], result[TOOL_MEDIA_PARTS_KEY][0].tobytes()

    template, borders, both = drawn(["template"]), drawn(["borders"]), drawn(
        ["template", "borders"])
    assert template[0].endswith("layers template") and both[0].endswith("template + borders")
    assert len({template[1], borders[1], both[1]}) == 3
    # The template was once called ``ara``; the name is still read.
    assert drawn(["ara"]) == template
    refused = look("atlas", positions_mm=[5.0], atlas_layers=["nissl"])
    assert refused["error"] == "UNKNOWN_LAYER" and "template, borders" in refused["message"]
    under = look("overlay", sections=["s0.png"], atlas_layers=["template"], atlas_opacity=0.6)
    assert "atlas template (images at 0.6)" in under["pictures"][0]["caption"]


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


def test_the_channel_tools_are_in_every_run(tmp_path: Path):
    _, _, box = _rgb(tmp_path)
    assert {"set_channel_properties", "set_preprocessed_channel_properties"} <= set(box.names)
    with pytest.raises(ValueError, match="agent_preprocessing"):
        JobSpec(image_folder=str(tmp_path), agent_preprocessing="yes")  # type: ignore[arg-type]
