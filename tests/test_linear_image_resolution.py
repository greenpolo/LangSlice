"""image_resolution sizes what the agent is shown, never what is computed.

Each level is two long edges (``render.PICTURE_EDGES``): the opening images of
the opening strips' tiles and every later picture; "auto" adds a ``resolution``
argument to the picture tools. Nothing is upsampled past its source.
"""

from __future__ import annotations

import hashlib
import inspect
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from langslice.agent.engine import build_context, build_seed_message
from langslice.core.captions import CAPTION_PX, _caption_font, caption, wrap_caption
from langslice.core.opening import COLUMN_GAP, section_tile, strip_layout
from langslice.core.sections import PREVIEW_LONG_EDGE, render_slice, shown_section
from langslice.core.sheets import SHEET_MAX_LONG_EDGE, stack_sheet
from langslice.core.sizes import PICTURE_EDGES, opening_edge
from langslice.core.spec import JobSpec
from langslice.core.transform import calibrate
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.media import package_result, strip_edge
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from tests.linear_tool_helpers import legacy_view
from tests.test_linear_physical import TwoRegionAtlas

#: A 6 x 4.5 mm field at 2.5 um/px (2400 x 1800 px) with 4 x 3 mm of tissue:
#: large enough that every level's sizes are reached without upsampling.
_UM_PER_PX = 2.5


def _section(size: tuple[int, int] = (2400, 1800)) -> Image.Image:
    width, height = size
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[height // 6 : height * 5 // 6, width // 6 : width * 5 // 6] = 120
    arr[height // 3 : height * 5 // 9, width // 3 : width * 7 // 12, 1] = 220
    arr[height * 11 // 18 : height * 13 // 18, width * 5 // 8 : width * 3 // 4, 0] = 250
    return Image.fromarray(arr)


def _run(folder: Path, resolution: str, *, size: tuple[int, int] = (2400, 1800),
         count: int = 2, um_per_px: float = _UM_PER_PX,
         tasks: tuple[str, ...] = ("reorder", "position", "transform"),
         ) -> tuple[Any, Any, dict[str, Any], JobSpec]:
    folder.mkdir()
    for index in range(count):
        _section(size).rotate(7 * index, fillcolor=(0, 0, 0)).save(
            folder / f"s{index}.tif", dpi=(25400 / um_per_px,) * 2)
    spec = JobSpec(image_folder=str(folder), model="fake", preprocess="none",
                   image_resolution=resolution, tasks=list(tasks))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: TwoRegionAtlas())
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 0.2 + 0.1 * index
    box = build_tools(state, ctx, spec)
    # Picture options given the old way (keywords) go into `view`.
    return state, ctx, {tool.__name__: legacy_view(tool) for tool in box.tools}, spec


def _images(result: dict[str, Any]) -> list[Image.Image]:
    # What the ADK agent receives: the door's JPEG parts.
    return [Image.open(io.BytesIO(part.inline_data.data))
            for part in package_result(result)[TOOL_MEDIA_PARTS_KEY] if part.inline_data]


def _widths(result: dict[str, Any]) -> list[int]:
    # Widths: a caption adds a band above every picture; the tissue here is
    # wider than tall, so a picture's width is its long edge.
    return [image.width for image in _images(result)]


def _entry(**extra: Any) -> dict[str, Any]:
    return {"id": "s0.tif", "rotation_deg": 4.0, "scale_x": 1.1, "scale_y": 0.95,
            "translate_x_mm": 0.12, "translate_y_mm": -0.05, "pivot": "tissue", **extra}


def _seed_sections(state: Any, ctx: Any) -> list[Image.Image]:
    """The opening strip(s), and each section's tile as the strip draws it."""
    parts = build_seed_message(state, ctx).parts or []
    strips = [Image.open(io.BytesIO(p.inline_data.data)) for p in parts if p.inline_data]
    _count, tile = strip_layout(strip_edge(ctx), opening_edge(ctx))
    assert strips[0].width == len(state.slices) * tile + (len(state.slices) - 1) * COLUMN_GAP
    return [section_tile(ctx, state, record, tile) for record in state.in_order()]


# --- each level's two sizes ------------------------------------------------


@pytest.mark.parametrize("level", ["low", "medium", "high", "auto"])
def test_each_level_opens_at_one_size_and_shows_later_pictures_at_another(
    tmp_path: Path, level: str,
):
    opening, later = PICTURE_EDGES[level]
    state, ctx, tools, _spec = _run(tmp_path / level, level)

    # Each tile is the opening size, shrunk by at most a few pixels to fill the strip.
    assert all(opening - 8 <= image.width <= opening for image in _seed_sections(state, ctx))
    assert _widths(tools["view_slices"](["s0.tif"])) == [later]
    assert _widths(tools["view_placement"]([{"id": "s0.tif"}], mode="overlay")) == [later]
    assert _widths(tools["view_placement"]([{"id": "s0.tif"}], mode="side_by_side"))[0] == later
    assert _widths(tools["fit_affine"](["s0.tif"], "silhouette")) == [later]
    # Every panel of a two-panel picture is the later size.
    assert _widths(tools["adjust_transforms"]([_entry(mode="ab")])) == [later, later]


def test_the_levels_are_the_designed_numbers():
    assert PICTURE_EDGES == {
        "low": (256, 512), "medium": (384, 768), "high": (512, 1024), "auto": (256, 512),
    }


def test_a_small_snapshot_is_never_upsampled(tmp_path: Path):
    """A 300 x 225 px snapshot stays at most its own pixels at every size."""
    state, ctx, tools, _spec = _run(tmp_path / "small", "high", size=(300, 225),
                                    um_per_px=20.0)
    framed = render_slice(ctx, state.slices[0], long_edge=4096, frame=True)
    assert max(framed.size) < 300
    assert _seed_sections(state, ctx)[0].width == framed.width
    assert _widths(tools["view_slices"](["s0.tif"])) == [framed.width]
    (overlay,) = _images(tools["view_placement"]([{"id": "s0.tif"}], mode="overlay"))
    assert overlay.width < PICTURE_EDGES["high"][1]


def test_a_zoom_magnifies_the_section_up_to_the_picture_size(tmp_path: Path):
    """A zoomed physical picture is drawn from a larger render, not upsampled."""
    _state, _ctx, tools, _spec = _run(tmp_path / "zoom", "low")
    (whole,) = _images(tools["view_placement"]([{"id": "s0.tif"}], mode="overlay"))
    (zoomed,) = _images(tools["view_placement"](
        [{"id": "s0.tif"}], mode="overlay", zoom=[0.25, 0.25, 0.75, 0.75]))
    assert whole.width == zoomed.width == PICTURE_EDGES["low"][1]


def test_the_contact_sheet_tiles_follow_the_level_and_stay_bounded(tmp_path: Path):
    state, ctx, _tools, _spec = _run(tmp_path / "few", "medium", count=2)
    few = stack_sheet(state, ctx)
    # Two tiles at the medium opening size plus the gap between them.
    assert few.width == 2 * PICTURE_EDGES["medium"][0] + 6
    state, ctx, _tools, _spec = _run(tmp_path / "many", "high", count=24,
                                     size=(1200, 900), um_per_px=5.0)
    many = stack_sheet(state, ctx)
    assert max(many.size) <= SHEET_MAX_LONG_EDGE
    assert many.width > 0.8 * SHEET_MAX_LONG_EDGE  # shrunk to fit, not to a stamp


# --- auto -------------------------------------------------------------------


def _has_resolution(tool: Any) -> bool:
    """Whether the tool's `view` schema carries the `resolution` key."""
    parameters = inspect.signature(tool, eval_str=True).parameters
    if "view" not in parameters:
        return "resolution" in parameters
    return "resolution" in parameters["view"].annotation.__annotations__


@pytest.mark.parametrize("level", ["low", "medium", "high", "auto"])
def test_resolution_exists_only_at_auto(tmp_path: Path, level: str):
    from google.adk.tools import FunctionTool

    from langslice.providers.openai_oauth import _json_schema_dict

    state, ctx, _tools, spec = _run(
        tmp_path / level, level, tasks=("reorder", "position", "transform", "nonlinear"))
    tools = {tool.__name__: tool for tool in build_tools(state, ctx, spec).tools}
    pictured = ["view_slices", "view_atlas", "view_placement", "view_stack", "set_positions",
                "orient_slices", "fit_affine", "fit_deformable", "adjust_transforms"]
    auto = level == "auto"
    for name in pictured:
        assert _has_resolution(tools[name]) is auto, name
        # What the model is sent: the view object's keys, written out.
        schema = _json_schema_dict(FunctionTool(tools[name])._get_declaration())
        view = schema["properties"]["view"]
        assert ("resolution" in view["properties"]) is auto, name
        assert {"mode", "channels", "atlas_channels", "zoom"} <= set(view["properties"])
    assert not _has_resolution(tools["status"])


def test_the_job_statement_names_resolution_only_at_auto(tmp_path: Path):
    from langslice.agent.prompt import display_lines

    names = ["view_slices", "fit_deformable"]
    assert not any("resolution" in line for line in display_lines(names))
    (line,) = [line for line in display_lines(names, resolution=2048) if "resolution" in line]
    assert "128 to 2048" in line and "512" in line and "`view` also takes `resolution`" in line


def test_auto_sizes_each_call_and_clamps(tmp_path: Path):
    _state, _ctx, tools, _spec = _run(tmp_path / "auto", "auto")
    plain = tools["view_slices"](["s0.tif"])
    assert _widths(plain) == [512] and plain["view"]["resolution"] == 512
    chosen = tools["view_slices"](["s0.tif"], resolution=1200)
    assert _widths(chosen) == [1200] and "resolution_note" not in chosen["view"]
    # The cap is the driver model's largest image (Nash 2026-10-03): the
    # OpenAI lanes' 2048 px here, the run's model being no other lane.
    big = tools["view_placement"]([{"id": "s0.tif"}], mode="overlay", resolution=5000)
    assert _widths(big) == [2048]
    assert "2048" in big["view"]["resolution_note"]
    small = tools["view_slices"](["s0.tif"], resolution=20)
    assert _widths(small) == [128] and "128" in small["view"]["resolution_note"]
    refused = tools["view_slices"](["s0.tif"], resolution="large")
    assert refused["error"] == "BAD_RESOLUTION"
    per_entry = tools["adjust_transforms"]([_entry(resolution=900)])
    assert _widths(per_entry) == [900]
    sheet = tools["view_stack"](resolution=200)
    assert _images(sheet)[0].width == 2 * 200 + 6


def test_resolution_is_ignored_and_invisible_below_auto(tmp_path: Path):
    _state, _ctx, tools, _spec = _run(tmp_path / "high", "high")
    result = tools["view_slices"](["s0.tif"])
    assert "resolution" not in result["view"]
    # Not in the schema, and refused with the reason when sent anyway.
    forced = tools["view_slices"](["s0.tif"], resolution=1200)
    assert forced["error"] == "UNKNOWN_ARGUMENTS"
    assert "fixed the picture size" in forced["message"]


# --- what is computed does not move ------------------------------------------


#: fit_affine and adjust_transforms on this stack, computed with the code
#: BEFORE the picture-size change (2026-10-01): the stored numbers must match.
_PINNED_S0_FIT = [-0.24853608803995778, 9.56406142034393e-16, 0.6245949944548352,
                  -2.2479381284208716e-15, -0.3341968926388023, 0.6672419978191492]
_PINNED_S1_ADJUST = [1.0973204552858067, 0.04970148754268928, -0.053191461442875475,
                     -0.10230949482471711, 0.947685847746833, 0.06586465442425882]


def _fixed_stack(folder: Path, level: str) -> tuple[Any, dict[str, Any]]:
    """The stack the pinned numbers were computed on (two 2400 x 1800 sections)."""
    folder.mkdir()
    for index in range(2):
        arr = np.zeros((1800, 2400, 3), dtype=np.uint8)
        arr[300:1500, 400:2000] = 120
        arr[600:1000, 800:1400, 1] = 220
        arr[1100:1300, 1500:1800, 0] = 250
        Image.fromarray(arr).rotate(7 * index, fillcolor=(0, 0, 0)).save(
            folder / f"s{index}.tif", dpi=(25400 / 2.5,) * 2)
    spec = JobSpec(image_folder=str(folder), model="fake", preprocess="none",
                   image_resolution=level)
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: TwoRegionAtlas())
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 0.2 + 0.1 * index
    return state, {tool.__name__: legacy_view(tool)
                   for tool in build_tools(state, ctx, spec).tools}


@pytest.mark.parametrize("level", ["low", "high", "auto"])
def test_fits_and_written_transforms_are_the_numbers_from_before(tmp_path: Path, level: str):
    state, tools = _fixed_stack(tmp_path / level, level)
    fit = tools["fit_affine"]([], "silhouette", **({"resolution": 1400} if level == "auto" else {}))
    assert [row["iou"] for row in fit["results"]] == [1.0, 1.0]
    assert state.by_id("s0.tif").transform["params"] == _PINNED_S0_FIT
    tools["adjust_transforms"]([{
        "id": "s1.tif", "rotation_deg": 4.0, "scale_x": 1.1, "scale_y": 0.95,
        "translate_x_mm": 0.12, "translate_y_mm": -0.05, "pivot": "tissue",
        # A left-out shear keeps the fit's (2026-10-03); the pin is shear-free.
        "shear": 0.0}])
    assert state.by_id("s1.tif").transform["params"] == _PINNED_S1_ADJUST


@pytest.mark.parametrize("level", ["medium", "high", "auto"])
def test_pictures_change_while_fits_and_transforms_stay(tmp_path: Path, level: str):
    low_state, low_ctx, low, _ = _run(tmp_path / "low", "low")
    big_state, big_ctx, big, _ = _run(tmp_path / level, level)
    extra = {"resolution": 1100} if level == "auto" else {}
    low_fit = low["fit_affine"](["s0.tif"], "silhouette")
    big_fit = big["fit_affine"](["s0.tif"], "silhouette", **extra)
    low_adjust = low["adjust_transforms"]([_entry(mode="ab")])
    big_adjust = big["adjust_transforms"]([_entry(mode="ab", **extra)])
    for record_low, record_big in zip(low_state.slices, big_state.slices, strict=True):
        section_low = render_slice(low_ctx, record_low, long_edge=PREVIEW_LONG_EDGE)
        section_big = render_slice(big_ctx, record_big, long_edge=PREVIEW_LONG_EDGE)
        assert section_low.size == section_big.size
        assert calibrate(low_state, low_ctx, record_low, section_low) == calibrate(
            big_state, big_ctx, record_big, section_big)
    assert low_fit["results"] == big_fit["results"]
    assert low_state.slices[0].transform == big_state.slices[0].transform
    assert low_adjust["results"][0]["physical"] == big_adjust["results"][0]["physical"]


def _digest(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def test_the_image_model_and_deformable_fit_inputs_do_not_depend_on_the_level(tmp_path: Path):
    from langslice.core import deformation
    from langslice.core.nonlinear.registration_handoff import prepare_linear_registration

    seen: dict[str, tuple[str, str, Any]] = {}
    for level in ("low", "high", "auto"):
        state, ctx, tools, _ = _run(tmp_path / level, level)
        tools["fit_affine"](["s0.tif"], "silhouette")
        prepared = prepare_linear_registration(state, ctx, "s0.tif")
        grid = deformation.fit_grid(state, ctx, state.by_id("s0.tif"))
        seen[level] = (_digest(prepared.image), _digest(grid.image), prepared.metadata)
    assert seen["low"] == seen["high"] == seen["auto"]


def test_low_draws_from_the_working_render_itself(tmp_path: Path):
    state, ctx, _tools, _spec = _run(tmp_path / "low", "low")
    record = state.slices[0]
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    shown, um, factors = shown_section(ctx, record, section, 9.0)
    assert shown is section and um == 9.0 and factors == (1.0, 1.0)


def test_a_larger_picture_shows_the_same_map(tmp_path: Path, monkeypatch):
    """Drawn larger, the adjusted section lands where the small picture puts it."""
    from langslice.core import canvas, placement

    drawn: list[np.ndarray] = []

    def capture(*args, **kwargs):
        frames: list[dict[str, Any]] = []
        images, iou = canvas.physical_views(*args, **kwargs, frames=frames)
        x0, y0, x1, y1 = frames[0]["content_box"]
        drawn.append(np.asarray(images[0].convert("L"))[y0:y1, x0:x1])
        return images, iou

    monkeypatch.setattr(placement, "physical_views", capture)
    _low_state, _low_ctx, low, _ = _run(tmp_path / "low", "low")
    _big_state, _big_ctx, big, _ = _run(tmp_path / "high", "high")
    entry = _entry(mode="section", pivot=[0.3, 0.6], rotation_deg=12.0)
    low["adjust_transforms"]([entry])
    big["adjust_transforms"]([entry])

    def tissue(body: np.ndarray) -> tuple[float, float, float]:
        ys, xs = np.nonzero(body > 60)
        return xs.mean() / body.shape[1], ys.mean() / body.shape[0], len(xs) / body.size

    small, large = tissue(drawn[0]), tissue(drawn[1])
    assert drawn[1].shape[1] / drawn[0].shape[1] == pytest.approx(2.0, rel=0.02)
    assert small == pytest.approx(large, abs=0.005)


# --- captions ------------------------------------------------------------------


def test_a_long_caption_wraps_instead_of_running_off_a_small_picture():
    font = _caption_font()
    text = "12: M04_D_08_Overlay.tif  9.00 mm (+0.40 to next)  [damaged: dorsal cortex torn]"
    picture = Image.new("RGB", (160, 120), (40, 40, 40))
    labelled = caption(picture, text)
    wrapped = wrap_caption(text, font, picture.width - 6)
    lines = wrapped.split("\n")
    assert len(lines) > 2
    assert all(font.getlength(line) <= picture.width - 6 for line in lines)
    # Nothing is dropped: the words come back in order.
    assert " ".join(wrapped.split()) == " ".join(text.split())
    # The band grows to hold every line; the picture keeps its width.
    assert labelled.width == picture.width
    assert labelled.height - picture.height >= len(lines) * CAPTION_PX
    # A word wider than the picture breaks between characters.
    assert all(font.getlength(line) <= 30 for line in
               wrap_caption("M04_D_08_Overlay.tif", font, 30).split("\n"))


def test_a_caption_that_fits_is_drawn_as_before():
    font = _caption_font()
    picture = Image.new("RGB", (512, 80), (0, 0, 0))
    text = "3: s0.tif  [flipped]\nsecond line"
    assert wrap_caption(text, font, 506) == text
    labelled = caption(picture, text)
    # The old caption: the text drawn as given, band sized to its box.
    probe = ImageDraw.Draw(picture.copy())
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    band = int(bottom - top + 6)
    expected = Image.new("RGB", (512, 80 + band), (0, 0, 0))
    expected.paste(picture, (0, band))
    ImageDraw.Draw(expected).text((3 - left, 3 - top), text, fill=(255, 255, 255), font=font)
    assert np.array_equal(np.asarray(labelled), np.asarray(expected))


def test_the_cap_is_the_driver_models_own(tmp_path: Path):
    """The door passes the driver model's largest image: Claude's for the MCP
    host, the model lane's for the ADK agent."""
    from langslice.agent.prompt import build_job_statement
    from langslice.core.opening import CLAUDE_MAX_VIEW_EDGE, OPENAI_MAX_IMAGE_EDGE

    state, ctx, _tools, spec = _run(tmp_path / "auto", "auto")
    claude = build_tools(state, ctx, spec, max_view_edge=CLAUDE_MAX_VIEW_EDGE)
    assert claude.max_view_edge == CLAUDE_MAX_VIEW_EDGE == 2000
    tool = next(tool for tool in claude.tools if tool.__name__ == "view_placement")
    big = tool([{"id": "s0.tif"}], view={"mode": "overlay", "resolution": 5000})
    assert _widths(big) == [2000] and "2000" in big["view"]["resolution_note"]
    ctx.model = "openai-oauth/gpt-6-astra"
    assert build_tools(state, ctx, spec).max_view_edge == OPENAI_MAX_IMAGE_EDGE
    text = build_job_statement(spec, state, tool_names=claude.names, species="mouse",
                               pos_lo=0.0, pos_hi=1.0, axis_ends=("anterior", "posterior"),
                               max_resolution=claude.max_view_edge)
    assert "128 to 2000" in text
