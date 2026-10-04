"""The view controls of the interactive alignment screen.

The alignment loop is a vision-action loop, so a view control that lies is
worse than one that does not exist: a zoom that does not magnify, a
checkerboard missing one of its two sources, or a scale bar that keeps its old
length after a crop would all be read as anatomy. These pin what each mode
actually puts on the screen, in pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_linear_physical import TwoRegionAtlas, _ctx

from langslice.linear.render import (
    canvas_geometry,
    physical_views,
    scale_bar_px,
)
from tests.linear_tool_helpers import single_adjust

_IDENTITY = {
    "rotation_deg": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "translate_x_mm": 0.0,
    "translate_y_mm": 0.0,
}
UM_PER_PX = 10.0


def _section(value: int = 120, size: tuple[int, int] = (300, 300)) -> Image.Image:
    """A dark field with a square of "tissue" of a known brightness."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[size[1] // 2 - 40 : size[1] // 2 + 40, size[0] // 2 - 40 : size[0] // 2 + 40] = value
    return Image.fromarray(arr, mode="RGB")


def _views(value: int = 120, **kwargs) -> tuple[list[np.ndarray], float]:
    images, iou = physical_views(
        _section(value),
        UM_PER_PX,
        TwoRegionAtlas(),
        0.2,
        "coronal",
        0.0,
        0.0,
        _IDENTITY,
        # Canvas pixels, one to one: these tests pin what each view draws.
        # The model-facing screen size is the pixel-size rule's business
        # (test_the_model_screen_is_sized_by_the_atlas_resolution).
        long_edge=None,
        **kwargs,
    )
    return [np.asarray(image.convert("RGB")) for image in images], iou


def test_the_screen_is_the_long_edge_and_never_an_upsample():
    """Each panel's long edge is *long_edge*, or the crop's own pixels when it
    has fewer: a small canvas stays small, and a zoom on it is a crop at the
    same scale (magnification needs a larger render, the toolbox's job)."""
    geometry = _geometry()
    canvas_long = max(geometry.size)
    (shrunk,) = _bodies(long_edge=128)
    assert max(shrunk.shape[:2]) == 128
    (whole,) = _bodies(long_edge=4 * canvas_long)
    assert whole.shape[1] == geometry.size[0], "never upsampled past the canvas"
    (zoomed,) = _bodies(long_edge=4 * canvas_long, zoom=[0.3, 0.3, 0.7, 0.7])
    assert zoomed.shape[1] == pytest.approx(0.4 * geometry.size[0], abs=2)


def _geometry():
    return canvas_geometry((300, 300), UM_PER_PX, TwoRegionAtlas(), 0.2, "coronal")


def _bar_length(rgb: np.ndarray) -> int:
    """The longest run of bar-bright pixels in the bottom strip."""
    strip = rgb[-40:, :]
    bright = (strip >= 230).all(axis=2)
    return int(bright.sum(axis=1).max())


# --- zoom ----------------------------------------------------------------


def test_zoom_magnifies_and_the_bar_is_still_one_millimetre():
    geometry = _geometry()
    box = [0.3, 0.3, 0.7, 0.7]
    (whole,), _ = _views()
    (zoomed,), _ = _views(zoom=box)

    # Canvas pixels one to one: the zoom is the crop, at its own size.
    assert zoomed.shape[1] == round(0.7 * geometry.size[0]) - round(0.3 * geometry.size[0])
    tissue = lambda rgb: float(((rgb >= 100) & (rgb <= 140)).all(axis=2).mean())  # noqa: E731
    assert tissue(zoomed) > 2.0 * tissue(whole), "the crop did not magnify the tissue"

    # The bar is drawn after the crop, at the magnified micrometres per pixel.
    crop_width = round(0.7 * geometry.size[0]) - round(0.3 * geometry.size[0])
    for image, width in ((whole, geometry.size[0]), (zoomed, crop_width)):
        factor = image.shape[1] / float(width)
        assert _bar_length(image) == pytest.approx(
            scale_bar_px(geometry.um_per_px / factor) + 1, abs=3
        )


def test_an_empty_zoom_is_the_whole_canvas():
    (plain,), _ = _views()
    (empty,), _ = _views(zoom=[])
    assert np.array_equal(plain, empty)


# --- the modes -----------------------------------------------------------


def test_side_by_side_returns_two_images_of_equal_size():
    images, _ = _views(mode="side_by_side")
    assert len(images) == 2
    assert images[0].shape == images[1].shape
    assert not np.array_equal(images[0], images[1])
    # Same crop, same scale: the 1 mm bar is the same length on both.
    assert _bar_length(images[0]) == _bar_length(images[1])


def test_checkerboard_carries_both_sources():
    """The section's tissue is 120-bright; this atlas's template renders white."""
    (board,), _ = _views(mode="checkerboard")
    geometry = _geometry()
    band = board.shape[0] - geometry.size[1]  # the caption above the canvas
    cx, cy = geometry.size[0] // 2, band + geometry.size[1] // 2
    middle = board[cy - 90 : cy + 90, cx - 90 : cx + 90]
    section_px = ((middle >= 110) & (middle <= 130)).all(axis=2).sum()
    template_px = (middle >= 250).all(axis=2).sum()
    assert section_px > 1000 and template_px > 1000


def test_outlines_mode_is_black_outside_the_lines():
    (lines,), _ = _views(mode="outlines")
    # No tissue, no template: the centre of the anatomy is bare canvas.
    geometry = _geometry()
    cx = geometry.size[0] // 2
    cy = lines.shape[0] - geometry.size[1] + geometry.size[1] // 2
    centre = lines[cy - 5 : cy + 5, cx - 5 : cx + 5]
    assert centre.max() == 0
    assert float((lines > 40).any(axis=2).mean()) < 0.05, "more than lines on screen"
    # Yellow atlas lines and a neutral grey tissue silhouette.
    values = set(np.unique(lines[lines > 40]))
    assert max(values) >= 200 and any(90 < v < 190 for v in values)


def test_the_atlas_opacity_is_a_dial_not_a_switch():
    # Inside the anatomy, outside the tissue: the atlas square is 1 mm (100 px)
    # and the tissue 80 px, both centred on the canvas, so the 10 px band just
    # inside the anatomy's left edge is template-only.
    geometry = _geometry()
    cx, cy = geometry.size[0] // 2, geometry.size[1] // 2
    box = (slice(cy - 30, cy + 30), slice(cx - 49, cx - 42))
    (none,), _ = _views()
    (half,), _ = _views(atlas_opacity=0.5)
    (full,), _ = _views(atlas_opacity=1.0)
    assert none[box].mean() < half[box].mean() < full[box].mean()


# --- the outline layers --------------------------------------------------


def _line_pixels(rgb: np.ndarray) -> int:
    """Hairline pixels in the picture, above the bar, below the caption.

    At canvas scale a hairline is one anti-aliased pixel wide, so the
    threshold sits below full brightness; the tissue (120) stays under it."""
    pixels = rgb[60:-40].astype(np.int16)
    return int(((pixels[..., 0] - pixels[..., 2] > 80)
                & (pixels[..., 1] - pixels[..., 2] > 80)).sum())


def test_the_outline_layer_picks_which_atlas_lines_are_drawn():
    """This atlas has two families: the 1 mm square and a region inside it."""
    (every,), _ = _views()
    (outer,), _ = _views(outlines="outer")
    (bare,), _ = _views(outlines="none")

    assert _line_pixels(bare) == 0, "outlines='none' still drew lines"
    assert _line_pixels(outer) > 300, "the root contour is missing"
    # The inner family's contour is what "outer" drops, and it is not small.
    assert _line_pixels(every) > _line_pixels(outer) + 200
    # Lines are the only thing that changes: they can cover tissue, never
    # uncover it, so the bare view shows at least as much of the section.
    tissue = lambda rgb: int(((rgb >= 110) & (rgb <= 130)).all(axis=2).sum())  # noqa: E731
    assert tissue(bare) >= tissue(every) > 0
    assert tissue(bare) - tissue(every) < 0.2 * tissue(bare)  # hairlines at canvas scale


# --- the tools, in the main toolbox --------------------------------------


def _tools(tmp_path: Path):
    """The transform tools of a one-section run, on a 10 um/px file."""
    from langslice.linear.toolbox import build_tools

    _section(200, (512, 512)).save(tmp_path / "s.tif", dpi=(2540.0, 2540.0))  # 10 um/px
    ctx, state = _ctx(tmp_path)
    record = state.slices[0]
    record.position_mm = 0.2
    box = build_tools(state, ctx, ctx.spec)
    return {tool.__name__: tool for tool in box.tools}, box, state


def test_the_adjust_payload_is_concise_while_local_history_stays_complete(tmp_path: Path):
    tools, box, _state = _tools(tmp_path)
    preview = single_adjust(tools["adjust_transforms"])

    first = preview("s.tif", 0.0, 1.0, 1.0, 0.25, 0.0)
    assert first["status"] == "ok"
    assert first["physical"]["translate_x_mm"] == 0.25
    assert first["physical"]["pivot"] == [0.5, 0.5]
    # No overlap number: silhouette overlap against the whole atlas plate
    # rewarded inflating a damaged remnant to fill it (luna, D_08, 2026-09-06:
    # 0.29 -> 0.51 at scale 1.35), and this loop exists for damaged sections.
    assert "silhouette_iou" not in first
    assert {key: first["view"][key] for key in ("mode", "zoom", "outlines")} == {
        "mode": "overlay",
        "zoom": [0.0, 0.0, 1.0, 1.0],
        "outlines": "all",
    }
    # The normalized matrix, derived pixel shift/decomposition, generic status
    # row and accumulated history stay host-side instead of growing each reply.
    assert not {
        "matrix_params", "translate_px", "decomposition", "changed", "history"
    } & set(first)

    preview("s.tif", 2.0, 1.0, 1.0, 0.0, 0.0)
    # The history is per section, kept on the toolbox for the whole run.
    assert len(box.transform_history["s.tif"]) == 2
    assert [entry["rotation_deg"] for entry in box.transform_history["s.tif"]] == [0.0, 2.0]


def test_the_adjust_tool_takes_the_view_controls(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY

    preview = single_adjust(_tools(tmp_path)[0]["adjust_transforms"])

    pair = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "side_by_side")
    assert len(pair[TOOL_MEDIA_PARTS_KEY]) == 2
    assert pair["view"]["mode"] == "side_by_side"

    zoomed = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [0.3, 0.3, 0.7, 0.7], 0.5,
                     atlas_channels=["ara", "borders"])
    assert len(zoomed[TOOL_MEDIA_PARTS_KEY]) == 1
    assert zoomed["view"]["zoom"] == [0.3, 0.3, 0.7, 0.7]

    assert preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "flicker")["error"] == "BAD_MODE"
    assert preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [0.3, 0.7])["error"] == "BAD_ZOOM"
    assert preview("ghost.tif", 0.0, 1.0, 1.0, 0.0, 0.0)["error"] == "UNKNOWN_SLICE_IDS"


def test_ab_returns_the_candidate_and_what_is_stored(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY

    tools, _box, _state = _tools(tmp_path)
    against_identity = single_adjust(tools["adjust_transforms"])(
        "s.tif", 6.0, 1.0, 1.0, 0.0, 0.0, "ab"
    )
    assert len(against_identity[TOOL_MEDIA_PARTS_KEY]) == 2
    assert against_identity["view"]["mode"] == "ab"
    assert "identity" in against_identity["description"]
    assert against_identity["ab_reference"]["source"] == "identity"

    single_adjust(tools["adjust_transforms"])("s.tif", 3.0, 1.0, 1.0, 0.0, 0.0)
    against_stored = single_adjust(tools["adjust_transforms"])(
        "s.tif", 6.0, 1.0, 1.0, 0.0, 0.0, "ab"
    )
    assert len(against_stored[TOOL_MEDIA_PARTS_KEY]) == 2
    assert "before" in against_stored["description"]
    # The B side is what the section carried before this call, which is the
    # before/after view; the call itself wrote the A side.
    assert against_stored["ab_reference"]["params"]["rotation_deg"] == 3.0
    assert _state.slices[0].transform["physical"]["rotation_deg"] == 6.0


def test_view_placement_draws_the_section_on_each_atlas_position(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY

    tools, _box, state = _tools(tmp_path)
    compare = tools["view_placement"]

    current = compare([{"id": "s.tif"}])
    assert current["status"] == "ok"
    assert current["compared"][0]["position_mm"] == 0.2
    assert current["compared"][0]["current_position_mm"] == 0.2
    assert len(current[TOOL_MEDIA_PARTS_KEY]) == 1  # side_by_side: one stitched image

    stepped = compare(
        [{"id": "s.tif", "positions_mm": [0.1, 0.2, 0.3]}],
        view={"mode": "overlay", "atlas_channels": ["ara", "borders"], "atlas_opacity": 0.3},
    )
    assert [row["position_mm"] for row in stepped["compared"]] == [0.1, 0.2, 0.3]
    assert len(stepped[TOOL_MEDIA_PARTS_KEY]) == 3
    assert stepped["render_failed"] == []
    # A look, not a write.
    assert state.slices[0].position_mm == 0.2

    # A batch: several sections, several candidates each, capped at 4 pairs
    # (one image each: the most any call returns).
    many = compare(
        [{"id": "s.tif", "positions_mm": [0.1] * 3}, {"id": "0", "positions_mm": [0.3] * 3}],
        view={"mode": "overlay"},
    )
    assert len(many["compared"]) == 4 and many["truncated"] and many["dropped_pairs"] == 2
    assert len(many[TOOL_MEDIA_PARTS_KEY]) == 4

    assert compare([{"id": "s.tif"}], view={"mode": "flicker"})["error"] == "BAD_MODE"
    assert compare([{"id": "ghost.tif"}])["error"] == "UNKNOWN_SLICE_IDS"
    state.slices[0].position_mm = None
    assert compare([{"id": "s.tif"}])["error"] == "NO_POSITION"


# --- the pivot -----------------------------------------------------------


def test_a_tissue_pivot_turns_the_section_about_its_own_centroid(tmp_path: Path):
    """A rotation about the tissue centroid leaves that centroid in place."""
    from langslice.image_prep import foreground_mask
    from langslice.linear.render import canvas_geometry, physical_views, pivot_on_canvas

    section = _section(200, (300, 200))  # tissue square well off the canvas centre
    arr = np.asarray(section).copy()
    arr[:] = 0
    arr[20:100, 30:110] = 200
    section = Image.fromarray(arr, mode="RGB")
    geometry = canvas_geometry(section.size, UM_PER_PX, TwoRegionAtlas(), 0.2, "coronal")
    pivot = pivot_on_canvas("tissue", section, geometry)
    assert pivot is not None and foreground_mask(section) is not None

    def centroid(images) -> tuple[float, float]:
        rgb = np.asarray(images[0].convert("RGB"))
        ys, xs = np.nonzero((rgb > 150).all(axis=2))
        return float(xs.mean()), float(ys.mean())

    turned = {"rotation_deg": 40.0, "scale_x": 1.0, "scale_y": 1.0,
              "translate_x_mm": 0.0, "translate_y_mm": 0.0}
    still, _ = physical_views(
        section, UM_PER_PX, TwoRegionAtlas(), 0.2, "coronal", 0.0, 0.0, _IDENTITY,
        mode="section", long_edge=None,
    )
    about_tissue, _ = physical_views(
        section, UM_PER_PX, TwoRegionAtlas(), 0.2, "coronal", 0.0, 0.0, turned,
        mode="section", pivot=pivot, long_edge=None,
    )
    about_canvas, _ = physical_views(
        section, UM_PER_PX, TwoRegionAtlas(), 0.2, "coronal", 0.0, 0.0, turned,
        mode="section", long_edge=None,
    )
    x0, y0 = centroid(still)
    xt, yt = centroid(about_tissue)
    xc, yc = centroid(about_canvas)
    assert (xt, yt) == pytest.approx((x0, y0), abs=2.0)  # the pivot held it
    assert abs(xc - x0) + abs(yc - y0) > 20.0  # the canvas centre swung it away


def test_the_pivot_rides_into_the_six_numbers_and_the_payload(tmp_path: Path):
    tools, _box, state = _tools(tmp_path)
    preview = single_adjust(tools["adjust_transforms"])

    centred = preview("s.tif", 10.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas")
    corner = preview("s.tif", 10.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, [0.25, 0.75])
    assert corner["physical"]["pivot"] == [0.25, 0.75]
    assert preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "middle")[
        "error"
    ] == "BAD_PIVOT"

    single_adjust(tools["adjust_transforms"])(
        "s.tif", 10.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, [0.25, 0.75]
    )
    stored = state.slices[0].transform
    assert stored["physical"]["pivot"] == [0.25, 0.75]
    # Same rotation, different pivot: the same map only up to a translation,
    # which is exactly what the six normalized numbers must carry.
    assert stored["params"][:2] == pytest.approx(
        [np.cos(np.radians(10.0)), np.sin(np.radians(10.0)) * 512 / 512], abs=1e-6
    )
    single_adjust(tools["adjust_transforms"])(
        "s.tif", 10.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas"
    )
    assert state.slices[0].transform["params"][2] != stored["params"][2]
    assert centred["physical"]["rotation_deg"] == pytest.approx(
        corner["physical"]["rotation_deg"]
    )


def _bodies(**kwargs) -> list[np.ndarray]:
    """Each panel below its caption band (the band's height varies with wrapping)."""
    frames: list[dict] = []
    long_edge = kwargs.pop("long_edge", None)
    images, _iou = physical_views(
        _section(), UM_PER_PX, TwoRegionAtlas(), 0.2, "coronal", 0.0, 0.0, _IDENTITY,
        long_edge=long_edge, frames=frames, **kwargs,
    )
    return [np.asarray(image.convert("RGB"))[frame["content_box"][1]:]
            for image, frame in zip(images, frames, strict=True)]


def test_clean_section_and_template_views_carry_no_outlines():
    (section_only,) = _bodies(mode="section")
    (template_only,) = _bodies(mode="template")
    (overlaid,) = _bodies(mode="overlay")
    pair = _bodies(mode="side_by_side")

    # The synthetic tissue is 120 grey; anti-aliased hairlines are far brighter.
    # Picture area only: below the caption band, above the scale bar.
    assert overlaid[:-40].max() >= 200
    assert section_only[:-40].max() < 200
    # The template alone is the side-by-side's second panel minus its lines.
    assert template_only.shape == pair[1].shape
    assert not np.array_equal(template_only[:-40], pair[1][:-40])


def test_the_adjust_tool_takes_the_outline_layer(tmp_path: Path):
    preview = single_adjust(_tools(tmp_path)[0]["adjust_transforms"])

    plain = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0)
    assert plain["view"]["outlines"] == "all"

    outer = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas", "outer")
    assert outer["view"]["outlines"] == "outer"
    assert "OUTER boundary" in outer["description"]

    bare = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas", "none")
    assert "No atlas outlines" in bare["description"]

    bad = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas", "midline")
    assert bad["error"] == "BAD_VIEW"


@pytest.mark.parametrize("color", ["cyan", "#00ffff"])
def test_border_color_changes_atlas_lines_without_changing_tissue_or_overlap(color):
    (yellow,), original_iou = _views()
    (cyan,), recolored_iou = _views(border_color=color)
    assert original_iou == recolored_iou
    # Both images carry the same tissue, scale bar and geometry. Only the
    # atlas lines change; their RGB channels swap red and blue for cyan.
    assert np.array_equal(yellow[..., 1], cyan[..., 1])
    assert np.array_equal(yellow[..., 0], cyan[..., 2])
    assert np.array_equal(yellow[..., 2], cyan[..., 0])
    assert _line_pixels(yellow) > 300
    assert _line_pixels(cyan) == 0


def test_border_thickness_is_measured_after_zoom_and_resize():
    from langslice.linear.render import _draw_polys

    square = np.asarray([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=float)
    widths = {}
    for thickness in (1, 4):
        widths[thickness] = []
        # Equivalent crops/scales change the square's on-screen size, but
        # the straight vertical edge should keep the requested pixel width.
        for factor, origin in ((1.0, (0, 0)), (0.5, (0, 0)), (2.0, (10, 10))):
            screen = np.zeros((200, 200, 3), dtype=np.uint8)
            _draw_polys(screen, [square], (255, 255, 0), factor=factor,
                        origin=origin, thickness=thickness)
            y = round((50 - origin[1]) * factor)
            x = round((20 - origin[0]) * factor)
            widths[thickness].append(int((screen[y, x-6:x+7, 0] > 127).sum()))
        assert len(set(widths[thickness])) == 1
    assert widths[4][0] > widths[1][0]


@pytest.mark.parametrize("style", [
    {"border_color": "not-a-color"}, {"border_color": "#abc"},
    {"border_thickness": 0}, {"border_thickness": 9},
    {"border_thickness": float("nan")}, {"border_thickness": True},
])
def test_invalid_border_style_does_not_write_a_transform(tmp_path: Path, style):
    tools, box, state = _tools(tmp_path)
    before = state.to_dict()
    response = single_adjust(tools["adjust_transforms"])("s.tif", 5.0, 1.0, 1.0, 0.0, 0.0, **style)
    assert response["error"] == "INVALID_BORDER_STYLE"
    assert state.to_dict() == before
    assert not box.job.undo_stack


def test_batch_border_style_changes_only_the_render(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY

    tools, box, state = _tools(tmp_path)
    entry = {"id": "s.tif", **_IDENTITY, "rotation_deg": 5.0}
    first = tools["adjust_transforms"]([entry])
    before = state.to_dict()
    undo_depth = len(box.job.undo_stack)
    second = tools["adjust_transforms"](
        [entry], view={"border_color": "cyan", "border_thickness": 3},
    )
    assert first["status"] == second["status"] == "ok"
    assert state.to_dict() == before
    assert len(box.job.undo_stack) == undo_depth
    first_image = first[TOOL_MEDIA_PARTS_KEY][0].inline_data.data
    second_image = second[TOOL_MEDIA_PARTS_KEY][0].inline_data.data
    assert first_image != second_image


@pytest.mark.parametrize("mode", ["overlay", "side_by_side", "checkerboard", "outlines"])
def test_border_style_reaches_all_atlas_bearing_panels(mode):
    defaults, default_iou = _views(mode=mode)
    styled, styled_iou = _views(mode=mode, border_color="cyan", border_thickness=4)
    assert default_iou == styled_iou
    for default, custom in zip(defaults, styled, strict=True):
        assert default.shape == custom.shape
        assert _line_pixels(default) > 0
        assert _line_pixels(custom) == 0
        assert _line_pixels(custom[..., ::-1]) > _line_pixels(default)


@pytest.mark.parametrize("mode", ["section", "template"])
def test_border_style_does_not_add_lines_to_clean_views(mode):
    (default,), _ = _views(mode=mode)
    (custom,), _ = _views(mode=mode, border_color="cyan", border_thickness=4)
    assert np.array_equal(default, custom)


def test_fractional_border_widths_are_distinct_and_preserve_geometry():
    from langslice.linear.render import _draw_polys, normalize_border_style

    square = np.asarray([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=float)
    ink = []
    for width in (0.25, 0.5, 0.75, 1.0):
        assert normalize_border_style("yellow", width)[1] == width
        screen = np.zeros((100, 100, 3), dtype=np.uint8)
        _draw_polys(screen, [square], (255, 255, 0), thickness=width)
        ink.append(int(screen[50, 15:26, 0].sum()))
        assert not screen[40:60, 40:60].any()
    assert all(a < b for a, b in zip(ink, ink[1:], strict=False))
