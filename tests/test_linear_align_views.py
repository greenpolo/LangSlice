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
    OVERLAY_LONG_EDGE,
    canvas_geometry,
    physical_views,
    scale_bar_px,
)

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
        long_edge=OVERLAY_LONG_EDGE,
        **kwargs,
    )
    return [np.asarray(image.convert("RGB")) for image in images], iou


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

    # Same output width; the height differs only by the caption band (the zoom
    # caption has a third line).
    assert whole.shape[1] == zoomed.shape[1], "the output width does not change"
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
    middle = board[200:570, 200:570]
    section_px = ((middle >= 110) & (middle <= 130)).all(axis=2).sum()
    template_px = (middle >= 250).all(axis=2).sum()
    assert section_px > 1000 and template_px > 1000


def test_outlines_mode_is_black_outside_the_lines():
    (lines,), _ = _views(mode="outlines")
    # No tissue, no template: the centre of the anatomy is bare canvas.
    centre = lines[lines.shape[0] // 2 - 5 : lines.shape[0] // 2 + 5,
                   lines.shape[1] // 2 - 5 : lines.shape[1] // 2 + 5]
    assert centre.max() == 0
    assert float((lines > 40).any(axis=2).mean()) < 0.05, "more than lines on screen"
    # Two greys: the atlas hairline and the section's own silhouette.
    values = set(np.unique(lines[lines > 40]))
    assert max(values) >= 200 and any(90 < v < 190 for v in values)


def test_the_template_opacity_is_a_dial_not_a_switch():
    box = (slice(360, 400), slice(360, 400))  # inside the anatomy, outside the tissue
    (none,), _ = _views()
    (half,), _ = _views(template_opacity=0.5)
    (full,), _ = _views(template_opacity=1.0)
    assert none[box].mean() < half[box].mean() < full[box].mean()


# --- the outline layers --------------------------------------------------


def _line_pixels(rgb: np.ndarray) -> int:
    """Hairline-bright pixels in the picture, above the bar, below the caption."""
    return int((rgb[60:-40] >= 200).all(axis=2).sum())


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
    assert tissue(bare) - tissue(every) < 0.1 * tissue(bare)


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


def test_the_preview_payload_carries_history_and_pixels_but_no_overlap(tmp_path: Path):
    tools, box, _state = _tools(tmp_path)
    preview = tools["preview_transform"]

    first = preview("s.tif", 0.0, 1.0, 1.0, 0.25, 0.0)
    assert first["status"] == "ok"
    # 0.25 mm on a 10 um/px canvas is 25 px, and the payload says so.
    assert first["translate_px"]["x"] == pytest.approx(25.0, abs=0.1)
    assert first["translate_px"]["y"] == 0.0
    assert first["translate_px"]["px_per_mm"] == pytest.approx(100.0)
    # No overlap number: silhouette overlap against the whole atlas plate
    # rewarded inflating a damaged remnant to fill it (luna, D_08, 2026-09-06:
    # 0.29 -> 0.51 at scale 1.35), and this loop exists for damaged sections.
    assert "silhouette_iou" not in first
    assert first["view"] == {
        "mode": "overlay",
        "zoom": [0.0, 0.0, 1.0, 1.0],
        "outlines": "all",
    }
    assert first["pivot"] == {"mode": "canvas", "canvas_frac": [0.5, 0.5]}
    # The decomposition names its translations for what they are: fractions.
    # The alignment payload's decomposition carries no translation fields at
    # all: the shift is the entered millimetres, and the matrix's fractions
    # (which absorb pivot-based scale/rotation) read as a contradiction.
    assert "translate_x_frac" not in first["decomposition"]
    assert "translate_x" not in first["decomposition"]
    kept = {"rotation_deg", "scale_x", "scale_y", "shear", "mirrored"}
    assert kept <= set(first["decomposition"])

    second = preview("s.tif", 2.0, 1.0, 1.0, 0.0, 0.0)
    assert [entry["rotation_deg"] for entry in second["history"]] == [0.0, 2.0]
    assert [entry["translate_x_mm"] for entry in second["history"]] == [0.25, 0.0]
    assert len(first["history"]) == 1  # oldest first, this preview last
    # The history is per section, kept on the toolbox for the whole run.
    assert len(box.preview_history["s.tif"]) == 2


def test_the_preview_tool_takes_the_view_controls(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY

    preview = _tools(tmp_path)[0]["preview_transform"]

    pair = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "side_by_side")
    assert len(pair[TOOL_MEDIA_PARTS_KEY]) == 2
    assert pair["view"]["mode"] == "side_by_side"

    zoomed = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [0.3, 0.3, 0.7, 0.7], 0.5)
    assert len(zoomed[TOOL_MEDIA_PARTS_KEY]) == 1
    assert zoomed["view"]["zoom"] == [0.3, 0.3, 0.7, 0.7]

    assert preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "flicker")["error"] == "BAD_MODE"
    assert preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [0.3, 0.7])["error"] == "BAD_ZOOM"
    assert preview("ghost.tif", 0.0, 1.0, 1.0, 0.0, 0.0)["error"] == "UNKNOWN_SLICE_IDS"


def test_ab_returns_the_candidate_and_what_is_stored(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY

    tools, _box, _state = _tools(tmp_path)
    against_identity = tools["preview_transform"]("s.tif", 6.0, 1.0, 1.0, 0.0, 0.0, "ab")
    assert len(against_identity[TOOL_MEDIA_PARTS_KEY]) == 2
    assert against_identity["view"]["mode"] == "ab"
    assert "identity" in against_identity["description"]
    assert against_identity["ab_reference"]["source"] == "identity"

    tools["set_transform"]("s.tif", 3.0, 1.0, 1.0, 0.0, 0.0, "")
    against_stored = tools["preview_transform"]("s.tif", 6.0, 1.0, 1.0, 0.0, 0.0, "ab")
    assert len(against_stored[TOOL_MEDIA_PARTS_KEY]) == 2
    assert "stored" in against_stored["description"]
    assert against_stored["ab_reference"]["params"]["rotation_deg"] == 3.0
    # A/B is a look, not a write: the stored transform is untouched.
    assert _state.slices[0].transform["physical"]["rotation_deg"] == 3.0


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
    preview = tools["preview_transform"]

    centred = preview("s.tif", 10.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas")
    corner = preview("s.tif", 10.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, [0.25, 0.75])
    assert corner["pivot"] == {"mode": "fractions", "canvas_frac": [0.25, 0.75]}
    assert preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "middle")[
        "error"
    ] == "BAD_PIVOT"

    tools["set_transform"]("s.tif", 10.0, 1.0, 1.0, 0.0, 0.0, "", [0.25, 0.75])
    stored = state.slices[0].transform
    assert stored["physical"]["pivot"] == [0.25, 0.75]
    # Same rotation, different pivot: the same map only up to a translation,
    # which is exactly what the six normalized numbers must carry.
    assert stored["params"][:2] == pytest.approx(
        [np.cos(np.radians(10.0)), np.sin(np.radians(10.0)) * 512 / 512], abs=1e-6
    )
    tools["set_transform"]("s.tif", 10.0, 1.0, 1.0, 0.0, 0.0, "", "canvas")
    assert state.slices[0].transform["params"][2] != stored["params"][2]
    assert centred["decomposition"]["rotation_deg"] == pytest.approx(
        corner["decomposition"]["rotation_deg"]
    )


# --- landmarks -----------------------------------------------------------


def test_a_pair_the_transform_already_maps_has_no_residual(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY

    tools, _box, _state = _tools(tmp_path)
    params = (4.0, 1.05, 0.95, 0.3, -0.2)

    from langslice.affine import physical_affine_matrix
    from langslice.linear.render import canvas_geometry, render_slice

    ctx = _ctx(tmp_path)[0]
    section = render_slice(ctx, _state.slices[0], long_edge=512)
    geometry = canvas_geometry(
        section.size, 10.0, TwoRegionAtlas(), 0.2, "coronal"
    )
    width, height = geometry.size
    matrix = physical_affine_matrix(
        size=geometry.size, um_per_px=10.0,
        rotation_deg=params[0], scale_x=params[1], scale_y=params[2],
        translate_x_mm=params[3], translate_y_mm=params[4],
    )
    here = np.array([0.4 * width, 0.35 * height])
    there = matrix[:, :2] @ here + matrix[:, 2]
    pair = {
        "section": [0.4, 0.35],
        "atlas": [there[0] / width, there[1] / height],
    }

    result = tools["landmarks"]("s.tif", [pair], *params)
    assert result["status"] == "ok"
    assert result["pairs"][0]["residual_mm"] == pytest.approx(0.0, abs=1e-3)
    assert result["rms_mm"] == pytest.approx(0.0, abs=1e-3)
    assert result["fit"] is None  # one pair supports no fit
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 1


def test_the_landmark_fit_recovers_a_known_shift_and_scale(tmp_path: Path):
    tools, _box, _state = _tools(tmp_path)
    # Four points on the canvas, moved by a known scale and shift. The file
    # is 512 px at 10 um/px, so the render is 1:1 and 1 mm is 100 canvas px;
    # the payload must report the shift in millimetres.
    points = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    scale, shift_mm = 1.2, 0.5
    from langslice.linear.render import canvas_geometry, render_slice

    ctx = _ctx(tmp_path)[0]
    section = render_slice(ctx, _state.slices[0], long_edge=512)
    geometry = canvas_geometry(section.size, 10.0, TwoRegionAtlas(), 0.2, "coronal")
    width, height = geometry.size
    cx, cy = width / 2.0, height / 2.0
    pairs = []
    for fx, fy in points:
        x, y = fx * width, fy * height
        moved = (
            cx + (x - cx) * scale + shift_mm * 1000.0 / 10.0,
            cy + (y - cy) * scale,
        )
        pairs.append(
            {"section": [fx, fy], "atlas": [moved[0] / width, moved[1] / height]}
        )

    result = tools["landmarks"]("s.tif", pairs, 0.0, 1.0, 1.0, 0.0, 0.0)
    fit = result["fit"]
    assert fit["kind"] == "affine" and fit["points"] == 4
    assert fit["scale_x"] == pytest.approx(scale, abs=1e-3)
    assert fit["scale_y"] == pytest.approx(scale, abs=1e-3)
    assert fit["rotation_deg"] == pytest.approx(0.0, abs=1e-3)
    assert fit["translate_x_mm"] == pytest.approx(shift_mm, abs=1e-3)
    assert fit["translate_y_mm"] == pytest.approx(0.0, abs=1e-3)
    assert fit["rms_mm"] == pytest.approx(0.0, abs=1e-3)
    # Under the identity the points are still where they were: the residual is
    # the move itself, in millimetres.
    assert result["rms_mm"] > 0.1

    two = tools["landmarks"]("s.tif", pairs[:2], 0.0, 1.0, 1.0, 0.0, 0.0)
    assert two["fit"]["kind"] == "similarity" and two["fit"]["points"] == 2
    assert tools["landmarks"]("s.tif", [], 0.0, 1.0, 1.0, 0.0, 0.0)["error"] == "BAD_ARGS"


def test_clean_section_and_template_views_carry_no_outlines():
    (section_only,), _ = _views(mode="section")
    (template_only,), _ = _views(mode="template")
    (overlaid,), _ = _views(mode="overlay")
    pair, _ = _views(mode="side_by_side")

    # The synthetic tissue is 120 grey; anti-aliased hairlines are far brighter.
    # Picture area only: below the caption band, above the scale bar.
    assert overlaid[60:-40].max() >= 200
    assert section_only[60:-40].max() < 200
    # The template alone is the side-by-side's second panel minus its lines.
    assert template_only.shape == pair[1].shape
    assert not np.array_equal(template_only[60:-40], pair[1][60:-40])


def test_the_preview_tool_takes_the_outline_layer(tmp_path: Path):
    preview = _tools(tmp_path)[0]["preview_transform"]

    plain = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0)
    assert plain["view"]["outlines"] == "all"

    outer = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas", "outer")
    assert outer["view"]["outlines"] == "outer"
    assert "OUTER boundary" in outer["description"]

    bare = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas", "none")
    assert "No atlas outlines" in bare["description"]

    bad = preview("s.tif", 0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [], 0.0, "canvas", "midline")
    assert bad["error"] == "BAD_OUTLINES"
