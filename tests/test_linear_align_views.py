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

    assert whole.shape == zoomed.shape, "the output frame does not change"
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


# --- the numbers the payload carries -------------------------------------


def test_silhouette_iou_is_a_fraction_and_measures_the_placement():
    _images, iou = _views()
    assert 0.0 <= iou <= 1.0
    # An 80 px square of tissue inside a 100 px square of atlas anatomy.
    assert iou == pytest.approx((80.0 / 100.0) ** 2, abs=0.05)


def _preview_tool(tmp_path: Path):
    from langslice.linear.transform import _build_align_tools

    _section(200, (512, 512)).save(tmp_path / "s.tif", dpi=(2540.0, 2540.0))  # 10 um/px
    ctx, state = _ctx(tmp_path)
    record = state.slices[0]
    record.position_mm = 0.2
    box = _build_align_tools(state, ctx, record)
    tool = next(t for t in box.tools if t.__name__ == "preview_transform")
    return tool, box


def test_the_preview_payload_carries_history_pixels_and_overlap(tmp_path: Path):
    preview, box = _preview_tool(tmp_path)

    first = preview(0.0, 1.0, 1.0, 0.25, 0.0)
    assert first["status"] == "ok"
    # 0.25 mm on a 10 um/px canvas is 25 px, and the payload says so.
    assert first["translate_px"]["x"] == pytest.approx(25.0, abs=0.1)
    assert first["translate_px"]["y"] == 0.0
    assert first["translate_px"]["px_per_mm"] == pytest.approx(100.0)
    assert 0.0 <= first["silhouette_iou"] <= 1.0
    assert first["view"] == {"mode": "overlay", "zoom": [0.0, 0.0, 1.0, 1.0]}
    # The decomposition names its translations for what they are: fractions.
    assert "translate_x_frac" in first["decomposition"]
    assert "translate_x" not in first["decomposition"]

    second = preview(2.0, 1.0, 1.0, 0.0, 0.0)
    assert [entry["rotation_deg"] for entry in second["history"]] == [0.0, 2.0]
    assert [entry["translate_x_mm"] for entry in second["history"]] == [0.25, 0.0]
    assert len(first["history"]) == 1  # oldest first, this preview last
    assert box.previews == 2


def test_the_preview_tool_takes_the_view_controls(tmp_path: Path):
    from langslice.adk import TOOL_MEDIA_PARTS_KEY

    preview, _box = _preview_tool(tmp_path)

    pair = preview(0.0, 1.0, 1.0, 0.0, 0.0, "side_by_side")
    assert len(pair[TOOL_MEDIA_PARTS_KEY]) == 2
    assert pair["view"]["mode"] == "side_by_side"

    zoomed = preview(0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [0.3, 0.3, 0.7, 0.7], 0.5)
    assert len(zoomed[TOOL_MEDIA_PARTS_KEY]) == 1
    assert zoomed["view"]["zoom"] == [0.3, 0.3, 0.7, 0.7]

    assert preview(0.0, 1.0, 1.0, 0.0, 0.0, "flicker")["error"] == "BAD_MODE"
    assert preview(0.0, 1.0, 1.0, 0.0, 0.0, "overlay", [0.3, 0.7])["error"] == "BAD_ZOOM"
