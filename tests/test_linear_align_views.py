"""The view controls of the interactive alignment screen.

The alignment loop is a vision-action loop, so a view control that lies is
worse than one that does not exist: a zoom that does not magnify, lines on
a view meant to be clean, or a scale bar that keeps its old length after a
crop would all be read as anatomy. These pin what each mode
actually puts on the screen, in pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from langslice.core.canvas import canvas_geometry, physical_views
from langslice.core.captions import scale_bar_px
from tests.test_linear_physical import TwoRegionAtlas, _ctx

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


def _views(value: int = 120, **kwargs) -> list[np.ndarray]:
    panels: list = []
    images = physical_views(
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
        panel_frames=panels,
        **kwargs,
    )
    # The picture content, above its caption band.
    return [np.asarray(image.convert("RGB"))[:panel.content_box[3]]
            for image, panel in zip(images, panels, strict=True)]


def test_the_screen_is_the_long_edge_and_a_whole_canvas_is_never_upsampled():
    """Each panel's long edge is *long_edge*; the whole canvas stays at its
    own pixels when it has fewer, while a zoom is drawn at *long_edge* (a
    small region at the size of any other picture), enlarged past the
    canvas's pixels and said so in its caption."""
    geometry = _geometry()
    canvas_long = max(geometry.size)
    (shrunk,) = _bodies(long_edge=128)
    assert max(shrunk.shape[:2]) == 128
    (whole,) = _bodies(long_edge=4 * canvas_long)
    assert whole.shape[1] == geometry.size[0], "never upsampled past the canvas"
    (zoomed,) = _bodies(long_edge=4 * canvas_long, zoom=[0.3, 0.3, 0.7, 0.7])
    assert max(zoomed.shape[:2]) == 4 * canvas_long
    panels: list = []
    (picture,) = physical_views(
        _section(), UM_PER_PX, TwoRegionAtlas(), 0.2, "coronal", 0.0, 0.0, _IDENTITY,
        long_edge=4 * canvas_long, zoom=[0.45, 0.3, 0.55, 0.7], panel_frames=panels)
    (panel,) = panels
    # A tall, narrow region: the caption band is laid out at the long edge.
    assert picture.width == 4 * canvas_long > panel.content_box[2]


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
    (whole,) = _views()
    (zoomed,) = _views(zoom=box)

    # Canvas pixels one to one: the zoom is the crop, at its own size.
    assert zoomed.shape[1] == round(0.7 * geometry.size[0]) - round(0.3 * geometry.size[0])
    tissue = lambda rgb: float(((rgb >= 100) & (rgb <= 140)).all(axis=2).mean())  # noqa: E731
    assert tissue(zoomed) > 2.0 * tissue(whole), "the crop did not magnify the tissue"
    assert np.array_equal(whole, _views(zoom=[])[0])  # an empty zoom is the whole canvas

    # The bar is drawn after the crop, at the magnified micrometres per pixel.
    crop_width = round(0.7 * geometry.size[0]) - round(0.3 * geometry.size[0])
    for image, width in ((whole, geometry.size[0]), (zoomed, crop_width)):
        factor = image.shape[1] / float(width)
        assert _bar_length(image) == pytest.approx(
            scale_bar_px(geometry.um_per_px / factor) + 1, abs=3
        )


# --- the modes -----------------------------------------------------------


def test_the_atlas_opacity_is_a_dial_not_a_switch():
    # Inside the anatomy, outside the tissue: the atlas square is 1 mm (100 px)
    # and the tissue 80 px, both centred on the canvas, so the 10 px band just
    # inside the anatomy's left edge is template-only.
    geometry = _geometry()
    cx, cy = geometry.size[0] // 2, geometry.size[1] // 2
    box = (slice(cy - 30, cy + 30), slice(cx - 49, cx - 42))
    (none,) = _views()
    (half,) = _views(atlas_opacity=0.5)
    (full,) = _views(atlas_opacity=1.0)
    assert none[box].mean() < half[box].mean() < full[box].mean()


# --- the outline layers --------------------------------------------------


def _line_pixels(rgb: np.ndarray, bare: np.ndarray) -> int:
    """Pixels the atlas lines changed: where *rgb* differs from the same view
    drawn without lines (cased lines are black or white, so a colour test
    cannot find them), above the bar, below the caption."""
    return int((rgb[60:-40] != bare[60:-40]).any(axis=-1).sum())


def test_the_outline_layer_picks_which_atlas_lines_are_drawn():
    """This atlas has two families: the 1 mm square and a region inside it."""
    (every,) = _views()
    (bare,) = _views(outlines="none")

    assert np.array_equal(bare, _views(mode="section")[0]), "outlines='none' still drew lines"
    assert _line_pixels(every, bare) > 300, "the region lines are missing"
    # Lines can cover tissue, never uncover it, so the bare view shows at
    # least as much of the section; cased hairlines at canvas scale cover little.
    tissue = lambda rgb: int(((rgb >= 110) & (rgb <= 130)).all(axis=2).sum())  # noqa: E731
    assert tissue(bare) >= tissue(every) > 0
    assert tissue(bare) - tissue(every) < 0.3 * tissue(bare)


# --- the tools, in the main toolbox --------------------------------------


def _tools(tmp_path: Path):
    """The transform tools of a one-section run, on a 10 um/px file."""
    from langslice.doors.tools.toolbox import build_tools

    _section(200, (512, 512)).save(tmp_path / "s.tif",
        resolution_unit=3, x_resolution=1000.0, y_resolution=1000.0)  # 10 um/px
    ctx, state = _ctx(tmp_path)
    record = state.slices[0]
    record.position_mm = 0.2
    box = build_tools(state, ctx, ctx.spec)
    return {tool.__name__: tool for tool in box.tools}, box, state


def test_look_draws_the_section_where_it_is_and_writes_nothing(tmp_path: Path):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    tools, _box, state = _tools(tmp_path)
    look = tools["look"]

    current = look("overlay", sections=["s.tif"])
    assert current["status"] == "ok"
    assert "at 0.20 mm" in current["pictures"][0]["caption"]
    assert len(current[TOOL_MEDIA_PARTS_KEY]) == 1

    # Candidate positions: the atlas at each, the section joined to its own.
    stepped = look("positioning", sections=["s.tif"], positions_mm=[0.1, 0.2, 0.3])
    assert stepped["status"] == "ok" and len(stepped[TOOL_MEDIA_PARTS_KEY]) == 1
    # A look, not a write.
    assert state.slices[0].position_mm == 0.2

    assert look("flicker", sections=["s.tif"])["error"] == "UNKNOWN_MODE"
    assert look("overlay", sections=["ghost.tif"])["error"] == "UNKNOWN_SLICE_IDS"
    state.slices[0].position_mm = None
    assert look("overlay", sections=["s.tif"])["error"] == "NO_POSITION"


# --- the pivot -----------------------------------------------------------


def _bodies(**kwargs) -> list[np.ndarray]:
    """Each panel above its caption band (the band's height varies with wrapping)."""
    panels: list = []
    long_edge = kwargs.pop("long_edge", None)
    images = physical_views(
        _section(), UM_PER_PX, TwoRegionAtlas(), 0.2, "coronal", 0.0, 0.0, _IDENTITY,
        long_edge=long_edge, panel_frames=panels, **kwargs,
    )
    return [np.asarray(image.convert("RGB"))[:panel.content_box[3]]
            for image, panel in zip(images, panels, strict=True)]


def test_clean_section_and_template_views_carry_no_outlines():
    (section_only,) = _bodies(mode="section")
    (template_only,) = _bodies(mode="template")
    (template_lined,) = _bodies(mode="template", template_lines=True)
    (overlaid,) = _bodies(mode="overlay")

    assert _line_pixels(overlaid, section_only) > 300
    # The template alone carries no lines unless asked for them.
    assert template_only.shape == template_lined.shape
    assert _line_pixels(template_lined, template_only) > 300


@pytest.mark.parametrize("core", ["black", "white"])
def test_border_core_changes_atlas_lines_without_changing_tissue(core):
    (bare,) = _views(outlines="none")
    (cased,) = _views(border_color=core)
    (other,) = _views(border_color="white" if core == "black" else "black")
    assert _line_pixels(cased, bare) > 300 and not np.array_equal(cased, other)
    tissue = lambda rgb: int((rgb == 120).all(axis=2).sum())  # noqa: E731
    assert 0.7 * tissue(bare) < tissue(cased) <= tissue(bare)


def test_border_thickness_is_measured_after_zoom_and_resize():
    from langslice.core.canvas import _draw_polys

    square = np.asarray([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=float)
    widths = {}
    for thickness in (1, 4):
        widths[thickness] = []
        # Equivalent crops/scales change the square's on-screen size, but
        # the straight vertical edge should keep the requested pixel width.
        for factor, origin in ((1.0, (0, 0)), (0.5, (0, 0)), (2.0, (10, 10))):
            screen = np.zeros((200, 200, 3), dtype=np.uint8)
            _draw_polys(screen, [square], "white", factor=factor,
                        origin=origin, thickness=thickness)
            y = round((50 - origin[1]) * factor)
            x = round((20 - origin[0]) * factor)
            widths[thickness].append(int((screen[y, x-6:x+7, 0] > 127).sum()))
        assert len(set(widths[thickness])) == 1
    assert widths[4][0] > widths[1][0]


@pytest.mark.parametrize("mode", ["overlay", "template"])
def test_border_style_reaches_all_atlas_bearing_panels(mode):
    bare = _views(mode=mode, template_lines=True, outlines="none")
    defaults = _views(mode=mode, template_lines=True)
    styled = _views(mode=mode, template_lines=True, border_color="white", border_thickness=4)
    for none, default, custom in zip(bare, defaults, styled, strict=True):
        assert default.shape == custom.shape == none.shape
        assert _line_pixels(default, none) > 0
        assert _line_pixels(custom, none) > _line_pixels(default, none)
