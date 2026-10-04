"""Geometry and compositing checks for the review renderer (no atlas needed)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from langslice.core.nonlinear import render


def _disc(size: int = 200, radius: int = 60, uid: int = 7) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    labels = np.zeros((size, size), dtype=np.int32)
    labels[(xs - size // 2) ** 2 + (ys - size // 2) ** 2 <= radius**2] = uid
    return labels


def test_region_contours_traces_each_region_once() -> None:
    labels = _disc()
    contours = render.region_contours(labels)
    assert set(contours) == {7}
    assert len(contours[7]) == 1
    points = contours[7]
    assert points[0].dtype == np.float64


def test_region_contours_skips_background_and_speckle() -> None:
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[40:44, 40:44] = 3  # 16px component, under the default 24px floor
    labels[10:30, 10:30] = 5
    assert set(render.region_contours(labels)) == {5}


def test_region_contours_finds_holes() -> None:
    labels = _disc()
    labels[90:110, 90:110] = 0
    contours = render.region_contours(labels, min_area_px=4.0)
    assert len(contours[7]) == 2  # outer silhouette plus the hole boundary


def test_smoothing_removes_the_staircase() -> None:
    """A smoothed circle sits closer to its true radius than the raster does."""
    labels = _disc()
    raw = render.region_contours(labels, smooth_window=0)[7][0]
    smooth = render.region_contours(labels, smooth_window=11)[7][0]
    center = np.array([100.0, 100.0])
    raw_spread = np.std(np.linalg.norm(raw - center, axis=1))
    smooth_spread = np.std(np.linalg.norm(smooth - center, axis=1))
    assert smooth_spread < raw_spread * 0.6


def test_smoothing_keeps_the_polygon_closed_and_sized() -> None:
    labels = _disc()
    smooth = render.region_contours(labels, smooth_window=11)[7][0]
    assert np.linalg.norm(smooth[0] - smooth[-1]) < 4.0
    center = np.array([100.0, 100.0])
    assert 57.0 < float(np.mean(np.linalg.norm(smooth - center, axis=1))) < 62.0


def test_label_anchor_lands_inside_a_crescent() -> None:
    """The centroid of a C falls outside it; the inscribed-circle center does not."""
    ys, xs = np.mgrid[0:200, 0:200]
    outer = (xs - 100) ** 2 + (ys - 100) ** 2 <= 80**2
    inner = (xs - 100) ** 2 + (ys - 70) ** 2 <= 55**2
    mask = outer & ~inner
    x, y, radius = render.label_anchor(mask)
    assert mask[int(round(y)), int(round(x))]
    assert radius > 5.0
    centroid_y = float(np.mean(np.nonzero(mask)[0]))
    centroid_x = float(np.mean(np.nonzero(mask)[1]))
    assert not mask[int(round(centroid_y)), int(round(centroid_x))]


def test_label_anchor_is_the_largest_inscribed_circle() -> None:
    mask = np.zeros((100, 300), dtype=bool)
    mask[45:55, 10:40] = True  # thin bar, radius ~5
    mask[20:80, 200:260] = True  # fat block, radius ~30
    x, y, radius = render.label_anchor(mask)
    assert x > 150
    assert 25.0 < radius < 32.0


def test_checkerboard_mask_alternates_and_spans_the_long_edge() -> None:
    mask = render.checkerboard_mask((64, 128), tiles=8)
    assert mask.shape == (64, 128)
    assert mask[0, 0] and not mask[0, 16] and mask[0, 32]
    assert not mask[16, 0]
    assert 0.4 < mask.mean() < 0.6


def test_checkerboard_takes_alternating_tiles_from_each_image() -> None:
    first = Image.new("RGB", (128, 128), (255, 0, 0))
    second = Image.new("RGB", (128, 128), (0, 0, 255))
    out = np.asarray(render.checkerboard(first, second, tiles=4))
    assert tuple(out[0, 0]) == (255, 0, 0)
    assert tuple(out[0, 40]) == (0, 0, 255)
    assert tuple(out[40, 0]) == (0, 0, 255)
    assert tuple(out[40, 40]) == (255, 0, 0)


def test_split_view_places_the_seam() -> None:
    first = Image.new("RGB", (200, 100), (255, 0, 0))
    second = Image.new("RGB", (200, 100), (0, 255, 0))
    out = np.asarray(render.split_view(first, second, position=0.25))
    assert tuple(out[50, 10]) == (255, 0, 0)
    assert tuple(out[50, 190]) == (0, 255, 0)


def test_polarity_detection_and_adaptive_border_color() -> None:
    assert render.is_dark_background(Image.new("RGB", (32, 32), (8, 8, 12)))
    assert not render.is_dark_background(Image.new("RGB", (32, 32), (230, 230, 225)))
    fill = (120, 200, 140)
    on_light = render.border_color(fill, dark_background=False)
    on_dark = render.border_color(fill, dark_background=True)
    # HSV value, not channel sum: saturating a pastel drops two channels while
    # the color itself gets brighter.
    assert max(on_light) < max(fill) < max(on_dark)
    gray_border = render.border_color((190, 190, 190), dark_background=True)
    assert len(set(gray_border)) == 1  # gray stays gray, no invented hue
    assert gray_border[0] > 190


def test_region_overlay_fills_and_outlines_without_touching_background() -> None:
    labels = _disc()
    histology = Image.fromarray(np.full((200, 200, 3), 200, dtype=np.uint8))
    out = np.asarray(
        render.region_overlay(
            histology,
            labels,
            lut={7: (255, 0, 0)},
            names={7: "CTX"},
            fill_alpha=0.5,
        )
    )
    assert tuple(out[2, 2]) == (200, 200, 200)  # background untouched
    assert out[130, 100][0] > out[130, 100][2]  # red fill blended in (clear of the acronym)
    ring = out[100, 155:168]  # crossing the boundary
    assert ring.std() > 3.0  # an anti-aliased edge, not a hard step


def test_region_overlay_labels_are_drawn_at_the_anchor() -> None:
    labels = _disc()
    histology = Image.fromarray(np.full((200, 200, 3), 200, dtype=np.uint8))
    common = {"lut": {7: (255, 0, 0)}, "fill_alpha": 0.5}
    with_text = np.asarray(render.region_overlay(histology, labels, names={7: "CTX"}, **common))
    without = np.asarray(render.region_overlay(histology, labels, **common))
    assert not np.array_equal(with_text[90:110, 85:115], without[90:110, 85:115])
    assert np.array_equal(with_text[130:150], without[130:150])  # only at the anchor


def test_region_overlay_borders_only_leaves_the_interior_alone() -> None:
    labels = _disc()
    histology = Image.fromarray(np.full((200, 200, 3), 200, dtype=np.uint8))
    out = np.asarray(
        render.region_overlay(histology, labels, lut={7: (255, 0, 0)}, fill_alpha=0.0)
    )
    assert tuple(out[100, 100]) == (200, 200, 200)
    assert not np.array_equal(out[100, 155:165], np.full((10, 3), 200, dtype=np.uint8))


def test_region_overlay_accepts_a_mismatched_canvas_size() -> None:
    labels = _disc(size=120, radius=40)
    histology = Image.new("RGB", (300, 300), (30, 30, 30))
    out = render.region_overlay(histology, labels, lut={7: (10, 200, 255)})
    assert out.size == (120, 120)


def test_marker_grid_reshapes_and_deformation_grid_renders() -> None:
    markers = [
        [float(x) + 5.0, float(y), float(x), float(y)]  # [overlay, image]
        for y in (0, 25, 50)
        for x in (0, 25, 50, 75)
    ]
    source, target = render._marker_grid(np.asarray(markers, dtype=np.float64))
    assert source.shape == (3, 4, 2)
    assert np.allclose(target[..., 0] - source[..., 0], 5.0)
    out = render.deformation_grid(Image.new("RGB", (100, 80), (20, 20, 20)), markers)
    assert out.size == (100, 80)
    assert np.asarray(out).std() > 1.0  # something was actually drawn


def test_contact_sheet_grids_captioned_panels() -> None:
    panels = [(f"panel {i}", Image.new("RGB", (200, 100), (i * 40, 10, 10))) for i in range(3)]
    sheet = render.contact_sheet(panels, columns=2, cell_px=200, title="demo")
    assert sheet.width == 400
    assert sheet.height > 200
    with pytest.raises(ValueError):
        render.contact_sheet([])


def _rings(size: int = 64) -> np.ndarray:
    """Two concentric discs on a low-resolution grid — an atlas in miniature."""
    ys, xs = np.mgrid[0:size, 0:size]
    r2 = (xs - size // 2) ** 2 + (ys - size // 2) ** 2
    labels = np.zeros((size, size), dtype=np.int32)
    labels[r2 <= (size // 3) ** 2] = 1
    labels[r2 <= (size // 6) ** 2] = 2
    return labels


def test_filled_regions_only_ever_paints_exact_palette_colors() -> None:
    """Nearest-color classification is downstream: no blended edge pixels."""
    lut = {1: (10, 200, 90), 2: (240, 60, 210)}
    rgb = render.filled_regions(_rings(), lut=lut, size=(512, 512))
    painted = {tuple(int(c) for c in color) for color in np.unique(rgb.reshape(-1, 3), axis=0)}
    assert painted <= {(0, 0, 0), *lut.values()}


def test_filled_regions_boundaries_carry_canvas_resolution_detail() -> None:
    """The upscaled outline is a smooth curve, not the NEAREST voxel staircase."""
    labels = _rings()
    lut = {1: (10, 200, 90), 2: (240, 60, 210)}
    size = (512, 512)
    smooth = render.filled_regions(labels, lut=lut, size=size)
    blocky = np.asarray(
        Image.fromarray(render.filled_regions(labels, lut=lut)).resize(
            size, resample=Image.Resampling.NEAREST
        )
    )

    def left_edge_steps(rgb: np.ndarray) -> int:
        """Distinct left-edge columns of the outer disc, row by row.

        A NEAREST upscale can only place that edge on multiples of the
        upscale factor, so its staircase has few distinct steps; a polygon
        filled at canvas resolution lands wherever the curve actually is.
        """
        painted = rgb.any(axis=2)
        rows = [row for row in range(rgb.shape[0]) if painted[row].any()]
        return len({int(np.argmax(painted[row])) for row in rows})

    assert left_edge_steps(smooth) > 2 * left_edge_steps(blocky)


def test_filled_regions_delineates_every_region_in_its_own_darker_color() -> None:
    """ARA plate style: hairlines, region's own hue, only value moves."""
    labels = _rings()
    lut = {1: (10, 200, 90), 2: (240, 60, 210)}
    size = (2048, 2048)
    flat = render.filled_regions(labels, lut=lut, size=size)
    bordered = render.filled_regions(labels, lut=lut, size=size, border_px=2.0)

    # interiors untouched, boundaries darkened
    assert (bordered[1024, 1024] == flat[1024, 1024]).all()
    changed = np.any(bordered != flat, axis=2)
    assert 0.0 < changed.mean() < 0.05  # hairlines, not a repaint

    # both regions are delineated, each in a darker version of its own color
    # (a shared boundary is one line: the neighbours draw over each other)
    painted = {tuple(int(c) for c in color) for color in np.unique(
        bordered.reshape(-1, 3), axis=0
    )}
    assert render.darker(lut[1]) in painted
    assert render.darker(lut[2]) in painted


def test_family_borders_are_heavier_than_the_leaf_borders_inside_them() -> None:
    labels = _rings()
    families = (labels != 0).astype(np.int32)  # both rings are one division
    lut = {1: (10, 200, 90), 2: (240, 60, 210)}
    leaf_only = render.filled_regions(labels, lut=lut, size=(2048, 2048), border_px=2.0)
    with_family = render.filled_regions(
        labels, lut=lut, size=(2048, 2048), border_px=2.0, families=families
    )
    outer = slice(0, 700)  # the outer silhouette, away from the inner disc
    assert (with_family[outer] != 0).sum() >= (leaf_only[outer] != 0).sum()
    assert np.any(with_family != leaf_only)


def test_filled_regions_keeps_small_regions_on_top_of_big_ones() -> None:
    lut = {1: (10, 200, 90), 2: (240, 60, 210)}
    rgb = render.filled_regions(_rings(), lut=lut, size=(256, 256))
    assert tuple(int(c) for c in rgb[128, 128]) == lut[2]  # inner disc survives
