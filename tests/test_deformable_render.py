"""Smooth drawing of the final (warped) borders on the original section."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from langslice.core.atlas.render import region_boundaries
from langslice.core.deformable import (
    FitSettings,
    draw_warped_borders,
    warped_border_layers,
)
from langslice.core.nonlinear.border_refinement import smooth_border_overlay
from tests.deformable_synthetic import (
    SMOOTH_FIELD,
    SyntheticAtlas,
    fit_section,
    placement,
    render_section,
)


def warped_border_coverage(record, atlas, **options):
    """``(faint, strong)`` layers of :func:`warped_border_layers`."""
    layers = warped_border_layers(record, atlas, **options)
    return layers["faint"], layers["strong"]


@pytest.fixture(scope="module")
def case():
    pytest.importorskip("ants", reason="the deformable fit needs antspyx")
    atlas = SyntheticAtlas()
    remove = np.zeros((260, 340), dtype=bool)
    remove[:, 250:] = True  # the right side of the section is missing
    image, _ = render_section(atlas, SMOOTH_FIELD(), remove=remove)
    record = fit_section(image, atlas, placement(), FitSettings(engine="ants", detail="coarse"))
    return atlas, image, record


def test_border_layers_strong_faint_and_marked(case):
    atlas, image, record = case
    # Without a highlight every border is strong.
    faint, strong = warped_border_coverage(record, atlas)
    assert strong.max() > 0.5 and not faint.any()
    # A highlight is strong over a faint outline of the rest.
    h_faint, h_strong = warped_border_coverage(record, atlas, highlight=["STR"])
    assert h_faint.max() > 0.2 and h_strong.max() > 0.5
    assert h_strong.sum() < strong.sum()  # only the selected region's edges
    assert ((h_faint + h_strong) > 0.5).sum() >= 0.9 * (strong > 0.5).sum()
    out = draw_warped_borders(image, record, atlas, highlight=["STR"])
    assert out.size == image.size
    with pytest.raises(ValueError):
        warped_border_coverage(record, atlas, highlight=["NOPE"])
    # Marked regions get their own layer; no edge is drawn twice.
    layers = warped_border_layers(record, atlas, highlight=["STR"], marked=["HY"])
    assert layers["marked"].max() > 0.5 and layers["strong"].max() > 0.5
    assert ((layers["marked"] > 0.5) & (layers["strong"] > 0.5)).sum() == 0
    plain = draw_warped_borders(image, record, atlas)
    none = draw_warped_borders(image, record, atlas, outlines="none")
    base = np.asarray(image.convert("RGB"), dtype=int)

    def inked(drawn) -> int:
        return int((np.abs(np.asarray(drawn, dtype=int) - base).sum(axis=-1) > 30).sum())

    assert inked(none) == 0 < inked(plain)
    with pytest.raises(ValueError):
        draw_warped_borders(image, record, atlas, outlines="some")


def test_smooth_borders_follow_the_nearest_label_edges(case):
    atlas, _, record = case
    _, strong = warped_border_coverage(record, atlas)
    edge = region_boundaries(record.labels)[0].astype(np.uint8)
    near = cv2.dilate(edge, np.ones((7, 7), np.uint8)) > 0
    drawn = strong > 0.5
    assert drawn.sum() > 100
    assert (drawn & near).sum() / drawn.sum() > 0.9  # drawn lines sit on the label edges


def test_resampled_record_draws_the_same_borders_smaller(case):
    from langslice.core.deformable import resampled_record

    atlas, _, record = case
    width, height = record.section_size
    half = resampled_record(record, (width // 2, height // 2))
    assert half.section_size == (width // 2, height // 2)
    assert half.mm_per_px == pytest.approx(record.mm_per_px * width / (width // 2))
    _, strong = warped_border_coverage(record, atlas, width_px=2.0)
    _, small = warped_border_coverage(half, atlas, width_px=1.0)
    shrunk = cv2.resize(strong, half.section_size, interpolation=cv2.INTER_AREA)
    near = cv2.dilate((small > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert ((shrunk > 0.3) & near).sum() / max(1, (shrunk > 0.3).sum()) > 0.9
    crop = resampled_record(record, (width // 2, height // 2), (10, 20, 90, 100))
    assert crop.section_size == (80, 80) and crop.field_mm.shape[:2] == (80, 80)


def test_warping_the_section_by_a_zero_field_changes_nothing(case):
    from dataclasses import replace

    from langslice.core.deformable import warp_section_image

    _, image, record = case
    still = replace(record, field_mm=np.zeros_like(record.field_mm),
                    inverse_field_mm=np.zeros_like(record.field_mm))
    out = warp_section_image(image, still)
    assert np.array_equal(np.asarray(out), np.asarray(image.convert("RGB")))
    moved = warp_section_image(image, record)
    assert not np.array_equal(np.asarray(moved), np.asarray(image.convert("RGB")))


def test_smooth_borders_draw_one_antialiased_line_per_shared_edge():
    labels = np.zeros((20, 20), dtype=np.int64)
    labels[4:16, 4:10] = 1
    labels[4:16, 10:16] = 2
    photo = Image.new("RGB", (100, 100), (0, 0, 0))
    atlas_to_image = np.array([[5.0, 0, 2.0], [0, 5.0, 2.0], [0, 0, 1]])  # 5x magnification
    drawn = np.asarray(smooth_border_overlay(photo, labels, atlas_to_image, width_px=2.0))
    row = drawn[50, 30:70, 0].astype(int)  # across the shared edge at x ~ 52
    lit = np.flatnonzero(row > 0)
    # One stroke about two pixels wide, not one line per region an atlas pixel apart.
    assert lit.size and lit.max() - lit.min() <= 3
    assert ((row > 0) & (row < 255)).any()  # antialiased edge pixels
