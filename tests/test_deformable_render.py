"""Smooth drawing of the final (warped) borders on the original section."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from langslice.core.deformable import (
    FitSettings,
    draw_warped_borders,
    fit_section,
    warped_border_coverage,
)
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, placement, render_section


@pytest.fixture(scope="module")
def case():
    atlas = SyntheticAtlas()
    remove = np.zeros((260, 340), dtype=bool)
    remove[:, 250:] = True  # the right side of the section is missing
    image, _ = render_section(atlas, SMOOTH_FIELD(), remove=remove)
    record = fit_section(image, atlas, placement(), FitSettings(engine="elastix", detail="coarse"))
    return atlas, image, record


def test_borders_are_drawn_only_over_tissue(case):
    atlas, _, record = case
    faint, strong = warped_border_coverage(record, atlas)
    assert strong.max() > 0.5 and not faint.any()
    reach = cv2.dilate(record.tissue.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert not strong[~reach].any()


def test_smooth_borders_follow_the_nearest_label_edges(case):
    atlas, _, record = case
    _, strong = warped_border_coverage(record, atlas)
    from langslice.core.atlas.render import family_labels

    merged = family_labels(record.labels, atlas)
    edge = np.zeros(merged.shape, np.uint8)
    edge[:, 1:] |= merged[:, 1:] != merged[:, :-1]
    edge[1:, :] |= merged[1:, :] != merged[:-1, :]
    edge[~record.tissue] = 0
    near = cv2.dilate(edge, np.ones((7, 7), np.uint8)) > 0
    drawn = strong > 0.5
    assert drawn.sum() > 100
    assert (drawn & near).sum() / drawn.sum() > 0.9  # drawn lines sit on the label edges


def test_highlight_is_strong_over_a_faint_outline(case):
    atlas, image, record = case
    faint, strong = warped_border_coverage(record, atlas, highlight=["STR"])
    both_faint, both_strong = warped_border_coverage(record, atlas)
    assert faint.max() > 0.2 and strong.max() > 0.5
    assert strong.sum() < both_strong.sum()  # only the selected region's edges
    # Together the two layers still draw every edge.
    assert ((faint + strong) > 0.5).sum() >= 0.9 * (both_strong > 0.5).sum()
    out = draw_warped_borders(image, record, atlas, highlight=["STR"])
    assert out.size == image.size
    with pytest.raises(ValueError):
        warped_border_coverage(record, atlas, highlight=["NOPE"])


def test_marked_regions_get_their_own_layer_and_outlines_limit_the_rest(case):
    from langslice.core.deformable import warped_border_layers

    atlas, image, record = case
    layers = warped_border_layers(record, atlas, highlight=["STR"], marked=["HY"])
    assert layers["marked"].max() > 0.5 and layers["strong"].max() > 0.5
    # No edge is drawn twice: marked edges leave the strong and faint layers.
    assert ((layers["marked"] > 0.5) & (layers["strong"] > 0.5)).sum() == 0
    plain = draw_warped_borders(image, record, atlas)
    none = draw_warped_borders(image, record, atlas, outlines="none")
    outer = draw_warped_borders(image, record, atlas, outlines="outer")
    base = np.asarray(image.convert("RGB"), dtype=int)

    def inked(out) -> int:
        return int((np.abs(np.asarray(out, dtype=int) - base).sum(axis=-1) > 30).sum())

    assert inked(none) == 0 < inked(outer) < inked(plain)
    with pytest.raises(ValueError):
        draw_warped_borders(image, record, atlas, outlines="some")


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
