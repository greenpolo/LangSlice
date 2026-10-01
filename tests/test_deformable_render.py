"""Smooth drawing of the final (warped) borders on the original section."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from langslice.deformable import (
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
    from langslice.atlas.render import family_labels

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
