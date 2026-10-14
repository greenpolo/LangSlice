"""Unit tests for the automatic, per-picture border colour
(:func:`langslice.core.canvas.pick_border_color`/:func:`resolve_border_color`).

Nash, 2026-10-14: "it should default to whatever is the highest contrast
against the slice image." These check the colour-choice function directly
(no atlas/section rendering needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from langslice.core.canvas import (
    AUTO_BORDER_COLOR,
    BORDER_CANDIDATES,
    pick_border_color,
    resolve_border_color,
)

#: Colours reserved for other drawing in a picture (landmark markers, fold
#: outlines, the deformable fit's excluded-region ink, the zoom grid) --
#: never offered to the automatic choice.
_RESERVED = {
    "cyan": (0, 255, 255), "orange": (255, 170, 0), "white": (255, 255, 255),
    "black": (0, 0, 0), "fold_magenta": (255, 43, 214), "marked_pink": (255, 64, 160),
    "grid_lime": (125, 255, 58),
}

YELLOW = (255, 255, 0)


def _flat(rgb: tuple[int, int, int], size: int = 60) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def _band_coverage(size: int = 60) -> np.ndarray:
    coverage = np.zeros((size, size), dtype=np.float32)
    coverage[size // 2 - 3 : size // 2 + 3, :] = 1.0
    return coverage


def test_candidates_exclude_every_reserved_colour():
    candidate_rgbs = {rgb for _name, rgb in BORDER_CANDIDATES}
    assert candidate_rgbs.isdisjoint(_RESERVED.values())


@pytest.mark.parametrize("pale", [(235, 215, 210), (245, 240, 238), (250, 248, 246)])
def test_pale_nissl_like_tissue_picks_a_non_yellow_colour(pale):
    """Yellow is nearly invisible on pale tissue (the problem Nash is
    fixing); the automatic choice must pick something else."""
    rgb, name = pick_border_color(_flat(pale), _band_coverage())
    assert name != "yellow"
    assert rgb != YELLOW


def test_dark_brown_timm_like_tissue_keeps_yellow():
    """Yellow, cyan or white would all read fine on dark tissue; cyan and
    white are reserved for landmark markers (not in BORDER_CANDIDATES), so
    yellow -- the highest-contrast candidate actually on offer -- is what
    the automatic choice keeps."""
    rgb, name = pick_border_color(_flat((110, 70, 30)), _band_coverage())
    assert name == "yellow"
    assert rgb == YELLOW


def test_no_coverage_falls_back_to_yellow_unused():
    rgb, name = pick_border_color(_flat((235, 215, 210)), np.zeros((60, 60), dtype=np.float32))
    assert rgb == YELLOW and name == "yellow"


def test_explicit_colour_bypasses_the_automatic_choice():
    rgb, thickness, word = resolve_border_color("red", 0.75)
    assert rgb == (255, 0, 0) and thickness == 0.75 and word == "red"


def test_explicit_auto_literal_is_rejected_by_normalize_border_style():
    from langslice.core.canvas import normalize_border_style

    with pytest.raises(ValueError, match="auto"):
        normalize_border_style(AUTO_BORDER_COLOR, 0.5)
