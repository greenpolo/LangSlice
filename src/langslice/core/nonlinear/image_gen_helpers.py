"""Label-map helpers for the border traces: colour families, crisp borders, line widths."""

from __future__ import annotations

from typing import Any

import numpy as np

from langslice.core.atlas.render import family_mapping as _family_mapping

#: Width of an ARA-style leaf delineation line, at a 2048px canvas.
_LEAF_BORDER_PX = 2.0


def line_width_px(long_edge: int) -> int:
    """Delineation-line width for a render whose long edge is *long_edge* px."""
    return max(1, round(_LEAF_BORDER_PX * long_edge / 2048))


def _merge_classified(classified_2d: np.ndarray, atlas: Any, merge_eps: float = 40.0) -> np.ndarray:
    """Map region ids onto one representative id per merged family color.

    The borders a model is shown and traces are family borders: the atlas's
    thin per-layer shade bands have no line of their own, so a trace never
    has to place them.
    """
    mapping = _family_mapping((int(u) for u in np.unique(classified_2d)), atlas, merge_eps)
    merged = np.zeros_like(classified_2d)
    for uid, rep_id in mapping.items():
        merged[classified_2d == uid] = rep_id
    return merged


def _extract_borders_from_classified(classified_2d: np.ndarray) -> np.ndarray:
    """Single-pixel region borders from classified ids, by neighbor difference.

    Interior borders mark one side of each id change; the outer silhouette
    marks the foreground pixels touching background. No contours, no
    smoothing — each boundary is one crisp line.
    """
    interior = np.zeros(classified_2d.shape, dtype=bool)
    interior[:, 1:] |= classified_2d[:, 1:] != classified_2d[:, :-1]
    interior[1:, :] |= classified_2d[1:, :] != classified_2d[:-1, :]

    fg = classified_2d != 0
    bg_padded = np.pad(classified_2d == 0, 1, constant_values=True)
    touches_bg = (
        bg_padded[2:, 1:-1] | bg_padded[:-2, 1:-1] | bg_padded[1:-1, 2:] | bg_padded[1:-1, :-2]
    )
    borders = ((interior & fg) | (fg & touches_bg)).astype(np.uint8) * 255
    return borders
