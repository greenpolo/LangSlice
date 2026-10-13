"""Where each image counts: the placed atlas footprint, exclusions.

Fixed mask = placed atlas footprint widened by :data:`MASK_MARGIN_MM`.
Moving (atlas) mask: the placed atlas footprint, widened by the
same margin, minus the excluded regions and a :data:`EXCLUSION_MARGIN_MM`
margin around them.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Widening of the placed atlas footprint so its outline sits inside the metric.
MASK_MARGIN_MM = 0.2
#: Margin around excluded regions also kept out of the moving mask. Excluded
#: pixels are blanked to background, which leaves an edge; without this margin
#: that artificial edge still pulls the section's tissue (measured on a
#: synthetic shifted region: Elastix moved it 92% as far as with no exclusion).
EXCLUSION_MARGIN_MM = 0.1


def _disk(radius_px: float) -> np.ndarray:
    r = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _disk(radius_px)) > 0
