"""Where each image counts: tissue, torn edges, exclusions, detected ventricles.

Fixed (section) mask: the tissue, widened by :data:`MASK_MARGIN_MM` so the
real brain outline sits inside the metric, minus a band along TORN edges.
A torn or cut edge is not a region boundary; letting the fit see it pulls
atlas borders onto the tear. The real outline must stay, because it is the
most reliable edge a section has.

Default torn-edge rule, chosen to be explainable: a stretch of the tissue
outline is torn when it lies more than :data:`TORN_EDGE_DEPTH_MM` inside the
placed atlas footprint (minus excluded regions) — the atlas says brain
continues well past where the tissue stops. The band is everything within
:data:`TORN_EDGE_BAND_MM` of such outline pixels. A caller may pass its own
band instead. Moving (atlas) mask: the placed atlas footprint, widened by the
same margin, minus the excluded regions and a :data:`EXCLUSION_MARGIN_MM`
margin around them.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from langslice.image_prep import foreground_mask

#: Widening of tissue and atlas footprints so their outlines sit inside the metric.
MASK_MARGIN_MM = 0.2
#: Margin around excluded regions also kept out of the moving mask. Excluded
#: pixels are blanked to background, which leaves an edge; without this margin
#: that artificial edge still pulls the section's tissue (measured on a
#: synthetic shifted region: Elastix moved it 92% as far as with no exclusion).
EXCLUSION_MARGIN_MM = 0.1
#: An outline stretch this far inside the placed atlas footprint is a tear.
TORN_EDGE_DEPTH_MM = 0.3
#: Half-width of the band removed from the fixed mask along a torn edge.
TORN_EDGE_BAND_MM = 0.15
#: Proxy long edge for tissue detection (``image_prep.foreground_mask``).
TISSUE_PROXY_EDGE = 1024
#: A hole in the tissue smaller than this is a bubble or speck, not a ventricle.
MIN_VENTRICLE_HOLE_MM2 = 0.004
#: A hole counts as a ventricle only within this distance of a placed atlas ventricle.
VENTRICLE_SEARCH_MM = 0.3


def tissue_masks(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """(filled, raw) tissue masks on the image's own grid.

    *raw* is ``image_prep``'s foreground rule (different from the slide
    background read off the frame border, every sizable piece kept); *filled*
    closes its interior holes, so ventricles and bubbles count as tissue for
    masking and label clipping.
    """
    proxy = foreground_mask(image, proxy_edge=TISSUE_PROXY_EDGE)
    if proxy is None:
        raise ValueError("No tissue could be separated from the slide background")
    raw = cv2.resize(proxy.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST) > 0
    filled = np.asarray(ndi.binary_fill_holes(raw), dtype=bool)
    return filled, raw


def _disk(radius_px: float) -> np.ndarray:
    r = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(mask: np.ndarray, radius_px: float) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _disk(radius_px)) > 0


def outline(mask: np.ndarray) -> np.ndarray:
    """Mask pixels with a 4-neighbour outside the mask (image border counts)."""
    padded = np.pad(mask.astype(bool), 1)
    inner = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask.astype(bool) & ~inner


def torn_edge_band(tissue: np.ndarray, footprint: np.ndarray, mm_per_px: float) -> np.ndarray:
    """Pixels near tissue outline that lies deep inside the atlas footprint."""
    depth = np.asarray(ndi.distance_transform_edt(footprint)) * mm_per_px
    torn = outline(tissue) & (depth > TORN_EDGE_DEPTH_MM)
    if not torn.any():
        return np.zeros(tissue.shape, dtype=bool)
    distance = np.asarray(ndi.distance_transform_edt(~torn)) * mm_per_px
    return distance <= TORN_EDGE_BAND_MM


def ventricle_holes(
    raw_tissue: np.ndarray, filled_tissue: np.ndarray, atlas_ventricles: np.ndarray,
    mm_per_px: float,
) -> np.ndarray:
    """Empty interior regions that plausibly are ventricles.

    A hole is background-like pixels enclosed by tissue (dark on fluorescence,
    bright on brightfield — the foreground rule handles both). It is kept when
    it is larger than :data:`MIN_VENTRICLE_HOLE_MM2` and lies within
    :data:`VENTRICLE_SEARCH_MM` of a placed atlas ventricle; other holes (tears,
    bubbles, a lifted fold) are left out rather than called ventricles.
    """
    holes = filled_tissue & ~raw_tissue
    if not holes.any() or not atlas_ventricles.any():
        return np.zeros(raw_tissue.shape, dtype=bool)
    near = dilate(atlas_ventricles, VENTRICLE_SEARCH_MM / mm_per_px)
    labelled, count = cast(tuple[np.ndarray, int], ndi.label(holes))
    components = np.asarray(labelled, dtype=np.intp)
    if not count:
        return np.zeros(raw_tissue.shape, dtype=bool)
    areas = np.bincount(components.ravel(), minlength=count + 1) * mm_per_px ** 2
    touching = np.bincount(components[near].ravel(), minlength=count + 1) > 0
    keep = (areas >= MIN_VENTRICLE_HOLE_MM2) & touching
    keep[0] = False
    return keep[components]
