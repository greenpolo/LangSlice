"""The silhouette prior: the atlas plane placed on the tissue by outline alone.

A model-free rough placement. The atlas plane at (``position_mm``,
``pitch_deg``, ``yaw_deg``) is placed onto the section's own silhouette by a
closed-form moments fit — centroid, principal axes, axis spreads. It is a
LINEAR placement: the boundaries are the atlas's, moved rigidly onto the
tissue, not deformed onto it.

Measured against the LSD_910 hand registrations: the placement alone scores
0.82 mean family dice, better than every image-model configuration tried.
Both border routes use it as the rough placement: route "supplied" only when
no placement is supplied and ``provider="none"`` (the model-free backbone,
:mod:`langslice.nonlinear.border_registration`); route "atlas" always (there
is no placement to supply, so the silhouette fit stands in for one, and one
or two model calls then draw/correct the boundaries against it).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from langslice.affine import _affine_from_pose, _moments_pose, silhouette_iou
from langslice.image_prep import foreground_mask

#: Sign patterns tried for the source eigenvectors: pure ROTATIONS only.
#: The two reflections score the same silhouette IoU on a near-symmetric
#: section and land the anatomy on the wrong hemisphere.
_SIGN_PATTERNS: tuple[tuple[int, int], ...] = ((1, 1), (-1, -1))


def tissue_mask(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Filled binary tissue silhouette of *image* at *size*, uint8 0/255.

    The cut is the shared :func:`langslice.image_prep.foreground_mask`:
    "different from the frame's own border", which reads dark-on-light
    brightfield and light-on-dark fluorescence alike, largest connected blob
    only (a slide carries fragments, dust and pen marks, any of which would
    drag the moments fit off the section). Holes — ventricles, tears, dim
    white matter — are filled here, because the placement matches the
    tissue's OUTLINE.

    A fixed threshold on the raw intensity is not an option: the canvas has
    already been through ``adaptive_preprocess`` (``--preprocess auto``,
    the default), and CLAHE lifts a dark fluorescence background into a
    textured gray that any absolute cut calls tissue — measured on the eval
    slices, a fixed cut claimed 99% of the frame where the section is ~85%,
    and Otsu on the same canvas kept only the bright cortical rim (25%).
    """
    mask = foreground_mask(image)
    if mask is None:
        raise ValueError(
            "Tissue silhouette could not be found on this image (the "
            "foreground covers almost none or almost all of the frame)."
        )
    # foreground_mask measures on a small proxy; NEAREST back up to the
    # canvas is plenty for a moments fit, which reads centroid and spread.
    scaled = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            size, Image.Resampling.NEAREST
        )
    ) > 0
    return np.asarray(binary_fill_holes(scaled), dtype=bool).astype(np.uint8) * 255


def place_plane_on_tissue(
    labels: np.ndarray, tissue: np.ndarray
) -> tuple[np.ndarray, tuple[int, int], float]:
    """Moments-affine the atlas plane onto the tissue mask, best sign wins.

    Returns ``(placed labels, sign pattern, tissue IoU)``. The fit is the
    shared silhouette core (:func:`langslice.affine._affine_from_pose`):
    centroids for translation, second-moment eigenvectors for rotation and
    principal axes, eigenvalue ratios for scale. Only the two rotations are
    tried; the one whose placed foreground overlaps the tissue better wins.
    """
    placed, signs, iou, _matrix = place_plane_on_tissue_with_matrix(labels, tissue)
    return placed, signs, iou


def place_plane_on_tissue_with_matrix(
    labels: np.ndarray, tissue: np.ndarray
) -> tuple[np.ndarray, tuple[int, int], float, np.ndarray]:
    """Return the moments placement and its native-label→tissue 3x3 affine.

    Keeping this initial map lets nonlinear residuals compose with placement
    when exporting correspondences. The original three-result API is retained.
    """
    if not np.any(labels) or not np.any(tissue):
        raise ValueError("Atlas and tissue silhouettes must both contain foreground")
    src = _moments_pose((labels != 0).astype(np.uint8) * 255)
    dst = _moments_pose(tissue)
    height, width = tissue.shape
    best: tuple[np.ndarray, tuple[int, int], float, np.ndarray] | None = None
    for signs in _SIGN_PATTERNS:
        matrix = _affine_from_pose(*src, *dst, signs)
        # float64 through cv2: it cannot warp integer label dtypes, and every
        # Allen id round-trips exactly through a double.
        placed = cv2.warpAffine(
            labels.astype(np.float64),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        ).astype(labels.dtype)
        iou = silhouette_iou(placed != 0, tissue > 0)
        if best is None or iou > best[2]:
            best = (placed, signs, iou, np.vstack([matrix, [0.0, 0.0, 1.0]]))
    assert best is not None
    return best


