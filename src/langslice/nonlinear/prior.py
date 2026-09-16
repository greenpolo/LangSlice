"""The silhouette prior: the atlas plane placed on the tissue, painted.

A model-free first guess at the registration. The atlas plane at
(``position_mm``, ``pitch_deg``, ``yaw_deg``) is placed onto the section's
own silhouette by a closed-form moments fit — centroid, principal axes,
axis spreads — and painted exactly like the model-facing atlas references
(flat palette fills, ventricles black, darker fill-change lines). It is a
LINEAR placement: the boundaries are the atlas's, moved rigidly onto the
tissue, not deformed onto it.

Measured against the LSD_910 hand registrations: the placement alone scores
0.82 mean family dice, better than every image-model configuration tried,
and handing it to the image model as the canvas to correct raises the
painting floor from ~0.5 to ~0.8. Hence the two uses in
:mod:`langslice.nonlinear.image_gen_registration`:

* ``provider="none"`` — the prior IS the painting (the model-free backbone).
* ``init="silhouette"`` with a real provider — the prior is the canvas the
  model edits, with the tissue alongside it as the reference.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from langslice.affine import _affine_from_pose, _moments_pose, silhouette_iou
from langslice.atlas.recolor import active_palette, color_lut
from langslice.image_prep import foreground_mask
from langslice.space import Plane

#: Sign patterns tried for the source eigenvectors: pure ROTATIONS only.
#: The two reflections score the same silhouette IoU on a near-symmetric
#: section and land the anatomy on the wrong hemisphere.
_SIGN_PATTERNS: tuple[tuple[int, int], ...] = ((1, 1), (-1, -1))

#: Soft-label smoothing sigma as a fraction of the canvas long edge — just
#: enough to take the voxel staircase off a 25um plane upscaled to canvas.
_SMOOTH_SIGMA_FRACTION = 1.0 / 400.0


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


def smooth_labels(labels: np.ndarray, sigma: float) -> np.ndarray:
    """Soft-label smoothing: blur each label's mask, take the per-pixel argmax.

    The plane arrives at atlas resolution and is warped onto a canvas many
    times larger, so every boundary is a voxel staircase. Blurring the
    one-hot masks and arg-maxing them puts each boundary back on the smooth
    curve the staircase samples, without ever blending two ids into a third.
    """
    if sigma <= 0:
        return labels
    from scipy import ndimage

    # ponytail: one pass per label, keeping only the running best — the
    # stacked version needs (labels x pixels) floats, ~1 GB on a real canvas.
    out = np.zeros_like(labels)
    best = np.zeros(labels.shape, dtype=np.float32)
    for uid in np.unique(labels):
        weight = ndimage.gaussian_filter((labels == uid).astype(np.float32), sigma)
        take = weight > best
        out[take] = uid
        best[take] = weight[take]
    return out


def build_silhouette_prior(
    canvas: Image.Image,
    *,
    atlas: Any,
    position_mm: float,
    plane: Plane = "coronal",
    image_axes: str | None = None,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> tuple[Image.Image, dict[str, Any]]:
    """The atlas plane placed on *canvas*'s tissue and painted, canvas-sized.

    *canvas* is the working canvas (already padded and aspect-snapped), and
    the prior comes back in exactly that frame, so it can be sent as the
    image to edit or registered as-is. The paint is
    :func:`langslice.nonlinear.render.paint_labels` — the same call the
    model-facing atlas reference makes — so every pixel is an exact palette
    color and classifies back losslessly.
    """
    from langslice.nonlinear.image_gen_helpers import _annotation_slice, line_width_px
    from langslice.nonlinear.render import paint_labels

    size = canvas.size
    tissue = tissue_mask(canvas, size)
    labels = _annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    if image_axes:
        from langslice.space import atlas_space_context, orient_slice_to_axes

        labels = orient_slice_to_axes(labels, atlas_space_context(atlas), plane, image_axes)
    placed, signs, iou = place_plane_on_tissue(labels, tissue)
    placed = smooth_labels(placed, sigma=max(size) * _SMOOTH_SIGMA_FRACTION)
    rgb = paint_labels(
        placed,
        lut=color_lut(atlas),
        line_px=line_width_px(max(size)),
        leaf_lines=active_palette() == "leaf-borders",
    )
    metadata = {
        "sign_pattern": list(signs),
        "tissue_iou": round(float(iou), 4),
        "tissue_fraction": round(float((tissue > 0).mean()), 4),
    }
    return Image.fromarray(rgb, mode="RGB"), metadata
