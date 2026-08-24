"""In-plane affine alignment of one histology section to one atlas section.

Shared by both methods, which is why it sits at the top level:
:mod:`langslice.nonlinear.quick_affine` wraps :func:`silhouette_affine` into a
warped RGBA preview for the 3D viewer, and the whole-brain ``transforms`` step
records its parameters as the proposed affine for an intact section. Pure
functions only — no CLI, no agent, no model calls, no file writes.

Two ways to get a 2x3 affine here:

* :func:`silhouette_affine` — closed-form fit from image moments. Registers
  SHAPES, not intensities: an Otsu silhouette of the tissue against the atlas
  root silhouette, centroid for translation, second-moment eigenvectors for
  rotation and principal axes, eigenvalue ratios for scale. The 4-way sign
  ambiguity on the eigenvectors (rotations vs reflections) is resolved by
  picking the candidate with the best silhouette IoU. ~150 ms warm, no
  Elastix, no itk.
* :func:`affine_matrix` — the same 2x3 built from human-readable knobs
  (rotation, per-axis scale, translation), which is what the interactive
  transform loop proposes.

PARAMETER CONVENTION. Matrices are OpenCV 2x3 row-major, in PIXELS of a
working frame::

    [x_dst]   [a  b  tx] [x_src]
    [y_dst] = [c  d  ty] [y_src]
                         [  1  ]

:func:`normalized_affine` converts one to the resolution-independent 6-vector
``[a, b, tx, c, d, ty]`` that gets handed back to hosts, where x is a fraction
of image width and y a fraction of image height, so applying it to the user's
full-resolution section needs nothing but the section's own size.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from langslice.atlas.core import get_root_mask
from langslice.space import Plane

logger = logging.getLogger(__name__)

#: Long-edge target for silhouette ops. Alignment of a brain outline does not
#: need resolution, and everything downstream is cheaper at 512.
AFFINE_LONG_EDGE = 512

# Sanity bounds on the tissue silhouette. Outside this band Otsu probably
# failed (uniform image, blank field) and a garbage fit is worse than an error.
_MIN_AREA_FRAC = 0.05
_MAX_AREA_FRAC = 0.90

# All four sign patterns for the source eigenvectors. (1,1) and (-1,-1) keep
# det(R) positive (pure rotations); (-1,1) and (1,-1) flip it (reflections).
# We try all four and pick the best by silhouette IoU.
_SIGN_PATTERNS: tuple[tuple[int, int], ...] = ((1, 1), (-1, 1), (1, -1), (-1, -1))


@dataclass
class SilhouetteFit:
    """One silhouette affine plus the pixels it was computed from.

    ``matrix`` maps slice pixels to atlas pixels within ``size``; both the
    resized slice and the atlas mask live in that frame.
    """

    matrix: np.ndarray
    iou: float
    size: tuple[int, int]
    slice_rgb: np.ndarray
    atlas_mask: np.ndarray
    sign_pattern: tuple[int, int]


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    """Scale *image* so its long edge is exactly *long_edge* px."""
    width, height = image.size
    current = max(width, height)
    if current == long_edge:
        return image
    scale = long_edge / float(current)
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=Image.Resampling.LANCZOS,
    )


def extract_slice_silhouette(image_gray: np.ndarray) -> np.ndarray:
    """Filled binary tissue silhouette of a grayscale histology section.

    Otsu picks the threshold, corner-pixel sampling resolves
    foreground/background polarity (histology can be dark-on-light or
    light-on-dark depending on stain), holes get filled, and only the largest
    connected component survives so staining debris drops out.
    """
    _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Corner-patch majority vote decides polarity. Background is whichever
    # class dominates the corners. Averaging four corners is more robust than
    # any one in case the tissue happens to touch a corner.
    h, w = binary.shape
    patch = max(4, min(h, w) // 32)
    corners = np.concatenate([
        binary[:patch, :patch].ravel(),
        binary[:patch, -patch:].ravel(),
        binary[-patch:, :patch].ravel(),
        binary[-patch:, -patch:].ravel(),
    ])
    if corners.mean() > 127:
        binary = 255 - binary

    # Fill ventricles, white-matter gaps etc. that Otsu classed as background.
    filled_bool = cast(np.ndarray, binary_fill_holes(binary > 0))
    filled = filled_bool.astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if num <= 1:
        return filled
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8) * 255


def _moments_pose(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centroid, eigenvalues_desc, eigenvectors_cols) of a binary mask.

    Centroid is in (x, y) image coordinates (col, row). Eigenvalues are sorted
    descending so column 0 of the eigenvector matrix is the major axis.
    """
    M = cv2.moments(mask, binaryImage=True)
    m00 = M["m00"]
    if m00 < 1e-6:
        raise ValueError("Empty mask — cannot compute moments.")
    cx = M["m10"] / m00
    cy = M["m01"] / m00
    cov = np.array([
        [M["mu20"] / m00, M["mu11"] / m00],
        [M["mu11"] / m00, M["mu02"] / m00],
    ])
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]  # cols reordered to match
    return np.array([cx, cy]), eigvals, eigvecs


def _affine_from_pose(
    src_c: np.ndarray, src_eigvals: np.ndarray, src_V: np.ndarray,
    dst_c: np.ndarray, dst_eigvals: np.ndarray, dst_V: np.ndarray,
    sign_pattern: tuple[int, int],
) -> np.ndarray:
    """Closed-form 2x3 affine that maps src silhouette -> dst silhouette.

    Derivation: project the source into its principal-axis frame (V_src^T),
    scale each axis by sqrt(lambda_dst / lambda_src) so the variance matches,
    rotate into the destination's frame (V_dst), then translate so centroids
    coincide. ``sign_pattern`` flips columns of V_src to explore the 4
    sign-ambiguity cases.
    """
    V_src_signed = src_V * np.array(sign_pattern)  # broadcast as row -> scales cols
    scale = np.sqrt(np.maximum(dst_eigvals, 1e-9) / np.maximum(src_eigvals, 1e-9))
    R = dst_V @ np.diag(scale) @ V_src_signed.T
    t = dst_c - R @ src_c
    return np.array([
        [R[0, 0], R[0, 1], t[0]],
        [R[1, 0], R[1, 1], t[1]],
    ], dtype=np.float64)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks."""
    a_bool = a > 0
    b_bool = b > 0
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    return inter / union if union > 0 else 0.0


def silhouette_affine(
    image: Image.Image,
    *,
    atlas: Any,
    position_mm: float,
    plane: Plane = "coronal",
    long_edge: int = AFFINE_LONG_EDGE,
) -> SilhouetteFit:
    """Fit a 2x3 affine aligning *image* to the atlas section at *position_mm*.

    Raises ``ValueError`` when the tissue silhouette is implausible (Otsu
    failed on a blank or uniform field) or no candidate could be computed.
    """
    slice_pil = resize_long_edge(image.convert("RGB"), long_edge)
    size = slice_pil.size  # (w, h)
    slice_rgb = np.asarray(slice_pil, dtype=np.uint8)
    slice_gray = cv2.cvtColor(slice_rgb, cv2.COLOR_RGB2GRAY)

    slice_mask = extract_slice_silhouette(slice_gray)
    area_frac = float((slice_mask > 0).sum()) / slice_mask.size
    if area_frac < _MIN_AREA_FRAC or area_frac > _MAX_AREA_FRAC:
        raise ValueError(
            f"Slice silhouette area fraction {area_frac:.2%} out of range "
            f"[{_MIN_AREA_FRAC:.0%}, {_MAX_AREA_FRAC:.0%}] — Otsu likely failed."
        )

    atlas_mask = get_root_mask(atlas, position_mm, size, plane=plane)

    src_c, src_eigvals, src_V = _moments_pose(slice_mask)
    dst_c, dst_eigvals, dst_V = _moments_pose(atlas_mask)

    h, w = atlas_mask.shape
    best_iou = -1.0
    best_affine: np.ndarray | None = None
    best_pattern: tuple[int, int] = (1, 1)
    for sign_pattern in _SIGN_PATTERNS:
        candidate = _affine_from_pose(
            src_c, src_eigvals, src_V,
            dst_c, dst_eigvals, dst_V,
            sign_pattern,
        )
        warped_mask = cv2.warpAffine(
            slice_mask, candidate, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        score = silhouette_iou(warped_mask, atlas_mask)
        if score > best_iou:
            best_iou = score
            best_affine = candidate
            best_pattern = sign_pattern

    if best_affine is None:
        raise ValueError("No affine candidate could be computed.")

    return SilhouetteFit(
        matrix=best_affine,
        iou=best_iou,
        size=size,
        slice_rgb=slice_rgb,
        atlas_mask=atlas_mask,
        sign_pattern=best_pattern,
    )


def affine_matrix(
    *,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    translate_x: float,
    translate_y: float,
    size: tuple[int, int],
) -> np.ndarray:
    """A 2x3 affine from human knobs, about the centre of a *size* image.

    Rotation is counter-clockwise on screen (the OpenCV convention), scales
    are multipliers per axis applied before the rotation, and translations are
    FRACTIONS of image width/height — positive x moves right, positive y moves
    down. Identity is ``rotation_deg=0, scale=1, translate=0``.
    """
    width, height = size
    cx, cy = width / 2.0, height / 2.0
    rad = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    a, b = cos_t * scale_x, sin_t * scale_y
    c, d = -sin_t * scale_x, cos_t * scale_y
    return np.array([
        [a, b, cx - (a * cx + b * cy) + translate_x * width],
        [c, d, cy - (c * cx + d * cy) + translate_y * height],
    ], dtype=np.float64)


def normalized_affine(matrix: np.ndarray, size: tuple[int, int]) -> list[float]:
    """A pixel-space 2x3 as the resolution-independent ``[a, b, tx, c, d, ty]``.

    Both source and destination are expressed in fractions of *size*, so the
    result can be applied to the section at any resolution. Source and
    destination frames are assumed to share *size* — which they do, because
    the atlas mask is built at the resized section's size.
    """
    width, height = size
    (a, b, tx), (c, d, ty) = matrix[0], matrix[1]
    return [
        float(a),
        float(b) * height / width,
        float(tx) / width,
        float(c) * width / height,
        float(d),
        float(ty) / height,
    ]
