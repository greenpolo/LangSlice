"""Silhouette-based affine preview registration for the GUI 3D viewer.

When the user locks an AP position, we want a brain-silhouette warped slice
to appear in the 3D viewer in well under a second — long before the full
nano-banana / Elastix pipeline finishes. Previous intensity-based Elastix
attempt was both slow (~15s cold-start) and fragile across modalities (the
NCC metric chases noise when comparing histology staining patterns to atlas
Nissl). Both problems vanish if we register *shapes* instead of intensities.

Pipeline:
    1. Extract a filled binary tissue silhouette from the slice
       (Otsu → polarity from corner pixels → fill holes → largest CC).
    2. Build the atlas root silhouette at the chosen plane/position
       (already exists as ``_build_atlas_root_mask``).
    3. Compute the closed-form affine that aligns the two silhouettes via
       image moments — centroid for translation, eigenvectors of the
       second-moment matrix for rotation/principal axes, eigenvalue ratios
       for scale.
    4. Resolve the 4-way sign ambiguity on the eigenvectors (rotations vs
       reflections) by picking the candidate with the highest silhouette
       IoU after warping.
    5. Apply the chosen 2×3 affine to the original slice RGB with
       ``cv2.warpAffine`` and use the atlas root silhouette as the alpha
       channel so the 3D viewer renders a brain-shaped sheet.

Cost: ~150ms in a warm Python process, no itk, no Elastix.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from langslice_harness.atlas import load_atlas
from langslice_harness.atlas.space import Plane
from langslice_harness.harness.registration.image_gen_helpers import (
    _build_atlas_root_mask,
)

logger = logging.getLogger(__name__)

# Long-edge target for silhouette ops. The warp is applied at this resolution
# (output PNG is target_size). Keep modest — silhouette alignment doesn't
# need high resolution, and the 3D viewer textures upload faster.
_TARGET_LONG_EDGE = 512

# Sanity bounds on the tissue silhouette. Outside this band, Otsu probably
# failed (uniform image, blank field) and we should error rather than emit
# a garbage preview.
_MIN_AREA_FRAC = 0.05
_MAX_AREA_FRAC = 0.90

# All four sign patterns for the source eigenvectors. (1,1) and (-1,-1) keep
# det(R) positive (pure rotations); (-1,1) and (1,-1) flip it (reflections).
# We try all four and pick the best by silhouette IoU.
_SIGN_PATTERNS: tuple[tuple[int, int], ...] = ((1, 1), (-1, 1), (1, -1), (-1, -1))


def _resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    width, height = image.size
    current = max(width, height)
    if current == long_edge:
        return image
    scale = long_edge / float(current)
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=Image.Resampling.LANCZOS,
    )


def _extract_slice_silhouette(image_gray: np.ndarray) -> np.ndarray:
    """Extract a filled binary tissue silhouette from a grayscale histology slice.

    Otsu picks the threshold, corner-pixel sampling resolves
    foreground/background polarity (histology can be dark-on-light or
    light-on-dark depending on stain), holes get filled, and only the
    largest connected component survives so we drop staining debris."""
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
    filled = binary_fill_holes(binary > 0).astype(np.uint8) * 255

    # Keep only the largest non-background CC — drops debris and staining
    # specks that survive thresholding.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if num <= 1:
        return filled
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8) * 255


def _moments_pose(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (centroid, eigenvalues_desc, eigenvectors_cols) of a binary mask.

    Centroid is in (x, y) image coordinates (col, row). Eigenvalues are sorted
    descending so column 0 of the eigenvector matrix is the major axis."""
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
    """Closed-form 2×3 affine that maps src silhouette → dst silhouette.

    Derivation: project the source into its principal-axis frame
    (V_src^T), scale each axis by sqrt(λ_dst / λ_src) so the variance
    matches, rotate back into the destination's frame (V_dst), then
    translate so centroids coincide. ``sign_pattern`` flips columns of
    V_src to explore the 4 sign-ambiguity cases."""
    V_src_signed = src_V * np.array(sign_pattern)  # broadcast as row → scales columns
    scale = np.sqrt(np.maximum(dst_eigvals, 1e-9) / np.maximum(src_eigvals, 1e-9))
    R = dst_V @ np.diag(scale) @ V_src_signed.T
    t = dst_c - R @ src_c
    return np.array([
        [R[0, 0], R[0, 1], t[0]],
        [R[1, 0], R[1, 1], t[1]],
    ], dtype=np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a > 0
    b_bool = b > 0
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    return inter / union if union > 0 else 0.0


def quick_affine_register(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    plane: Plane = "coronal",
    out_path: Path,
) -> dict[str, Any]:
    """Silhouette-based affine alignment of slice → atlas root.

    Returns ``{warped_slice_path, elapsed_s, silhouette_iou}``. The warped
    PNG is RGBA at ~512px long-edge, alpha = atlas root silhouette, so the
    3D viewer can drop it in unchanged."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    atlas = load_atlas(atlas_name)

    # Downsample the slice for silhouette extraction; warp output ends up at
    # this resolution. _build_atlas_root_mask resizes to match.
    slice_pil = _resize_long_edge(image.convert("RGB"), _TARGET_LONG_EDGE)
    target_size = slice_pil.size  # (w, h)
    slice_rgb = np.asarray(slice_pil, dtype=np.uint8)
    slice_gray = cv2.cvtColor(slice_rgb, cv2.COLOR_RGB2GRAY)

    slice_mask = _extract_slice_silhouette(slice_gray)
    area_frac = float((slice_mask > 0).sum()) / slice_mask.size
    if area_frac < _MIN_AREA_FRAC or area_frac > _MAX_AREA_FRAC:
        raise ValueError(
            f"Slice silhouette area fraction {area_frac:.2%} out of range "
            f"[{_MIN_AREA_FRAC:.0%}, {_MAX_AREA_FRAC:.0%}] — Otsu likely failed."
        )

    atlas_mask = _build_atlas_root_mask(atlas, position_mm, target_size, plane=plane)

    src_c, src_eigvals, src_V = _moments_pose(slice_mask)
    dst_c, dst_eigvals, dst_V = _moments_pose(atlas_mask)

    h, w = atlas_mask.shape
    best_iou = -1.0
    best_affine: np.ndarray | None = None
    best_pattern: tuple[int, int] | None = None
    for sign_pattern in _SIGN_PATTERNS:
        candidate = _affine_from_pose(
            src_c, src_eigvals, src_V,
            dst_c, dst_eigvals, dst_V,
            sign_pattern,
        )
        warped_mask = cv2.warpAffine(
            slice_mask, candidate, (w, h),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
        score = _iou(warped_mask, atlas_mask)
        if score > best_iou:
            best_iou = score
            best_affine = candidate
            best_pattern = sign_pattern

    assert best_affine is not None

    # Apply the winning affine to the RGB slice with linear interpolation for
    # smoothness; clip to the atlas silhouette via alpha so the 3D viewer
    # renders a brain-shaped sheet rather than a rectangular slab.
    warped_rgb = cv2.warpAffine(
        slice_rgb, best_affine, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_rgba = np.dstack([warped_rgb, atlas_mask])
    Image.fromarray(warped_rgba, mode="RGBA").save(out_path)

    elapsed = time.perf_counter() - started
    logger.info(
        "quick_affine_register: %s @ %.2fmm (%s) IoU=%.3f sign=%s -> %s in %.2fs",
        atlas_name, position_mm, plane, best_iou, best_pattern, out_path, elapsed,
    )
    return {
        "warped_slice_path": str(out_path.resolve()),
        "elapsed_s": round(elapsed, 2),
        "silhouette_iou": round(best_iou, 3),
    }
