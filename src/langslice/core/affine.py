"""In-plane affine alignment of one histology section to one atlas section.

The linear ``fit_affine`` tool's silhouette method records
:func:`silhouette_affine`'s parameters as the proposed affine for an intact
section; the matrix builders and the normalized six-number convention are
shared by every reader of a stored transform. Pure
functions only — no CLI, no agent, no model calls, no file writes.

Two ways to get a 2x3 affine here:

* :func:`silhouette_affine` — closed-form fit from image moments. Registers
  SHAPES, not intensities: an Otsu silhouette of the tissue against the atlas
  root silhouette, centroid for translation, second-moment eigenvectors for
  rotation and principal axes, eigenvalue ratios for scale. The 4-way sign
  ambiguity on the eigenvectors (rotations vs reflections) is resolved by
  picking the candidate with the best silhouette IoU. ~150 ms warm, no
  Elastix, no itk. :func:`mask_affine` is its core, taking two prepared masks
  in one frame — what an ROI-restricted fit hands it.
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from langslice.core.atlas.core import get_root_mask
from langslice.core.space import Plane

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


#: The identity's six normalized numbers (:func:`normalized_affine`): what a
#: section without a transform is drawn and mapped with.
IDENTITY_PARAMS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def pixel_center_map(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """3x3 map of pixel centres from one grid onto a resized (and shifted) grid.

    Pixel centres sit at integer indices (OpenCV's convention), so a resize
    by ``s`` maps ``x`` to ``s * x + (s - 1) / 2``; each axis keeps its own
    factor (resize rounding), and *offset* adds padding or a placement on a
    larger canvas, in target pixels. The one copy: the image tool's canvas,
    the nonlinear canvas and the deformable working grid all use it.
    """
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    return np.array([
        [sx, 0.0, offset[0] + (sx - 1.0) / 2.0],
        [0.0, sy, offset[1] + (sy - 1.0) / 2.0],
        [0.0, 0.0, 1.0],
    ])


def coerce_affine_matrix(matrix: Any) -> np.ndarray:
    """*matrix* as a finite 3x3 float64 array (a copy); ``ValueError`` otherwise."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine matrix, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Affine matrix contains non-finite values")
    return arr.copy()


def apply_affine_to_points(matrix: Any, points: Any) -> np.ndarray:
    """A homogeneous 3x3 affine applied to ``(N, 2)`` points."""
    arr = coerce_affine_matrix(matrix)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {pts.shape}")
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (arr @ homogeneous.T).T[:, :2]


def affine_matrix_from_legacy_params(
    image_width: int,
    image_height: int,
    rotation_deg: float = 0.0,
    translate_x_pct: float = 0.0,
    translate_y_pct: float = 0.0,
) -> np.ndarray:
    """A rotation about the image centre plus a shift in percent of the image
    size, as a 3x3 matrix (the QuickNII export's knob form)."""
    cx, cy = float(image_width) / 2.0, float(image_height) / 2.0
    tx = float(image_width) * (translate_x_pct / 100.0)
    ty = float(image_height) * (translate_y_pct / 100.0)
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def shift(x: float, y: float) -> np.ndarray:
        return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])

    rotation = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    return shift(cx + tx, cy + ty) @ rotation @ shift(-cx, -cy)


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


def axis_ratio(mask: np.ndarray) -> float:
    """Long over short principal axis of a binary mask (1.0: no preferred axis).

    The moments fit turns the section so the two masks' long axes meet; near
    1.0 that axis, and so the fitted rotation, is set by small outline noise.
    """
    _centre, eigvals, _vectors = _moments_pose(mask)
    return float(np.sqrt(max(eigvals[0], 1e-12) / max(eigvals[1], 1e-12)))


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


def mask_affine(
    src_mask: np.ndarray, dst_mask: np.ndarray
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """The moments fit of one binary mask onto another: ``(2x3, iou, pattern)``.

    The core both silhouette routes share — the whole-tissue fit below and the
    linear tool's ROI-restricted variant, which prepares its own two masks on
    the physical canvas. Both masks live in the SAME frame; the affine maps
    *src_mask* pixels to *dst_mask* pixels.

    Raises ``ValueError`` when either mask is empty or no candidate survives.
    """
    src_c, src_eigvals, src_V = _moments_pose(src_mask)
    dst_c, dst_eigvals, dst_V = _moments_pose(dst_mask)

    h, w = dst_mask.shape
    best_iou = -1.0
    best_affine: np.ndarray | None = None
    best_pattern: tuple[int, int] = (1, 1)
    for sign_pattern in _SIGN_PATTERNS:
        candidate = _affine_from_pose(
            src_c, src_eigvals, src_V,
            dst_c, dst_eigvals, dst_V,
            sign_pattern,
        )
        # Reflections are not the fit's to make: a mirrored section is an
        # ORIENTATION correction (the section's flip flag), and the atlas
        # silhouette is left-right symmetric anyway, so a reflected candidate
        # ties the proper one on IoU and would win by handedness noise.
        if np.linalg.det(candidate[:, :2]) <= 0:
            continue
        warped_mask = cv2.warpAffine(
            src_mask, candidate, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        score = silhouette_iou(warped_mask, dst_mask)
        if score > best_iou:
            best_iou = score
            best_affine = candidate
            best_pattern = sign_pattern

    if best_affine is None:
        raise ValueError("No affine candidate could be computed.")
    return best_affine, best_iou, best_pattern


def tissue_silhouette(
    image: Image.Image, long_edge: int = AFFINE_LONG_EDGE
) -> tuple[np.ndarray, np.ndarray]:
    """``(mask, rgb)``: the section's filled tissue silhouette in the fit frame.

    The section resized to *long_edge* (the frame every silhouette fit works
    in) and its Otsu silhouette, 0/255. Raises ``ValueError`` when the
    silhouette is implausible (Otsu failed on a blank or uniform field).
    """
    slice_pil = resize_long_edge(image.convert("RGB"), long_edge)
    slice_rgb = np.asarray(slice_pil, dtype=np.uint8)
    slice_gray = cv2.cvtColor(slice_rgb, cv2.COLOR_RGB2GRAY)

    slice_mask = extract_slice_silhouette(slice_gray)
    area_frac = float((slice_mask > 0).sum()) / slice_mask.size
    if area_frac < _MIN_AREA_FRAC or area_frac > _MAX_AREA_FRAC:
        raise ValueError(
            f"Slice silhouette area fraction {area_frac:.2%} out of range "
            f"[{_MIN_AREA_FRAC:.0%}, {_MAX_AREA_FRAC:.0%}] — Otsu likely failed."
        )
    return slice_mask, slice_rgb


def silhouette_affine(
    image: Image.Image,
    *,
    atlas: Any,
    position_mm: float,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    long_edge: int = AFFINE_LONG_EDGE,
    atlas_mask_at: Callable[[tuple[int, int]], np.ndarray] | None = None,
) -> SilhouetteFit:
    """Fit a 2x3 affine aligning *image* to the atlas section at *position_mm*.

    The one silhouette wrapper: the linear ``fit_affine`` silhouette method
    (``core.transform.fit_silhouette``) calls it. The atlas tissue silhouette is
    the root mask of the plane at *position_mm* and the cutting angles
    (:func:`langslice.core.atlas.core.get_root_mask`), so an angled stack is
    measured against the plane every other picture in the run shows (until
    2026-09-10 it measured against the flat plane on a 13-degree brain and
    said so with ``flat_atlas_fit``). *atlas_mask_at* (a ``(w, h)`` size ->
    mask) replaces that silhouette with the caller's own.

    Raises ``ValueError`` when the tissue silhouette is implausible (Otsu
    failed on a blank or uniform field) or no candidate could be computed.
    """
    slice_mask, slice_rgb = tissue_silhouette(image, long_edge)
    size = (slice_mask.shape[1], slice_mask.shape[0])  # (w, h)

    atlas_mask = (
        atlas_mask_at(size)
        if atlas_mask_at is not None
        else get_root_mask(atlas, position_mm, size, plane=plane,
                           pitch_deg=pitch_deg, yaw_deg=yaw_deg)
    )
    matrix, iou, pattern = mask_affine(slice_mask, atlas_mask)

    return SilhouetteFit(
        matrix=matrix,
        iou=iou,
        size=size,
        slice_rgb=slice_rgb,
        atlas_mask=atlas_mask,
        sign_pattern=pattern,
    )


def affine_matrix(
    *,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    translate_x: float,
    translate_y: float,
    size: tuple[int, int],
    pivot: tuple[float, float] | None = None,
    shear: float = 0.0,
) -> np.ndarray:
    """A 2x3 affine from human knobs, about the centre of a *size* image.

    Rotation is counter-clockwise on screen (the OpenCV convention), scales
    are multipliers per axis applied before the rotation, and translations are
    FRACTIONS of image width/height — positive x moves right, positive y moves
    down. Identity is ``rotation_deg=0, scale=1, translate=0, shear=0``.

    *shear* is :func:`decompose_affine`'s: the linear part is
    ``R(rotation) . [[scale_x, shear * scale_x], [0, scale_y]]``, so before
    the rotation a point ``(x, y)`` (from the pivot) moves sideways by
    ``shear * y``, in units of ``scale_x``: a dimensionless slant, ``0.1``
    shifting each row by a tenth of its distance below the pivot. The knobs
    and :func:`decompose_affine` round-trip exactly (up to its rounding).

    *pivot* moves the point rotation and scale happen about, in PIXELS of the
    same frame *size* describes; ``None`` is the frame's centre. The
    translation is unaffected — it is applied after, whatever the pivot.
    """
    width, height = size
    cx, cy = (width / 2.0, height / 2.0) if pivot is None else (float(pivot[0]), float(pivot[1]))
    rad = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    a, b = cos_t * scale_x, sin_t * scale_y
    c, d = -sin_t * scale_x, cos_t * scale_y
    if shear:
        b += cos_t * shear * scale_x
        d -= sin_t * shear * scale_x
    return np.array([
        [a, b, cx - (a * cx + b * cy) + translate_x * width],
        [c, d, cy - (c * cx + d * cy) + translate_y * height],
    ], dtype=np.float64)


def physical_affine_matrix(
    *,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    translate_x_mm: float,
    translate_y_mm: float,
    size: tuple[int, int],
    um_per_px: float,
    pivot: tuple[float, float] | None = None,
    shear: float = 0.0,
) -> np.ndarray:
    """:func:`affine_matrix` with the shifts given in MILLIMETRES.

    The registration-software convention (ABBA's): rotation in degrees about
    the frame's centre, unitless per-axis scales, translations in mm on a
    frame whose pixels are *um_per_px* micrometres wide. One millimetre is
    ``1000 / um_per_px`` pixels, whatever the frame's size — which is the
    whole point of calibrating: the same numbers mean the same displacement
    at any working resolution.

    *pivot* is the rotation/scale centre in pixels of *size*; ``None`` is the
    frame's centre. Since the frame's width and height cancel out of the
    translation, the SAME numbers build the same map on the section's frame
    and on the padded canvas around it — as long as the pivot is expressed in
    whichever frame *size* names.
    """
    if um_per_px <= 0:
        raise ValueError("um_per_px must be positive")
    px_per_mm = 1000.0 / float(um_per_px)
    width, height = size
    return affine_matrix(
        rotation_deg=rotation_deg,
        scale_x=scale_x,
        scale_y=scale_y,
        translate_x=translate_x_mm * px_per_mm / width,
        translate_y=translate_y_mm * px_per_mm / height,
        size=size,
        pivot=pivot,
        shear=shear,
    )


def normalized_physical_affine(
    *,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    translate_x_mm: float,
    translate_y_mm: float,
    size: tuple[int, int],
    um_per_px: float,
    pivot: tuple[float, float] | None = None,
    shear: float = 0.0,
) -> list[float]:
    """Physical parameters as the six normalized numbers hosts consume.

    *size* is the SECTION's own frame, not the padded canvas the parameters
    were chosen on: the pad is symmetric, so both frames share a centre, and
    a rotation about the canvas centre is the same map as a rotation about
    the section centre. Only the frame the numbers are expressed in changes.

    That equivalence is exactly what a *pivot* breaks, so a pivot chosen on
    the canvas must be handed over in the SECTION's frame (subtract the
    canvas's section offset). The six numbers then describe the same map the
    canvas showed, still on the section's own frame.
    """
    return normalized_affine(
        physical_affine_matrix(
            rotation_deg=rotation_deg,
            scale_x=scale_x,
            scale_y=scale_y,
            translate_x_mm=translate_x_mm,
            translate_y_mm=translate_y_mm,
            size=size,
            um_per_px=um_per_px,
            pivot=pivot,
            shear=shear,
        ),
        size,
    )


def decompose_affine(
    params_or_matrix: Any, size: tuple[int, int] | None = None
) -> dict[str, Any]:
    """A 2x3 affine as the numbers a human reads: rotation, scale, shear, shift.

    Accepts either the normalized 6-vector ``[a, b, tx, c, d, ty]`` or a 2x3
    matrix. The decomposition is ``M = R(rotation) . [[sx, shear*sx], [0, sy]]``
    read back out by QR: ``scale_x`` is the length of the first column,
    ``rotation_deg`` its angle (counter-clockwise on screen, the
    :func:`affine_matrix` convention), ``shear`` the residual x/y coupling in
    units of ``scale_x``, and ``scale_y`` the determinant over ``scale_x`` —
    so it goes NEGATIVE exactly when the map includes a reflection.
    ``mirrored`` reports that determinant sign on its own.

    A normalized 6-vector lives in a fractional frame (x/width, y/height),
    which is anisotropic on a non-square image: a pure 5-degree rotation reads
    as 7 degrees plus shear there. Pass *size* to decompose in the pixel frame
    instead — angles and scales then mean what they say.

    Translations come back as the input's own ``tx``/``ty``, which for the
    normalized 6-vector every payload passes are FRACTIONS of width and
    height. Hence the names: a 0.15 mm shift on a 20 mm-wide frame reads as
    ``translate_x_frac = 0.0075``, and reading that as millimetres is exactly
    the confusion the suffix exists to stop.
    """
    values = np.asarray(params_or_matrix, dtype=np.float64).reshape(2, 3)
    (a, b, tx), (c, d, ty) = values[0], values[1]
    if size is not None:
        width, height = float(size[0]), float(size[1])
        b, c = b * width / height, c * height / width
    scale_x = float(math.hypot(a, c))
    determinant = float(a * d - b * c)
    rotation = math.degrees(math.atan2(-c, a)) if scale_x > 0 else 0.0
    shear = float((a * b + c * d) / (scale_x**2)) if scale_x > 0 else 0.0
    scale_y = determinant / scale_x if scale_x > 0 else 0.0
    return {
        "rotation_deg": round(rotation, 3),
        "scale_x": round(scale_x, 4),
        "scale_y": round(scale_y, 4),
        "shear": round(shear, 4),
        "translate_x_frac": round(float(tx), 4),
        "translate_y_frac": round(float(ty), 4),
        "mirrored": determinant < 0,
    }


def denormalized_affine(params: Any, size: tuple[int, int]) -> np.ndarray:
    """The inverse of :func:`normalized_affine`: six numbers back to pixels.

    What a stored transform has to go through to be DRAWN again — the six
    numbers are exact, where the reported knobs (shear included) are
    rounded, so a picture of "what is stored" is built from these and not
    from the knobs.
    """
    a, b, tx, c, d, ty = (float(v) for v in params)
    width, height = float(size[0]), float(size[1])
    return np.array(
        [[a, b * width / height, tx * width], [c * height / width, d, ty * height]],
        dtype=np.float64,
    )


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
