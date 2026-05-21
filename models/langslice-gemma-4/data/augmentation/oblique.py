"""Oblique atlas slice extraction — tilted coronal/sagittal/horizontal sections.

Real histological sections are rarely cut perfectly orthogonal to the atlas
axes.  Yaw, pitch, and roll tilts of up to ~8° are common artefacts of
mounting, sectioning angle, and brain positioning.  This module extracts
tilted slices from the 3D atlas volumes by computing per-output-pixel sample
coordinates via a 3D rotation matrix and then sampling with
``scipy.ndimage.map_coordinates``.

Public API
----------
get_oblique_slice(atlas, *, base_position_mm, plane, yaw_deg, pitch_deg,
                  roll_deg, output_size)
    Extract one tilted reference + annotation slice pair.

sample_oblique_angles(rng, *, max_yaw_deg, max_pitch_deg, max_roll_deg)
    Sample random tilt angles uniformly within caller-supplied bounds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import map_coordinates

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "get_oblique_slice",
    "sample_oblique_angles",
]

_MAX_ANGLE_DEG: float = 8.0


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _rotation_matrix_zyx(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return a 3×3 ZYX intrinsic Euler rotation matrix (right-hand).

    Convention used here (matching a standard neuroimaging frame):
        yaw   — rotation around the DV (Z/vertical) axis
        pitch — rotation around the ML (Y/left-right) axis
        roll  — rotation around the AP (X/front-back) axis

    The combined matrix is R = Rz(yaw) @ Ry(pitch) @ Rx(roll), applied to
    3-vectors [ap, ml, dv] expressed as column vectors.
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    cos_y, sin_y = math.cos(y), math.sin(y)
    cos_p, sin_p = math.cos(p), math.sin(p)
    cos_r, sin_r = math.cos(r), math.sin(r)

    # Rz (yaw)
    Rz = np.array(
        [
            [cos_y, -sin_y, 0.0],
            [sin_y, cos_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Ry (pitch)
    Ry = np.array(
        [
            [cos_p, 0.0, sin_p],
            [0.0, 1.0, 0.0],
            [-sin_p, 0.0, cos_p],
        ],
        dtype=np.float64,
    )

    # Rx (roll)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_r, -sin_r],
            [0.0, sin_r, cos_r],
        ],
        dtype=np.float64,
    )

    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _get_atlas_context(atlas: object) -> tuple[int, int, int, int]:
    """Return (ap_axis, ml_axis, dv_axis, n_ap) from a BrainGlobe atlas."""
    from langslice_harness.atlas.space import atlas_space_context

    ctx = atlas_space_context(atlas)
    return ctx.ap_axis_index, ctx.ml_axis_index, ctx.dv_axis_index, ctx.shape[ctx.ap_axis_index]


def _canonical_slice_shape(
    atlas: object,
    plane: str,
    ap_axis: int,
    ml_axis: int,
    dv_axis: int,
) -> tuple[int, int]:
    """Return the (H, W) shape of a canonical slice for the given plane."""
    vol_shape = np.asarray(atlas.reference).shape  # (s0, s1, s2)
    if plane == "coronal":
        # Slice normal = AP axis; in-plane = DV × ML
        in_axes = [dv_axis, ml_axis]
    elif plane == "sagittal":
        # Slice normal = ML axis; in-plane = DV × AP
        in_axes = [dv_axis, ap_axis]
    elif plane == "horizontal":
        # Slice normal = DV axis; in-plane = AP × ML
        in_axes = [ap_axis, ml_axis]
    else:
        raise ValueError(f"Unknown plane: {plane!r}")
    return int(vol_shape[in_axes[0]]), int(vol_shape[in_axes[1]])


def get_oblique_slice(
    atlas: object,
    *,
    base_position_mm: float,
    plane: str = "coronal",
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a tilted slice from the atlas reference + annotation volumes.

    Parameters
    ----------
    atlas:
        A BrainGlobe atlas object exposing ``.reference``, ``.annotation``,
        ``.resolution``, ``.orientation``, and ``.atlas_name``.
    base_position_mm:
        The position (in mm) along the slice-normal axis of the canonical
        plane.  For coronal this is the AP coordinate.
    plane:
        Canonical plane to tilt from: ``"coronal"``, ``"sagittal"``, or
        ``"horizontal"``.
    yaw_deg:
        Rotation (degrees) around the DV (vertical) axis.  |angle| ≤ 8°.
    pitch_deg:
        Rotation (degrees) around the ML (left-right) axis.  |angle| ≤ 8°.
    roll_deg:
        Rotation (degrees) around the AP (front-back) axis.  |angle| ≤ 8°.
    output_size:
        ``(H, W)`` of the returned arrays.  Defaults to the native canonical
        slice shape.

    Returns
    -------
    reference_slice_uint8 : np.ndarray, shape (H, W), dtype uint8
        Interpolated (bilinear) reference intensity slice.
    annotation_slice_int32 : np.ndarray, shape (H, W), dtype int32
        Nearest-neighbour annotation IDs (region IDs preserved exactly).

    Raises
    ------
    ValueError
        If any |angle| > 8°.
    """
    for name, angle in (("yaw_deg", yaw_deg), ("pitch_deg", pitch_deg), ("roll_deg", roll_deg)):
        if abs(angle) > _MAX_ANGLE_DEG:
            raise ValueError(
                f"|{name}| = {abs(angle):.2f}° exceeds the {_MAX_ANGLE_DEG:.0f}° safety limit. "
                "Larger tilts can produce slices that don't intersect tissue meaningfully."
            )

    from langslice_harness.atlas.core import position_mm_to_index
    from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

    ctx = atlas_space_context(atlas)
    ap_axis = ctx.ap_axis_index
    ml_axis = ctx.ml_axis_index
    dv_axis = ctx.dv_axis_index

    # Convert base position to voxel index along the slice-normal axis.
    normal_axis = slice_axis_index(ctx, plane)  # type: ignore[arg-type]
    base_idx = position_mm_to_index(atlas, base_position_mm, plane=plane)  # type: ignore[arg-type]

    vol_shape = np.asarray(atlas.reference).shape  # (s0, s1, s2) — 3D
    n0, n1, n2 = int(vol_shape[0]), int(vol_shape[1]), int(vol_shape[2])

    # Determine canonical slice shape (H, W) and which two axes form the plane.
    if plane == "coronal":
        row_axis, col_axis = dv_axis, ml_axis
    elif plane == "sagittal":
        row_axis, col_axis = dv_axis, ap_axis
    elif plane == "horizontal":
        row_axis, col_axis = ap_axis, ml_axis
    else:
        raise ValueError(f"Unknown plane: {plane!r}")

    native_h = int(vol_shape[row_axis])
    native_w = int(vol_shape[col_axis])

    if output_size is None:
        out_h, out_w = native_h, native_w
    else:
        out_h, out_w = int(output_size[0]), int(output_size[1])

    # Build output (row, col) grid in pixel-centre coordinates.
    # Each output pixel i maps to native-space coordinate i * (native_n / out_n),
    # then centred so that the native volume runs from offset 0 to native_n-1.
    # We scale so that the output grid spans the same native-space extent as the
    # canonical slice, regardless of out_h/out_w.
    row_scale = native_h / out_h
    col_scale = native_w / out_w
    row_offsets = (np.arange(out_h) * row_scale) - (native_h - 1) / 2.0
    col_offsets = (np.arange(out_w) * col_scale) - (native_w - 1) / 2.0
    grid_cols, grid_rows = np.meshgrid(col_offsets, row_offsets)  # both (out_h, out_w)

    # Flatten to (out_h * out_w,) vectors for batch rotation.
    flat_rows = grid_rows.ravel()
    flat_cols = grid_cols.ravel()
    # Assemble per-point vectors in (AP, ML, DV) anatomical space, then rotate.
    # Map row_axis/col_axis/normal_axis slots back to (AP, ML, DV) order.
    # We work in a frame where the slice-normal direction is 0, row is the
    # second anatomical axis, col the third.
    vecs = np.zeros((3, len(flat_rows)), dtype=np.float64)
    # Assign anatomical-axis coordinates:
    vecs[ap_axis] = 0.0  # will be overridden for normal_axis below
    vecs[ml_axis] = 0.0
    vecs[dv_axis] = 0.0
    # In-plane directions
    vecs[row_axis] = flat_rows
    vecs[col_axis] = flat_cols
    # Normal direction stays 0 (we anchor to the base slice)

    # Apply rotation in (AP, ML, DV) anatomical frame.
    R = _rotation_matrix_zyx(yaw_deg, pitch_deg, roll_deg)
    # vecs shape: (3, N) — R is (3×3), so rotated = R @ vecs → (3, N)
    rotated = R @ vecs  # (3, N)

    # Translate the rotation centre to (base_idx, ml_centre, dv_centre) in
    # voxel space.  Centres are at (n-1)/2 so that pixel 0 maps to voxel 0
    # and pixel n-1 maps to voxel n-1 exactly under zero tilt.
    centres = [(n0 - 1) / 2.0, (n1 - 1) / 2.0, (n2 - 1) / 2.0]
    centres[normal_axis] = float(base_idx)

    # Sample coordinates in (axis0, axis1, axis2) voxel order.
    sample_coords = np.empty((3, len(flat_rows)), dtype=np.float64)
    for axis in range(3):
        sample_coords[axis] = rotated[axis] + centres[axis]

    # Clamp to valid voxel range to suppress cval edge artefacts.
    sample_coords[0] = np.clip(sample_coords[0], 0.0, n0 - 1)
    sample_coords[1] = np.clip(sample_coords[1], 0.0, n1 - 1)
    sample_coords[2] = np.clip(sample_coords[2], 0.0, n2 - 1)

    # Reference: linear interpolation → float → normalize → uint8
    ref_vol = np.asarray(atlas.reference).astype(np.float32)
    ref_flat = map_coordinates(ref_vol, sample_coords, order=1, mode="constant", cval=0.0)
    ref_2d = ref_flat.reshape(out_h, out_w)
    # Normalize to uint8 like the canonical pipeline.
    max_val = float(ref_2d.max())
    if max_val > 0.0:
        ref_uint8 = np.clip(ref_2d / max_val * 255.0, 0, 255).astype(np.uint8)
    else:
        ref_uint8 = np.zeros((out_h, out_w), dtype=np.uint8)

    # Annotation: nearest-neighbour → int32 (preserves region IDs exactly)
    ann_vol = np.asarray(atlas.annotation)
    ann_flat = map_coordinates(
        ann_vol.astype(np.float64), sample_coords, order=0, mode="constant", cval=0.0
    )
    ann_2d = np.round(ann_flat).astype(np.int32).reshape(out_h, out_w)

    return ref_uint8, ann_2d


# ---------------------------------------------------------------------------
# Convenience sampler
# ---------------------------------------------------------------------------


def sample_oblique_angles(
    rng: np.random.Generator,
    *,
    max_yaw_deg: float = 8.0,
    max_pitch_deg: float = 8.0,
    max_roll_deg: float = 8.0,
) -> tuple[float, float, float]:
    """Sample random tilt angles uniformly within the specified bounds.

    Parameters
    ----------
    rng:
        A ``numpy.random.Generator`` (e.g. ``np.random.default_rng(seed)``).
    max_yaw_deg, max_pitch_deg, max_roll_deg:
        Symmetric bounds for each axis; the returned angle is in
        ``[-max, +max]``.  Must not exceed ``8.0°``.

    Returns
    -------
    (yaw_deg, pitch_deg, roll_deg) : tuple[float, float, float]
    """
    for name, limit in (
        ("max_yaw_deg", max_yaw_deg),
        ("max_pitch_deg", max_pitch_deg),
        ("max_roll_deg", max_roll_deg),
    ):
        if limit > _MAX_ANGLE_DEG:
            raise ValueError(f"{name}={limit:.1f}° exceeds the {_MAX_ANGLE_DEG:.0f}° safety limit.")

    yaw = float(rng.uniform(-max_yaw_deg, max_yaw_deg))
    pitch = float(rng.uniform(-max_pitch_deg, max_pitch_deg))
    roll = float(rng.uniform(-max_roll_deg, max_roll_deg))
    return yaw, pitch, roll
