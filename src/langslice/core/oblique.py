"""Oblique-plane sampling of a BrainGlobe atlas: every plane LangSlice draws or maps.

Adapted from `brainglobe-registration <https://github.com/brainglobe/brainglobe-registration>`_,
copyright the BrainGlobe developers, BSD-3-Clause: the Euler-angle
rotation-matrix construction of
``brainglobe_registration.utils.transforms.create_rotation_matrix``. One
deliberate departure from upstream: **sample the plane, don't rotate the
volume.** Upstream rotates the whole atlas volume (dask ``affine_transform``)
and then takes a z slice; here a single arbitrary plane is sampled with
:func:`scipy.ndimage.map_coordinates`, three orders of magnitude cheaper.

Angle convention (defined on the raw atlas axes, before any display swap):

* ``pitch_deg`` rotates about the plane's **column** axis. For a coronal plane
  in an ``asr``-style atlas that is the ML axis, i.e. the usual "cutting angle"
  that makes the section tilt anterior-dorsal / posterior-ventral. A pitched
  plane is still left-right symmetric.
* ``yaw_deg`` rotates about the plane's **row** axis (DV for coronal), i.e. the
  section samples a more anterior AP level on one side than the other. This is
  the *only* component that breaks left-right symmetry.

Both are right-handed rotations in the atlas index frame, applied pitch first.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from langslice.core.atlas.core import orient_slice_for_display
from langslice.core.space import Plane, atlas_space_context, slice_axis_index

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def plane_axes(atlas: Any, plane: Plane = "coronal") -> tuple[int, int, int]:
    """``(normal_axis, row_axis, col_axis)`` of *plane* in atlas array order.

    Row/col are the two non-normal axes in ascending index order, which is the
    layout :func:`numpy.take` produces and therefore what
    :func:`langslice.core.atlas.core.get_reference_slice` starts from.
    """
    context = atlas_space_context(atlas)
    normal = slice_axis_index(context, plane)
    row, col = (a for a in range(3) if a != normal)
    return normal, row, col


def build_rotation_matrix(
    pitch_deg: float, yaw_deg: float, *, row_axis: int, col_axis: int
) -> np.ndarray:
    """3x3 rotation: *pitch_deg* about *col_axis*, then *yaw_deg* about *row_axis*.

    Extrinsic (fixed-frame) composition, so the returned matrix is
    ``R_yaw @ R_pitch``. Axes are atlas array axes, not anatomical names.
    """
    letters = "".join("xyz"[axis] for axis in (col_axis, row_axis))
    return np.asarray(
        Rotation.from_euler(letters, [pitch_deg, yaw_deg], degrees=True).as_matrix(),
        dtype=np.float64,
    )


_volume_cache: dict[tuple[str, str, int, int], tuple[np.ndarray, tuple[float, float, float]]] = {}


def _scaled_volume(
    atlas: Any, kind: str, downsample: int
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Contiguous float32 volume strided by *downsample*, plus its mm resolution.

    Cached per (atlas, kind, downsample): a 25 um mouse template is 300 MB as
    float32 and the fit re-samples it hundreds of times per section.
    """
    key = (str(getattr(atlas, "atlas_name", "?")), kind, int(downsample), id(atlas))
    cache = _volume_cache
    hit = cache.get(key)
    if hit is not None:
        return hit
    raw = np.asarray(getattr(atlas, kind))
    step = max(1, int(downsample))
    volume = np.ascontiguousarray(raw[::step, ::step, ::step], dtype=np.float32)
    context = atlas_space_context(atlas)
    res_mm = cast(
        "tuple[float, float, float]",
        tuple(r / 1000.0 * step for r in context.resolution_um),
    )
    if len(cache) > 8:  # ponytail: tiny manual LRU; atlases are few and huge
        cache.clear()
    cache[key] = (volume, res_mm)
    return volume, res_mm


def _plane_basis(
    shape: tuple[int, ...],
    res_mm: tuple[float, float, float],
    axes: tuple[int, int, int],
    normal_index: float,
    pitch_deg: float,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(centre, step_row, step_col)`` of a plane in volume index coordinates.

    The plane passes through the volume's in-plane centre at *normal_index*
    along the normal axis; raw plane pixel ``(r, c)`` sits at ``centre +
    (r - (H - 1) / 2) * step_row + (c - (W - 1) / 2) * step_col``. In-plane
    unit directions in physical space, expressed on the atlas axes, are
    converted to index steps one voxel wide along their own axis; dividing
    component-wise by the per-axis resolution keeps anisotropic atlases
    square.
    """
    normal_axis, row_axis, col_axis = axes
    height, width = shape[row_axis], shape[col_axis]
    rotation = build_rotation_matrix(pitch_deg, yaw_deg, row_axis=row_axis, col_axis=col_axis)
    res = np.asarray(res_mm, dtype=np.float64)
    basis_row = np.zeros(3)
    basis_row[row_axis] = 1.0
    basis_col = np.zeros(3)
    basis_col[col_axis] = 1.0
    step_row = (rotation @ basis_row) * res[row_axis] / res
    step_col = (rotation @ basis_col) * res[col_axis] / res

    centre = np.zeros(3)
    centre[normal_axis] = normal_index
    centre[row_axis] = (height - 1) / 2.0
    centre[col_axis] = (width - 1) / 2.0
    return centre, step_row, step_col


def _plane_coordinates(
    shape: tuple[int, ...],
    res_mm: tuple[float, float, float],
    axes: tuple[int, int, int],
    normal_index: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """(3, H, W) volume index coordinates of a plane, in raw (undisplayed) order
    (:func:`_plane_basis`)."""
    _normal_axis, row_axis, col_axis = axes
    height, width = shape[row_axis], shape[col_axis]
    centre, step_row, step_col = _plane_basis(
        shape, res_mm, axes, normal_index, pitch_deg, yaw_deg)

    rows = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0)[:, None]
    cols = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0)[None, :]
    coords = np.empty((3, height, width), dtype=np.float64)
    for axis in range(3):
        coords[axis] = centre[axis] + rows * step_row[axis] + cols * step_col[axis]
    return coords


def plane_index_coordinates(
    atlas: Any,
    position_mm: float,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """(3, H, W) full-resolution atlas index of every pixel of a display-oriented plane.

    The pixel grid is exactly :func:`langslice.core.atlas.render.annotation_slice`'s
    at the same arguments: a flat plane sits on the rounded voxel index that
    function takes, a tilted plane on :func:`sample_oblique_plane`'s geometry.
    Use it to sample any volume co-registered with the atlas but stored on
    its own grid (a different resolution, another file) at the same points.
    """
    normal_axis, row_axis, col_axis = plane_axes(atlas, plane)
    context = atlas_space_context(atlas)
    res_mm = cast(
        "tuple[float, float, float]", tuple(r / 1000.0 for r in context.resolution_um)
    )
    if pitch_deg or yaw_deg:
        normal_index = position_mm / res_mm[normal_axis]
    else:
        from langslice.core.atlas.core import position_mm_to_index

        normal_index = float(position_mm_to_index(atlas, position_mm, plane=plane))
    coords = _plane_coordinates(
        context.shape, res_mm, (normal_axis, row_axis, col_axis),
        normal_index, pitch_deg, yaw_deg,
    )
    if plane in {"sagittal", "horizontal"}:
        coords = np.swapaxes(coords, 1, 2)
    return np.ascontiguousarray(coords)


def plane_index_affine(
    atlas: Any,
    position_mm: float,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """3x3 map from a display-oriented plane pixel ``[row, col, 1]`` to its
    atlas voxel index (3 axes, atlas array order, voxel centres at integers).

    The same plane as :func:`plane_index_coordinates` (and so
    :func:`langslice.core.atlas.render.annotation_slice`) at the same arguments,
    built from the same basis (:func:`_plane_basis`), without the full grid.
    Multiply each row by the axis resolution for BrainGlobe micrometres.
    """
    normal_axis, row_axis, col_axis = plane_axes(atlas, plane)
    context = atlas_space_context(atlas)
    res_mm = cast(
        "tuple[float, float, float]", tuple(r / 1000.0 for r in context.resolution_um)
    )
    if pitch_deg or yaw_deg:
        normal_index = position_mm / res_mm[normal_axis]
    else:
        from langslice.core.atlas.core import position_mm_to_index

        normal_index = float(position_mm_to_index(atlas, position_mm, plane=plane))
    shape = context.shape
    centre, step_row, step_col = _plane_basis(
        shape, res_mm, (normal_axis, row_axis, col_axis), normal_index, pitch_deg, yaw_deg)
    origin = (centre - (shape[row_axis] - 1) / 2.0 * step_row
              - (shape[col_axis] - 1) / 2.0 * step_col)
    if plane in {"sagittal", "horizontal"}:  # displayed transposed
        step_row, step_col = step_col, step_row
    return np.column_stack([step_row, step_col, origin])


def sample_oblique_plane(
    atlas: Any,
    position_mm: float,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    *,
    volume: str | np.ndarray = "template",
    downsample: int = 1,
    order: int = 1,
) -> np.ndarray:
    """Sample one arbitrary oblique plane out of an atlas volume.

    The plane passes through the volume's in-plane centre at *position_mm*
    along the slice-normal axis (atlas-native millimetres from the anterior
    edge for a coronal plane), tilted by *pitch_deg* and *yaw_deg*.

    At ``pitch_deg == yaw_deg == 0`` and ``downsample=1`` this reproduces
    :func:`langslice.core.atlas.core.get_reference_slice`'s array exactly (same
    shape, same display orientation, same voxels) for any *position_mm* that
    lands on the voxel grid.

    Args:
        atlas: BrainGlobe-style atlas exposing ``template``/``annotation``.
        position_mm: Position along the slice-normal axis.
        plane: Slicing plane.
        pitch_deg: Rotation about the plane's column axis (degrees).
        yaw_deg: Rotation about the plane's row axis (degrees).
        volume: ``"template"``, ``"annotation"``, or a full-resolution array
            with the same shape as the template.
        downsample: Integer stride applied to all three volume axes. The
            returned plane shrinks by the same factor.
        order: ``map_coordinates`` spline order. 1 (linear) is right for a
            search loop; 3 for a final render.

    Returns:
        2D float32 array, oriented like the rest of ``langslice.core.atlas.core``.
    """
    normal_axis, row_axis, col_axis = plane_axes(atlas, plane)
    if isinstance(volume, str):
        vol, res_mm = _scaled_volume(atlas, volume, downsample)
    else:
        step = max(1, int(downsample))
        vol = np.ascontiguousarray(np.asarray(volume)[::step, ::step, ::step], dtype=np.float32)
        context = atlas_space_context(atlas)
        res_mm = cast(
            "tuple[float, float, float]",
            tuple(r / 1000.0 * step for r in context.resolution_um),
        )

    coords = _plane_coordinates(
        vol.shape, res_mm, (normal_axis, row_axis, col_axis),
        position_mm / res_mm[normal_axis], pitch_deg, yaw_deg,
    )
    sampled = map_coordinates(vol, coords, order=order, mode="constant", cval=0.0)
    return np.asarray(orient_slice_for_display(sampled, plane), dtype=np.float32)


_annotation_index_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _compact_annotation(atlas: Any) -> tuple[np.ndarray, np.ndarray]:
    """(sorted ids, float32 volume of their dense indices), cached per atlas."""
    key = (str(getattr(atlas, "atlas_name", "?")), id(atlas))
    hit = _annotation_index_cache.get(key)
    if hit is not None:
        return hit
    annotation = np.asarray(atlas.annotation)
    ids = np.unique(annotation)
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.float32)
    lookup[ids] = np.arange(len(ids), dtype=np.float32)
    _annotation_index_cache.clear()  # ponytail: one atlas at a time; these are ~300 MB
    _annotation_index_cache[key] = (ids, lookup[annotation])
    return _annotation_index_cache[key]


def sample_oblique_annotation(
    atlas: Any,
    position_mm: float,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    *,
    downsample: int = 1,
) -> np.ndarray:
    """Annotation ids on an oblique plane, sampled nearest-neighbour.

    Not ``sample_oblique_plane(volume="annotation")``: that samples in
    float32, whose 24-bit mantissa cannot hold the larger Allen structure ids
    (484682516 lands on 484682528). Compacting the volume to dense indices
    first makes every sampled value round-trip exactly.
    """
    ids, index_volume = _compact_annotation(atlas)
    sampled = sample_oblique_plane(
        atlas,
        position_mm,
        plane,
        pitch_deg,
        yaw_deg,
        volume=index_volume,
        downsample=downsample,
        order=0,
    )
    return ids[np.rint(sampled).astype(np.intp).clip(0, len(ids) - 1)]
