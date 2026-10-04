"""A linear registration made elsewhere, as LangSlice's own placement terms.

The inverse of the one geometry path from a stored placement to the atlas
(:func:`langslice.core.maps.section_frame`: the file's pixels oriented onto
the :data:`~langslice.core.sections.PREVIEW_LONG_EDGE` render, the six stored
numbers placing the native atlas plane there through
:func:`langslice.core.handoff.linear_placement_matrix`, the plane taken to
BrainGlobe micrometres by :func:`langslice.core.oblique.plane_index_affine`).
Given one section's exact map from its image file's pixels ``[row, col, 1]``
to atlas micrometres (what ``registration.json`` stores, and what a QuickNII
anchoring is once converted to micrometres by the job layer,
``langslice.job.imports``), :func:`placement_from_pixel_map` returns the
position, cutting angles, flip, quarter turn and six normalized numbers
that reproduce it.

Degrees of freedom: the map is affine from the image plane into the volume,
nine numbers. LangSlice's placement is also nine: the plane's normal (pitch,
yaw), its position along the normal axis, and the full in-plane affine (six
numbers, so anisotropic scale and shear survive). Every map whose plane is
within :data:`MAX_TILT_DEG` of the job's section plane is therefore
represented exactly, with three exceptions the result reports rather than
hides:

- a FLAT plane (both angles 0) is drawn by LangSlice at the nearest voxel
  of the normal axis (:func:`langslice.core.atlas.core.position_mm_to_index`),
  so a position between voxels is kept as given but drawn up to half a voxel
  away (``out_of_plane_um``);
- an angle smaller than :data:`ANGLE_EPSILON_DEG` is read as 0 (the rounding
  of a written file), which makes the plane flat (above);
- flip and quarter turn are not in the map at all (the six numbers can hold
  any of the eight): unless the caller fixes them (*orientation*), they are
  chosen so the six numbers are unmirrored and turn the least
  (:func:`placement_from_pixel_map`).

The pixel size is not in the map either: the six numbers are normalized
against a render at a given micrometres per pixel, and absorb any scale. The
caller passes the pixel size the job will map the file at (its own
calibration when it has one; :func:`implied_pixel_size_um` otherwise).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from langslice.core.affine import decompose_affine, normalized_affine, pixel_center_map
from langslice.core.canvas import canvas_geometry
from langslice.core.handoff import linear_placement_matrix
from langslice.core.maps import SWAP, RenderSizes, orientation_matrix
from langslice.core.oblique import build_rotation_matrix, plane_axes, plane_index_affine
from langslice.core.space import Plane, atlas_space_context
from langslice.core.state import ROTATIONS

#: Cutting angles smaller than this (degrees) are read as 0: a written file's
#: rounding (QuickNII anchorings carry six decimals of a voxel, about 1e-8
#: degrees), never a cut.
ANGLE_EPSILON_DEG = 1e-6
#: The largest tilt (degrees) between an imported plane and the job's section
#: plane: beyond it the registration is of another plane (a sagittal
#: registration read into a coronal job), refused.
MAX_TILT_DEG = 45.0


def implied_pixel_size_um(pixel_to_atlas_um: Any) -> float:
    """Micrometres per pixel the map itself implies: the square root of the
    atlas area one image pixel covers (anisotropic or sheared maps get
    their geometric mean)."""
    matrix = np.asarray(pixel_to_atlas_um, dtype=np.float64)
    return float(math.sqrt(np.linalg.norm(np.cross(matrix[:, 0], matrix[:, 1]))))


def plane_angles(normal: np.ndarray, atlas: Any, plane: Plane) -> tuple[float, float]:
    """``(pitch_deg, yaw_deg)`` of the plane whose PHYSICAL normal (atlas axis
    order) is *normal*: the inverse of the basis
    :func:`langslice.core.oblique.build_rotation_matrix` builds (pitch about
    the column axis first, then yaw about the row axis), in
    :mod:`langslice.core.oblique`'s convention. ``ValueError`` beyond
    :data:`MAX_TILT_DEG`."""
    normal_axis, row_axis, col_axis = plane_axes(atlas, plane)
    unit = np.asarray(normal, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    if unit[normal_axis] < 0:
        unit = -unit
    tilt = math.degrees(math.acos(min(1.0, float(unit[normal_axis]))))
    if tilt > MAX_TILT_DEG:
        raise ValueError(f"the registered plane is tilted {tilt:.1f} degrees from a "
                         f"{plane} section (at most {MAX_TILT_DEG:.0f})")
    axes = np.eye(3)
    s_row = float(np.cross(axes[col_axis], axes[normal_axis])[row_axis])
    s_col = float(np.cross(axes[row_axis], axes[normal_axis])[col_axis])
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, s_row * float(unit[row_axis])))))
    yaw = math.degrees(math.atan2(s_col * float(unit[col_axis]), float(unit[normal_axis])))
    pitch = 0.0 if abs(pitch) < ANGLE_EPSILON_DEG else pitch
    yaw = 0.0 if abs(yaw) < ANGLE_EPSILON_DEG else yaw
    check = build_rotation_matrix(pitch, yaw, row_axis=row_axis, col_axis=col_axis)
    miss = np.linalg.norm(np.cross(check[:, normal_axis], unit))
    if miss > math.radians(1e-3):  # a convention slip, never data
        raise ValueError("the cutting angles do not reproduce the registered plane")
    return pitch, yaw


def plane_position_mm(origin_um: np.ndarray, normal: np.ndarray, atlas: Any,
                      plane: Plane) -> float:
    """The position (atlas-native mm along the normal axis) at which the plane
    through *origin_um* with physical *normal* crosses the volume's in-plane
    centre, the point :func:`langslice.core.oblique.plane_index_affine`'s
    plane passes through."""
    normal_axis, row_axis, col_axis = plane_axes(atlas, plane)
    context = atlas_space_context(atlas)
    res = np.asarray(context.resolution_um, dtype=np.float64)
    shape = context.shape
    n = np.asarray(normal, dtype=np.float64)
    centre = (n[row_axis] * (shape[row_axis] - 1) / 2.0 * res[row_axis]
              + n[col_axis] * (shape[col_axis] - 1) / 2.0 * res[col_axis])
    along = (float(n @ np.asarray(origin_um, dtype=np.float64)) - centre) / n[normal_axis]
    return along / 1000.0


@dataclass(frozen=True)
class RecoveredPlacement:
    """One section's placement in LangSlice's terms (see the module text).

    ``params`` are the six normalized numbers on the oriented render the
    section is drawn at, at ``render_um_per_px`` (``file_um_per_px`` times
    the render's shrink): stored with that calibration they reproduce the
    map. ``out_of_plane_um`` is the largest distance, over the image's four
    corners, between the given map and the plane LangSlice draws (the
    flat-plane voxel snap; zero up to rounding otherwise);
    ``in_plane_error_px`` the largest in-plane disagreement at the corners
    after the round trip, in file pixels. ``in_plane`` is the six numbers
    read as rotation, scales and shear on the render's pixel frame
    (:func:`langslice.core.affine.decompose_affine`).
    """

    position_mm: float
    pitch_deg: float
    yaw_deg: float
    rotation_deg: int
    flip: bool
    params: tuple[float, float, float, float, float, float]
    file_um_per_px: float
    render_um_per_px: float
    out_of_plane_um: float
    in_plane_error_px: float
    in_plane: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["params"] = list(self.params)
        return out


def _native_to_um(atlas: Any, position_mm: float, plane: Plane, pitch: float,
                  yaw: float) -> np.ndarray:
    res = np.asarray(atlas_space_context(atlas).resolution_um, dtype=np.float64)
    return np.diag(res) @ plane_index_affine(atlas, position_mm, plane, pitch, yaw)


def placement_from_pixel_map(
    pixel_to_atlas_um: Any,
    *,
    atlas: Any,
    plane: Plane,
    sizes: RenderSizes,
    file_um_per_px: float,
    orientation: tuple[int, bool] | None = None,
) -> RecoveredPlacement:
    """LangSlice's placement of a section whose image file's pixel
    ``[row, col, 1]`` (centres at integers) maps to atlas micrometres by the
    3x3 *pixel_to_atlas_um*.

    *sizes* is how the job draws the file (:func:`langslice.core.maps.render_sizes`),
    *file_um_per_px* the pixel size the job maps it at. *orientation*
    ``(rotation_deg, flip)`` fixes the quarter turn and flip; None chooses
    them: of the eight, the one whose six numbers are unmirrored and turn
    least (``|rotation| <= 45`` degrees on the render's pixel frame; on an
    exact tie the smaller quarter turn, unflipped first). ``ValueError`` for a
    degenerate map, a plane of another orientation, or a position the atlas
    cannot draw.
    """
    matrix = np.asarray(pixel_to_atlas_um, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("pixel_to_atlas_um must be a finite 3x3 matrix")
    if not np.isfinite(file_um_per_px) or file_um_per_px <= 0:
        raise ValueError("the pixel size must be finite and positive")
    normal = np.cross(matrix[:, 0], matrix[:, 1])
    if np.linalg.norm(normal) <= 1e-12 * max(1.0, float(np.abs(matrix[:, :2]).max()) ** 2):
        raise ValueError("the map is degenerate (its image axes are parallel)")
    pitch, yaw = plane_angles(normal, atlas, plane)
    normal_axis, row_axis, col_axis = plane_axes(atlas, plane)
    exact_normal = build_rotation_matrix(pitch, yaw, row_axis=row_axis,
                                         col_axis=col_axis)[:, normal_axis]
    position = plane_position_mm(matrix[:, 2], exact_normal, atlas, plane)
    native_to_um = _native_to_um(atlas, position, plane, pitch, yaw)

    # File pixel -> native plane pixel ([row, col]), least squares onto the
    # plane LangSlice draws (exact unless the plane was snapped).
    basis = native_to_um[:, :2]
    pinv = np.linalg.pinv(basis)
    linear = pinv @ matrix[:, :2]
    offset = pinv @ (matrix[:, 2] - native_to_um[:, 2])
    file_to_native_rc = np.array([[*linear[0], offset[0]], [*linear[1], offset[1]],
                                  [0.0, 0.0, 1.0]])
    width, height = sizes.file_size
    corners = np.array([[-0.5, -0.5, 1.0], [-0.5, width - 0.5, 1.0],
                        [height - 0.5, -0.5, 1.0], [height - 0.5, width - 0.5, 1.0]]).T
    drawn = native_to_um @ file_to_native_rc @ corners
    given = matrix @ corners
    out_of_plane = float(np.abs(exact_normal @ (given - drawn)).max())

    file_to_native = SWAP @ file_to_native_rc @ SWAP
    unturned = (int(sizes.unturned_render[0]), int(sizes.unturned_render[1]))
    render_um = float(file_um_per_px) * float(sizes.render_scale)
    file_to_unturned = pixel_center_map(sizes.file_size, unturned)
    geometries: dict[tuple[int, int], Any] = {}

    def candidate(rotation: int, flip: bool) -> tuple[np.ndarray, tuple[int, int], Any]:
        render_size = (unturned[1], unturned[0]) if rotation % 180 == 90 else unturned
        if render_size not in geometries:
            geometries[render_size] = canvas_geometry(render_size, render_um, atlas, position,
                                                      plane, pitch, yaw)
        geometry = geometries[render_size]
        sx, sy = geometry.section_offset
        ax, ay = geometry.atlas_offset
        atlas_to_frame = np.array([[geometry.atlas_scale, 0.0, ax - sx],
                                   [0.0, geometry.atlas_scale, ay - sy], [0.0, 0.0, 1.0]])
        file_to_render = orientation_matrix(unturned, rotation, flip) @ file_to_unturned
        atlas_to_render = file_to_render @ np.linalg.inv(file_to_native)
        return atlas_to_frame @ np.linalg.inv(atlas_to_render), render_size, geometry

    if orientation is not None:
        rotation, flip = int(orientation[0]) % 360, bool(orientation[1])
        if rotation not in ROTATIONS:
            raise ValueError(f"rotation_deg must be one of {list(ROTATIONS)}")
        chosen = (rotation, flip)
    else:
        scored = []
        for flip in (False, True):
            for rotation in ROTATIONS:
                placed, render_size, _geometry = candidate(rotation, flip)
                parts = decompose_affine(normalized_affine(placed[:2], render_size),
                                         render_size)
                scored.append((bool(parts["mirrored"]), abs(float(parts["rotation_deg"])),
                               rotation, flip))
        _mirrored, _turn, rotation, flip = min(scored)
        chosen = (rotation, flip)
    placed, render_size, _geometry = candidate(*chosen)
    params = tuple(float(v) for v in normalized_affine(placed[:2], render_size))

    # The round trip, as core.maps draws it, at the corners.
    atlas_to_render, _ = linear_placement_matrix(render_size, render_um, atlas, position,
                                                 plane, pitch, yaw, params)
    file_to_render = orientation_matrix(unturned, chosen[0], chosen[1]) @ file_to_unturned
    back = SWAP @ np.linalg.inv(atlas_to_render) @ file_to_render @ SWAP
    in_plane_error = float(np.abs(back @ corners - file_to_native_rc @ corners).max())
    # Native pixels -> file pixels (along x) for the error in file pixels.
    native_px_in_file = float(np.sqrt(abs(np.linalg.det(np.linalg.inv(linear)))))
    return RecoveredPlacement(
        position_mm=float(position), pitch_deg=float(pitch), yaw_deg=float(yaw),
        rotation_deg=int(chosen[0]), flip=bool(chosen[1]),
        params=params,  # type: ignore[arg-type]
        file_um_per_px=float(file_um_per_px), render_um_per_px=render_um,
        out_of_plane_um=out_of_plane, in_plane_error_px=in_plane_error * native_px_in_file,
        in_plane=decompose_affine(list(params), render_size),
    )
