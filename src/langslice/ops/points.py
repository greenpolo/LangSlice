"""Points of a section and the atlas, both ways, through the section's registration.

What a script's ``job.context.to_atlas_um`` / ``from_atlas_um`` /
``region_at`` read: a section's points (pixels of the image file as stored,
or of its fit grid, ``FitFrame.image``) carried to atlas micrometres through
the section's current registration, its applied deformation included
(:func:`langslice.core.maps.native_points`), and atlas micrometres carried
back (:func:`langslice.core.maps.file_points`, through the record's inverse
field). Writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.ops.refusal import Refused, unknown_sections

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

#: The frames a section's points are given in: the image file's own pixels
#: (``section_px``), or the fit grid's (``FitFrame.image``).
FRAMES = ("file", "fit")


@dataclass(frozen=True)
class AtlasPoints:
    """Section points in the atlas: ``um`` ``(N, 3)`` atlas micrometres (the
    atlas's axis order; NaN outside the atlas volume), ``native`` ``(N, 2)``
    native atlas-plane pixels ``(x, y)``, ``ids`` ``(N,)`` the atlas id under
    each (0 outside the atlas), ``inside`` ``(N,)`` within the volume."""

    um: np.ndarray
    native: np.ndarray
    ids: np.ndarray
    inside: np.ndarray


@dataclass(frozen=True)
class SectionPoints:
    """Atlas points on a section: ``xy`` ``(N, 2)`` in the asked frame and
    ``off_plane_um`` ``(N,)``, how far each atlas point lies off the
    section's plane (it is projected onto it)."""

    xy: np.ndarray
    off_plane_um: np.ndarray


def _placed(job: Job, workspace: Workspace, section: Any) -> tuple[Any, Any, Any]:
    """The section's record, linear frame and applied deformation record."""
    from langslice.core.maps import placement_problem, section_frame
    from langslice.ops.exports import applied_record

    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [str(section)])
    problem = placement_problem(job.state, record)
    if problem is not None:
        raise Refused("INVALID_LINEAR_PLACEMENT", message=f"{record.id}: {problem}")
    frame = section_frame(job.state, workspace, record)
    try:
        warp = applied_record(job, job.state, record)
    except ValueError as exc:
        raise Refused("DEFORMATION_UNREADABLE", message=f"{record.id}: {exc}") from exc
    return record, frame, warp


def _fit_matrix(job: Job, workspace: Workspace, record: Any, frame: Any, warp: Any,
                ) -> np.ndarray:
    """3x3: file pixel centres -> fit-grid pixel centres."""
    from langslice.core import deformation
    from langslice.core.maps import file_to_fit

    if warp is not None:
        size = (int(warp.section_size[0]), int(warp.section_size[1]))
    else:
        width, height = deformation.fit_grid(job.state, workspace, record).image.size
        size = (int(width), int(height))
    return file_to_fit(frame, size)


def _points(value: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise Refused("BAD_ARGS", message=f"{name} is N x {width} finite numbers.")
    return array


def _frame_of(frame_name: str) -> str:
    if frame_name not in FRAMES:
        raise Refused("BAD_ARGS", message=f"frame is one of {list(FRAMES)}.")
    return frame_name


def atlas_points(job: Job, workspace: Workspace, section: Any, xy: Any, *,
                 frame: str = "file") -> AtlasPoints:
    """Section points *xy* (``(N, 2)`` ``[x, y]`` pixels of *frame*: ``file``,
    the image file's own, or ``fit``, the fit grid's) in the atlas, through the
    section's current registration (deformation included). Refused:
    ``UNKNOWN_SLICE_IDS``, ``INVALID_LINEAR_PLACEMENT``, ``DEFORMATION_UNREADABLE``,
    ``BAD_ARGS``."""
    from langslice.core.deformable.geometry import sample_native
    from langslice.core.maps import apply, native_points, point_rows

    points = _points(xy, 2, "xy")
    record, held, warp = _placed(job, workspace, section)
    x, y = points[:, 0], points[:, 1]
    if _frame_of(frame) == "fit":
        x, y = apply(np.linalg.inv(_fit_matrix(job, workspace, record, held, warp)), x, y)
    count = len(x)
    nx, ny = native_points(held, warp, point_rows(x), point_rows(y))
    nx, ny = np.ravel(nx)[:count], np.ravel(ny)[:count]
    index = held.native_to_index @ np.stack([ny, nx, np.ones_like(nx)])
    shape = np.asarray(held.atlas["shape"], dtype=np.float64)[:, None]
    inside = np.all((index >= -0.5) & (index < shape - 0.5), axis=0)
    um = (held.native_to_um @ np.stack([ny, nx, np.ones_like(nx)])).T
    um[~inside] = np.nan
    native = np.stack([nx, ny], axis=-1)
    ids = sample_native(held.annotation, native).astype(np.uint32)
    ids[~inside] = 0
    return AtlasPoints(um=um, native=native, ids=ids, inside=inside)


def section_points(job: Job, workspace: Workspace, section: Any, xyz_um: Any, *,
                   frame: str = "file") -> SectionPoints:
    """Atlas points *xyz_um* (``(N, 3)`` atlas micrometres, the atlas's axis
    order) on the section: each projected onto the section's atlas plane,
    then carried back through the section's current registration (its
    deformation's inverse included) to ``[x, y]`` pixels of *frame*. Refused
    as :func:`atlas_points`; ``DEFORMATION_UNREADABLE`` too for a
    deformation without an inverse."""
    from langslice.core.maps import apply, file_points

    points = _points(xyz_um, 3, "xyz_um")
    record, held, warp = _placed(job, workspace, section)
    to_um = np.asarray(held.native_to_um, dtype=np.float64)
    solved, *_ = np.linalg.lstsq(to_um[:, :2], (points - to_um[:, 2]).T, rcond=None)
    rows, cols = solved[0], solved[1]
    off = np.linalg.norm((to_um[:, :2] @ solved).T + to_um[:, 2] - points, axis=1)
    try:
        x, y = file_points(held, warp, cols, rows)
    except ValueError as exc:
        raise Refused("DEFORMATION_UNREADABLE", message=f"{record.id}: {exc}") from exc
    if _frame_of(frame) == "fit":
        x, y = apply(_fit_matrix(job, workspace, record, held, warp), x, y)
    return SectionPoints(xy=np.stack([np.ravel(x), np.ravel(y)], axis=-1),
                         off_plane_um=np.ravel(off))


__all__ = ["FRAMES", "AtlasPoints", "SectionPoints", "atlas_points", "section_points"]
