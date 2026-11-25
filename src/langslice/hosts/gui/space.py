"""The job in three dimensions, for the viewer's 3D view.

The 3D view draws the atlas brain as BrainGlobe's own surface of it and
every placed section as a thin slab where its linear placement puts it.
This module gives the page the numbers for that, from the same frames
``registration.json`` is written from
(:func:`langslice.core.maps.section_maps.section_frame`), so a section in
the 3D view sits exactly where its pictures say it does:

- the atlas brain (:func:`atlas_shell`): the atlas's ``root`` mesh, the
  one brainrender draws, its vertices as float32 and its triangles as
  uint32, base64 encoded;
- each section's picture corners (:func:`section_corners`), once the host
  or a tool has given it a position (the job's initial spacing is not a
  placement): the top-left,
  top-right and bottom-left corners of its thumbnail (the section turned and
  flipped as it is kept, as ``render/thumb`` draws it), the fourth corner the
  parallelogram's;
- the brain's cross-section at each step from back to front
  (:func:`cross_sections`), which a section's picture is clipped to, so a
  section not yet aligned keeps its true size and nothing of it hangs
  outside the atlas;
- the section thickness the job was given, else 10 micrometres.

Atlas micrometres follow BrainGlobe: voxel ``i``'s centre at
``i * resolution``, axes as the atlas's ``orientation`` letters name them.
The page gets every point in the view's frame instead (:func:`view_frame`):
millimetres centred on the atlas, anterior +z, superior +y and the animal's
right at -x, so it holds no axis convention of its own.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from langslice.core.maps.section_maps import (
    orientation_matrix,
    placement_problem,
    section_frame,
)
from langslice.core.stack.section import INITIAL, Section
from langslice.core.stack.stack import Stack
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

DEFAULT_THICKNESS_UM = 10.0
ROOT_STRUCTURE = "root"
VIEW_AXES: dict[str, tuple[int, float]] = {
    "a": (2, -1.0),
    "p": (2, 1.0),
    "s": (1, -1.0),
    "i": (1, 1.0),
    "r": (0, 1.0),
    "l": (0, -1.0),
}


@dataclass(frozen=True)
class Room:
    """How finely the brain's cross-section is kept for the 3D view's clip.

    ``bins`` steps from the brain's back to its front, each the extent of
    the mesh's vertices within a step of it, ``margin_mm`` wider on every
    side and beyond both ends, so a section on the brain's surface is never
    cut.
    """

    margin_mm: float = 0.15
    bins: int = 96


ROOM = Room()

Corners = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class ShellError(Exception):
    """An atlas without a brain mesh the 3D view can draw; says why."""


class _MeshCells(Protocol):
    """A block of a mesh's cells, as meshio keeps it."""

    type: str
    data: npt.NDArray[np.int64]


class _Mesh(Protocol):
    """A mesh as meshio reads it: points and blocks of cells."""

    points: npt.NDArray[np.float64]
    cells: list[_MeshCells]


class _MeshAtlas(Protocol):
    """An atlas that keeps meshes of its structures (BrainGlobe's)."""

    orientation: str
    resolution: tuple[float, float, float]
    shape: tuple[int, int, int]

    def mesh_from_structure(self, structure: str) -> _Mesh:
        """Read the mesh of *structure*."""
        ...


@dataclass(frozen=True)
class AtlasShell:
    """The atlas brain's surface: BrainGlobe's root mesh in the view's frame.

    ``positions`` float32 x, y, z per vertex and ``indices`` uint32 per
    triangle, wound so each face looks out, base64 encoded; ``size_mm`` the
    atlas volume's size along x, y and z; ``room`` float32 left, right,
    bottom and top of the brain per :data:`ROOM` step from ``room_span``'s
    back to its front (:func:`cross_sections`).
    """

    size_mm: tuple[float, float, float]
    vertices: int
    triangles: int
    positions: str
    indices: str
    room: str
    room_span: tuple[float, float]


@dataclass(frozen=True)
class SpaceSection:
    """One section in 3D: its thumbnail's corners in the view's frame, None while unplaced."""

    id: str
    order: int
    corners_mm: Corners | None


@dataclass(frozen=True)
class Space:
    """The 3D view's facts (``GET jobs/<n>/space``)."""

    thickness_um: float
    given_thickness: bool
    sections: tuple[SpaceSection, ...]


def view_frame(orientation: str, extent_um: tuple[float, float, float]) -> npt.NDArray[np.float64]:
    """Make the 3x4 map from atlas micrometres to the 3D view's frame.

    Parameters
    ----------
    orientation
        The atlas's orientation letters (BrainGlobe: each names the side its
        axis starts from).
    extent_um
        The atlas volume's size along its three axes.

    Returns
    -------
    numpy.ndarray
        On ``[x, y, z, 1]`` in atlas micrometres: millimetres centred on the
        atlas, anterior +z, superior +y, the animal's right at -x.
    """
    frame = np.zeros((3, 4))
    for k, letter in enumerate(orientation.lower()):
        axis, sign = VIEW_AXES[letter]
        frame[axis, k] = sign / 1000.0
        frame[axis, 3] = -sign * extent_um[k] / 2000.0
    return frame


def _atlas_frame(
    atlas: _MeshAtlas,
) -> tuple[npt.NDArray[np.float64], tuple[float, float, float]]:
    """The atlas's view frame, and its volume's size along x, y and z in millimetres."""
    r, n = atlas.resolution, atlas.shape
    extent = (float(r[0]) * n[0], float(r[1]) * n[1], float(r[2]) * n[2])
    frame = view_frame(str(atlas.orientation), extent)
    size = np.abs(frame[:, :3]) @ np.array(extent)
    return frame, (float(size[0]), float(size[1]), float(size[2]))


def outward(points: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Wind *faces* so their normals point away from the mesh's centre, on the whole."""
    centred = points - points.mean(axis=0)
    a, b, c = (points[faces[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    if float(np.einsum("ij,ij->", normals, centred[faces[:, 0]])) >= 0:
        return faces
    return faces[:, [0, 2, 1]]


def cross_sections(
    points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], tuple[float, float]]:
    """Find the brain's extent at each :data:`ROOM` step from back to front.

    Parameters
    ----------
    points
        The mesh's vertices in the view's frame.

    Returns
    -------
    tuple
        Per step the least and greatest x and y of the vertices within a
        step of it, :data:`ROOM`'s margin wider, as left, right, bottom, top
        (a step with no vertex takes its nearest filled neighbour's), and
        the steps' back and front along z.
    """
    margin, bins = ROOM.margin_mm, ROOM.bins
    back = float(points[:, 2].min()) - margin
    front = float(points[:, 2].max()) + margin
    at = np.floor((points[:, 2] - back) / (front - back) * bins).astype(np.int64)
    sides = np.tile(np.array([np.inf, -np.inf, np.inf, -np.inf]), (bins, 1))
    for shift in (-1, 0, 1):
        step = at + shift
        inside = (step >= 0) & (step < bins)
        rows, x, y = step[inside], points[inside, 0], points[inside, 1]
        np.minimum.at(sides[:, 0], rows, x - margin)
        np.maximum.at(sides[:, 1], rows, x + margin)
        np.minimum.at(sides[:, 2], rows, y - margin)
        np.maximum.at(sides[:, 3], rows, y + margin)
    for order in (range(bins), range(bins - 1, -1, -1)):
        last = -1
        for row in order:
            if np.isfinite(sides[row, 0]):
                last = row
            elif last >= 0:
                sides[row] = sides[last]
    return sides, (back, front)


def atlas_shell(workspace: Workspace) -> AtlasShell:
    """Read the atlas's root mesh, the brain's surface BrainGlobe ships.

    Parameters
    ----------
    workspace
        The job's workspace, whose atlas is read.

    Returns
    -------
    AtlasShell
        The mesh's vertices and outward-wound triangles in the view's frame
        (:func:`view_frame`), and the brain's cross-sections.

    Raises
    ------
    ShellError
        When the atlas keeps no root mesh or it cannot be read.
    """
    atlas = cast(_MeshAtlas, workspace.atlas)
    try:
        mesh = atlas.mesh_from_structure(ROOT_STRUCTURE)
    except (AttributeError, KeyError, OSError, ValueError) as exc:
        raise ShellError(f"This atlas has no brain mesh to draw: {exc}") from exc
    triangles = [c.data for c in mesh.cells if c.type == "triangle"]
    if not triangles:
        raise ShellError("This atlas's brain mesh has no triangles.")
    frame, size = _atlas_frame(atlas)
    points = np.asarray(mesh.points, dtype=np.float64) @ frame[:, :3].T + frame[:, 3]
    faces = outward(points, np.concatenate(triangles).astype(np.int64))
    sides, span = cross_sections(points)
    return AtlasShell(
        size,
        len(points),
        len(faces),
        _b64(np.ascontiguousarray(points, dtype="<f4")),
        _b64(np.ascontiguousarray(faces, dtype="<u4")),
        _b64(np.ascontiguousarray(sides, dtype="<f4")),
        span,
    )


def _b64(values: npt.NDArray[np.generic]) -> str:
    """Base64 of an array's bytes, as the page decodes it."""
    return base64.b64encode(values.tobytes()).decode("ascii")


def job_space(workspace: Workspace, state: Stack, *, thickness_um: float | None) -> Space:
    """Place every section of *state* in the atlas, as far as it can be.

    Parameters
    ----------
    workspace
        The job's workspace.
    state
        The stack as the watcher reads it.
    thickness_um
        The section thickness the job was given, None when it was not.

    Returns
    -------
    Space
        The sections in stack order, each with its corners in the view's
        frame (:func:`view_frame`) or None.
    """
    frame, _size = _atlas_frame(cast(_MeshAtlas, workspace.atlas))
    sections = tuple(
        SpaceSection(section.id, order, _in_view(frame, section_corners(workspace, state, section)))
        for order, section in enumerate(state.in_order(), 1)
    )
    given = thickness_um is not None and thickness_um > 0
    thickness = (
        float(thickness_um)
        if thickness_um is not None and thickness_um > 0
        else DEFAULT_THICKNESS_UM
    )
    return Space(thickness, given, sections)


def _in_view(frame: npt.NDArray[np.float64], corners: Corners | None) -> Corners | None:
    """Corners in atlas micrometres, taken to the view's frame in millimetres."""
    if corners is None:
        return None
    view = frame @ np.vstack([np.array(corners, dtype=np.float64).T, np.ones(3)])
    return _as_corners(view, digits=5)


def section_corners(workspace: Workspace, state: Stack, section: Section) -> Corners | None:
    """Find where the corners of a section's thumbnail sit in the atlas.

    Parameters
    ----------
    workspace
        The job's workspace.
    state
        The stack the section belongs to.
    section
        The section whose thumbnail is placed.

    Returns
    -------
    Corners or None
        The thumbnail's top-left, top-right and bottom-left corners in
        atlas micrometres; None for a section still at the job's initial
        spacing (not placed by anyone yet), one without a linear placement
        or one whose frame cannot be made (logged).
    """
    if section.position_source == INITIAL:
        return None
    if placement_problem(state, section) is not None:
        return None
    try:
        frame = section_frame(state, workspace, section)
    except (ValueError, OSError):
        logger.warning("No frame for %s", section.id, exc_info=True)
        return None
    turn = orientation_matrix(frame.file_size, frame.rotation_deg, frame.flip)
    turned = turned_corners(frame.file_size, frame.rotation_deg)
    file_xy = np.linalg.inv(turn) @ turned
    rowcol = np.stack([file_xy[1], file_xy[0], np.ones(3)])
    atlas_um = frame.pixel_to_atlas_um() @ rowcol
    return _as_corners(atlas_um)


def turned_corners(file_size: tuple[int, int], rotation_deg: int) -> npt.NDArray[np.float64]:
    """List the turned picture's top-left, top-right and bottom-left corners.

    Parameters
    ----------
    file_size
        The file's width and height in pixels.
    rotation_deg
        The section's quarter turn.

    Returns
    -------
    numpy.ndarray
        3x3, one ``[x, y, 1]`` column per corner, in the turned picture's
        pixel-centre coordinates (a corner half a pixel outside the
        outermost centres).
    """
    width, height = float(file_size[0]), float(file_size[1])
    if int(rotation_deg) % 180 == 90:
        width, height = height, width
    return np.array(
        [
            [-0.5, width - 0.5, -0.5],
            [-0.5, -0.5, height - 0.5],
            [1.0, 1.0, 1.0],
        ]
    )


def _as_corners(columns3: npt.NDArray[np.float64], digits: int = 2) -> Corners:
    """Round a 3x3 of corner columns to the corners' tuples."""
    columns = cast(list[list[float]], columns3.T.tolist())
    points = [(round(x, digits), round(y, digits), round(z, digits)) for x, y, z in columns]
    return (points[0], points[1], points[2])
