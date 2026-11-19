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
  flipped as it is kept, as ``render/thumb`` draws it) in atlas micrometres,
  in the atlas's own axis order, the fourth corner the parallelogram's;
- the section thickness the job was given, else 10 micrometres.

Atlas micrometres follow BrainGlobe: voxel ``i``'s centre at
``i * resolution``, axes as the atlas's ``orientation`` letters name them.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from langslice.core.atlas.load import BrainGlobeAtlas
from langslice.core.maps.section_maps import (
    orientation_matrix,
    placement_problem,
    section_frame,
)
from langslice.core.space import atlas_space_context, slice_axis_index
from langslice.core.stack.section import INITIAL, Section
from langslice.core.stack.stack import Stack
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

DEFAULT_THICKNESS_UM = 10.0
ROOT_STRUCTURE = "root"

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
    """The atlas brain's surface: BrainGlobe's root mesh in atlas um."""

    orientation: str
    extent_um: tuple[float, float, float]
    vertices: int
    triangles: int
    positions: str
    indices: str


@dataclass(frozen=True)
class SpaceSection:
    """One section in 3D: its thumbnail's corners, None while unplaced."""

    id: str
    order: int
    corners_um: Corners | None


@dataclass(frozen=True)
class Space:
    """The 3D view's facts (``GET jobs/<n>/space``)."""

    thickness_um: float
    given_thickness: bool
    plane_axis: int
    sections: tuple[SpaceSection, ...]


def atlas_shell(workspace: Workspace) -> AtlasShell:
    """Read the atlas's root mesh, the brain's surface BrainGlobe ships.

    Parameters
    ----------
    workspace
        The job's workspace, whose atlas is read.

    Returns
    -------
    AtlasShell
        The mesh's vertices (x, y, z in atlas micrometres, the atlas's axis
        order, the frame of :func:`section_corners`) and triangles.

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
    points = np.ascontiguousarray(mesh.points, dtype="<f4")
    faces = np.ascontiguousarray(np.concatenate(triangles), dtype="<u4")
    resolution = tuple(float(value) for value in atlas.resolution)
    full = tuple(int(value) for value in atlas.shape)
    extent_um = (
        resolution[0] * full[0],
        resolution[1] * full[1],
        resolution[2] * full[2],
    )
    return AtlasShell(
        str(atlas.orientation),
        extent_um,
        len(points),
        len(faces),
        base64.b64encode(points.tobytes()).decode("ascii"),
        base64.b64encode(faces.tobytes()).decode("ascii"),
    )


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
        The sections in stack order, each with its corners or None.
    """
    sections = tuple(
        SpaceSection(section.id, order, section_corners(workspace, state, section))
        for order, section in enumerate(state.in_order(), 1)
    )
    given = thickness_um is not None and thickness_um > 0
    thickness = (
        float(thickness_um)
        if thickness_um is not None and thickness_um > 0
        else DEFAULT_THICKNESS_UM
    )
    context = atlas_space_context(cast(BrainGlobeAtlas, workspace.atlas))
    plane_axis = slice_axis_index(context, workspace.settings.plane)
    return Space(thickness, given, plane_axis, sections)


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


def _as_corners(atlas_um: npt.NDArray[np.float64]) -> Corners:
    """Round a 3x3 of corner columns to the corners' tuples."""
    columns = cast(list[list[float]], atlas_um.T.tolist())
    points = [(round(x, 2), round(y, 2), round(z, 2)) for x, y, z in columns]
    return (points[0], points[1], points[2])
