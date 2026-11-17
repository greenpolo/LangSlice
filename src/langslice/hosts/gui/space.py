"""The job in three dimensions, for the viewer's 3D view.

The 3D view draws the atlas brain as a shell and every placed section as a
thin slab where its linear placement puts it. This module gives the page
the numbers for that, from the same frames ``registration.json`` is written
from (:func:`langslice.core.maps.section_maps.section_frame`), so a section
in the 3D view sits exactly where its pictures say it does:

- the atlas brain as a coarse mask (:func:`atlas_shell`): the annotation
  sampled every ``step`` voxels, so its longest side has at most
  :data:`SHELL_MAX_VOXELS` samples, one bit per sample, packed and base64
  encoded; the page draws its surface;
- each section's picture corners (:func:`section_corners`): the top-left,
  top-right and bottom-left corners of its thumbnail (the section turned and
  flipped as it is kept, as ``render/thumb`` draws it) in atlas micrometres,
  in the atlas's own axis order, the fourth corner the parallelogram's;
- the section thickness the job was given, else :data:`DEFAULT_THICKNESS_UM`.

Atlas micrometres follow BrainGlobe: voxel ``i``'s centre at
``i * resolution``, axes as the atlas's ``orientation`` letters name them.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from langslice.core.maps.section_maps import (
    orientation_matrix,
    placement_problem,
    section_frame,
)
from langslice.core.space import atlas_space_context, slice_axis_index
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

SHELL_MAX_VOXELS = 112
DEFAULT_THICKNESS_UM = 10.0

Corners = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True)
class AtlasShell:
    """The atlas brain as a coarse mask of samples ``step_um`` apart."""

    orientation: str
    shape: tuple[int, int, int]
    step_um: tuple[float, float, float]
    extent_um: tuple[float, float, float]
    bits: str


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
    """Sample the atlas's annotated brain on a coarse grid.

    Parameters
    ----------
    workspace
        The job's workspace, whose atlas is read.

    Returns
    -------
    AtlasShell
        The mask of annotated samples, row-major over the atlas's axes,
        packed eight samples to a byte (``numpy.packbits``).
    """
    atlas = workspace.atlas
    annotation = np.asarray(atlas.annotation)
    resolution_um = tuple(float(value) for value in atlas.resolution)
    stride = max(1, int(np.ceil(max(annotation.shape) / SHELL_MAX_VOXELS)))
    sampled = annotation[::stride, ::stride, ::stride] > 0
    packed = np.packbits(sampled.astype(np.uint8), axis=None)
    shape = (sampled.shape[0], sampled.shape[1], sampled.shape[2])
    step_um = (
        resolution_um[0] * stride,
        resolution_um[1] * stride,
        resolution_um[2] * stride,
    )
    extent_um = (
        resolution_um[0] * annotation.shape[0],
        resolution_um[1] * annotation.shape[1],
        resolution_um[2] * annotation.shape[2],
    )
    return AtlasShell(
        str(atlas.orientation),
        shape,
        step_um,
        extent_um,
        base64.b64encode(packed.tobytes()).decode("ascii"),
    )


def job_space(
    workspace: Workspace, state: Stack, *, thickness_um: float | None
) -> Space:
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
        SpaceSection(
            section.id, order, section_corners(workspace, state, section)
        )
        for order, section in enumerate(state.in_order(), 1)
    )
    given = thickness_um is not None and thickness_um > 0
    thickness = float(thickness_um) if given else DEFAULT_THICKNESS_UM
    context = atlas_space_context(workspace.atlas)
    plane_axis = slice_axis_index(context, workspace.settings.plane)
    return Space(thickness, given, plane_axis, sections)


def section_corners(
    workspace: Workspace, state: Stack, section: Section
) -> Corners | None:
    """Find where the corners of a section's thumbnail sit in the atlas.

    Parameters
    ----------
    workspace
        The job's workspace.
    state
        The stack the section belongs to.
    section
        The section.

    Returns
    -------
    Corners or None
        The thumbnail's top-left, top-right and bottom-left corners in
        atlas micrometres; None for a section without a linear placement
        or one whose frame cannot be made (logged).
    """
    if placement_problem(state, section) is not None:
        return None
    try:
        frame = section_frame(state, workspace, section)
    except (ValueError, OSError):
        logger.warning("No frame for %s", section.id, exc_info=True)
        return None
    turn = orientation_matrix(
        frame.file_size, frame.rotation_deg, frame.flip
    )
    turned = turned_corners(frame.file_size, frame.rotation_deg)
    file_xy = np.linalg.inv(turn) @ turned
    rowcol = np.stack([file_xy[1], file_xy[0], np.ones(3)])
    atlas_um = frame.pixel_to_atlas_um() @ rowcol
    return _as_corners(atlas_um)


def turned_corners(
    file_size: tuple[int, int], rotation_deg: int
) -> npt.NDArray[np.float64]:
    """The turned picture's top-left, top-right and bottom-left corners.

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
    points = [
        (
            round(float(atlas_um[0, k]), 2),
            round(float(atlas_um[1, k]), 2),
            round(float(atlas_um[2, k]), 2),
        )
        for k in range(3)
    ]
    return (points[0], points[1], points[2])
