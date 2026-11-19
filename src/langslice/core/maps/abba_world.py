"""A placed section in ABBA's own world frame, for an ABBA project file.

A project file written from a job opens the job's own images in ABBA, so
the section's pixels are carried into ABBA's world here.

ABBA's frame, for its Allen CCFv3 atlas started coronal (x "RL", y "SI",
z "AP"), as read off the transforms ABBA 0.25 writes into a QuPath project
(``ABBA-Transform-*.json``) and checked against the lab's saves::

    world = A(rx, ry) (atlas_mm - c) + (0, 0, AXIS_OFFSET_MM)

``atlas_mm`` is the atlas point in millimetres in the order (ML, DV, AP),
the BrainGlobe ``asr`` axes reversed; ``c`` is :data:`CENTRE_MM`; ``A`` is
:func:`reslice_matrix` of ABBA's slicing rotations ``rx``, ``ry`` (radians),
which are LangSlice's ``-pitch`` and ``-yaw``. A coronal section of the
stack's cutting angles lies in one world plane ``z = location``, the slice's
position in ABBA; its file pixels go to world ``x, y`` by an affine map.

In ABBA the section's file pixel ``(x, y)`` (centres at integers) is first
a source point in millimetres, ``((x - (W - 1) / 2) v, (y - (H - 1) / 2)
v)`` (ABBA centres a source whose view transform is a bare voxel scale
``v``), then placed by the registrations. :func:`abba_placement` returns
the affine step that takes those source millimetres to the world, and the
two world maps a deformation's landmark pairs are placed with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from langslice.core.maps.section_maps import SWAP

if TYPE_CHECKING:
    from langslice.core.maps.section_maps import SectionFrame

CENTRE_MM = (5.69, 4.0)
AXIS_OFFSET_MM = 0.985
LOCATION_TO_ATLAS_Z_MM = 0.99
ABBA_ATLASES = frozenset(
    {
        "allen_mouse_10um",
        "allen_mouse_25um",
        "allen_mouse_50um",
        "allen_mouse_100um",
    }
)
ABBA_ORIENTATIONS = frozenset({"asr"})
ABBA_PLANES = frozenset({"coronal"})
PLANE_TOLERANCE_MM = 1e-4
FloatArray = npt.NDArray[np.float64]
MM_PER_UM = 1e-3


def reslice_matrix(rotation_x: float, rotation_y: float) -> np.ndarray:
    """Build ABBA's linear map from atlas (ML, DV, AP) mm to world mm.

    Parameters
    ----------
    rotation_x, rotation_y
        ABBA's slicing rotations, in radians.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        The world z is the atlas AP sheared by the tangents of the two
        rotations; x and y are rescaled so the plane keeps its metric.
    """
    tan_x, tan_y = math.tan(rotation_x), math.tan(rotation_y)
    cos_x, cos_y = math.cos(rotation_x), math.cos(rotation_y)
    return np.array(
        [
            [1.0 / cos_y, -tan_x * tan_y, 0.0],
            [0.0, 1.0 / cos_x, 0.0],
            [tan_y, -tan_x / cos_y, 1.0],
        ],
        dtype=np.float64,
    )


def rotations(pitch_deg: float, yaw_deg: float) -> tuple[float, float]:
    """Convert LangSlice's cutting angles to ABBA's slicing rotations.

    Parameters
    ----------
    pitch_deg, yaw_deg
        The stack's cutting angles, in degrees.

    Returns
    -------
    tuple of float
        ABBA's ``(rx, ry)``, in radians.
    """
    return -math.radians(pitch_deg), -math.radians(yaw_deg)


def atlas_um_to_world(rotation_x: float, rotation_y: float) -> np.ndarray:
    """Build the map from BrainGlobe ``asr`` micrometres to ABBA's world.

    Parameters
    ----------
    rotation_x, rotation_y
        ABBA's slicing rotations, in radians.

    Returns
    -------
    numpy.ndarray, shape (3, 4)
        Atlas micrometres (AP, DV, ML) to world millimetres.
    """
    reorder = np.zeros((3, 3))
    reorder[0, 2] = reorder[1, 1] = reorder[2, 0] = MM_PER_UM
    matrix = reslice_matrix(rotation_x, rotation_y)
    centre = np.array([*CENTRE_MM, 0.0])
    result = np.zeros((3, 4))
    result[:, :3] = matrix @ reorder
    result[:, 3] = -matrix @ centre + np.array([0.0, 0.0, AXIS_OFFSET_MM])
    return result


@dataclass(frozen=True)
class AbbaPlacement:
    """One section's linear placement in ABBA's world frame.

    Attributes
    ----------
    location_mm
        The slice's position on ABBA's slicing axis (its world z).
    affine_mm
        3x3 on ``[x, y, 1]``: ABBA's centred source millimetres to world
        ``x, y`` (the affine registration step).
    file_to_world, native_to_world
        3x3 on ``[x, y, 1]``: a file pixel, and a native atlas-plane pixel,
        to world ``x, y`` after that step.
    voxel_mm
        The source's voxel size.
    """

    location_mm: float
    affine_mm: np.ndarray
    file_to_world: np.ndarray
    native_to_world: np.ndarray
    voxel_mm: float


def check_atlas(name: str, orientation: str, plane: str) -> None:
    """Refuse an atlas or plane ABBA's frame here does not hold.

    Parameters
    ----------
    name, orientation
        The BrainGlobe atlas's name and axis orientation.
    plane
        The stack's plane.

    Raises
    ------
    ValueError
        Saying what ABBA cannot open.
    """
    if name not in ABBA_ATLASES or orientation not in ABBA_ORIENTATIONS:
        raise ValueError(f"ABBA's Allen CCFv3 atlas cannot hold a registration to {name}")
    if plane not in ABBA_PLANES:
        raise ValueError(f"ABBA opens coronal sections only, not {plane}")


def _planar(matrix: np.ndarray) -> np.ndarray:
    """Keep the x and y rows of a 3x3 world map, as a 2-D affine."""
    return np.vstack([matrix[:2], [0.0, 0.0, 1.0]])


def abba_placement(frame: SectionFrame, rotation_x: float, rotation_y: float) -> AbbaPlacement:
    """Carry a section's linear placement into ABBA's world.

    Parameters
    ----------
    frame
        The placed section's frame.
    rotation_x, rotation_y
        The stack's slicing rotations in ABBA (:func:`rotations`).

    Returns
    -------
    AbbaPlacement
        The slice's location, its affine step and its world maps.

    Raises
    ------
    ValueError
        When the atlas is not ABBA's Allen atlas, the section is not
        coronal, it does not lie in one ABBA slice at these rotations, or
        it has no pixel size.
    """
    facts = cast("dict[str, object]", frame.atlas)
    check_atlas(
        str(facts.get("name", "")),
        str(facts.get("orientation", "")),
        frame.plane,
    )
    to_world = atlas_um_to_world(rotation_x, rotation_y)
    native_to_um = np.asarray(frame.native_to_um, dtype=np.float64) @ SWAP
    native_world = to_world[:, :3] @ native_to_um
    native_world[:, 2] += to_world[:, 3]
    file_to_native = np.asarray(frame.file_to_native(), dtype=np.float64)
    file_world = native_world @ file_to_native

    width, height = frame.file_size
    corners = np.array(
        [
            [-0.5, -0.5, 1.0],
            [width - 0.5, -0.5, 1.0],
            [-0.5, height - 0.5, 1.0],
            [width - 0.5, height - 0.5, 1.0],
        ]
    )
    depth = cast("FloatArray", corners @ file_world[2])
    spread = float(np.ptp(depth))
    if spread > PLANE_TOLERANCE_MM:
        raise ValueError(
            "the section is not cut at the stack's cutting angles in ABBA "
            f"(its depth changes by {spread * 1000:.2f} um)"
        )
    voxel = float(frame.file_um_per_px) * MM_PER_UM
    if not math.isfinite(voxel) or voxel <= 0:
        raise ValueError("the section has no pixel size")
    source_to_file = np.array(
        [
            [1.0 / voxel, 0.0, (width - 1) / 2.0],
            [0.0, 1.0 / voxel, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    file_to_world = _planar(file_world)
    return AbbaPlacement(
        location_mm=float(np.mean(depth)),
        affine_mm=file_to_world @ source_to_file,
        file_to_world=file_to_world,
        native_to_world=_planar(native_world),
        voxel_mm=voxel,
    )
