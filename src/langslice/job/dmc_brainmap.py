"""A job's linear registrations as a DMC-BrainMap project.

DMC-BrainMap (napari-dmc-brainmap, written against 0.2.0b2) keeps a
section's registration in its SHARPy-track step: one ``registration.json``
per channel folder, ``<animal>/sharpy_track/<channel>/registration.json``,
beside the section images resized to the atlas plane's voxel grid
(``X`` x ``Y``). Per section, keyed by the image's place in the folder's
natural-sorted ``*.tif`` listing:

- ``atlasLocation``: ``[x_angle_deg, y_angle_deg, z_mm]``, the cutting
  plane. ``z_mm`` is millimetres from DMC's bregma voxel along the plane's
  normal axis, positive towards index 0 (``round(-z_mm / res) + bregma``
  is the plane's centre voxel). The plane is a SHEAR of the volume, not a
  rotation: the depth voxel at atlas-plane pixel ``(x, y)`` is
  ``int(z + ys * (y - Y/2) / (Y/2) + xs * (x - X/2) / (X/2 - 1))`` with
  the whole-voxel shifts ``xs = int(tan(x_angle) * X/2)`` and
  ``ys = int(tan(y_angle) * Y/2)`` (``SliceHandle.get_z_plane``).
- ``atlasDots`` / ``sampleDots``: at least five ``[x, y]`` point pairs on
  the atlas plane and on the resized image, from which DMC fits one
  projective transform (``fitGeoTrans``); an atlas point ``(x, y)`` is the
  voxel ``(int(x), int(y))``.
- ``imgName``: the image's filename.

A cell measured on an image of ``W`` x ``H`` pixels at ``(c, r)`` (pixel
centres at integers) is read by DMC at ``ceil(c * (X - 1) / (W - 1))`` on the
resized image (``results_creator._scale_positions``).

:func:`dmc_placement` writes a section's linear map (``registration.json``'s
``pixel_to_atlas_um``) in these terms. LangSlice's plane is exact; DMC's
holds it to whole voxels: its depth is a whole-voxel centre plus whole-voxel
shifts, and every index is truncated. The pairs reproduce the in-plane
map exactly (an affine is a projective transform), each placed half a voxel
(and half a resized pixel) on, so DMC's truncations land on average where
LangSlice's nearest voxel is. The shifts are the nearest whole voxels to
LangSlice's slopes, the centre voxel the one that agrees best with
LangSlice's depth over the section. DMC holds no deformation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

#: The channel folder names DMC-BrainMap offers (its params widget).
DMC_CHANNELS = ("dapi", "green", "n3", "cy3", "cy5")
#: DMC's own bregma voxel per atlas (``utils/atlas_utils.get_bregma``);
#: any other atlas gets DMC's estimate (:func:`dmc_bregma`).
DMC_BREGMA: dict[str, tuple[int, int, int]] = {
    "allen_mouse_10um": (540, 0, 570),
    "whs_sd_rat_39um": (371, 72, 266),
    "azba_zfish_4um": (360, 0, 335),
}
#: DMC's names of the three anatomical axes (BrainGlobe's
#: ``axes_description``); its results table knows no others.
DMC_AXES = frozenset({"ap", "si", "rl"})
#: DMC fits a section's transform only from this many pairs or more.
MIN_PAIRS = 5
#: Grids of candidate pairs over the image, tried in turn until enough
#: pairs fall on the atlas plane: the coarsest gives at most 25 pairs.
PAIR_GRIDS = (5, 9, 17)
#: The grid over the image on which the plane's centre voxel is chosen.
DEPTH_GRID = 33
#: Decimals the pairs and angles are written with: 1e-4 voxel, far below
#: the whole voxels DMC rounds to.
DECIMALS = 4
#: How far past its whole-voxel shift an angle's tangent puts the shift, in
#: voxels: far above the angle's rounding to 1e-6 degree, far below a voxel.
ANGLE_MARGIN = 0.01

#: BrainGlobe's two-letter axis name for each origin letter (its
#: ``axes_description``: the letter is where the axis starts).
_AXIS_NAMES: dict[str, str] = {
    "a": "ap",
    "p": "pa",
    "s": "si",
    "i": "is",
    "r": "rl",
    "l": "lr",
}
#: The axis each cutting plane is normal to.
_NORMAL_AXIS: dict[str, str] = {
    "coronal": "ap",
    "horizontal": "si",
    "sagittal": "rl",
}


@dataclass(frozen=True)
class DmcAxis:
    """One axis of DMC's section frame, as its ``xyz_dict`` holds it."""

    #: BrainGlobe's two-letter name (``ap``, ``si``, ``rl``).
    name: str
    #: The axis's index in the atlas volume.
    index: int
    size: int
    resolution_um: float


@dataclass(frozen=True)
class DmcFrame:
    """DMC's frame of an atlas cut in one plane.

    ``x`` is the columns and ``y`` the rows of its atlas plane, ``z`` the
    plane's normal (``utils/atlas_utils.get_xyz``); ``bregma`` its bregma
    voxel.
    """

    atlas: str
    plane: str
    x: DmcAxis
    y: DmcAxis
    z: DmcAxis
    bregma: tuple[int, int, int]

    @property
    def z_bregma(self) -> int:
        """The bregma voxel along the plane's normal axis."""
        return self.bregma[self.z.index]


def dmc_frame(
    atlas: str,
    orientation: str,
    shape: Sequence[int],
    resolution_um: Sequence[float],
    plane: str,
) -> DmcFrame:
    """Build DMC's frame of an atlas cut in *plane*, as DMC builds it.

    Parameters
    ----------
    atlas
        The BrainGlobe atlas name DMC opens the atlas by.
    orientation
        The atlas's BrainGlobe origin string, such as ``asr``.
    shape
        The atlas volume's shape in its own axis order.
    resolution_um
        The voxel size of each axis, in micrometres.
    plane : {"coronal", "horizontal", "sagittal"}
        The cutting plane.

    Returns
    -------
    DmcFrame
        The axes and the bregma voxel DMC uses for this atlas and plane.

    Raises
    ------
    ValueError
        When the atlas's axes are not ones DMC names (``ap``, ``si``, ``rl``).
    """
    names = tuple(_AXIS_NAMES[letter] for letter in orientation)
    if frozenset(names) != DMC_AXES:
        raise ValueError(
            f"DMC-BrainMap reads only atlases whose axes run ap, si and rl; "
            f"{atlas} has {', '.join(names)}"
        )

    # the plane's normal is z; DMC's rows and columns are the other two
    # axes, in the volume's order (BrainGlobe's ``index_pairs``)
    normal = names.index(_NORMAL_AXIS[plane])
    rows, cols = (i for i in range(3) if i != normal)
    axes = [
        DmcAxis(names[i], i, int(shape[i]), float(resolution_um[i]))
        for i in range(3)
    ]
    return DmcFrame(
        atlas=atlas,
        plane=plane,
        x=axes[cols],
        y=axes[rows],
        z=axes[normal],
        bregma=dmc_bregma(atlas, names, shape),
    )


def dmc_bregma(
    atlas: str, names: Sequence[str], shape: Sequence[int]
) -> tuple[int, int, int]:
    """Give the bregma voxel DMC uses for an atlas.

    Parameters
    ----------
    atlas
        The BrainGlobe atlas name.
    names
        The atlas's axis names in its own order (``ap``, ``si``, ``rl``).
    shape
        The atlas volume's shape in its own axis order.

    Returns
    -------
    tuple of int
        DMC's table entry, else its estimate: half the volume along the
        two axes other than ``si`` and 0 along ``si``.
    """
    if atlas in DMC_BREGMA:
        return DMC_BREGMA[atlas]
    vertical = list(names).index("si")
    halves = [int(int(size) / 2) for size in shape]
    halves[vertical] = 0
    return halves[0], halves[1], halves[2]


@dataclass(frozen=True)
class DmcPlacement:
    """One section's registration in DMC's terms.

    ``atlas_dots`` and ``sample_dots`` are ``[x, y]`` pairs on the atlas
    plane and on the image resized to it; ``x_shift`` and ``y_shift`` are
    the whole-voxel shifts the angles give DMC, ``z_voxel`` the plane's
    centre voxel.
    """

    x_angle_deg: float
    y_angle_deg: float
    z_mm: float
    z_voxel: int
    x_shift: int
    y_shift: int
    atlas_dots: list[list[float]]
    sample_dots: list[list[float]]

    def location(self) -> list[float]:
        """Give ``atlasLocation``'s entry.

        Returns
        -------
        list of float
            ``[x_angle_deg, y_angle_deg, z_mm]``.
        """
        return [self.x_angle_deg, self.y_angle_deg, self.z_mm]


def dmc_placement(
    pixel_to_atlas_um: np.ndarray,
    file_size: tuple[int, int],
    frame: DmcFrame,
) -> DmcPlacement:
    """Write a section's linear map as DMC's plane and point pairs.

    Parameters
    ----------
    pixel_to_atlas_um : array, shape (3, 3)
        The image file's pixel ``[row, col, 1]`` (centres at integers) to
        BrainGlobe atlas micrometres in the atlas's axis order.
    file_size
        The image file's ``(width, height)`` in pixels.
    frame
        DMC's frame of the atlas and plane (:func:`dmc_frame`).

    Returns
    -------
    DmcPlacement
        The angles, depth and pairs DMC reads.

    Raises
    ------
    ValueError
        When the section's plane contains the normal axis, when fewer than
        five pairs fall on the atlas plane, or when the plane leaves the
        atlas volume somewhere on DMC's atlas plane (DMC refuses that
        section).
    """
    width, height = int(file_size[0]), int(file_size[1])
    size_x, size_y = frame.x.size, frame.y.size
    voxels = _voxel_map(pixel_to_atlas_um, frame)

    # depth as a function of the atlas plane's x and y
    in_plane = voxels[:2, :2]
    if abs(cast("float", np.linalg.det(in_plane))) < 1e-9:
        raise ValueError(
            "the section's plane contains the atlas plane's normal"
        )
    slopes = cast("np.ndarray", voxels[2, :2] @ np.linalg.inv(in_plane))
    slope_x, slope_y = cast("list[float]", slopes.tolist())
    x_shift = int(round(slope_x * (size_x / 2 - 1)))
    y_shift = int(round(slope_y * size_y / 2))
    x_angle = _shift_angle(x_shift, size_x)
    y_angle = _shift_angle(y_shift, size_y)

    # the centre voxel that agrees best with LangSlice's depth
    depth_x, depth_y, depth = _points(voxels, width, height, DEPTH_GRID, frame)
    if depth.size == 0:
        raise ValueError("the section does not reach DMC's atlas plane")
    z_voxel = _best_centre(depth_x, depth_y, depth, x_shift, y_shift, frame)
    _check_inside(z_voxel, x_shift, y_shift, frame)

    atlas_dots, sample_dots = _pairs(voxels, width, height, frame)
    return DmcPlacement(
        x_angle_deg=x_angle,
        y_angle_deg=y_angle,
        z_mm=_z_mm(z_voxel, frame),
        z_voxel=z_voxel,
        x_shift=x_shift,
        y_shift=y_shift,
        atlas_dots=atlas_dots,
        sample_dots=sample_dots,
    )


def _voxel_map(pixel_to_atlas_um: np.ndarray, frame: DmcFrame) -> np.ndarray:
    """Map file pixels ``[col, row, 1]`` to DMC's voxels (3x3).

    The continuous voxel coordinates (centres at integers) along DMC's x, y
    and z.
    """
    matrix = np.asarray(pixel_to_atlas_um, dtype=np.float64)
    atlas_rows = cast("list[list[float]]", matrix.tolist())
    rows: list[list[float]] = []
    for axis in (frame.x, frame.y, frame.z):
        per_um = [
            value / axis.resolution_um for value in atlas_rows[axis.index]
        ]
        rows.append([per_um[1], per_um[0], per_um[2]])
    return np.asarray(rows, dtype=np.float64)


def _project(
    voxels: np.ndarray, cols: np.ndarray, rows: np.ndarray
) -> list[np.ndarray]:
    """Map file pixels to DMC's continuous voxel x, y and depth."""
    points = np.stack([cols, rows, np.ones(cols.size)])
    return cast("list[np.ndarray]", list(voxels @ points))


def _shift_angle(shift: int, size: int) -> float:
    """Give the angle whose ``int(tan(angle) * size / 2)`` is *shift*.

    Its tangent puts the shift just past the whole voxel
    (``ANGLE_MARGIN``), so truncation never lands short and the angle reads
    as the slope it stands for.
    """
    if shift == 0:
        return 0.0
    target = shift + math.copysign(ANGLE_MARGIN, shift)
    angle = round(math.degrees(math.atan(target / (size / 2))), 6)
    assert int(math.tan(math.radians(angle)) * (size / 2)) == shift
    return angle


def dmc_depth(
    z_voxel: int,
    x_shift: int,
    y_shift: int,
    frame: DmcFrame,
    xy: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Give DMC's plane depth at atlas-plane pixels, before truncation.

    Parameters
    ----------
    z_voxel
        The plane's centre voxel along the normal axis.
    x_shift, y_shift
        The whole-voxel shifts DMC's angles give.
    frame
        DMC's frame of the atlas and plane.
    xy
        The atlas-plane pixels ``(x, y)``, whole numbers, as arrays.

    Returns
    -------
    numpy.ndarray
        The depth, in voxels, computed as ``SliceHandle.get_z_plane`` does.
    """
    grid_x, grid_y = xy
    if x_shift == 0 and y_shift == 0:
        return np.full(np.shape(grid_x), float(z_voxel))
    size_x, size_y = frame.x.size, frame.y.size
    # the plane through the centre, its right edge and its top edge, each
    # vector (z, y, x) from the centre; their cross product is its normal
    right = (float(x_shift), 0.0, size_x - 1 - size_x / 2)
    top = (float(-y_shift), -size_y / 2, 0.0)
    normal_z = right[1] * top[2] - right[2] * top[1]
    normal_y = right[2] * top[0] - right[0] * top[2]
    normal_x = right[0] * top[1] - right[1] * top[0]
    across = -normal_y * (grid_y - size_y / 2) - normal_x * (
        grid_x - size_x / 2
    )
    return across / normal_z + z_voxel


def _points(
    voxels: np.ndarray, width: int, height: int, n: int, frame: DmcFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the file on an n x n grid of pixels on DMC's atlas plane.

    Returns the continuous voxel x, y and depth of each pixel that maps onto
    the plane.
    """
    cols, rows = np.meshgrid(
        np.linspace(0.0, width - 1.0, n), np.linspace(0.0, height - 1.0, n)
    )
    x, y, z = _project(voxels, cols.ravel(), rows.ravel())
    inside = _on_plane(x + 0.5, y + 0.5, frame)
    return x[inside], y[inside], z[inside]


def _on_plane(x: np.ndarray, y: np.ndarray, frame: DmcFrame) -> np.ndarray:
    """Mark the DMC coordinates on its atlas plane (its own check)."""
    inside_x = (x >= 0) & (x <= frame.x.size - 1)
    return inside_x & (y >= 0) & (y <= frame.y.size - 1)


def _best_centre(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    x_shift: int,
    y_shift: int,
    frame: DmcFrame,
) -> int:
    """Choose the centre voxel whose DMC depths best match *depth*."""
    grid = (np.floor(x + 0.5), np.floor(y + 0.5))
    tilt = dmc_depth(0, x_shift, y_shift, frame, grid)
    middle = int(round(float(np.mean(depth - tilt))))
    candidates = range(middle - 2, middle + 3)
    errors = [
        float(
            np.mean(
                np.abs(
                    np.trunc(dmc_depth(z, x_shift, y_shift, frame, grid))
                    - depth
                )
            )
        )
        for z in candidates
    ]
    return candidates[int(np.argmin(errors))]


def _check_inside(
    z_voxel: int, x_shift: int, y_shift: int, frame: DmcFrame
) -> None:
    """Refuse a plane DMC refuses.

    That is a plane whose depth leaves the volume anywhere on DMC's atlas
    plane (the depth is linear, so at a corner).
    """
    corners_x = np.array([0.0, frame.x.size - 1.0, 0.0, frame.x.size - 1.0])
    corners_y = np.array([0.0, 0.0, frame.y.size - 1.0, frame.y.size - 1.0])
    depth = dmc_depth(z_voxel, x_shift, y_shift, frame, (corners_x, corners_y))
    if np.any(depth < 0) or np.any(depth >= frame.z.size):
        raise ValueError(
            "at its cutting angles the plane leaves the atlas volume at a "
            "corner of DMC-BrainMap's atlas plane, which DMC-BrainMap refuses"
        )


def _pairs(
    voxels: np.ndarray, width: int, height: int, frame: DmcFrame
) -> tuple[list[list[float]], list[list[float]]]:
    """Give point pairs (atlas plane, resized image) of the linear map.

    The pairs lie on the first grid of :data:`PAIR_GRIDS` that puts enough
    of them on the atlas plane.
    """
    size_x, size_y = frame.x.size, frame.y.size
    scale_x = (size_x - 1) / max(width - 1, 1)
    scale_y = (size_y - 1) / max(height - 1, 1)
    for n in PAIR_GRIDS:
        cols, rows = np.meshgrid(
            np.linspace(0.0, width - 1.0, n), np.linspace(0.0, height - 1.0, n)
        )
        cols, rows = cols.ravel(), rows.ravel()
        x, y, _depth = _project(voxels, cols, rows)
        inside = _on_plane(x + 0.5, y + 0.5, frame)
        if np.count_nonzero(inside) < MIN_PAIRS or not _spread(
            cols[inside], rows[inside]
        ):
            continue
        atlas = np.stack([x[inside] + 0.5, y[inside] + 0.5], axis=1)
        sample = np.stack(
            [cols[inside] * scale_x + 0.5, rows[inside] * scale_y + 0.5],
            axis=1,
        )
        return (
            np.round(atlas, DECIMALS).tolist(),
            np.round(sample, DECIMALS).tolist(),
        )
    raise ValueError(
        f"fewer than {MIN_PAIRS} points of the section fall on DMC-BrainMap's "
        "atlas plane"
    )


def _spread(cols: np.ndarray, rows: np.ndarray) -> bool:
    """Check that points span an area (not all on one line)."""
    points = np.stack([cols, rows, np.ones(cols.size)], axis=1)
    return cast("int", np.linalg.matrix_rank(points)) == 3


def _z_mm(z_voxel: int, frame: DmcFrame) -> float:
    """``atlasLocation``'s ``z_mm`` for a centre voxel, as DMC writes it."""
    resolution = frame.z.resolution_um
    millimetres = (frame.z_bregma - z_voxel) * resolution / 1000.0
    value = round(millimetres, dmc_decimals(resolution))
    return 0.0 if value == 0 else value


def dmc_decimals(resolution_um: float) -> int:
    """Give the decimals DMC writes a millimetre position with.

    Parameters
    ----------
    resolution_um
        The voxel size along the plane's normal axis, in micrometres.

    Returns
    -------
    int
        The fewest decimals (two at least) within 1% of one voxel
        (``utils/atlas_utils.get_decimal``).
    """
    step = resolution_um / 1000.0
    decimals = 2
    while abs(round(step, decimals) - step) >= 0.01 * step:
        decimals += 1
    return decimals


def natural_key(name: str) -> tuple[str | int, ...]:
    """Give a filename's natural sort key, as ``natsort.natsorted`` orders.

    Parameters
    ----------
    name
        The filename to sort.

    Returns
    -------
    tuple
        The text and the runs of digits in turn, text first (empty when the
        name starts with a digit), the digits as numbers.
    """
    key: list[str | int] = []
    for i, part in enumerate(re.split(r"(\d+)", name)):
        if i % 2:
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)
