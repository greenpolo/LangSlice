"""DeepSlice's batch steps on its anchorings: the angle vote and the spacing.

Vendored from DeepSlice 1.2.8 (Carey et al. 2023, Nature Communications;
https://github.com/PolarBean/DeepSlice, commit
b0d6965ee1861403bc5686eccab4f226c8b15c8b), its
``coord_post_processing`` package, mouse only. Copyright (c) 2023 Harry
Carey; MIT License, in ``LICENSE`` beside this file. Ported from its pandas
tables to numpy arrays operation for operation, so the numbers are
DeepSlice's own.

An anchoring is QuickNII's nine numbers ``(ox, oy, oz, ux, uy, uz, vx, vy,
vz)``: the plane's corner and its two edge vectors in the 25 um Allen
volume's voxels (528 x 320 x 456), ``oy`` the depth along the cutting axis.

- :func:`propagated` is ``DSModel.propagate_angles``: two passes, each a
  vote of the good sections' own dorsoventral and mediolateral angles,
  weighted by a Gaussian of each one's depth at the brain's centre (the
  middle of the brain counts most), then every section turned to the voted
  pair. DeepSlice's own vote fails on two voting sections or fewer (its
  weight list is then too short to index), and so does this one.
- :func:`spaced` is ``DSModel.enforce_index_spacing``: the good sections'
  depths reordered to follow the cutting index, then every section moved to
  an even spacing along it (the interval given, else DeepSlice's weighted
  estimate), fitted to the predicted depths.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

#: An array of anchorings, one row of nine numbers per section.
Anchorings = npt.NDArray[np.float64]

#: The 25 um Allen volume DeepSlice's mouse anchorings live in, voxels.
VOLUME = (528, 320, 456)
#: The voxel, um, DeepSlice counts depths and spacing in.
VOXEL_UM = 25.0
#: The depth range of the mouse volume DeepSlice weights by.
DEPTH_MAX = 528
#: The two angles a section is turned by, in DeepSlice's order.
DORSOVENTRAL, MEDIOLATERAL = "DV", "ML"
#: DeepSlice's names for the two ways the index can run.
ROSTRO_CAUDAL, CAUDAL_ROSTRO = "rostro-caudal", "caudal-rostro"
#: The share of the depth steps ``trim_mean`` leaves out, percent.
TRIM_PERCENT = 10


@dataclass(frozen=True)
class Spacing:
    """Hold the spacing DeepSlice used along the cutting order.

    Parameters
    ----------
    thickness_voxels
        The signed interval it spaced by, in voxels.
    estimated
        Whether DeepSlice estimated it.
    direction
        Its reading of the reordered depths, ``rostro-caudal`` or
        ``caudal-rostro``.
    """

    thickness_voxels: float
    estimated: bool
    direction: str


def propagated(
    anchorings: Anchorings, bad: npt.NDArray[np.bool_]
) -> tuple[Anchorings, list[tuple[float, float]]]:
    """Turn every section to the angles its good sections vote, twice.

    Parameters
    ----------
    anchorings
        The sections' anchorings, one row each.
    bad
        Which sections are left out of the vote.

    Returns
    -------
    Anchorings
        The anchorings, turned.
    list of tuple
        The (DV, ML) pair each of the two passes voted.
    """
    votes: list[tuple[float, float]] = []
    for _ in range(2):  # one angle's turn moves the other a little
        good = anchorings[~bad]
        depths = [_center_depth(row) for row in good]
        dv_angle = _mean_angle(
            [_angle(row, DORSOVENTRAL) for row in good], depths
        )
        ml_angle = _mean_angle(
            [_angle(row, MEDIOLATERAL) for row in good], depths
        )
        votes.append((dv_angle, ml_angle))
        anchorings = np.array(
            [
                _turned(
                    _turned(row, DORSOVENTRAL, dv_angle),
                    MEDIOLATERAL,
                    ml_angle,
                )
                for row in anchorings
            ]
        )
    return anchorings, votes


def spaced(
    anchorings: Anchorings,
    numbers: npt.NDArray[np.int64],
    bad: npt.NDArray[np.bool_],
    *,
    thickness_um: float | None,
) -> tuple[Anchorings, Spacing]:
    """Space the sections evenly along their cutting index.

    Parameters
    ----------
    anchorings
        The sections' anchorings, in index order.
    numbers
        Each section's index, ascending.
    bad
        Which sections are left out of the reordering and the fit.
    thickness_um
        DeepSlice's signed ``section_thickness``; None to estimate it.

    Returns
    -------
    Anchorings
        The anchorings, their depths spaced.
    Spacing
        The interval used and DeepSlice's reading of the direction.
    """
    anchorings = _ordered(anchorings, bad)
    depths = np.array([_center_depth(row) for row in anchorings])
    direction = _direction(depths)
    if thickness_um:
        thickness = thickness_um / VOXEL_UM
    else:
        thickness = _average_thickness(numbers, depths, bad)

    index_depths = numbers * thickness
    weights = _depth_weights(numbers, depths, bad)
    offset = np.average(depths - index_depths, weights=weights)
    anchorings = anchorings.copy()
    anchorings[:, 1] = anchorings[:, 1] + (index_depths + offset - depths)
    spacing = Spacing(
        thickness_voxels=float(thickness),
        estimated=not thickness_um,
        direction=direction,
    )
    return anchorings, spacing


# --- the plane of one anchoring ------------------------------------------------


def _plane(section: Anchorings) -> tuple[Anchorings, float]:
    """Return the plane's normal (over 9) and its constant k."""
    a, b, c = section[0:3], section[3:6], section[6:9]
    cross = np.cross(b, c)
    cross /= 9
    k = -((a[0] * cross[0]) + (a[1] * cross[1]) + (a[2] * cross[2]))
    return cross, float(k)


def _center_depth(section: Anchorings) -> float:
    """Return the plane's depth at the middle of the brain's other axes."""
    cross, k = _plane(section)
    width, height = 456, 320
    linear_point = ((width / 2) * cross[0]) + ((height / 2) * cross[2]) + k
    return float(-(linear_point / cross[1]))


def _angle(section: Anchorings, direction: str) -> float:
    """Return the plane's dorsoventral or mediolateral angle, degrees."""
    cross, k = _plane(section)
    points = section.copy()
    for i in range(3):  # edge vectors to absolute points
        points[i + 3] += points[i]
        points[i + 6] += points[i]
    if direction == MEDIOLATERAL:
        a = points[0:2]
        linear_point = (
            ((points[0] - 100) * cross[0]) + (points[2] * cross[2]) + k
        )
        depth = -(linear_point / cross[1])
        b = np.array((points[0] - 100, depth))
        c = b + [100, 0]
    else:
        a = points[1:3]
        linear_point = (
            (points[0] * cross[0]) + ((points[2] - 100) * cross[2]) + k
        )
        depth = -(linear_point / cross[1])
        b = np.array((depth, points[2] - 100))
        c = b + [0, 100]

    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = float(np.degrees(np.arccos(cosine)))
    if direction == MEDIOLATERAL and b[1] > a[1]:
        angle *= -1
    if direction == DORSOVENTRAL and b[0] < a[0]:
        angle *= -1
    return angle


# --- the angle vote --------------------------------------------------------------


def _gaussian_weights(size: int) -> npt.NDArray[np.float64]:
    """Return DeepSlice's weights over ``size`` depths, peaked mid-brain."""
    x = np.linspace(-np.pi, np.pi, size)
    weights = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    weights[weights > size - 1] = size - 1
    weights[weights < 0] = 0
    return weights


def _mean_angle(angles: Sequence[float], depths: Sequence[float]) -> float:
    """Return the weighted mean of the good sections' own angles."""
    gaussian: Sequence[float] = [1.0] * len(depths)
    if len(depths) > 2:
        gaussian = list(_gaussian_weights(DEPTH_MAX))
    centers = np.array(depths)
    centers[centers < 0] = 0
    centers[centers > DEPTH_MAX] = DEPTH_MAX - 1
    weights = [gaussian[int(y)] for y in centers]
    return float(np.average(angles, weights=weights))


def _turned(section: Anchorings, direction: str, mean: float) -> Anchorings:
    """Turn a plane about its centre so that its angle is ``mean``."""
    difference = _angle(section, direction) - mean
    return _rotated(section, -difference, direction)


def _rotated(
    section: Anchorings, degrees: float, direction: str
) -> Anchorings:
    """Rotate a plane about the axis DeepSlice picks through the volume."""
    cross, k = _plane(section)
    points = section.copy()
    for i in range(3):
        points[i + 3] += points[i]
        points[i + 6] += points[i]
    volume = np.array(VOLUME)
    centre = volume / 2
    plane, through = _section_plane(volume, cross, k, centre)

    axis = _axis(volume - np.array(through), cross, k, plane, direction)
    rotation = _rotation(axis, degrees)
    rotated = np.dot(points.reshape(3, 3) - through, rotation) + through
    rotated = rotated.reshape(9)
    for i in range(3):  # back to edge vectors
        rotated[i + 3] -= rotated[i]
        rotated[i + 6] -= rotated[i]
    return rotated


def _section_plane(
    volume: npt.NDArray[np.int64],
    cross: Anchorings,
    k: float,
    centre: Anchorings,
) -> tuple[int, tuple[float, float, float]]:
    """Return which plane a section is nearest (0 coronal, 1 sagittal,
    2 horizontal) and the point on it through the volume's middle."""
    cor_linear = (
        ((volume[0] / 2) * cross[0]) + ((volume[2] / 2) * cross[2]) + k
    )
    cor_y = -(cor_linear / cross[1])
    sag_linear = (
        ((volume[1] / 2) * cross[1]) + ((volume[2] / 2) * cross[2]) + k
    )
    sag_x = -(sag_linear / cross[0])
    horz_linear = (
        ((volume[0] / 2) * cross[0]) + ((volume[1] / 2) * cross[1]) + k
    )
    horz_z = -(horz_linear / cross[2])
    offsets = (cor_y - centre[1], sag_x - centre[0], horz_z - centre[2])
    plane = int(np.argmin(np.abs(offsets)))
    candidates = (
        (volume[0] / 2, cor_y, volume[2] / 2),
        (sag_x, volume[1] / 2, volume[2] / 2),
        (volume[0] / 2, volume[1] / 2, horz_z),
    )
    x, y, z = candidates[plane]
    return plane, (float(x), float(y), float(z))


def _axis(
    translated: Anchorings,
    cross: Anchorings,
    k: float,
    plane: int,
    direction: str,
) -> Anchorings:
    """Return the axis a plane turns about (DeepSlice's ``get_axis``)."""
    half = translated / 2
    if plane == 0:
        cor_y = -(
            (((half[0]) * cross[0]) + ((half[2]) * cross[2]) + k) / cross[1]
        )
        axis = (half[0], cor_y, half[2])
        if direction == DORSOVENTRAL:
            linear = (translated[0] * cross[0]) + (half[2] * cross[2]) + k
            other = (translated[0], -(linear / cross[1]), half[2])
        else:
            linear = (half[0] * cross[0]) + (translated[2] * cross[2]) + k
            other = (half[0], -(linear / cross[1]), translated[2])
    elif plane == 1:
        sag_x = -(
            (((half[1]) * cross[1]) + ((half[2]) * cross[2]) + k) / cross[0]
        )
        axis = (sag_x, half[1], half[2])
        if direction == DORSOVENTRAL:
            linear = (translated[1] * cross[1]) + (half[2] * cross[2]) + k
            other = (-(linear / cross[0]), translated[1], half[2])
        else:
            linear = (half[1] * cross[1]) + (translated[2] * cross[2]) + k
            other = (-(linear / cross[0]), half[1], translated[2])
    else:
        horz_z = -(
            (((half[0]) * cross[0]) + ((half[1]) * cross[1]) + k) / cross[2]
        )
        axis = (half[0], half[1], horz_z)
        if direction == DORSOVENTRAL:
            linear = (translated[0] * cross[0]) + (half[1] * cross[1]) + k
            other = (translated[0], half[1], -(linear / cross[2]))
        else:
            linear = (half[0] * cross[0]) + (translated[1] * cross[1]) + k
            other = (half[0], translated[1], -(linear / cross[2]))
    return np.array(axis) - np.array(other)


def _rotation(axis: Anchorings, degrees: float) -> Anchorings:
    """Return the 3x3 rotation about an axis (Euler-Rodrigues)."""
    angle = np.radians(degrees)
    unit = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2.0)
    b, c, d = unit * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
            [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
            [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc],
        ]
    )


# --- the spacing -----------------------------------------------------------------


def _direction(depths: Anchorings) -> str:
    """Return which way the depths run along the index (a trimmed mean)."""
    steps = depths[1:] - depths[:-1]
    n = len(steps)
    trim = int(round(n * (float(TRIM_PERCENT) / 100) / 2))
    if np.mean(steps[trim + 1 : n - trim]) > 0:
        return ROSTRO_CAUDAL
    return CAUDAL_ROSTRO


def _ordered(anchorings: Anchorings, bad: npt.NDArray[np.bool_]) -> Anchorings:
    """Give the good sections their depths in index order.

    The good sections' ``oy`` are sorted by depth at the brain's centre, in
    the direction the depths run, and handed out along the index; the bad
    sections keep theirs.
    """
    depths = np.array([_center_depth(row) for row in anchorings])
    good = np.flatnonzero(~bad)
    order = np.argsort(depths[good], kind="stable")
    if _direction(depths) == CAUDAL_ROSTRO:
        order = np.argsort(-depths[good], kind="stable")
    anchorings = anchorings.copy()
    anchorings[good, 1] = anchorings[good, 1][order]
    return anchorings


def _depth_weights(
    numbers: npt.NDArray[np.int64],
    depths: Anchorings,
    bad: npt.NDArray[np.bool_] | None,
) -> list[float]:
    """Return DeepSlice's weight of each section in the spacing fit."""
    gaussian = _gaussian_weights(DEPTH_MAX + 1)
    clipped = np.array(depths)
    clipped[clipped < 0] = 0
    clipped[clipped > DEPTH_MAX] = DEPTH_MAX
    weights = [float(gaussian[int(y)]) for y in clipped]
    if len(numbers) <= 2:
        weights = [0.5, 0.5]
    if bad is not None:
        weights = [
            0.0 if is_bad else w
            for w, is_bad in zip(weights, bad, strict=False)
        ]
    return weights


def _average_thickness(
    numbers: npt.NDArray[np.int64],
    depths: Anchorings,
    bad: npt.NDArray[np.bool_],
) -> float:
    """Return DeepSlice's weighted estimate of the interval, voxels."""
    numbers, depths = numbers[~bad], depths[~bad]
    number_steps = numbers[:-1] - numbers[1:]
    depth_steps = depths[:-1] - depths[1:]
    weights = _depth_weights(numbers, depths, None)
    thicknesses = depth_steps / number_steps
    return float(np.average(thicknesses, weights=weights[1:]))
