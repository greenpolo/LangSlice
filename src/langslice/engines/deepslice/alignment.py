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
from typing import cast

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


@dataclass(frozen=True)
class _Plane:
    """A plane as DeepSlice writes it: normal over 9, and constant ``k``."""

    nx: np.float64
    ny: np.float64
    nz: np.float64
    k: np.float64


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
        good = _rows(anchorings[~bad])
        depths = [_center_depth(row) for row in good]
        dv_angle = _mean_angle(
            [_angle(row, DORSOVENTRAL) for row in good], depths
        )
        ml_angle = _mean_angle(
            [_angle(row, MEDIOLATERAL) for row in good], depths
        )
        votes.append((dv_angle, ml_angle))
        turned = [
            _turned(
                _turned(row, DORSOVENTRAL, dv_angle), MEDIOLATERAL, ml_angle
            )
            for row in _rows(anchorings)
        ]
        anchorings = np.array(turned)
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
    depths = _depths(anchorings)
    direction = _direction(depths)
    if thickness_um:
        thickness = thickness_um / VOXEL_UM
    else:
        thickness = _average_thickness(numbers, depths, bad)

    index_depths = numbers * thickness
    weights = _depth_weights(numbers, depths, bad)
    offset = _average(depths - index_depths, weights)
    anchorings = anchorings.copy()
    anchorings[:, 1] = anchorings[:, 1] + (index_depths + offset - depths)
    spacing = Spacing(
        thickness_voxels=thickness,
        estimated=not thickness_um,
        direction=direction,
    )
    return anchorings, spacing


# --- numpy's untyped edges ---------------------------------------------------


def _at(values: npt.NDArray[np.float64], index: int) -> np.float64:
    """Return one element, as the float64 it is (numpy leaves it untyped)."""
    return cast("np.float64", values[index])


def _rows(anchorings: Anchorings) -> list[Anchorings]:
    """Return an array's rows, each an anchoring."""
    return cast("list[Anchorings]", list(anchorings))


def _floats(values: npt.NDArray[np.float64]) -> list[float]:
    """Return an array's elements as Python floats."""
    return cast("list[float]", values.tolist())


def _average(
    values: npt.NDArray[np.float64], weights: Sequence[float]
) -> float:
    """Return numpy's weighted average (a float64 numpy leaves untyped)."""
    return float(cast("np.float64", np.average(values, weights=weights)))


# --- the plane of one anchoring ----------------------------------------------


def _plane(section: Anchorings) -> _Plane:
    """Return the plane through an anchoring (DeepSlice's equation)."""
    a, b, c = section[0:3], section[3:6], section[6:9]
    cross = cast("npt.NDArray[np.float64]", np.cross(b, c))
    cross /= 9
    nx, ny, nz = _at(cross, 0), _at(cross, 1), _at(cross, 2)
    k = -((_at(a, 0) * nx) + (_at(a, 1) * ny) + (_at(a, 2) * nz))
    return _Plane(nx, ny, nz, k)


def _center_depth(section: Anchorings) -> float:
    """Return the plane's depth at the middle of the brain's other axes."""
    plane = _plane(section)
    width, height = 456, 320
    linear_point = (
        ((width / 2) * plane.nx) + ((height / 2) * plane.nz) + plane.k
    )
    return float(-(linear_point / plane.ny))


def _depths(anchorings: Anchorings) -> npt.NDArray[np.float64]:
    return np.array([_center_depth(row) for row in _rows(anchorings)])


def _absolute(section: Anchorings) -> Anchorings:
    """Return the anchoring's corner and the ends of its two edges."""
    points = section.copy()
    for i in range(3):
        points[i + 3] += points[i]
        points[i + 6] += points[i]
    return points


def _angle(section: Anchorings, direction: str) -> float:
    """Return the plane's dorsoventral or mediolateral angle, degrees."""
    plane = _plane(section)
    points = _absolute(section)
    x, y, z = _at(points, 0), _at(points, 1), _at(points, 2)
    if direction == MEDIOLATERAL:
        a = points[0:2]
        linear_point = ((x - 100) * plane.nx) + (z * plane.nz) + plane.k
        depth = -(linear_point / plane.ny)
        b = np.array((x - 100, depth))
        c = b + [100, 0]
        flipped = depth > y  # b[1] > a[1]
    else:
        a = points[1:3]
        linear_point = (x * plane.nx) + ((z - 100) * plane.nz) + plane.k
        depth = -(linear_point / plane.ny)
        b = np.array((depth, z - 100))
        c = b + [0, 100]
        flipped = depth < y  # b[0] < a[0]

    ba, bc = a - b, c - b
    dot = cast("np.float64", np.dot(ba, bc))
    norms = cast("np.float64", np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = dot / norms
    radians = cast("np.float64", np.arccos(cosine))
    angle = float(cast("np.float64", np.degrees(radians)))
    return -angle if flipped else angle


# --- the angle vote ----------------------------------------------------------


def _gaussian_weights(size: int) -> npt.NDArray[np.float64]:
    """Return DeepSlice's weights over ``size`` depths, peaked mid-brain."""
    x = np.linspace(-np.pi, np.pi, size)
    weights = cast(
        "npt.NDArray[np.float64]", np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    )
    weights[weights > size - 1] = size - 1
    weights[weights < 0] = 0
    return weights


def _mean_angle(angles: Sequence[float], depths: Sequence[float]) -> float:
    """Return the weighted mean of the good sections' own angles."""
    gaussian = [1.0] * len(depths)
    if len(depths) > 2:
        gaussian = _floats(_gaussian_weights(DEPTH_MAX))
    centers = np.array(depths)
    centers[centers < 0] = 0
    centers[centers > DEPTH_MAX] = DEPTH_MAX - 1
    weights = [gaussian[int(y)] for y in _floats(centers)]
    return _average(np.array(angles), weights)


def _turned(section: Anchorings, direction: str, mean: float) -> Anchorings:
    """Turn a plane about its centre so that its angle is ``mean``."""
    difference = _angle(section, direction) - mean
    return _rotated(section, -difference, direction)


def _rotated(
    section: Anchorings, degrees: float, direction: str
) -> Anchorings:
    """Rotate a plane about the axis DeepSlice picks through the volume."""
    plane = _plane(section)
    points = _absolute(section)
    nearest, through = _section_plane(plane)
    translated = np.array(VOLUME) - np.array(through)
    axis = _axis(translated, plane, nearest, direction)
    rotation = _rotation(axis, degrees)

    moved = points.reshape(3, 3) - through
    rotated = cast("Anchorings", np.dot(moved, rotation)) + through
    rotated = rotated.reshape(9)
    for i in range(3):  # back to edge vectors
        rotated[i + 3] -= rotated[i]
        rotated[i + 6] -= rotated[i]
    return rotated


def _section_plane(plane: _Plane) -> tuple[int, tuple[float, float, float]]:
    """Return the plane a section is nearest and its point mid-volume.

    The plane is 0 coronal, 1 sagittal or 2 horizontal.
    """
    vx, vy, vz = VOLUME
    cor_linear = ((vx / 2) * plane.nx) + ((vz / 2) * plane.nz) + plane.k
    cor_y = -(cor_linear / plane.ny)
    sag_linear = ((vy / 2) * plane.ny) + ((vz / 2) * plane.nz) + plane.k
    sag_x = -(sag_linear / plane.nx)
    horz_linear = ((vx / 2) * plane.nx) + ((vy / 2) * plane.ny) + plane.k
    horz_z = -(horz_linear / plane.nz)

    offsets = (cor_y - vy / 2, sag_x - vx / 2, horz_z - vz / 2)
    nearest = int(np.argmin(np.abs(offsets)))
    candidates = (
        (vx / 2, float(cor_y), vz / 2),
        (float(sag_x), vy / 2, vz / 2),
        (vx / 2, vy / 2, float(horz_z)),
    )
    return nearest, candidates[nearest]


def _axis(
    translated: Anchorings, plane: _Plane, nearest: int, direction: str
) -> Anchorings:
    """Return the axis a plane turns about (DeepSlice's ``get_axis``)."""
    tx, ty, tz = _at(translated, 0), _at(translated, 1), _at(translated, 2)
    hx, hy, hz = tx / 2, ty / 2, tz / 2
    nx, ny, nz, k = plane.nx, plane.ny, plane.nz, plane.k
    dorsoventral = direction == DORSOVENTRAL
    if nearest == 0:
        axis = (hx, -(((hx * nx) + (hz * nz) + k) / ny), hz)
        if dorsoventral:
            other = (tx, -(((tx * nx) + (hz * nz) + k) / ny), hz)
        else:
            other = (hx, -(((hx * nx) + (tz * nz) + k) / ny), tz)
    elif nearest == 1:
        axis = (-(((hy * ny) + (hz * nz) + k) / nx), hy, hz)
        if dorsoventral:
            other = (-(((ty * ny) + (hz * nz) + k) / nx), ty, hz)
        else:
            other = (-(((hy * ny) + (tz * nz) + k) / nx), hy, tz)
    else:
        axis = (hx, hy, -(((hx * nx) + (hy * ny) + k) / nz))
        if dorsoventral:
            other = (tx, hy, -(((tx * nx) + (hy * ny) + k) / nz))
        else:
            other = (hx, ty, -(((hx * nx) + (ty * ny) + k) / nz))
    return np.array(axis) - np.array(other)


def _rotation(axis: Anchorings, degrees: float) -> Anchorings:
    """Return the 3x3 rotation about an axis (Euler-Rodrigues)."""
    angle = float(cast("np.float64", np.radians(degrees)))
    unit = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2.0)
    scaled = unit * math.sin(angle / 2.0)
    b, c, d = _at(scaled, 0), _at(scaled, 1), _at(scaled, 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
            [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
            [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc],
        ]
    )


# --- the spacing -------------------------------------------------------------


def _direction(depths: npt.NDArray[np.float64]) -> str:
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
    depths = _depths(anchorings)
    good = np.flatnonzero(~bad)
    order = np.argsort(depths[good], kind="stable")
    if _direction(depths) == CAUDAL_ROSTRO:
        order = np.argsort(-depths[good], kind="stable")
    anchorings = anchorings.copy()
    anchorings[good, 1] = anchorings[good, 1][order]
    return anchorings


def _depth_weights(
    numbers: npt.NDArray[np.int64],
    depths: npt.NDArray[np.float64],
    bad: npt.NDArray[np.bool_] | None,
) -> list[float]:
    """Return DeepSlice's weight of each section in the spacing fit."""
    gaussian = _floats(_gaussian_weights(DEPTH_MAX + 1))
    clipped = np.array(depths)
    clipped[clipped < 0] = 0
    clipped[clipped > DEPTH_MAX] = DEPTH_MAX
    weights = [gaussian[int(y)] for y in _floats(clipped)]
    if len(numbers) <= 2:
        weights = [0.5, 0.5]
    if bad is None:
        return weights
    kept: list[float] = []
    flags = cast("list[bool]", bad.tolist())
    for weight, is_bad in zip(weights, flags, strict=False):
        if is_bad:  # a bad section weighs nothing
            kept.append(0.0)
        else:
            kept.append(weight)
    return kept


def _average_thickness(
    numbers: npt.NDArray[np.int64],
    depths: npt.NDArray[np.float64],
    bad: npt.NDArray[np.bool_],
) -> float:
    """Return DeepSlice's weighted estimate of the interval, voxels."""
    numbers, depths = numbers[~bad], depths[~bad]
    number_steps = numbers[:-1] - numbers[1:]
    depth_steps = depths[:-1] - depths[1:]
    weights = _depth_weights(numbers, depths, None)
    return _average(depth_steps / number_steps, weights[1:])
