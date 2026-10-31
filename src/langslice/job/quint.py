"""A job's registrations as QUINT JSON (QuickNII / VisuAlign), and back.

The QUINT anchoring vector ``[ox, oy, oz, ux, uy, uz, vx, vy, vz]`` defines
how a 2-D section image maps into a 3-D atlas reference volume:

    atlas_point = o + u * (px / W) + v * (py / H)

where ``(px, py)`` are pixel coordinates in the section image of size ``W x H``,
``o`` is the top-left corner in atlas voxel space, and ``u`` / ``v`` are
direction vectors pointing toward the top-right and bottom-left corners.

:func:`job_export` writes it from each section's exact pixel -> atlas map
(``registration.json``'s ``pixel_to_atlas_um``), so any plane and cutting angle
is represented; :mod:`langslice.job.imports.readers` reads it back through the
inverses here.

QuickNII and VisuAlign make ``u`` and ``v`` perpendicular when they read a
file, and Nutil and PyNutil use them as written, so a sheared anchoring means
different placements in different programs. The export therefore writes the
nearest perpendicular anchoring (:func:`perpendicular_anchoring`), which every
reader takes as written, and ``visualign.json`` carries the shear it drops in
its markers, with the deformation (:func:`markers_on_anchoring`).
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from langslice import __version__
from langslice.core.atlas.images import AUGMENTED_AP_OFFSET_MM, AUGMENTED_ATLAS
from langslice.core.maps.section_maps import marker_grid

#: The QuickNII ``.cutlas`` target of a BrainGlobe atlas. QuickNII and
#: VisuAlign ship the Allen CCFv3 at 25 um only
#: (``ABA_Mouse_CCFv3_2017_25um.cutlas``; DeepSlice writes the same name),
#: so every Allen resolution is exported to it (:data:`_TARGET_RESOLUTION_UM`).
#: The augmented CCFv3 atlas is the Allen 25 um volume with 38 more planes at
#: the front and 24 behind, so it is exported to the Allen target on the Allen
#: grid, shifted by ``AUGMENTED_AP_OFFSET_MM`` (:func:`_on_allen_grid`).
#: The rat name is DeepSlice's, the one its QuickNII files open with. The
#: Kim atlas name follows the same pattern; any other atlas is exported to
#: ``<name>.cutlas``.
_KNOWN_TARGETS: dict[str, str] = {
    "allen_mouse_25um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_10um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_50um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_100um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "whs_sd_rat_39um": "WHS_Rat_v4_39um.cutlas",
    AUGMENTED_ATLAS: "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "kim_unified_25um": "Kim_UnifiedMouse_v1_25um.cutlas",
}
#: A target whose voxels differ from the BrainGlobe atlas's: its voxel size.
#: QuickNII coordinates are voxel EDGES (physical / voxel size), so an
#: anchoring moves to the target's grid by the factor of the two sizes (the
#: Allen volumes at every resolution span the same 13.2 x 8.0 x 11.4 mm).
_TARGET_RESOLUTION_UM: dict[str, float] = {
    "ABA_Mouse_CCFv3_2017_25um.cutlas": 25.0,
}


#: The Allen CCFv3 25 um volume's shape in BrainGlobe's ``asr`` order.
_CCFV3_25UM_SHAPE = (528, 320, 456)


def _on_allen_grid(atlas: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """For the augmented atlas, the Allen 25 um atlas record and the shift
    (atlas micrometres) from the job's coordinates to the Allen volume's:
    the front extension is cut off. Any other atlas: itself, no shift."""
    shift = np.zeros(3)
    if atlas.get("name") != AUGMENTED_ATLAS:
        return atlas, shift
    orientation = str(atlas["orientation"])
    if orientation != "asr":
        raise ValueError(f"{AUGMENTED_ATLAS}: expected orientation asr, got {orientation}")
    shift[0] = AUGMENTED_AP_OFFSET_MM * 1000.0
    return {**atlas, "name": "allen_mouse_25um", "shape": list(_CCFV3_25UM_SHAPE)}, shift


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnchoringVector:
    """Nine-element anchoring vector used by QuickNII / VisuAlign / Nutil."""

    ox: float
    oy: float
    oz: float
    ux: float
    uy: float
    uz: float
    vx: float
    vy: float
    vz: float

    def to_list(self) -> list[float]:
        return [
            self.ox,
            self.oy,
            self.oz,
            self.ux,
            self.uy,
            self.uz,
            self.vx,
            self.vy,
            self.vz,
        ]


@dataclass
class SliceExport:
    """Single slice entry in a QUINT JSON."""

    filename: str
    anchoring: AnchoringVector
    width: int
    height: int
    nr: int = 1
    markers: list[Any] = field(default_factory=list)


@dataclass
class QUINTExport:
    """Top-level QUINT / ABBA JSON structure."""

    target: str
    slices: list[SliceExport]
    name: str = ""
    aligner: str = f"langslice_{__version__}"


# ---------------------------------------------------------------------------
# Anchoring computation
# ---------------------------------------------------------------------------


def _resolve_target(atlas_name: str) -> str:
    """Map a BrainGlobe atlas name to a ``.cutlas`` target identifier."""
    if atlas_name in _KNOWN_TARGETS:
        return _KNOWN_TARGETS[atlas_name]
    return f"{atlas_name}.cutlas"


def to_target_grid(anchoring: AnchoringVector, atlas_name: str,
                   resolution_um: Sequence[float]) -> AnchoringVector:
    """*anchoring*, in the voxel space of the atlas *atlas_name* (voxels of
    *resolution_um*), in its QuickNII target's voxel space."""
    target = _TARGET_RESOLUTION_UM.get(_resolve_target(atlas_name))
    sizes = {float(value) for value in resolution_um}
    if target is None or sizes == {target}:
        return anchoring
    if len(sizes) != 1:
        raise ValueError(f"{atlas_name}: anisotropic voxels cannot be exported to its target")
    factor = sizes.pop() / target
    return AnchoringVector(*(round(float(value) * factor, 6) for value in anchoring.to_list()))


def export_to_dict(export: QUINTExport) -> dict[str, Any]:
    """Serialise a :class:`QUINTExport` to a JSON-ready dict."""
    return {
        "name": export.name,
        "target": export.target,
        "aligner": export.aligner,
        "slices": [
            {
                "filename": s.filename,
                "anchoring": s.anchoring.to_list(),
                "height": s.height,
                "width": s.width,
                "nr": s.nr,
                "markers": s.markers,
            }
            for s in export.slices
        ],
    }


# ---------------------------------------------------------------------------
# BrainGlobe atlas micrometres <-> QuickNII voxels
# ---------------------------------------------------------------------------
#
# The conversion goes through the two volumes' anatomical spaces
# (``brainglobe_space``) and the voxel-edge shift below; on the Allen CCFv3
# 25 um atlas it agrees with DeepSlice and with QUINT registrations made by
# hand. For other atlases the QuickNII target is taken to share the
# BrainGlobe voxel grid.

#: QuickNII / VisuAlign volume axes (x, y, z) as a BrainGlobe origin string:
#: x runs left -> right (ML), y posterior -> anterior (AP), z inferior ->
#: superior (DV). Against a BrainGlobe ``asr`` volume every axis is reversed.
QUICKNII_ORIGIN = "lpi"
#: QuickNII coordinates are continuous voxel-EDGE coordinates (voxel i spans
#: [i, i+1)); BrainGlobe micrometres put voxel i's CENTRE at i * resolution.
VOXEL_EDGE_TO_CENTRE = 0.5

_AXIS_OF_LETTER = {"a": "ap", "p": "ap", "s": "dv", "i": "dv", "l": "ml", "r": "ml"}


def _quicknii_spaces(atlas: dict[str, Any]) -> tuple[Any, Any]:
    from brainglobe_space import AnatomicalSpace

    orientation = str(atlas["orientation"])
    shape = [int(v) for v in atlas["shape"]]
    index = {_AXIS_OF_LETTER[letter]: axis for axis, letter in enumerate(orientation)}
    quicknii_shape = tuple(shape[index[name]] for name in ("ml", "ap", "dv"))
    return (AnatomicalSpace(orientation, shape=tuple(shape)),
            AnatomicalSpace(QUICKNII_ORIGIN, shape=quicknii_shape))


def atlas_um_to_quicknii_points(points_um: Any, atlas: dict[str, Any]) -> np.ndarray:
    """BrainGlobe atlas micrometres (``(n, 3)``, the atlas's axis order) as
    QuickNII voxel coordinates (x ML, y AP, z DV, voxel edges). *atlas* is
    ``registration.json``'s atlas record (orientation, shape, resolution_um)."""
    atlas, shift = _on_allen_grid(atlas)
    bg_space, quicknii_space = _quicknii_spaces(atlas)
    resolution = np.asarray(atlas["resolution_um"], dtype=np.float64)
    edges = (np.asarray(points_um, dtype=np.float64).reshape(-1, 3) - shift) / resolution
    edges = edges + VOXEL_EDGE_TO_CENTRE
    return np.asarray(bg_space.map_points_to(quicknii_space, edges), dtype=np.float64)


def quicknii_points_to_atlas_um(points: Any, atlas: dict[str, Any]) -> np.ndarray:
    """The inverse of :func:`atlas_um_to_quicknii_points`: QuickNII voxel
    coordinates (``(n, 3)``: x ML, y AP, z DV, voxel edges, on the atlas's
    own grid) as BrainGlobe atlas micrometres in the atlas's axis order.
    For the augmented atlas the anchoring is on the Allen grid and the
    micrometres are the augmented atlas's (0.35 mm further from the front)."""
    atlas, shift = _on_allen_grid(atlas)
    bg_space, quicknii_space = _quicknii_spaces(atlas)
    forward = np.asarray(bg_space.transformation_matrix_to(quicknii_space), dtype=np.float64)
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack([values, np.ones(len(values))])
    edges = (np.linalg.inv(forward) @ homogeneous.T).T[:, :3]
    resolution = np.asarray(atlas["resolution_um"], dtype=np.float64)
    return (edges - VOXEL_EDGE_TO_CENTRE) * resolution + shift


def quicknii_target(atlas_name: str) -> str:
    """The QuickNII ``.cutlas`` target a BrainGlobe atlas is exported to."""
    return _resolve_target(atlas_name)


def from_target_grid(anchoring: Sequence[float], atlas_name: str,
                     resolution_um: Sequence[float]) -> list[float]:
    """The inverse of :func:`to_target_grid`: nine anchoring numbers in the
    QuickNII target's voxel space, in the voxel space of the BrainGlobe atlas
    *atlas_name* (voxels of *resolution_um*). Not rounded."""
    target = _TARGET_RESOLUTION_UM.get(_resolve_target(atlas_name))
    sizes = {float(value) for value in resolution_um}
    values = [float(value) for value in anchoring]
    if target is None or sizes == {target}:
        return values
    if len(sizes) != 1:
        raise ValueError(f"{atlas_name}: anisotropic voxels cannot be read from its target")
    factor = target / sizes.pop()
    return [value * factor for value in values]


def anchoring_from_pixel_map(
    pixel_to_atlas_um: Any, width: int, height: int, atlas: dict[str, Any],
) -> AnchoringVector:
    """The QuickNII anchoring of an image of *width* x *height* pixels whose
    pixel ``[row, col, 1]`` (centres at integers) maps to atlas micrometres
    by the 3x3 *pixel_to_atlas_um*. QuickNII measures the image from its
    top-left CORNER as fractions of width and height: ``atlas = o + u * x/W +
    v * y/H``."""
    matrix = np.asarray(pixel_to_atlas_um, dtype=np.float64)
    corner = matrix @ np.array([-0.5, -0.5, 1.0])
    right = corner + matrix[:, 1] * float(width)
    down = corner + matrix[:, 0] * float(height)
    o, top_right, bottom_left = atlas_um_to_quicknii_points(
        np.stack([corner, right, down]), atlas)
    u, v = top_right - o, bottom_left - o
    return AnchoringVector(*(round(float(value), 6) for value in (*o, *u, *v)))


#: A section whose perpendicular anchoring moves no corner of its image by
#: more than this many pixels from its exact one needs no markers for the
#: difference (the rounding of the written numbers moves it far less).
SHEAR_TOLERANCE_PX = 0.01
#: Steps of the marker grid of a section without a deformation: its markers
#: carry the shear alone, an affine change any markers around the image
#: reproduce exactly, so a few do.
SHEAR_MARKER_DIVISIONS = 2


def perpendicular_anchoring(anchoring: AnchoringVector) -> AnchoringVector:
    """Make an anchoring's ``u`` and ``v`` perpendicular, as near as can be.

    QuickNII and VisuAlign make ``u`` and ``v`` perpendicular when they
    read a file (``u`` kept, ``v`` turned to it: QuickNII's
    ``SliceData.unpack``, VisuAlign's ``Slice.recompose``), so they show
    a sheared anchoring as another placement. The anchoring returned here
    lies in the same plane, has the same image centre, and of all
    anchorings with perpendicular axes it is the nearest over the image
    (least squares); those programs read it as written.

    Parameters
    ----------
    anchoring
        The exact anchoring, sheared or not, in QuickNII voxels.

    Returns
    -------
    AnchoringVector
        The perpendicular anchoring, rounded as the export writes it.

    Raises
    ------
    ValueError
        When ``u`` is zero or ``v`` is parallel to it.

    Notes
    -----
    In the plane's basis ``e1 = u / |u|`` and ``e2`` (the part of ``v``
    perpendicular to ``e1``, normalized), ``u = (u1, 0)`` and ``v = (v1,
    v2)``. New axes along ``r1 = (cos a, sin a)`` and ``r2 = (-sin a, cos
    a)`` with lengths ``r1 . u`` and ``r2 . v`` miss ``u`` and ``v`` by
    ``|u|^2 + |v|^2 - (r1 . u)^2 - (r2 . v)^2``, least at ``tan 2a = -2 v1
    v2 / (u1^2 + v2^2 - v1^2)``. A point at image fractions ``(s, t)``
    moves by ``(u' - u)(s - 1/2) + (v' - v)(t - 1/2)``, whose mean square
    over the image is that miss over 12.
    """
    values = np.asarray(anchoring.to_list(), dtype=np.float64)
    origin, u, v = values[:3], values[3:6], values[6:]
    u1 = float(np.linalg.norm(u))
    if u1 == 0.0:
        raise ValueError("the anchoring's u is zero")
    e1 = u / u1
    v1 = _dot(v, e1)
    across = v - v1 * e1
    v2 = float(np.linalg.norm(across))
    if v2 <= 1e-9 * float(np.linalg.norm(v)):
        raise ValueError("the anchoring's v is parallel to its u")
    e2 = across / v2

    # the axes turned by a, each as long as the old axis along it
    angle = 0.5 * math.atan2(-2.0 * v1 * v2, u1 * u1 + v2 * v2 - v1 * v1)
    r1 = math.cos(angle) * e1 + math.sin(angle) * e2
    r2 = math.cos(angle) * e2 - math.sin(angle) * e1
    new_u = _dot(r1, u) * r1
    new_v = _dot(r2, v) * r2
    new_origin = origin + (u + v - new_u - new_v) / 2.0
    numbers = np.concatenate([new_origin, new_u, new_v])
    values = cast("list[float]", numbers.tolist())
    return AnchoringVector(*(round(value, 6) for value in values))


def _dot(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Return the dot product of two vectors."""
    return (a @ b).item()


def anchoring_pixel_map(
    source: AnchoringVector,
    target: AnchoringVector,
    width: int,
    height: int,
) -> np.ndarray:
    """Map an image point under one anchoring to the point under another.

    Parameters
    ----------
    source, target
        Two anchorings of one image in one plane.
    width, height
        The image's size in pixels, which the anchorings divide.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        On ``[x, y, 1]`` in continuous pixel coordinates (top-left corner
        at 0): the point ``target`` takes to the atlas point ``source``
        takes ``(x, y)`` to.
    """
    old = np.asarray(source.to_list(), dtype=np.float64)
    new = np.asarray(target.to_list(), dtype=np.float64)
    new_axes = np.column_stack([new[3:6] / width, new[6:] / height])
    old_affine = np.column_stack([old[3:6] / width, old[6:] / height,
                                  old[:3] - new[:3]])
    change = np.linalg.pinv(new_axes) @ old_affine
    return np.vstack([change, [0.0, 0.0, 1.0]])


def markers_on_anchoring(
    markers: Sequence[Sequence[float]],
    exact: AnchoringVector,
    anchoring: AnchoringVector,
    size: tuple[int, int],
) -> list[list[float]]:
    """Carry VisuAlign markers from one anchoring onto another.

    Parameters
    ----------
    markers
        ``[x, y, nx, ny]`` rows against *exact* (continuous pixels: the
        section point ``(nx, ny)`` shows the point ``(x, y)`` of the
        anchored plane); empty for a section without a deformation.
    exact
        The anchoring the markers were made against, sheared or not.
    anchoring
        The anchoring they are written with.
    size
        The image's ``(width, height)`` in pixels.

    Returns
    -------
    list of list of float
        The rows with ``(x, y)`` moved so that *anchoring* takes them where
        *exact* took them. With no *markers*, a coarse identity grid
        (:data:`SHEAR_MARKER_DIVISIONS`) so moved, or no rows when the two
        anchorings differ by less than :data:`SHEAR_TOLERANCE_PX` anywhere
        on the image.
    """
    width, height = size
    change = anchoring_pixel_map(exact, anchoring, width, height)
    rows = np.asarray(markers, dtype=np.float64).reshape(-1, 4)
    if not len(rows):
        corners = np.array([[0.0, 0.0, 1.0], [width, 0.0, 1.0],
                            [0.0, height, 1.0], [width, height, 1.0]])
        moved = corners @ change.T
        if np.abs(moved - corners).max() <= SHEAR_TOLERANCE_PX:
            return []
        grid_x, grid_y = marker_grid((width, height),
                                     divisions=SHEAR_MARKER_DIVISIONS)
        rows = np.column_stack([grid_x, grid_y, grid_x, grid_y])
    points = np.column_stack([rows[:, :2], np.ones(len(rows))])
    rows[:, :2] = (points @ change.T)[:, :2]
    return [list(map(float, row)) for row in rows]


def job_export(
    sections: Sequence[dict[str, Any]],
    atlas: dict[str, Any],
    *,
    name: str = "",
    write_markers: bool = False,
) -> dict[str, Any]:
    """Write a job's placed sections as QuickNII / VisuAlign JSON.

    Every anchoring is the section's exact one made perpendicular
    (:func:`perpendicular_anchoring`), which QuickNII and VisuAlign show as
    written. With *write_markers* (``visualign.json``) each section's
    markers carry what that drops, so the file reproduces LangSlice's
    mapping in every reader; without them (``quicknii.json``) the file
    holds the perpendicular placement only.

    Parameters
    ----------
    sections
        One row per section: ``filename``, ``width``, ``height``, ``nr``,
        ``pixel_to_atlas_um`` and optionally ``markers`` (VisuAlign's
        ``[x, y, nx, ny]`` against the exact anchoring, from
        :func:`langslice.core.maps.section_maps.residual_markers`).
    atlas
        The atlas record of ``registration.json`` (name, orientation,
        shape, resolution).
    name
        The series name written into the file.
    write_markers
        Whether to write markers (VisuAlign) or none (QuickNII).

    Returns
    -------
    dict
        The JSON document.
    """
    atlas_name = cast("str", atlas["name"])
    slices: list[SliceExport] = []
    resolution = cast("list[float]", atlas["resolution_um"])
    for entry in sections:
        size = cast("tuple[int, int]", (entry["width"], entry["height"]))
        exact = to_target_grid(
            anchoring_from_pixel_map(entry["pixel_to_atlas_um"], *size, atlas),
            atlas_name, resolution)
        anchoring = perpendicular_anchoring(exact)
        markers: list[list[float]] = []
        if write_markers:
            given = cast("list[list[float]]", entry.get("markers") or [])
            markers = markers_on_anchoring(given, exact, anchoring, size)
        filename = cast("str", entry["filename"])
        slices.append(SliceExport(
            filename=os.path.basename(filename), anchoring=anchoring,
            width=size[0], height=size[1], nr=cast("int", entry["nr"]),
            markers=markers))
    return export_to_dict(QUINTExport(target=_resolve_target(atlas_name),
                                      slices=slices, name=name))
