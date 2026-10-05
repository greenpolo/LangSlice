"""A job's registrations as QUINT JSON (QuickNII / VisuAlign), and back.

The QUINT anchoring vector ``[ox, oy, oz, ux, uy, uz, vx, vy, vz]`` defines
how a 2-D section image maps into a 3-D atlas reference volume:

    atlas_point = o + u * (px / W) + v * (py / H)

where ``(px, py)`` are pixel coordinates in the section image of size ``W x H``,
``o`` is the top-left corner in atlas voxel space, and ``u`` / ``v`` are
direction vectors pointing toward the top-right and bottom-left corners.

:func:`job_export` writes it from each section's exact pixel -> atlas map
(``registration.json``'s ``pixel_to_atlas_um``), so any plane and cutting
angle is represented; :mod:`langslice.job.imports` reads it back through
the inverses here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from langslice import __version__

#: The QuickNII ``.cutlas`` target of a BrainGlobe atlas. QuickNII and
#: VisuAlign ship the Allen CCFv3 at 25 um only
#: (``ABA_Mouse_CCFv3_2017_25um.cutlas``; DeepSlice writes the same name),
#: so every Allen resolution is exported to it (:data:`_TARGET_RESOLUTION_UM`).
#: The rat name is DeepSlice's, the one its QuickNII files open with. The
#: Kim atlas name follows the same pattern; any other atlas is exported to
#: ``<name>.cutlas``.
_KNOWN_TARGETS: dict[str, str] = {
    "allen_mouse_25um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_10um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_50um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_100um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "whs_sd_rat": "WHS_Rat_v4_39um.cutlas",
    "whs_sd_rat_39um": "WHS_Rat_v4_39um.cutlas",
    "kim_unified_25um": "Kim_UnifiedMouse_v1_25um.cutlas",
}
#: A target whose voxels differ from the BrainGlobe atlas's: its voxel size.
#: QuickNII coordinates are voxel EDGES (physical / voxel size), so an
#: anchoring moves to the target's grid by the factor of the two sizes (the
#: Allen volumes at every resolution span the same 13.2 x 8.0 x 11.4 mm).
_TARGET_RESOLUTION_UM: dict[str, float] = {
    "ABA_Mouse_CCFv3_2017_25um.cutlas": 25.0,
}


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
    bg_space, quicknii_space = _quicknii_spaces(atlas)
    resolution = np.asarray(atlas["resolution_um"], dtype=np.float64)
    edges = np.asarray(points_um, dtype=np.float64).reshape(-1, 3) / resolution
    edges = edges + VOXEL_EDGE_TO_CENTRE
    return np.asarray(bg_space.map_points_to(quicknii_space, edges), dtype=np.float64)


def quicknii_points_to_atlas_um(points: Any, atlas: dict[str, Any]) -> np.ndarray:
    """The inverse of :func:`atlas_um_to_quicknii_points`: QuickNII voxel
    coordinates (``(n, 3)``: x ML, y AP, z DV, voxel edges, on the atlas's
    own grid) as BrainGlobe atlas micrometres in the atlas's axis order."""
    bg_space, quicknii_space = _quicknii_spaces(atlas)
    forward = np.asarray(bg_space.transformation_matrix_to(quicknii_space), dtype=np.float64)
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack([values, np.ones(len(values))])
    edges = (np.linalg.inv(forward) @ homogeneous.T).T[:, :3]
    resolution = np.asarray(atlas["resolution_um"], dtype=np.float64)
    return (edges - VOXEL_EDGE_TO_CENTRE) * resolution


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


def job_export(
    sections: Sequence[dict[str, Any]], atlas: dict[str, Any], *, name: str = "",
) -> dict[str, Any]:
    """QuickNII / VisuAlign JSON of a job's placed sections.

    Each of *sections* gives ``filename``, ``width``, ``height``, ``nr``,
    ``pixel_to_atlas_um`` and optionally ``markers`` (VisuAlign's
    ``[x_overlay, y_overlay, x_image, y_image]`` in the image's pixels, from
    :func:`langslice.core.maps.residual_markers`).
    """
    slices = [
        SliceExport(
            filename=os.path.basename(str(entry["filename"])),
            anchoring=to_target_grid(
                anchoring_from_pixel_map(entry["pixel_to_atlas_um"], int(entry["width"]),
                                         int(entry["height"]), atlas),
                str(atlas["name"]), atlas["resolution_um"]),
            width=int(entry["width"]), height=int(entry["height"]), nr=int(entry["nr"]),
            markers=list(entry.get("markers") or []),
        )
        for entry in sections
    ]
    return export_to_dict(QUINTExport(target=_resolve_target(str(atlas["name"])),
                                      slices=slices, name=name))
