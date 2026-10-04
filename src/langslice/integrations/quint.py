"""Export registration results as QUINT/ABBA-compatible JSON.

The QUINT anchoring vector ``[ox, oy, oz, ux, uy, uz, vx, vy, vz]`` defines
how a 2-D section image maps into a 3-D atlas reference volume:

    atlas_point = o + u * (px / W) + v * (py / H)

where ``(px, py)`` are pixel coordinates in the section image of size ``W x H``,
``o`` is the top-left corner in atlas voxel space, and ``u`` / ``v`` are
direction vectors pointing toward the top-right and bottom-left corners.

For coronal sections in the Allen Mouse Brain Atlas (CCFv3, 25 um):

    QuickNII x  <->  atlas ML axis  (size 456)
    QuickNII y  <->  atlas AP axis  (size 528)
    QuickNII z  <->  atlas DV axis  (size 320)

This module is atlas-agnostic: it reads axis sizes and resolution from any
BrainGlobe atlas and builds a coronal anchoring vector in atlas voxel space.
The canonical path accepts a full 3x3 affine matrix that maps source-image
pixels into a chosen coronal output frame.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from langslice import __version__
from langslice.nonlinear import (
    affine_matrix_from_legacy_params,
    apply_affine_to_points,
    coerce_affine_matrix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atlas <-> QuickNII axis mapping
# ---------------------------------------------------------------------------
# BrainGlobe atlases are stored in ASR orientation:
#   axis-0 = Anterior -> Posterior  (AP)
#   axis-1 = Superior -> Inferior   (DV)
#   axis-2 = Left -> Right          (ML)
#
# QuickNII uses a different axis order:
#   x = ML,  y = AP,  z = DV
#
# We map: BG axis-0 -> QN y, BG axis-1 -> QN z, BG axis-2 -> QN x

_KNOWN_TARGETS: dict[str, str] = {
    "allen_mouse_25um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_10um": "ABA_Mouse_CCFv3_2017_10um.cutlas",
    "allen_mouse_50um": "ABA_Mouse_CCFv3_2017_50um.cutlas",
    "allen_mouse_100um": "ABA_Mouse_CCFv3_2017_100um.cutlas",
    "whs_sd_rat": "WHS_SD_Rat_v4_39um.cutlas",
    "whs_sd_rat_39um": "WHS_SD_Rat_v4_39um.cutlas",
    "kim_unified_25um": "Kim_UnifiedMouse_v1_25um.cutlas",
}

CORONAL_FRAME_PADDING_FACTOR = 1.1


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


@dataclass(frozen=True)
class CoronalFrameGeometry:
    """Coronal frame geometry shared by export and calibrated previews."""

    frame_width: int
    frame_height: int
    atlas_left_px: float
    atlas_top_px: float
    atlas_width_px: float
    atlas_height_px: float
    origin_x: float
    origin_z: float
    u_x: float
    u_z: float
    v_x: float
    v_z: float

    def frame_to_atlas(self, px: float, py: float, ap_voxel: float) -> np.ndarray:
        """Map a frame pixel position to atlas voxel coordinates."""
        u_frac = px / self.frame_width
        v_frac = py / self.frame_height
        return np.array(
            [
                self.origin_x + self.u_x * u_frac + self.v_x * v_frac,
                ap_voxel,
                self.origin_z + self.u_z * u_frac + self.v_z * v_frac,
            ],
            dtype=np.float64,
        )


def compute_coronal_frame_geometry(
    atlas_shape: Sequence[int],
    frame_width: int,
    frame_height: int,
    padding_factor: float = CORONAL_FRAME_PADDING_FACTOR,
) -> CoronalFrameGeometry:
    """Compute coronal frame geometry used by QUINT/ABBA anchoring math."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    if padding_factor <= 0.0:
        raise ValueError("padding_factor must be positive")

    n_dv = int(atlas_shape[1])
    n_ml = int(atlas_shape[2])
    if n_dv <= 0 or n_ml <= 0:
        raise ValueError("atlas_shape must contain positive DV/ML dimensions")

    u_mag = n_ml * padding_factor
    v_mag = n_dv * padding_factor

    center_x = n_ml / 2.0
    center_z = n_dv / 2.0

    u_x = -u_mag
    u_z = 0.0
    v_x = 0.0
    v_z = -v_mag

    origin_x = center_x - (u_x * 0.5 + v_x * 0.5)
    origin_z = center_z - (u_z * 0.5 + v_z * 0.5)

    atlas_width_px = frame_width / padding_factor
    atlas_height_px = frame_height / padding_factor
    atlas_left_px = (frame_width - atlas_width_px) / 2.0
    atlas_top_px = (frame_height - atlas_height_px) / 2.0

    return CoronalFrameGeometry(
        frame_width=frame_width,
        frame_height=frame_height,
        atlas_left_px=atlas_left_px,
        atlas_top_px=atlas_top_px,
        atlas_width_px=atlas_width_px,
        atlas_height_px=atlas_height_px,
        origin_x=origin_x,
        origin_z=origin_z,
        u_x=u_x,
        u_z=u_z,
        v_x=v_x,
        v_z=v_z,
    )


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


def compute_anchoring(
    position_mm: float,
    atlas_shape: Sequence[int],
    atlas_resolution: Sequence[float],
    image_width: int,
    image_height: int,
    affine_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    rotation_deg: float = 0.0,
    translate_x_pct: float = 0.0,
    translate_y_pct: float = 0.0,
) -> AnchoringVector:
    """Build a QUINT anchoring vector for a coronal section.

    Parameters
    ----------
    position_mm : float
        Physical position in mm from the anterior edge of the volume.
    atlas_shape : (n_ap, n_dv, n_ml)
        Number of voxels along each BrainGlobe axis.
    atlas_resolution : (res_ap, res_dv, res_ml)
        Voxel resolution in micrometers.
    image_width, image_height : int
        Pixel dimensions of the source section image.
    affine_matrix : array-like of shape (3, 3), optional
        Homogeneous transform mapping source-image pixels into the output frame.
        When omitted, the legacy rotation/translation parameters are converted
        into a matrix for backward compatibility.
    output_width, output_height : int, optional
        Pixel dimensions of the output frame. Defaults to the source image size.
    rotation_deg : float
        Legacy in-plane rotation in degrees. Used only when *affine_matrix* is
        omitted.
    translate_x_pct, translate_y_pct : float
        Legacy translation as percentage of image size. Used only when
        *affine_matrix* is omitted.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    frame_width = int(output_width or image_width)
    frame_height = int(output_height or image_height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("output_width and output_height must be positive")

    matrix = coerce_affine_matrix(
        affine_matrix
        if affine_matrix is not None
        else affine_matrix_from_legacy_params(
            image_width=image_width,
            image_height=image_height,
            rotation_deg=rotation_deg,
            translate_x_pct=translate_x_pct,
            translate_y_pct=translate_y_pct,
        )
    )

    # Convert physical position mm -> voxel index (BG axis-0).
    res_ap_mm = float(atlas_resolution[0]) / 1000.0
    ap_voxel = position_mm / res_ap_mm
    geometry = compute_coronal_frame_geometry(
        atlas_shape=atlas_shape,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    transformed_corners = apply_affine_to_points(
        matrix,
        [
            (0.0, 0.0),
            (float(image_width), 0.0),
            (0.0, float(image_height)),
        ],
    )
    atlas_origin = geometry.frame_to_atlas(
        px=float(transformed_corners[0][0]),
        py=float(transformed_corners[0][1]),
        ap_voxel=ap_voxel,
    )
    atlas_top_right = geometry.frame_to_atlas(
        px=float(transformed_corners[1][0]),
        py=float(transformed_corners[1][1]),
        ap_voxel=ap_voxel,
    )
    atlas_bottom_left = geometry.frame_to_atlas(
        px=float(transformed_corners[2][0]),
        py=float(transformed_corners[2][1]),
        ap_voxel=ap_voxel,
    )
    u_vec = atlas_top_right - atlas_origin
    v_vec = atlas_bottom_left - atlas_origin

    return AnchoringVector(
        ox=round(float(atlas_origin[0]), 6),
        oy=round(float(atlas_origin[1]), 6),
        oz=round(float(atlas_origin[2]), 6),
        ux=round(float(u_vec[0]), 6),
        uy=round(float(u_vec[1]), 6),
        uz=round(float(u_vec[2]), 6),
        vx=round(float(v_vec[0]), 6),
        vy=round(float(v_vec[1]), 6),
        vz=round(float(v_vec[2]), 6),
    )


# ---------------------------------------------------------------------------
# High-level export helpers
# ---------------------------------------------------------------------------


def build_quint_export(
    filename: str,
    position_mm: float,
    atlas_name: str,
    atlas_shape: Sequence[int],
    atlas_resolution: Sequence[float],
    image_width: int,
    image_height: int,
    affine_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    rotation_deg: float = 0.0,
    translate_x_pct: float = 0.0,
    translate_y_pct: float = 0.0,
    nr: int = 1,
) -> QUINTExport:
    """Build a complete QUINT export structure for a single slice."""
    anchoring = compute_anchoring(
        position_mm=position_mm,
        atlas_shape=atlas_shape,
        atlas_resolution=atlas_resolution,
        image_width=image_width,
        image_height=image_height,
        affine_matrix=affine_matrix,
        output_width=output_width,
        output_height=output_height,
        rotation_deg=rotation_deg,
        translate_x_pct=translate_x_pct,
        translate_y_pct=translate_y_pct,
    )
    slice_entry = SliceExport(
        filename=os.path.basename(filename),
        anchoring=anchoring,
        width=image_width,
        height=image_height,
        nr=nr,
    )
    return QUINTExport(
        target=_resolve_target(atlas_name),
        slices=[slice_entry],
    )


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


def save_quint_json(export: QUINTExport, path: str) -> None:
    """Write QUINT JSON to *path*."""
    data = export_to_dict(export)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved QUINT JSON: %s", path)


# ---------------------------------------------------------------------------
# A job's registrations as QuickNII / VisuAlign JSON (formats phase, 2026-10-04)
# ---------------------------------------------------------------------------
#
# The anchoring comes straight from a section's exact pixel -> atlas map
# (``registration.json``'s ``pixel_to_atlas_um``, BrainGlobe micrometres,
# pixel centres at integers), so any plane and any cutting angle is
# represented, unlike the coronal-frame path above. The conversion to
# QuickNII's voxel space is SliceBench's (``slicebench/geometry.py``,
# ``atlas_um_to_quicknii``), checked there against DeepSlice and human QUINT
# registrations on the Allen CCFv3 25 um atlas; for other atlases the
# QuickNII target is assumed to share the BrainGlobe voxel grid (untested).

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
    ``[x, y, nx, ny]`` in the image's pixel units, from
    :func:`langslice.core.maps.residual_markers`).
    """
    slices = [
        SliceExport(
            filename=os.path.basename(str(entry["filename"])),
            anchoring=anchoring_from_pixel_map(entry["pixel_to_atlas_um"], int(entry["width"]),
                                               int(entry["height"]), atlas),
            width=int(entry["width"]), height=int(entry["height"]), nr=int(entry["nr"]),
            markers=list(entry.get("markers") or []),
        )
        for entry in sections
    ]
    return export_to_dict(QUINTExport(target=_resolve_target(str(atlas["name"])),
                                      slices=slices, name=name))
