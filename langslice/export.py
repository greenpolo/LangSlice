"""Export registration results as QUINT/ABBA-compatible JSON.

The QUINT anchoring vector ``[ox, oy, oz, ux, uy, uz, vx, vy, vz]`` defines
how a 2-D section image maps into a 3-D atlas reference volume:

    atlas_point = o + u * (px / W) + v * (py / H)

where ``(px, py)`` are pixel coordinates in the section image of size ``W × H``,
``o`` is the top-left corner in atlas *voxel* space, and ``u`` / ``v`` are direction
vectors pointing toward the top-right and bottom-left corners respectively.

For coronal sections the mapping in the Allen Mouse Brain Atlas (CCFv3, 25 µm)
coordinate system used by QuickNII / DeepSlice is:

    QuickNII x  ↔  atlas ML axis  (size 456)
    QuickNII y  ↔  atlas AP axis  (size 528)
    QuickNII z  ↔  atlas DV axis  (size 320)

This module is atlas-agnostic: it reads axis sizes and resolution from any
BrainGlobe atlas and builds a correct anchoring vector for coronal cuts.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from langslice import __version__

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atlas ↔ QuickNII axis mapping
# ---------------------------------------------------------------------------
# BrainGlobe atlases are stored in ASR orientation:
#   axis-0 = Anterior → Posterior  (AP)
#   axis-1 = Superior → Inferior   (DV)
#   axis-2 = Left → Right          (ML)
#
# QuickNII uses a different axis order:
#   x = ML,  y = AP,  z = DV
#
# We map:  BG axis-0 → QN y,  BG axis-1 → QN z,  BG axis-2 → QN x

_KNOWN_TARGETS: dict[str, str] = {
    "allen_mouse_25um": "ABA_Mouse_CCFv3_2017_25um.cutlas",
    "allen_mouse_10um": "ABA_Mouse_CCFv3_2017_10um.cutlas",
    "allen_mouse_50um": "ABA_Mouse_CCFv3_2017_50um.cutlas",
    "allen_mouse_100um": "ABA_Mouse_CCFv3_2017_100um.cutlas",
    "whs_sd_rat": "WHS_SD_Rat_v4_39um.cutlas",
    "whs_sd_rat_39um": "WHS_SD_Rat_v4_39um.cutlas",
    "kim_unified_25um": "Kim_UnifiedMouse_v1_25um.cutlas",
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
        return [self.ox, self.oy, self.oz, self.ux, self.uy, self.uz, self.vx, self.vy, self.vz]


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
    # Fallback: construct a reasonable target string
    return f"{atlas_name}.cutlas"


def compute_anchoring(
    ap_mm: float,
    atlas_shape: Sequence[int],
    atlas_resolution: Sequence[float],
    origin_index: int,
    image_width: int,
    image_height: int,
    rotation_deg: float = 0.0,
    translate_x_pct: float = 0.0,
    translate_y_pct: float = 0.0,
) -> AnchoringVector:
    """Build a QUINT anchoring vector for a coronal section.

    Parameters
    ----------
    ap_mm : float
        Anterior-posterior position in mm (positive = anterior of origin).
    atlas_shape : (n_ap, n_dv, n_ml)
        Number of voxels along each BrainGlobe axis.
    atlas_resolution : (res_ap, res_dv, res_ml)
        Voxel resolution in **micrometers**.
    origin_index : int
        Index along the AP axis that corresponds to 0 mm (e.g. Bregma).
    image_width, image_height : int
        Pixel dimensions of the section image.
    rotation_deg : float
        In-plane rotation in degrees (clockwise positive).
    translate_x_pct, translate_y_pct : float
        Translation as percentage of image size (−50 to 50).
    """
    n_ap, n_dv, n_ml = int(atlas_shape[0]), int(atlas_shape[1]), int(atlas_shape[2])

    # --- Convert AP mm → voxel index (BG axis-0) ---
    res_ap_mm = float(atlas_resolution[0]) / 1000.0
    ap_voxel = origin_index - (ap_mm / res_ap_mm)  # keep as float for smooth positioning

    # --- Determine image → atlas scale ---
    # We want the atlas coronal slice to be centered and *fit* within the
    # image bounds.  The section image typically includes background around
    # the brain, so we scale the atlas to occupy ~90 % of the smaller
    # dimension to leave a border, matching DeepSlice conventions.
    #
    # u spans image width  → atlas ML axis (QN x)
    # v spans image height → atlas DV axis (QN z)
    padding_factor = 1.1  # atlas slice occupies ~90% of image → vectors 10% larger

    u_mag = n_ml * padding_factor  # voxels spanned across image width
    v_mag = n_dv * padding_factor  # voxels spanned across image height

    # --- Compute un-rotated origin ---
    # For a centred, upright coronal section:
    #   o  = top-left   →  (+x, y_ap, +z) = (ML_high, ap_voxel, DV_high)
    #   u  = rightward  →  (-u_mag, 0, 0)         [image x → decreasing ML]
    #   v  = downward   →  (0, 0, -v_mag)          [image y → decreasing DV]
    center_x = n_ml / 2.0  # ML centre
    center_z = n_dv / 2.0  # DV centre

    ox0 = center_x + u_mag / 2.0
    oz0 = center_z + v_mag / 2.0

    ux0 = -u_mag
    uz0 = 0.0
    vx0 = 0.0
    vz0 = -v_mag

    # --- Apply in-plane rotation ---
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Rotate u and v around the image centre (the origin shifts accordingly)
    ux = ux0 * cos_t - uz0 * sin_t
    uz = ux0 * sin_t + uz0 * cos_t
    vx = vx0 * cos_t - vz0 * sin_t
    vz = vx0 * sin_t + vz0 * cos_t

    # Recompute origin so that the centre of the image still maps to atlas centre
    # Centre in atlas space = o + u*0.5 + v*0.5
    ox = center_x - (ux * 0.5 + vx * 0.5)
    oz = center_z - (uz * 0.5 + vz * 0.5)

    # --- Apply translation ---
    # translate_x_pct and translate_y_pct shift the image relative to the atlas.
    # Positive translate_x → image moves right → atlas appears shifted left → origin shifts in +x
    tx_vox = (translate_x_pct / 100.0) * u_mag
    tz_vox = (translate_y_pct / 100.0) * v_mag
    ox += tx_vox
    oz += tz_vox

    # --- Assemble ---
    # For a coronal section the y-components (AP axis) are essentially zero for
    # u and v (the cut plane is perpendicular to AP), and the origin y is the
    # AP voxel position.
    return AnchoringVector(
        ox=round(ox, 6),
        oy=round(ap_voxel, 6),
        oz=round(oz, 6),
        ux=round(ux, 6),
        uy=0.0,
        uz=round(uz, 6),
        vx=round(vx, 6),
        vy=0.0,
        vz=round(vz, 6),
    )


# ---------------------------------------------------------------------------
# High-level export helpers
# ---------------------------------------------------------------------------


def build_quint_export(
    filename: str,
    ap_mm: float,
    atlas_name: str,
    atlas_shape: Sequence[int],
    atlas_resolution: Sequence[float],
    origin_index: int,
    image_width: int,
    image_height: int,
    rotation_deg: float = 0.0,
    translate_x_pct: float = 0.0,
    translate_y_pct: float = 0.0,
    nr: int = 1,
) -> QUINTExport:
    """Build a complete QUINT export structure for a single slice."""
    anchoring = compute_anchoring(
        ap_mm=ap_mm,
        atlas_shape=atlas_shape,
        atlas_resolution=atlas_resolution,
        origin_index=origin_index,
        image_width=image_width,
        image_height=image_height,
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
