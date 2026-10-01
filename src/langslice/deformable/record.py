"""The canonical per-section result of a deformable fit, and its plausibility report.

Field convention (brainglobe-registration's): ``field_mm[y, x]`` is the
residual displacement, in millimetres, of section pixel (x, y). The section
point ``p_mm = (x, y) * mm_per_px`` corresponds to the point
``p_mm + field_mm[y, x]`` of the LINEARLY PLACED atlas plane (the placed
plane is drawn on the section's own pixel grid). The full map to the native
atlas plane composes the linear placement after it::

    native_xy = inverse(atlas_to_section) @ (pixel + field_mm / mm_per_px)

which is ``registration_handoff``'s and ``border_registration``'s order
(``composed_native_map``: residual first, then undo the placement), so ABBA,
VisuAlign and BrainGlobe exports can all be derived from
:meth:`DeformableRecord.native_coordinates` and
:meth:`DeformableRecord.volume_coordinates` without the atlas object. The
inverse field lives on the same grid in the placed-atlas frame: placed point
q corresponds to section point ``q + inverse_field_mm(q)``.

A sequential fit is a chain: each record's field is the TOTAL residual
(earlier steps composed in), and ``parent`` is the record before this step,
so undoing a step is ``record.parent``.

Plausibility diagnostics are reported, never enforced: a flag says the warp
compresses, expands or folds tissue more than a rough biological limit, and
a person or agent decides what that means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from langslice.deformable.geometry import Placement, sample_native
from langslice.deformable.settings import FitSettings

#: Area ratio (warped / placed) limits for solid tissue regions.
TISSUE_AREA_RATIO_LIMITS = (0.5, 2.0)
#: Looser limits for the ventricular system, which collapses or dilates freely.
VENTRICLE_AREA_RATIO_LIMITS = (0.1, 10.0)
#: Regions smaller than this (placed area, mm^2) are reported but never flagged.
MIN_FLAG_AREA_MM2 = 0.02
#: Fraction of tissue pixels with a non-positive Jacobian above which FOLDS is flagged.
FOLD_FRACTION_LIMIT = 0.001

RECORD_VERSION = 1


def jacobian_determinant(field_mm: np.ndarray, mm_per_px: float) -> np.ndarray:
    """det D(p + u(p)) of the section-to-placed-atlas map, per section pixel."""
    u = field_mm[..., 0].astype(np.float64) / mm_per_px
    v = field_mm[..., 1].astype(np.float64) / mm_per_px
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    return (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx


def diagnose(
    field_mm: np.ndarray,
    mm_per_px: float,
    warped: np.ndarray,
    placed: np.ndarray,
    tissue: np.ndarray,
    *,
    ventricles: frozenset[int],
    excluded: frozenset[int],
    acronyms: dict[int, str],
) -> dict[str, Any]:
    """Per-region area ratios from the Jacobian, fold fraction, and flags.

    For region R, the warped region is the set of section pixels whose atlas
    point lies in R; its placed area follows from the Jacobian J of the
    section-to-placed map, so ``area_ratio = N_R / sum_{p in R} J(p)``
    (warped over placed; below 1 the warp compresses the region onto the
    section). The raster ratio of pixel counts is reported alongside.
    """
    jacobian = jacobian_determinant(field_mm, mm_per_px)
    in_tissue = jacobian[tissue]
    fold_fraction = float(np.mean(in_tissue <= 0)) if in_tissue.size else 0.0
    ids = np.union1d(np.unique(warped), np.unique(placed))
    ids = ids[ids != 0]
    index_warped = np.searchsorted(ids, warped)
    valid = warped != 0
    count = np.bincount(index_warped[valid], minlength=len(ids))
    jac_sum = np.bincount(index_warped[valid], weights=jacobian[valid], minlength=len(ids))
    placed_count = np.bincount(np.searchsorted(ids, placed)[placed != 0], minlength=len(ids))
    pixel_mm2 = mm_per_px ** 2
    regions: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    for k, uid in enumerate(int(i) for i in ids):
        ventricle = uid in ventricles
        limits = VENTRICLE_AREA_RATIO_LIMITS if ventricle else TISSUE_AREA_RATIO_LIMITS
        placed_mm2 = float(placed_count[k] * pixel_mm2)
        # None: the warp turns this region inside out (non-positive total Jacobian).
        ratio: float | None = 0.0
        if count[k]:
            ratio = float(count[k] / jac_sum[k]) if jac_sum[k] > 0 else None
        entry = {
            "id": uid, "acronym": acronyms.get(uid, str(uid)),
            "placed_area_mm2": placed_mm2, "warped_area_mm2": float(count[k] * pixel_mm2),
            "area_ratio": ratio,
            "raster_area_ratio": float(count[k] / placed_count[k]) if placed_count[k] else None,
            "ventricle": ventricle, "excluded": uid in excluded,
            "limits": list(limits),
        }
        regions.append(entry)
        if uid in excluded or placed_mm2 < MIN_FLAG_AREA_MM2:
            continue
        code = None
        if count[k] == 0:
            code = "REGION_VANISHED"
        elif ratio is None:
            code = "REGION_FOLDED"
        elif ratio < limits[0]:
            code = "REGION_COMPRESSED"
        elif ratio > limits[1]:
            code = "REGION_EXPANDED"
        if code:
            flags.append({"code": code, "id": uid, "acronym": entry["acronym"],
                          "area_ratio": ratio, "limits": list(limits),
                          "ventricle": ventricle})
    if fold_fraction > FOLD_FRACTION_LIMIT:
        flags.append({"code": "FOLDS", "fold_fraction": fold_fraction,
                      "limit": FOLD_FRACTION_LIMIT})
    return {
        "fold_fraction": fold_fraction,
        "jacobian_in_tissue": {
            "min": float(in_tissue.min()) if in_tissue.size else None,
            "p01": float(np.percentile(in_tissue, 1)) if in_tissue.size else None,
            "p99": float(np.percentile(in_tissue, 99)) if in_tissue.size else None,
            "max": float(in_tissue.max()) if in_tissue.size else None,
        },
        "max_displacement_mm": float(np.linalg.norm(field_mm, axis=-1)[tissue].max())
        if tissue.any() else 0.0,
        "regions": regions,
        "flags": flags,
        "thresholds": {
            "tissue_area_ratio": list(TISSUE_AREA_RATIO_LIMITS),
            "ventricle_area_ratio": list(VENTRICLE_AREA_RATIO_LIMITS),
            "min_flag_area_mm2": MIN_FLAG_AREA_MM2,
            "fold_fraction": FOLD_FRACTION_LIMIT,
        },
        "note": "Reported, never enforced; thresholds are rough biological defaults.",
    }


@dataclass
class DeformableRecord:
    """One section's linear placement plus its fitted residual deformation."""

    placement: Placement
    section_size: tuple[int, int]
    settings: FitSettings
    field_mm: np.ndarray
    inverse_field_mm: np.ndarray | None
    inverse_source: str
    labels: np.ndarray
    tissue: np.ndarray
    torn_band: np.ndarray
    excluded_ids: list[int]
    engine: dict[str, Any]
    diagnostics: dict[str, Any]
    atlas: dict[str, Any]
    step: int = 0
    parent: DeformableRecord | None = field(default=None, repr=False)

    @property
    def mm_per_px(self) -> float:
        return self.placement.section_mm_per_px

    def native_coordinates(self) -> np.ndarray:
        """(H, W, 2) native atlas-plane pixel [x, y] for every section pixel."""
        height, width = self.field_mm.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float64)
        x = xx + self.field_mm[..., 0] / self.mm_per_px
        y = yy + self.field_mm[..., 1] / self.mm_per_px
        inverse = np.linalg.inv(self.placement.atlas_to_section)
        return np.stack([inverse[k, 0] * x + inverse[k, 1] * y + inverse[k, 2]
                         for k in range(2)], axis=-1)

    def volume_coordinates(self) -> np.ndarray:
        """(H, W, 3) atlas volume index coordinates in the atlas's own axis order."""
        native = self.native_coordinates()
        matrix = np.asarray(self.atlas["native_to_volume_index"], dtype=np.float64)
        return native[..., 0:1] * matrix[:, 0] + native[..., 1:2] * matrix[:, 1] + matrix[:, 2]

    def placed_labels(self, native: np.ndarray) -> np.ndarray:
        """The linear placement alone, as leaf labels on the section grid (unclipped)."""
        height, width = self.field_mm.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float64)
        inverse = np.linalg.inv(self.placement.atlas_to_section)
        coords = np.stack([inverse[k, 0] * xx + inverse[k, 1] * yy + inverse[k, 2]
                           for k in range(2)], axis=-1)
        return sample_native(native, coords)

    def undo(self) -> DeformableRecord | None:
        """The record before this step of a sequential fit (None for the first)."""
        return self.parent

    def metadata(self) -> dict[str, Any]:
        return {
            "version": RECORD_VERSION, "step": self.step,
            "placement": self.placement.to_dict(),
            "section_size": list(self.section_size),
            "settings": self.settings.to_dict(),
            "field_convention": (
                "field_mm[y, x] = [dx, dy] mm: section pixel (x, y) at (x, y)*mm_per_px "
                "corresponds to placed-atlas point (x, y)*mm_per_px + field; native = "
                "inv(atlas_to_section) @ (pixel + field/mm_per_px)"
            ),
            "inverse_source": self.inverse_source,
            "labels": "leaf atlas ids under the composed warp, 0 outside tissue",
            "excluded_ids": list(self.excluded_ids),
            "engine": self.engine, "diagnostics": self.diagnostics, "atlas": self.atlas,
        }

    def save(self, directory: str | Path) -> Path:
        """Write ``record.json`` + ``arrays.npz``; the parent chain goes in ``parent/``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / "record.json").write_text(json.dumps(self.metadata(), indent=2,
                                                    default=_json_default) + "\n")
        arrays: dict[str, np.ndarray] = {
            "field_mm": self.field_mm.astype(np.float32), "labels": self.labels,
            "tissue": self.tissue, "torn_band": self.torn_band,
        }
        if self.inverse_field_mm is not None:
            arrays["inverse_field_mm"] = self.inverse_field_mm.astype(np.float32)
        np.savez_compressed(out / "arrays.npz", **arrays)  # pyright: ignore[reportArgumentType]
        if self.parent is not None:
            self.parent.save(out / "parent")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> DeformableRecord:
        root = Path(directory)
        meta = json.loads((root / "record.json").read_text())
        with np.load(root / "arrays.npz") as data:
            arrays = {key: data[key] for key in data.files}
        parent = cls.load(root / "parent") if (root / "parent" / "record.json").exists() else None
        return cls(
            placement=Placement.from_dict(meta["placement"]),
            section_size=(int(meta["section_size"][0]), int(meta["section_size"][1])),
            settings=FitSettings.from_dict(meta["settings"]),
            field_mm=arrays["field_mm"], inverse_field_mm=arrays.get("inverse_field_mm"),
            inverse_source=str(meta["inverse_source"]), labels=arrays["labels"],
            tissue=arrays["tissue"], torn_band=arrays["torn_band"],
            excluded_ids=[int(i) for i in meta["excluded_ids"]], engine=meta["engine"],
            diagnostics=meta["diagnostics"], atlas=meta["atlas"], step=int(meta["step"]),
            parent=parent,
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
