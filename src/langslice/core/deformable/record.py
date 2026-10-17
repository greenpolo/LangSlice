"""The canonical per-section result of a deformable fit, and its plausibility report.

Field convention (brainglobe-registration's): ``field_mm[y, x]`` is the
residual displacement, in millimetres, of section pixel (x, y). The section
point ``p_mm = (x, y) * mm_per_px`` corresponds to the point
``p_mm + field_mm[y, x]`` of the LINEARLY PLACED atlas plane (the placed
plane is drawn on the section's own pixel grid). The full map to the native
atlas plane composes the linear placement after it::

    native_xy = inverse(atlas_to_section) @ (pixel + field_mm / mm_per_px)

(residual first, then undo the placement; ``core.maps.native_points``
composes it the same way, and ``atlas["native_to_volume_index"]`` carries a
native pixel to the atlas volume), so ABBA, VisuAlign and BrainGlobe exports
need no atlas object. The
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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from langslice.core.deformable.geometry import Placement
from langslice.core.deformable.settings import FitSettings

#: Area ratio (warped / placed) limits for solid tissue regions.
TISSUE_AREA_RATIO_LIMITS = (0.5, 2.0)
#: Looser limits for the ventricular system, which collapses or dilates freely.
VENTRICLE_AREA_RATIO_LIMITS = (0.1, 10.0)
#: Regions smaller than this (placed area, mm^2, adult mouse; times the brain
#: scale squared) are reported but never flagged.
MIN_FLAG_AREA_MM2 = 0.02
#: Fraction of footprint pixels with a non-positive Jacobian above which FOLDS is flagged.
FOLD_FRACTION_LIMIT = 0.001
#: DISPLACEMENT_OUTSIZED: the largest displacement in the footprint above this
#: fraction of the footprint's longest extent (about 1 mm on a 10 mm coronal
#: section): a blow-up can stay under every fold and area flag while it
#: drags the outline over a millimetre.
OUTSIZED_MAX_FRACTION = 0.1
#: DISPLACEMENT_OUTSIZED also fires when the median displacement in the footprint
#: exceeds this, in mm: a wholesale drift a uniform field would hide from the
#: maximum rule. It sits above the medians of fits whose borders looked
#: plausible. Adult mouse; times the brain scale.
OUTSIZED_MEDIAN_MM = 0.6

RECORD_VERSION = 1
#: The keys ``record.json`` holds (:meth:`DeformableRecord.metadata`); any
#: other is refused on loading.
_RECORD_KEYS = ("version", "step", "placement", "section_size", "settings", "field_convention",
                "inverse_source", "labels", "excluded_ids", "engine", "diagnostics", "atlas",
                "provenance")


def jacobian_determinant(field_mm: np.ndarray, mm_per_px: float) -> np.ndarray:
    """det D(p + u(p)) of the section-to-placed-atlas map, per section pixel."""
    u = field_mm[..., 0].astype(np.float64) / mm_per_px
    v = field_mm[..., 1].astype(np.float64) / mm_per_px
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    return np.asarray((1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx, dtype=np.float64)


def displacement_report(
    field_mm: np.ndarray, mm_per_px: float, footprint: np.ndarray, brain_scale: float = 1.0,
) -> dict[str, Any]:
    """Max and median displacement in *footprint*, their limits, and whether either is passed.

    The maximum's limit is :data:`OUTSIZED_MAX_FRACTION` of the footprint's
    longest extent (the longer side of its bounding box); the median's is
    :data:`OUTSIZED_MEDIAN_MM` times the atlas's *brain_scale*.
    """
    magnitude = np.linalg.norm(field_mm, axis=-1)[footprint]
    rows, cols = np.nonzero(footprint)
    extent_mm = (float(max(rows.max() - rows.min(), cols.max() - cols.min()) + 1) * mm_per_px
                 if rows.size else 0.0)
    largest = float(magnitude.max()) if magnitude.size else 0.0
    median = float(np.median(magnitude)) if magnitude.size else 0.0
    max_limit = OUTSIZED_MAX_FRACTION * extent_mm
    median_limit = OUTSIZED_MEDIAN_MM * brain_scale
    return {
        "max_mm": largest, "median_mm": median, "tissue_extent_mm": extent_mm,
        "max_limit_mm": max_limit, "median_limit_mm": median_limit,
        "outsized": bool(magnitude.size) and (largest > max_limit or median > median_limit),
    }


def diagnose(
    field_mm: np.ndarray,
    mm_per_px: float,
    warped: np.ndarray,
    placed: np.ndarray,
    footprint: np.ndarray,
    *,
    ventricles: frozenset[int],
    excluded: frozenset[int],
    acronyms: dict[int, str],
    brain_scale: float = 1.0,
) -> dict[str, Any]:
    """Per-region area ratios from the Jacobian, fold fraction, and flags; the
    millimetre limits are the adult mouse's times the atlas's *brain_scale*.

    For region R, the warped region is the set of section pixels whose atlas
    point lies in R; its placed area follows from the Jacobian J of the
    section-to-placed map, so ``area_ratio = N_R / sum_{p in R} J(p)``
    (warped over placed; below 1 the warp compresses the region onto the
    section). The raster ratio of pixel counts is reported alongside.
    """
    jacobian = jacobian_determinant(field_mm, mm_per_px)
    in_tissue = jacobian[footprint]
    fold_fraction = float(np.mean(in_tissue <= 0)) if in_tissue.size else 0.0
    ids = np.union1d(np.unique(warped), np.unique(placed))
    ids = ids[ids != 0]
    index_warped = np.searchsorted(ids, warped)
    valid = warped != 0
    count = np.bincount(index_warped[valid], minlength=len(ids))
    jac_sum = np.bincount(index_warped[valid], weights=jacobian[valid], minlength=len(ids))
    placed_count = np.bincount(np.searchsorted(ids, placed)[placed != 0], minlength=len(ids))
    pixel_mm2 = mm_per_px ** 2
    min_flag_area_mm2 = MIN_FLAG_AREA_MM2 * brain_scale ** 2
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
        if uid in excluded or placed_mm2 < min_flag_area_mm2:
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
    displacement = displacement_report(field_mm, mm_per_px, footprint, brain_scale)
    if displacement["outsized"]:
        flags.append({"code": "DISPLACEMENT_OUTSIZED",
                      **{key: displacement[key] for key in (
                          "max_mm", "median_mm", "max_limit_mm", "median_limit_mm")}})
    return {
        "fold_fraction": fold_fraction,
        "jacobian_in_tissue": {
            "min": float(in_tissue.min()) if in_tissue.size else None,
            "p01": float(np.percentile(in_tissue, 1)) if in_tissue.size else None,
            "p99": float(np.percentile(in_tissue, 99)) if in_tissue.size else None,
            "max": float(in_tissue.max()) if in_tissue.size else None,
        },
        "max_displacement_mm": float(np.linalg.norm(field_mm, axis=-1)[footprint].max())
        if footprint.any() else 0.0,
        "displacement": displacement,
        "regions": regions,
        "flags": flags,
        "thresholds": {
            "tissue_area_ratio": list(TISSUE_AREA_RATIO_LIMITS),
            "ventricle_area_ratio": list(VENTRICLE_AREA_RATIO_LIMITS),
            "min_flag_area_mm2": min_flag_area_mm2,
            "fold_fraction": FOLD_FRACTION_LIMIT,
            "outsized_max_fraction": OUTSIZED_MAX_FRACTION,
            "outsized_median_mm": displacement["median_limit_mm"],
        },
        "note": "Reported, never enforced; thresholds are rough biological defaults.",
    }


@dataclass
class DeformableRecord:
    """One section's linear placement plus its fitted residual deformation."""

    placement: Placement
    section_size: tuple[int, int]
    #: The engine's settings; None for a warp no engine computed (the landmark warp).
    settings: FitSettings | None
    field_mm: np.ndarray
    inverse_field_mm: np.ndarray | None
    inverse_source: str
    labels: np.ndarray
    footprint: np.ndarray
    excluded_ids: list[int]
    engine: dict[str, Any]
    diagnostics: dict[str, Any]
    atlas: dict[str, Any]
    step: int = 0
    parent: DeformableRecord | None = field(default=None, repr=False)
    #: Who made this record and from what: free-form, saved with the metadata
    #: (the linear agent's tool stores its section id, the linear handoff
    #: metadata and the fit inputs it chose).
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Section pixels (on the section grid) the fit's metric left out: the EM
    #: section mask's rejects (``masks.em_masks``); None for a fit with the
    #: footprint masks. For review pictures; nothing reads it back into a fit.
    left_out: np.ndarray | None = None

    @property
    def mm_per_px(self) -> float:
        return self.placement.section_mm_per_px

    def metadata(self) -> dict[str, Any]:
        return {
            "version": RECORD_VERSION, "step": self.step,
            "placement": self.placement.to_dict(),
            "section_size": list(self.section_size),
            "settings": None if self.settings is None else self.settings.to_dict(),
            "field_convention": (
                "field_mm[y, x] = [dx, dy] mm: section pixel (x, y) at (x, y)*mm_per_px "
                "corresponds to placed-atlas point (x, y)*mm_per_px + field; native = "
                "inv(atlas_to_section) @ (pixel + field/mm_per_px)"
            ),
            "inverse_source": self.inverse_source,
            "labels": "leaf atlas ids under the composed warp",
            "excluded_ids": list(self.excluded_ids),
            "engine": self.engine, "diagnostics": self.diagnostics, "atlas": self.atlas,
            "provenance": self.provenance,
        }

    def save(self, directory: str | Path) -> Path:
        """Write ``arrays.npz``, the parent chain in ``parent/``, then ``record.json``
        last, so an interrupted save leaves no ``record.json`` (the store rewrites it)."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "field_mm": self.field_mm.astype(np.float32), "labels": self.labels,
            "footprint": self.footprint,
        }
        if self.inverse_field_mm is not None:
            arrays["inverse_field_mm"] = self.inverse_field_mm.astype(np.float32)
        if self.left_out is not None:
            arrays["left_out"] = self.left_out.astype(bool)
        np.savez_compressed(out / "arrays.npz", **arrays)  # pyright: ignore[reportArgumentType]
        if self.parent is not None:
            self.parent.save(out / "parent")
        (out / "record.json").write_text(json.dumps(self.metadata(), indent=2,
                                                    default=_json_default) + "\n")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> DeformableRecord:
        """The record saved in *directory*; ``ValueError`` naming its
        ``record.json`` when it (or its arrays) cannot be read."""
        root = Path(directory)
        parent = cls.load(root / "parent") if (root / "parent" / "record.json").exists() else None
        try:
            meta = json.loads((root / "record.json").read_text())
            unknown = sorted(set(meta) - set(_RECORD_KEYS))
            if unknown:
                raise ValueError(f"Unknown record key(s) {unknown}")
            with np.load(root / "arrays.npz") as data:
                arrays = {key: data[key] for key in data.files}
            return cls(
                placement=Placement.from_dict(meta["placement"]),
                section_size=(int(meta["section_size"][0]), int(meta["section_size"][1])),
                settings=None if meta["settings"] is None else FitSettings.from_dict(
                    meta["settings"]),
                field_mm=arrays["field_mm"], inverse_field_mm=arrays.get("inverse_field_mm"),
                inverse_source=str(meta["inverse_source"]), labels=arrays["labels"],
                footprint=arrays["footprint"],
                excluded_ids=[int(i) for i in meta["excluded_ids"]], engine=meta["engine"],
                diagnostics=meta["diagnostics"], atlas=meta["atlas"], step=int(meta["step"]),
                parent=parent, provenance=dict(meta["provenance"]),
                left_out=arrays.get("left_out"),
            )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(f"{root / 'record.json'}: {exc}") from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
