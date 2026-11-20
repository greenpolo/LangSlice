"""The canonical per-section result of a deformable fit, and its plausibility report.

Field convention (brainglobe-registration's): ``field_mm[y, x]`` is the
residual displacement, in millimetres, of section pixel (x, y). The section
point ``p_mm = (x, y) * mm_per_px`` corresponds to the point
``p_mm + field_mm[y, x]`` of the LINEARLY PLACED atlas plane (the placed
plane is drawn on the section's own pixel grid). The full map to the native
atlas plane composes the linear placement after it::

    native_xy = inverse(atlas_to_section) @ (pixel + field_mm / mm_per_px)

(residual first, then undo the placement;
``core.maps.section_maps.native_points`` composes it the same way, and
``atlas["native_to_volume_index"]`` carries a native pixel to the atlas
volume), so ABBA, VisuAlign and BrainGlobe exports need no atlas object. The
inverse field lives on the same grid in the placed-atlas frame: placed point q
corresponds to section point ``q + inverse_field_mm(q)``.

A sequential fit is a chain: each record's field is the TOTAL residual
(earlier steps composed in), and ``parent`` is the record before this step,
so undoing a step is ``record.parent``.

On disk (:meth:`DeformableRecord.save`) a record is a folder: its arrays in
``warp.ome.zarr`` (:mod:`langslice.core.fits.deformable.ome_warp`) and
``record.json`` (the metadata, written last), which names the parent by its
folder, a sibling in the same parent folder (in a job, the section's
``deformable/``; the folder name is the record's key's first 24 hex
digits). A record keeps the step that made its fields where it has one
(``stored_step``: the engine's own fields on its working grid, or a landmark
mesh) and loading rebuilds the fields from it, composed with the parent's,
as the fit made them (the ``fields`` module); a map made elsewhere and a
restricted fit store their fields whole, as float16.

Plausibility diagnostics are reported, never enforced: a flag says the warp
folds tissue, drops or turns a region inside out, or moves the tissue further
than its brain allows, and a person or agent decides what that means. Each
region's area ratio is reported without a flag: a large change in a region's
area is as often the warp correcting the placement as distorting the tissue.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from langslice.core.fits.deformable import ome_warp
from langslice.core.fits.deformable.fields import (
    EngineStep,
    MeshStep,
    StoredStep,
    rebuilt_fields,
)
from langslice.core.fits.deformable.geometry import Placement, WorkingGrid
from langslice.core.fits.deformable.ome_warp import (
    WARP_STORE,
    at_storage_precision,
)
from langslice.core.fits.deformable.settings import FitSettings
from langslice.core.fits.flags import FitFlag
from langslice.core.jsonable import JsonObject, to_jsonable
from langslice.core.stack.json_reads import (
    list_at,
    number_at,
    numbers_at,
    object_at,
    optional_at,
    text_at,
)

logger = logging.getLogger(__name__)

MIN_FLAG_AREA_MM2 = 0.02
FOLD_FRACTION_LIMIT = 0.001
OUTSIZED_MAX_FRACTION = 0.1
OUTSIZED_MEDIAN_MM = 0.6

RECORD_VERSION = 3
_RECORD_KEYS = (
    "version",
    "step",
    "parent",
    "placement",
    "section_size",
    "settings",
    "field_convention",
    "fields",
    "inverse_source",
    "labels",
    "excluded_ids",
    "engine",
    "diagnostics",
    "atlas",
    "provenance",
    "fields_sha256",
)
FIELDS_WHOLE = "section_grid"
FIELDS_ENGINE = "engine_grid"
FIELDS_MESH = "landmark_mesh"
FIELDS_KINDS = (FIELDS_WHOLE, FIELDS_ENGINE, FIELDS_MESH)


def jacobian_determinant(field_mm: np.ndarray, mm_per_px: float) -> np.ndarray:
    """det D(p + u(p)) of the section-to-placed-atlas map, per section pixel."""
    u = field_mm[..., 0].astype(np.float64) / mm_per_px
    v = field_mm[..., 1].astype(np.float64) / mm_per_px
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    return np.asarray((1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx, dtype=np.float64)


def displacement_report(
    field_mm: np.ndarray,
    mm_per_px: float,
    footprint: np.ndarray,
    brain_scale: float = 1.0,
) -> dict[str, Any]:
    """Max and median displacement in *footprint*, their limits, and whether either is passed.

    The maximum's limit is :data:`OUTSIZED_MAX_FRACTION` of the footprint's
    longest extent (the longer side of its bounding box); the median's is
    :data:`OUTSIZED_MEDIAN_MM` times the atlas's *brain_scale*.
    """
    magnitude = np.linalg.norm(field_mm, axis=-1)[footprint]
    rows, cols = np.nonzero(footprint)
    extent_mm = (
        float(max(rows.max() - rows.min(), cols.max() - cols.min()) + 1) * mm_per_px
        if rows.size
        else 0.0
    )
    largest = float(magnitude.max()) if magnitude.size else 0.0
    median = float(np.median(magnitude)) if magnitude.size else 0.0
    max_limit = OUTSIZED_MAX_FRACTION * extent_mm
    median_limit = OUTSIZED_MEDIAN_MM * brain_scale
    return {
        "max_mm": largest,
        "median_mm": median,
        "tissue_extent_mm": extent_mm,
        "max_limit_mm": max_limit,
        "median_limit_mm": median_limit,
        "outsized": bool(magnitude.size) and (largest > max_limit or median > median_limit),
    }


LANDMARK_ENGINE = "landmark_triangulation"


def landmark_base_field(record: DeformableRecord | None) -> np.ndarray | None:
    """The field of the most recent landmark application in *record*'s chain
    (itself included), or None when the chain has none."""
    while record is not None:
        if record.engine.get("name") == LANDMARK_ENGINE:
            return record.field_mm
        record = record.parent
    return None


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
    base_field_mm: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-region area ratios from the Jacobian, fold fraction, and flags; the
    millimetre limits are the adult mouse's times the atlas's *brain_scale*.

    Folds and area ratios are measured on the whole *field_mm*. With
    *base_field_mm* (the agent's landmark warp, :func:`landmark_base_field`)
    DISPLACEMENT_OUTSIZED measures only what *field_mm* adds to it; without,
    the field's displacement from the linear placement.

    For region R, the warped region is the set of section pixels whose atlas
    point lies in R; its placed area follows from the Jacobian J of the
    section-to-placed map, so ``area_ratio = N_R / sum_{p in R} J(p)``
    (warped over placed; below 1 the warp compresses the region onto the
    section). The raster ratio of pixel counts is reported alongside.
    Area ratios raise no flag (a region whose area changes most under the
    warp is as often moved closer to its true outline as further from
    it); a region that vanishes or folds over itself does.
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
    pixel_mm2 = mm_per_px**2
    min_flag_area_mm2 = MIN_FLAG_AREA_MM2 * brain_scale**2
    regions: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    for k, uid in enumerate(int(i) for i in ids):
        ventricle = uid in ventricles
        placed_mm2 = float(placed_count[k] * pixel_mm2)
        ratio: float | None = 0.0
        if count[k]:
            ratio = float(count[k] / jac_sum[k]) if jac_sum[k] > 0 else None
        entry = {
            "id": uid,
            "acronym": acronyms.get(uid, str(uid)),
            "placed_area_mm2": placed_mm2,
            "warped_area_mm2": float(count[k] * pixel_mm2),
            "area_ratio": ratio,
            "raster_area_ratio": float(count[k] / placed_count[k]) if placed_count[k] else None,
            "ventricle": ventricle,
            "excluded": uid in excluded,
        }
        regions.append(entry)
        if uid in excluded or placed_mm2 < min_flag_area_mm2:
            continue
        code = None
        if count[k] == 0:
            code = FitFlag.REGION_VANISHED
        elif ratio is None:
            code = FitFlag.REGION_FOLDED
        if code:
            flags.append(
                {
                    "code": code,
                    "id": uid,
                    "acronym": entry["acronym"],
                    "area_ratio": ratio,
                    "ventricle": ventricle,
                }
            )
    if fold_fraction > FOLD_FRACTION_LIMIT:
        flags.append(
            {"code": FitFlag.FOLDS, "fold_fraction": fold_fraction, "limit": FOLD_FRACTION_LIMIT}
        )
    displacement = displacement_report(
        field_mm if base_field_mm is None else field_mm - base_field_mm,
        mm_per_px,
        footprint,
        brain_scale,
    )
    displacement["beyond_landmark_warp"] = base_field_mm is not None
    if displacement["outsized"]:
        flags.append(
            {
                "code": FitFlag.DISPLACEMENT_OUTSIZED,
                **{
                    key: displacement[key]
                    for key in ("max_mm", "median_mm", "max_limit_mm", "median_limit_mm")
                },
            }
        )
    return {
        "fold_fraction": fold_fraction,
        "jacobian_in_tissue": {
            "min": float(in_tissue.min()) if in_tissue.size else None,
            "p01": float(np.percentile(in_tissue, 1)) if in_tissue.size else None,
            "p99": float(np.percentile(in_tissue, 99)) if in_tissue.size else None,
            "max": float(in_tissue.max()) if in_tissue.size else None,
        },
        "max_displacement_mm": float(np.linalg.norm(field_mm, axis=-1)[footprint].max())
        if footprint.any()
        else 0.0,
        "displacement": displacement,
        "regions": regions,
        "flags": flags,
        "thresholds": {
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
    provenance: dict[str, Any] = field(default_factory=dict)
    left_out: np.ndarray | None = None
    weights: np.ndarray | None = None
    saved_as: str | None = field(default=None, repr=False, compare=False)
    stored_step: StoredStep | None = field(default=None, repr=False, compare=False)

    @property
    def mm_per_px(self) -> float:
        return self.placement.section_mm_per_px

    def round_to_storage(self) -> None:
        """Round the fields to the precision they are saved at
        (:func:`~.ome_warp.at_storage_precision`), so this record equals the
        one read back."""
        self.field_mm = at_storage_precision(self.field_mm)
        if self.inverse_field_mm is not None:
            self.inverse_field_mm = at_storage_precision(self.inverse_field_mm)
        if self.weights is not None:
            self.weights = np.asarray(self.weights).astype(np.float16)

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "version": RECORD_VERSION,
            "step": self.step,
            "parent": None if self.parent is None else self.parent.saved_as,
            "placement": self.placement.to_dict(),
            "section_size": list(self.section_size),
            "settings": None if self.settings is None else self.settings.to_dict(),
            "field_convention": (
                "field_mm[y, x] = [dx, dy] mm: section pixel (x, y) at (x, y)*mm_per_px "
                "corresponds to placed-atlas point (x, y)*mm_per_px + field; native = "
                "inv(atlas_to_section) @ (pixel + field/mm_per_px); stored in "
                f"{WARP_STORE} as `fields` says"
            ),
            "fields": self.fields_kind,
            "inverse_source": self.inverse_source,
            "labels": "leaf atlas ids under the composed warp",
            "excluded_ids": list(self.excluded_ids),
            "engine": self.engine,
            "diagnostics": self.diagnostics,
            "atlas": self.atlas,
            "provenance": self.provenance,
        }

        if self.stored_step is not None:
            meta["fields_sha256"] = fields_digest(self.field_mm, _present(self.inverse_field_mm))
        return meta

    @property
    def fields_kind(self) -> str:
        """Name how the fields are saved (``record.json``'s ``fields``)."""
        if isinstance(self.stored_step, EngineStep):
            return FIELDS_ENGINE
        if isinstance(self.stored_step, MeshStep):
            return FIELDS_MESH
        return FIELDS_WHOLE

    def save(self, directory: str | Path) -> Path:
        """Write the record in *directory*: its arrays (``warp.ome.zarr``), then
        ``record.json`` last, so an interrupted save leaves no ``record.json``
        (the store rewrites it). The fields are rounded to the precision they
        are saved at first (:meth:`round_to_storage`). The parent is named by its folder, a sibling
        of *directory*; a parent not saved there is saved there first.
        ``ValueError`` for a parent that has no folder name
        (:attr:`saved_as`): save it first, beside this one."""
        out = Path(directory)
        if self.parent is not None:
            name = self.parent.saved_as
            if not name:
                raise ValueError(
                    "The parent record has no folder: save it first, in the same parent folder"
                )
            if not (out.parent / name / "record.json").exists():
                self.parent.save(out.parent / name)
        out.mkdir(parents=True, exist_ok=True)
        (out / "record.json").unlink(missing_ok=True)
        self.round_to_storage()
        ome_warp.write(out, self)
        metadata = to_jsonable(self.metadata(), strict=True)
        (out / "record.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        self.saved_as = out.name
        return out

    @classmethod
    def load(
        cls, directory: str | Path, _children: frozenset[str] = frozenset()
    ) -> DeformableRecord:
        """The record saved in *directory*, its parent chain from the sibling
        folders ``record.json`` names; ``ValueError`` naming the
        ``record.json`` that (or whose arrays) cannot be read, or whose chain
        comes back to itself."""
        root = Path(directory)
        meta = cls._metadata(root)
        if root.name in _children:
            raise ValueError(f"{root / 'record.json'}: its parent chain comes back to it")
        parent = (
            None
            if meta["parent"] is None
            else cls.load(root.parent / str(meta["parent"]), _children | {root.name})
        )
        try:
            arrays = ome_warp.read(root)
            field_mm, inverse_mm, stored_step = _read_fields(meta, arrays, parent)
            _check_rebuilt(root, meta, field_mm, inverse_mm)
            return cls(
                placement=Placement.from_dict(meta["placement"]),
                section_size=(int(meta["section_size"][0]), int(meta["section_size"][1])),
                settings=None
                if meta["settings"] is None
                else FitSettings.from_dict(meta["settings"]),
                field_mm=field_mm,
                inverse_field_mm=inverse_mm,
                inverse_source=str(meta["inverse_source"]),
                labels=_present(arrays["labels"]),
                footprint=_present(arrays["footprint"]),
                excluded_ids=[int(i) for i in meta["excluded_ids"]],
                engine=meta["engine"],
                diagnostics=meta["diagnostics"],
                atlas=meta["atlas"],
                step=int(meta["step"]),
                parent=parent,
                provenance=dict(meta["provenance"]),
                left_out=arrays["left_out"],
                weights=arrays["weights"],
                saved_as=root.name,
                stored_step=stored_step,
            )
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{root / 'record.json'}: {exc}") from exc

    @staticmethod
    def _metadata(root: Path) -> dict[str, Any]:
        """*root*'s ``record.json``, checked: its keys and its format."""
        try:
            meta = json.loads((root / "record.json").read_text(encoding="utf-8"))
            unknown = sorted(set(meta) - set(_RECORD_KEYS))
            if unknown:
                raise ValueError(f"Unknown record key(s) {unknown}")
            if meta.get("version") != RECORD_VERSION:
                raise ValueError(
                    f"record format {meta.get('version')!r}, which "
                    f"this LangSlice does not read (it reads {RECORD_VERSION})"
                )
            if "parent" not in meta:
                raise KeyError("parent")
            return meta
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{root / 'record.json'}: {exc}") from exc


def _read_fields(
    meta: JsonObject,
    arrays: dict[str, np.ndarray | None],
    parent_record: DeformableRecord | None,
) -> tuple[np.ndarray, np.ndarray | None, StoredStep | None]:
    """Read a record's fields: as stored whole, or rebuilt from its step.

    Returns the forward and inverse fields on the section grid at storage
    precision (the inverse None for a record saved without one) and the
    step they were rebuilt from (None for fields saved whole).
    """
    kind = text_at(meta, "fields")
    if kind == FIELDS_WHOLE:
        return _present(arrays["field_mm"]), arrays["inverse_field_mm"], None
    width, height = (int(v) for v in numbers_at(meta, "section_size"))
    placement = object_at(meta, "placement")
    mm_per_px = number_at(placement, "section_mm_per_px")
    engine = object_at(meta, "engine")
    step: StoredStep
    if kind == FIELDS_ENGINE:
        working = numbers_at(object_at(engine, "working_grid"), "size")
        working_size = (int(working[0]), int(working[1]))
        grid = WorkingGrid((width, height), mm_per_px, working_size)
        step = EngineStep(
            grid, _present(arrays["step_field_mm"]), _present(arrays["step_inverse_field_mm"])
        )
    elif kind == FIELDS_MESH:
        mesh = object_at(engine, "native_parameters")
        source = np.asarray(list_at(mesh, "source_xy"), dtype=np.float64)
        target = np.asarray(list_at(mesh, "target_xy"), dtype=np.float64)
        triangles = np.asarray(list_at(mesh, "simplices"), dtype=np.int32)
        step = MeshStep(source, target, triangles)
    else:
        raise ValueError(f"fields {kind!r} is none of {FIELDS_KINDS}")
    previous = None
    if parent_record is not None:
        parent_inverse_mm = _present(parent_record.inverse_field_mm)
        previous = (parent_record.field_mm, parent_inverse_mm)
    field_mm, inverse_mm = rebuilt_fields(step, (width, height), mm_per_px, previous)
    field_mm = at_storage_precision(field_mm)
    return field_mm, at_storage_precision(inverse_mm), step


def fields_digest(field_mm: np.ndarray, inverse_field_mm: np.ndarray) -> str:
    """Hash a record's fields as stored: float16, forward then inverse.

    Parameters
    ----------
    field_mm
        The forward field, ``(height, width, 2)``, at storage precision.
    inverse_field_mm
        The inverse field, likewise.

    Returns
    -------
    str
        The SHA-256 of both fields' float16 bytes in C order, hex.
    """
    digest = hashlib.sha256()
    for plane in (field_mm, inverse_field_mm):
        stored = np.ascontiguousarray(plane, dtype=np.float16)
        digest.update(stored.tobytes())
    return digest.hexdigest()


def _check_rebuilt(
    root: Path,
    meta: JsonObject,
    field_mm: np.ndarray,
    inverse_field_mm: np.ndarray | None,
) -> None:
    """Warn when rebuilt fields differ from the bits the record was made with.

    A record whose ``record.json`` has no ``fields_sha256`` is not checked.
    """
    made = optional_at(meta, "fields_sha256", text_at)
    if made is None or inverse_field_mm is None:
        return
    if fields_digest(field_mm, inverse_field_mm) == made:
        return
    logger.warning(
        "%s: the fields were rebuilt with different bits on this machine "
        "than the ones the record was made with; a fit started from this "
        "record can differ from one started on the machine that made it",
        root / "record.json",
    )


def _present(array: np.ndarray | None) -> np.ndarray:
    if array is None:
        raise ValueError("an array is missing")
    return array
