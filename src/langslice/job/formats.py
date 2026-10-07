"""The job folder's public files: ``registration.json`` and each section's maps.

``docs/file_formats.md`` describes every field. ``state.json`` stays the
job's one working source; what is written here is derived from it, in
public units, for scripts and other programs, and is never read back as
input:

- ``registration.json`` (top of the job folder): every section's
  registration parameters (atlas, pixel size, plane, orientation, in-plane
  affine, the applied deformation's record) and, for a placed section, the
  matrix from the image file's pixels ``[row, col, 1]`` to BrainGlobe atlas
  micrometres, plus the section's derived files. Small; rewritten atomically
  on every checkpoint (:func:`write_registration`, called by
  ``Job.checkpoint``).
- per section (``sections/<name>/``), on submit and on demand
  (``export_maps``, :func:`write_section_maps`): ``coords.tif``,
  ``labels.tif``, ``labels_fiji.tif`` + ``labels.csv`` (over the
  section's footprint, its filled outline), ``tissue.png`` (the
  threshold's tissue estimate), ``residual.tif`` (only with a deformation)
  and ``maps.json`` (what the maps were written from: grid, matrix, the
  parameters' digest).

Writers never raise into a write of the state: a failure is logged (and the
section's entry says why it has no matrix).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import langslice
from langslice.job.checkpoint import write_json_atomic
from langslice.job.layout import (
    IMAGES_ARE_PARENT,
    REGISTRATION_FILE,
    JobLayout,
    job_folder_for,
)

if TYPE_CHECKING:
    from langslice.core.maps import SectionFrame, SectionMaps
    from langslice.core.state import SliceState, StackState
    from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

#: ``registration.json``'s format.
REGISTRATION_FORMAT_VERSION = 1
#: ``maps.json``'s format.
MAPS_FORMAT_VERSION = 1
COORDS_FILE = "coords.tif"
LABELS_FILE = "labels.tif"
LABELS_FIJI_FILE = "labels_fiji.tif"
LABELS_CSV_FILE = "labels.csv"
RESIDUAL_FILE = "residual.tif"
TISSUE_FILE = "tissue.png"
MAPS_FILE = "maps.json"
#: Under ``exports/``: the QuickNII anchoring of every placed section, and the
#: same with VisuAlign markers of each applied deformation.
QUICKNII_FILE = "quicknii.json"
VISUALIGN_FILE = "visualign.json"
#: Every per-section file, by its artifact kind.
SECTION_FILES: dict[str, str] = {
    "coords": COORDS_FILE, "labels": LABELS_FILE, "labels_fiji": LABELS_FIJI_FILE,
    "labels_csv": LABELS_CSV_FILE, "tissue": TISSUE_FILE, "residual": RESIDUAL_FILE,
    "maps": MAPS_FILE,
}
#: Stated in ``registration.json``, so a script reading it needs nothing else.
CONVENTION = (
    "Atlas coordinates are BrainGlobe micrometres in the atlas's own axis order "
    "(atlas.orientation; packaged atlases are 'asr': axis 0 anterior to posterior, "
    "1 superior to inferior, 2 right to left), voxel i's centre at i * resolution_um. "
    "Image pixels are [row, col] of the image FILE as stored (no rotation or flip "
    "applied; those are in the mapping), pixel centres at integers, row 0 at the top. "
    "pixel_to_atlas_um maps [row, col, 1] to atlas micrometres under the linear "
    "placement; with a deformation, coords.tif holds the complete mapping and "
    "residual.tif the displacement d such that atlas_um = pixel_to_atlas_um @ "
    "[row + d_row, col + d_col, 1] on that grid."
)
#: What ``registration.json`` says about itself.
NOTE = (
    "Derived from state.json on every change and never read back: edit nothing "
    "here. Change a registration through the verbs (langslice-job FOLDER VERB, or "
    "langslice.open_job in Python). Edited maps or label images become a "
    "registration only through a fit (see docs/file_formats.md)."
)

# --- the registration parameters ----------------------------------------------------


def digest(value: Any) -> str:
    """SHA-256 of a JSON value (sorted keys)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def applied_deformation(state: StackState, record: SliceState) -> dict[str, Any] | None:
    """The section's deformation as it stands on its current placement, or None."""
    from langslice.core.deformation import linear_key

    held = record.deformation
    if not held or held.get("linear_key") != linear_key(state, record):
        return None
    return held


def parameters(state: StackState, record: SliceState, atlas: dict[str, Any] | None,
               ) -> dict[str, Any]:
    """The section's registration parameters in public units (the truth, as
    ``state.json`` holds it)."""
    transform = record.transform or {}
    held = applied_deformation(state, record)
    deformation: dict[str, Any] | None
    if held is None:
        deformation = None
    elif "keep_linear" in held:
        deformation = {"kind": "none", "reason": str(held.get("keep_linear") or "")}
    else:
        deformation = {"kind": "residual", "record": held.get("record"),
                       "key": held.get("key"), "steps": held.get("steps") or [],
                       "inverse_source": held.get("inverse_source")}
    position = record.position_mm
    affine: dict[str, Any] | None = None
    if transform:
        affine = {
            "kind": transform.get("kind"),
            "params": transform.get("params"),
            "params_convention": (
                "normalized 2x3 [a, b, tx, c, d, ty] on the oriented section render: "
                "x as a fraction of its width, y of its height (langslice.core.affine)"),
            "physical": transform.get("physical"),
            "mirrored": bool(transform.get("mirrored", False)),
        }
    return {
        "atlas": None if atlas is None else {"name": atlas.get("name"),
                                             "version": atlas.get("version")},
        "plane": {
            "name": state.plane,
            "position_mm": position,
            "position_um": None if position is None else float(position) * 1000.0,
            "pitch_deg": record.pitch_deg,
            "yaw_deg": record.yaw_deg,
        },
        "orientation": {"rotation_deg": int(record.rotation_deg), "flip": bool(record.flip),
                        "order": "rotate (counter-clockwise quarter turns) first, then "
                                 "flip left-right"},
        "affine": affine,
        "deformation": deformation,
        "damaged": bool(record.damaged),
        "damaged_regions": list(record.damaged_regions),
    }


def section_entry(
    state: StackState, workspace: Workspace | None, layout: JobLayout, record: SliceState,
    atlas: dict[str, Any] | None,
) -> dict[str, Any]:
    """One section of ``registration.json``."""
    from langslice.core.maps import placement_problem, scale_problem, section_frame

    folder = layout.section_dir(record.id)
    entry: dict[str, Any] = {
        "id": record.id,
        "order": int(record.index_corrected),
        "folder": layout.relative(folder),
        "parameters": parameters(state, record, atlas),
    }
    frame: SectionFrame | None = None
    problem = placement_problem(state, record)
    if problem is None and workspace is None:
        problem = "not computed (no workspace)"
    if problem is None:
        assert workspace is not None
        try:
            frame = section_frame(state, workspace, record)
        except Exception as exc:  # an unreadable file, a missing atlas: said, not raised
            logger.warning("No frame for %s", record.id, exc_info=True)
            problem = str(exc)
        else:
            problem = scale_problem(frame)
    entry["image"] = None if frame is None else {
        "file": record.id, "size": list(frame.file_size),
        "pixel_size_um": frame.file_um_per_px, "pixel_size_source": frame.calibration_source,
    }
    entry["pixel_to_atlas_um"] = None if frame is None else frame.pixel_to_atlas_um().tolist()
    entry["mapping"] = None
    if frame is not None:
        entry["mapping"] = "linear" if record.transform else "linear (identity in-plane: " \
                                                            "no transform written)"
    if frame is not None and entry["parameters"]["deformation"] is not None \
            and entry["parameters"]["deformation"]["kind"] == "residual":
        entry["mapping"] = "linear + residual"
    entry["problem"] = problem
    entry["parameters_digest"] = digest({"parameters": entry["parameters"],
                                         "matrix": entry["pixel_to_atlas_um"],
                                         "image": entry["image"]})
    entry["maps"] = maps_status(layout, record.id, entry["parameters_digest"])
    return entry


def maps_status(layout: JobLayout, section_id: str, current: str) -> dict[str, Any] | None:
    """The section's written maps (``maps.json``) and whether they are current."""
    folder = layout.section_dir(section_id)
    try:
        held = json.loads((folder / MAPS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    files = {kind: layout.relative(folder / name) for kind, name in SECTION_FILES.items()
             if (folder / name).exists()}
    return {
        "files": files,
        "grid_size": held.get("grid_size"),
        "full_resolution": held.get("full_resolution"),
        "pixel_to_atlas_um": held.get("pixel_to_atlas_um"),
        "current": held.get("parameters_digest") == current,
    }


def registration_document(
    state: StackState, workspace: Workspace | None, layout: JobLayout,
) -> dict[str, Any]:
    """``registration.json`` for *state* (see the module text)."""
    atlas: dict[str, Any] | None = None
    if workspace is not None:
        try:
            from langslice.core.layers import atlas_facts

            atlas = atlas_facts(workspace.atlas)
        except Exception:  # an atlas that cannot load: the document says so
            logger.warning("registration.json: the atlas could not be read", exc_info=True)
    exports = {kind: layout.relative(layout.exports_dir / name)
               for kind, name in (("quicknii", QUICKNII_FILE), ("visualign", VISUALIGN_FILE))
               if (layout.exports_dir / name).exists()}
    return {
        "format_version": REGISTRATION_FORMAT_VERSION,
        "written_by": f"LangSlice {langslice.__version__}",
        "note": NOTE,
        "convention": CONVENTION,
        "atlas": atlas if atlas is not None else {"name": state.atlas},
        "plane": state.plane,
        # The stack's one plane; null when the sections' angles differ (a
        # registration supplied per section): each section's "plane" has its own.
        "cutting_angles_deg": (None if state.mixed_angles else
                               dict(zip(("pitch", "yaw"), state.stack_angles, strict=True))),
        "image_folder": _image_folder(layout, workspace.image_folder if workspace is not None
                                      else state.image_folder),
        "submitted": bool(state.submitted),
        "sections": [section_entry(state, workspace, layout, record, atlas)
                     for record in state.in_order()],
        "exports": exports,
    }


def _image_folder(layout: JobLayout, images: str) -> str:
    """The image folder as ``job.json`` stores it: the job folder's parent
    (:data:`~langslice.job.layout.IMAGES_ARE_PARENT`) for the default job
    folder, so the file stays right when the folder moves with its images;
    else the absolute path."""
    if layout.folder == job_folder_for(images):
        return IMAGES_ARE_PARENT
    return str(images)


def write_registration(
    layout: JobLayout, state: StackState, workspace: Workspace | None,
) -> Path | None:
    """Write ``registration.json`` atomically; never raises (None on failure)."""
    path = layout.folder / REGISTRATION_FILE
    try:
        write_json_atomic(str(path), registration_document(state, workspace, layout))
    except Exception:
        logger.warning("Could not write %s", path, exc_info=True)
        return None
    return path


# --- the per-section maps -------------------------------------------------------------


#: Float maps are deflate-compressed WITHOUT the floating-point predictor:
#: ImageJ 1.x cannot decode it (``ij/io/TiffDecoder.java`` logs "unsupported
#: predictor value of 3" and reads the bytes as they are), so Fiji would show
#: garbage. The price is file size: the float maps compress far less.
FLOAT_COMPRESSION = "zlib"


def _imagej_resolution(um_per_px: float) -> dict[str, Any]:
    return {"resolution": (1.0 / um_per_px, 1.0 / um_per_px)}


def write_float_channels(path: Path, planes: np.ndarray, names: Iterable[str],
                         um_per_px: float, info: dict[str, Any]) -> None:
    """An ImageJ hyperstack of float32 channels ``(channels, rows, cols)``,
    calibrated in micrometres, *info* as JSON in its Info property."""
    import tifffile

    data = np.ascontiguousarray(planes, dtype=np.float32)
    ranges: list[float] = []
    for plane in data:
        finite = plane[np.isfinite(plane)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        ranges += [low, high if high > low else low + 1.0]
    tifffile.imwrite(
        path, data, imagej=True, compression=FLOAT_COMPRESSION, **_imagej_resolution(um_per_px),
        metadata={"axes": "CYX", "unit": "um", "Labels": list(names), "Ranges": tuple(ranges),
                  "Info": json.dumps(info)},
    )


def write_labels(folder: Path, labels: np.ndarray, atlas: Any, um_per_px: float,
                 info: dict[str, Any]) -> list[tuple[Path, str]]:
    """``labels.tif`` (uint32 atlas ids), ``labels_fiji.tif`` (uint16 dense
    index, with an ImageJ lookup table giving indexes 1-255 their atlas
    colours) and ``labels.csv`` (index, id, acronym, name, r, g, b, every
    index)."""
    import tifffile

    from langslice.core.atlas.recolor import structure_rows

    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size > 65535:
        raise ValueError("More than 65535 regions in one section")
    dense = np.zeros(labels.shape, dtype=np.uint16)
    if ids.size:
        dense = (np.searchsorted(ids, labels) + 1).astype(np.uint16)
        dense[labels == 0] = 0
    rows = structure_rows(atlas)
    table: list[tuple[int, int, str, str, int, int, int]] = []
    for index, uid in enumerate(int(v) for v in ids):
        entry = rows.get(uid, {})
        rgb = entry.get("rgb_triplet") or (255, 255, 255)
        table.append((index + 1, uid, str(entry.get("acronym", uid)),
                      str(entry.get("name", uid)), int(rgb[0]), int(rgb[1]), int(rgb[2])))
    tifffile.imwrite(folder / LABELS_FILE, labels.astype(np.uint32), compression="zlib",
                     predictor=True, description=json.dumps(info))
    lut = np.zeros((3, 256), dtype=np.uint8)
    for index, _uid, _acronym, _name, red, green, blue in table:
        if index < 256:
            lut[:, index] = (red, green, blue)
    tifffile.imwrite(
        folder / LABELS_FIJI_FILE, dense, imagej=True, compression="zlib",
        **_imagej_resolution(um_per_px),
        metadata={"axes": "YX", "unit": "um", "mode": "color", "LUTs": [lut],
                  "Ranges": (0.0, 255.0),
                  "Info": json.dumps({**info, "index": "labels.csv: index -> atlas id"})},
    )
    with (folder / LABELS_CSV_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "id", "acronym", "name", "r", "g", "b"])
        writer.writerows(table)
    return [(folder / LABELS_FILE, "labels"), (folder / LABELS_FIJI_FILE, "labels_fiji"),
            (folder / LABELS_CSV_FILE, "labels_csv")]


def write_section_maps(
    layout: JobLayout, maps: SectionMaps, atlas: Any, *, parameters_digest: str,
    deformation_record: str | None,
) -> list[tuple[Path, str]]:
    """Write one section's maps (see the module text); return ``(path, kind)``
    of every file written. A residual left by an earlier export is removed
    when the section has no deformation now."""
    folder = layout.section_dir(maps.section_id)
    folder.mkdir(parents=True, exist_ok=True)
    info = {
        "section": maps.section_id, "grid_size": list(maps.grid_size),
        "full_resolution": maps.full_resolution, "um_per_px": maps.um_per_px,
        "pixel_to_atlas_um": maps.pixel_to_atlas_um.tolist(),
        "parameters_digest": parameters_digest, "written_by": f"LangSlice {langslice.__version__}",
    }
    written: list[tuple[Path, str]] = []
    write_float_channels(folder / COORDS_FILE, np.moveaxis(maps.coords, -1, 0),
                         ("axis0_um", "axis1_um", "axis2_um"), maps.um_per_px,
                         {**info, "content": "atlas micrometres per pixel, NaN outside the "
                                             "section's footprint or the atlas"})
    written.append((folder / COORDS_FILE, "coords"))
    written += write_labels(folder, maps.labels, atlas, maps.um_per_px,
                            {**info, "content": "atlas ids per pixel, 0 outside the "
                                                "section's footprint or the atlas"})
    from PIL import Image

    Image.fromarray(np.where(maps.tissue, 255, 0).astype(np.uint8)).save(
        folder / TISSUE_FILE, format="PNG", optimize=True)
    written.append((folder / TISSUE_FILE, "tissue"))
    residual_path = folder / RESIDUAL_FILE
    if maps.residual is not None:
        write_float_channels(residual_path, np.moveaxis(maps.residual, -1, 0),
                             ("d_row_px", "d_col_px"), maps.um_per_px,
                             {**info, "content": "displacement (drow, dcol) in grid pixels: "
                                                 "atlas_um = pixel_to_atlas_um @ [row + drow, "
                                                 "col + dcol, 1]",
                              "deformation_record": deformation_record})
        written.append((residual_path, "residual"))
    elif residual_path.exists():
        residual_path.unlink()
    record = {"format_version": MAPS_FORMAT_VERSION, **info,
              "tissue_found": maps.tissue_found,
              "footprint_fraction": float(np.mean(maps.footprint)),
              "tissue_fraction": float(np.mean(maps.tissue)),
              "deformation_record": deformation_record,
              "files": {kind: name for kind, name in SECTION_FILES.items()
                        if kind != "maps" and (folder / name).exists()}}
    write_json_atomic(str(folder / MAPS_FILE), record)
    written.append((folder / MAPS_FILE, "maps"))
    return written


def derived_files(layout: JobLayout, state: StackState) -> list[tuple[Path, str]]:
    """``(path, kind)`` of every derived file the job folder holds now:
    ``registration.json``, the exports and each section's maps."""
    found: list[tuple[Path, str]] = []
    top = layout.folder / REGISTRATION_FILE
    if top.exists():
        found.append((top, "registration"))
    for name, kind in ((QUICKNII_FILE, "quicknii"), (VISUALIGN_FILE, "visualign")):
        if (layout.exports_dir / name).exists():
            found.append((layout.exports_dir / name, kind))
    for record in state.in_order():
        folder = layout.section_dir(record.id)
        found += [(folder / name, kind) for kind, name in SECTION_FILES.items()
                  if (folder / name).exists()]
    return found
