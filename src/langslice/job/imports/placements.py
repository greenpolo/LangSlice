"""Imported placements and the inputs they become."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from langslice.core.maps.imported import (
    ANGLE_EPSILON_DEG,
    RecoveredPlacement,
    implied_pixel_size_um,
    placement_from_pixel_map,
)
from langslice.core.maps.section_maps import SWAP, orientation_matrix
from langslice.job.quint import (
    anchoring_from_pixel_map,
    from_target_grid,
    quicknii_points_to_atlas_um,
    quicknii_target,
    to_target_grid,
)

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

from langslice.job.imports.matching import match_sections
from langslice.job.imports.readers import (
    DEEPSLICE_DEFAULT_TARGET,
    RegistrationEntry,
    RegistrationFile,
    read_registration,
)

#: Relative aspect-ratio difference between the registered image and the
#: job's file above which the placement is said to be by fractions only.
ASPECT_TOLERANCE = 0.01
#: The transform ``kind`` an imported placement carries.
IMPORTED_KIND = "imported"


@dataclass(frozen=True)
class ImportedPlacement:
    """One section's imported placement (see the module text).

    ``placement`` is the geometry in LangSlice's terms;
    ``pixel_to_atlas_um`` the imported map on the job file's pixels
    (``[row, col, 1]`` -> atlas micrometres) it was recovered from.
    ``markers`` are the VisuAlign markers on the job file's continuous pixel
    coordinates (None when the file gives none or they cannot be scaled),
    ``markers_raw`` as written.
    """

    section_id: str
    source_filename: str
    match: str
    nr: int | None
    placement: RecoveredPlacement
    pixel_size_um: float
    pixel_size_source: str
    pixel_to_atlas_um: np.ndarray
    markers: list[list[float]] | None
    markers_raw: list[list[float]]
    problems: list[str]

    @property
    def position_mm(self) -> float:
        return self.placement.position_mm

    @property
    def pitch_deg(self) -> float:
        return self.placement.pitch_deg

    @property
    def yaw_deg(self) -> float:
        return self.placement.yaw_deg

    @property
    def flip(self) -> bool:
        return self.placement.flip

    @property
    def rotation_deg(self) -> int:
        return self.placement.rotation_deg

    @property
    def params(self) -> tuple[float, ...]:
        return self.placement.params

    def transform(self) -> dict[str, Any]:
        """The stored-transform dictionary this placement is supplied as
        (``kind`` :data:`IMPORTED_KIND`, the six numbers, the knobs every
        stored transform carries (``physical``), their reading on the render
        (``in_plane``), the calibration they are relative to)."""
        parts = dict(self.placement.in_plane)
        mirrored = bool(parts.pop("mirrored", False))
        parts.pop("translate_x_frac", None)
        parts.pop("translate_y_frac", None)
        return {
            "kind": IMPORTED_KIND,
            "params": list(self.placement.params),
            "physical": self.placement.physical(),
            "in_plane": parts,
            "mirrored": mirrored,
            "calibration": {"section_um_per_px": self.placement.render_um_per_px,
                            "source": self.pixel_size_source},
            "note": f"imported from {self.source_filename}",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id, "source_filename": self.source_filename,
            "match": self.match, "nr": self.nr, **self.placement.to_dict(),
            "pixel_size_um": self.pixel_size_um, "pixel_size_source": self.pixel_size_source,
            "pixel_to_atlas_um": self.pixel_to_atlas_um.tolist(), "markers": self.markers,
            "markers_raw": self.markers_raw, "problems": list(self.problems),
        }


@dataclass(frozen=True)
class ImportResult:
    """Every matched section's placement, and what did not match."""

    source: RegistrationFile
    placements: list[ImportedPlacement]
    unmatched: list[str]
    missing: list[str]
    refused: dict[str, str]
    pixel_size_um: float | None
    pixel_size_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.source.path, "format": self.source.format,
            "target": self.source.target, "aligner": self.source.aligner,
            "notes": list(self.source.notes),
            "placements": [placement.to_dict() for placement in self.placements],
            "unmatched": list(self.unmatched), "missing": list(self.missing),
            "refused": dict(self.refused), "pixel_size_um": self.pixel_size_um,
            "pixel_size_source": self.pixel_size_source,
        }


# --- to the job's atlas ----------------------------------------------------------------


def _fraction_map_from_anchoring(anchoring: Sequence[float], atlas: dict[str, Any],
                                 ) -> np.ndarray:
    """3x3: ``[fy, fx, 1]`` (fractions of the image's height and width from
    its top-left corner) -> atlas micrometres, for an anchoring already on
    the atlas's own voxel grid."""
    o, u, v = (np.asarray(anchoring[i:i + 3], dtype=np.float64) for i in (0, 3, 6))
    corner, right, down = quicknii_points_to_atlas_um(np.stack([o, o + u, o + v]), atlas)
    return np.column_stack([down - corner, right - corner, corner])


def _fraction_map_from_pixels(pixel_to_atlas_um: np.ndarray, size: tuple[int, int],
                              ) -> np.ndarray:
    width, height = float(size[0]), float(size[1])
    to_pixels = np.array([[height, 0.0, -0.5], [0.0, width, -0.5], [0.0, 0.0, 1.0]])
    return np.asarray(pixel_to_atlas_um, dtype=np.float64) @ to_pixels


def pixel_map_on(fraction_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The fraction map on the pixels of an image of *size* (width, height):
    ``[row, col, 1]`` (centres at integers) -> atlas micrometres."""
    width, height = float(size[0]), float(size[1])
    to_fractions = np.array([[1.0 / height, 0.0, 0.5 / height],
                             [0.0, 1.0 / width, 0.5 / width], [0.0, 0.0, 1.0]])
    return fraction_map @ to_fractions


def _entry_fraction_map(source: RegistrationFile, entry: RegistrationEntry,
                        atlas: dict[str, Any], target: str | None) -> np.ndarray:
    """The entry's placement as a fraction map in the job atlas's micrometres
    (``ValueError`` when it cannot be read into that atlas)."""
    job_target = quicknii_target(str(atlas["name"]))
    if entry.pixel_to_atlas_um is not None:
        if entry.size is None:
            raise ValueError("registration.json gives no image size")
        held = source.atlas or {}
        if held.get("name") == atlas["name"]:
            return _fraction_map_from_pixels(entry.pixel_to_atlas_um, entry.size)
        if not all(key in held for key in ("name", "orientation", "shape", "resolution_um")):
            raise ValueError("registration.json does not describe its atlas")
        held_target = quicknii_target(str(held["name"]))
        if held_target != job_target:
            raise ValueError(f"registered to {held['name']}, which has no common QuickNII "
                             f"target with the job's atlas {atlas['name']}")
        vector = to_target_grid(
            anchoring_from_pixel_map(entry.pixel_to_atlas_um, entry.size[0], entry.size[1],
                                     held), str(held["name"]), held["resolution_um"])
        anchoring: Sequence[float] = vector.to_list()
    else:
        if entry.anchoring is None:
            raise ValueError(entry.problem or "no anchoring")
        if target is not None and target != job_target:
            raise ValueError(f"registered to the target {target}; the job's atlas "
                             f"{atlas['name']} is exported to {job_target}")
        anchoring = entry.anchoring
    on_grid = from_target_grid(anchoring, str(atlas["name"]), atlas["resolution_um"])
    return _fraction_map_from_anchoring(on_grid, atlas)


def _scaled_markers(entry: RegistrationEntry, file_size: tuple[int, int],
                    ) -> list[list[float]] | None:
    if not entry.markers:
        return None
    if entry.size is None:
        return None
    sx = file_size[0] / float(entry.size[0])
    sy = file_size[1] / float(entry.size[1])
    return [[m[0] * sx, m[1] * sy, m[2] * sx, m[3] * sy] for m in entry.markers]


def import_placements(
    source: RegistrationFile | str | os.PathLike[str],
    workspace: Workspace,
    *,
    section_ids: Iterable[str] | None = None,
    orientations: Mapping[str, tuple[int, bool]] | None = None,
    planes: Mapping[str, tuple[float, float, float]] | None = None,
    registered_on: str = "file",
) -> ImportResult:
    """Every matched section's placement in the job's terms (see the module
    text). *workspace* gives the job's atlas, plane, section files and
    calibration; *section_ids* the sections (default: the folder's images,
    named as the job names them). *orientations* fixes a section's
    ``(rotation_deg, flip)`` (default: chosen per section). *planes* fixes a
    section's ``(position_mm, pitch_deg, yaw_deg)`` (default: the file's own):
    its six numbers are then the file's map projected onto that plane.
    A section whose entry cannot be placed is listed under ``refused`` with
    the reason; an ambiguous match raises :class:`AmbiguousMatch`.

    *registered_on* says which image the file's entries were made on:
    ``"file"`` (default) the section's image file as stored, ``"oriented"`` the
    section as its orientation turns and flips it (what a tool that drew
    ``section_raster`` and registered that picture holds); the entry's width,
    height and fractions then refer to the oriented image, and the placement is
    read for the orientation in *orientations* (default: none)."""
    from langslice.core.maps.section_maps import render_sizes
    from langslice.core.pictures.layers import atlas_facts
    from langslice.core.sections.discovery import discover_slices

    if registered_on not in ("file", "oriented"):
        raise ValueError(f"registered_on is 'file' or 'oriented'; got {registered_on!r}")
    held = source if isinstance(source, RegistrationFile) else read_registration(source)
    if section_ids is None:
        section_ids = [os.path.basename(path)
                       for path in discover_slices(workspace.image_folder)]
    ids = list(section_ids)
    names = [entry.filename for entry in held.entries]
    matched = match_sections(names, ids)
    atlas_object = workspace.atlas
    atlas = atlas_facts(atlas_object)
    plane = workspace.settings.plane
    file_target = held.target or (
        DEEPSLICE_DEFAULT_TARGET if held.format == "deepslice-csv" else None)

    refused: dict[str, str] = {}
    prepared: list[tuple[int, str, str, Any, np.ndarray, list[str]]] = []
    for index, (section_id, rule) in sorted(matched.items()):
        entry = held.entries[index]
        problems: list[str] = []
        try:
            fractions = _entry_fraction_map(held, entry, atlas, file_target)
            sizes = render_sizes(workspace, section_id)
        except (OSError, ValueError) as exc:
            refused[section_id] = str(exc)
            continue
        turn = (orientations or {}).get(section_id, (0, False)) if registered_on == "oriented" \
            else (0, False)
        shown = (sizes.file_size[1], sizes.file_size[0]) if int(turn[0]) % 180 == 90 \
            else sizes.file_size
        if entry.size is not None:
            registered = entry.size[0] / float(entry.size[1])
            here = shown[0] / float(shown[1])
            if abs(registered / here - 1.0) > ASPECT_TOLERANCE:
                problems.append(
                    f"registered on a {entry.size[0]} x {entry.size[1]} image, the file is "
                    f"{shown[0]} x {shown[1]}: placed by the same "
                    "fractions of width and height")
        matrix = pixel_map_on(fractions, shown)
        if registered_on == "oriented":  # file pixels -> oriented pixels, as [row, col, 1]
            matrix = matrix @ SWAP @ orientation_matrix(sizes.file_size, int(turn[0]),
                                                        bool(turn[1])) @ SWAP
        prepared.append((index, section_id, rule, sizes, matrix, problems))

    implied = [implied_pixel_size_um(matrix) for *_rest, matrix, _p in prepared]
    known = {section_id: workspace.calibration(section_id) for _i, section_id, *_r in prepared}
    stack_size: float | None = float(np.median(implied)) if implied else None
    stack_source = "imported"
    if known and all(value is not None for value, _source in known.values()):
        sources = {source for _value, source in known.values()}
        stack_source = sources.pop() if len(sources) == 1 else "file"
        values = {float(value) for value, _source in known.values()}  # type: ignore[arg-type]
        stack_size = values.pop() if len(values) == 1 else None

    placements: list[ImportedPlacement] = []
    for index, section_id, rule, sizes, matrix, problems in prepared:
        entry = held.entries[index]
        value, source_name = known[section_id]
        if value is None:
            assert stack_size is not None
            value, source_name = stack_size, "imported"
        try:
            placement = placement_from_pixel_map(
                matrix, atlas=atlas_object, plane=plane, sizes=sizes,
                file_um_per_px=float(value),
                orientation=(orientations or {}).get(section_id),
                at=(planes or {}).get(section_id))
        except ValueError as exc:
            refused[section_id] = str(exc)
            continue
        if placement.out_of_plane_um > 1e-3:
            problems.append(
                f"drawn {placement.out_of_plane_um:.2f} um from the registered plane (a flat "
                "plane is drawn at the nearest voxel)")
        markers = _scaled_markers(entry, sizes.file_size)
        if entry.markers and markers is None:
            problems.append("VisuAlign markers without the registered image size: kept raw")
        placements.append(ImportedPlacement(
            section_id=section_id, source_filename=entry.filename, match=rule, nr=entry.nr,
            placement=placement, pixel_size_um=float(value), pixel_size_source=source_name,
            pixel_to_atlas_um=matrix, markers=markers, markers_raw=list(entry.markers),
            problems=problems))
    return ImportResult(
        source=held, placements=placements,
        unmatched=[name for index, name in enumerate(names) if index not in matched],
        missing=[sid for sid in ids if sid not in {s for s, _r in matched.values()}],
        refused=refused, pixel_size_um=stack_size, pixel_size_source=stack_source,
    )


# --- as a job's supplied inputs ----------------------------------------------------------

#: The warning when the file carries VisuAlign markers: only the linear
#: registration is imported.
MARKERS_NOT_IMPORTED = (
    "The file's VisuAlign nonlinear markers ({count} section(s)) were not imported: "
    "only its linear registration is. LangSlice's nonlinear step replaces them.")


@dataclass(frozen=True)
class RegistrationInputs:
    """An imported registration as a job's supplied inputs.

    ``inputs`` holds the ``JobSpec.inputs`` keys it supplies: ``positions``,
    ``angles`` (the stack-wide form, the median, when every placed section's
    pitch and yaw agree to within
    :data:`~langslice.core.maps.imported.ANGLE_EPSILON_DEG`, a written file's
    rounding; else per section), ``orientation``, ``transforms`` and, when
    neither the caller nor the files give a pixel size, ``pixel_size_um``.
    ``report`` is what the caller is told (the :class:`ImportResult` in brief:
    the file, the sections placed and how each matched, what did not match, the
    refused sections with the reason, ``warnings``). ``result`` is the full
    :class:`ImportResult` (the VisuAlign markers kept raw and rescaled there,
    never imported).
    """

    inputs: dict[str, Any]
    report: dict[str, Any]
    result: ImportResult


def _plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def registration_inputs(
    source: str | os.PathLike[str] | RegistrationFile,
    workspace: Workspace,
) -> RegistrationInputs:
    """Import *source* (:func:`import_placements`) into the job of
    *workspace*'s settings, as the supplied inputs a job is made from.

    The pixel size: the caller's (``inputs.pixel_size_um``) or each file's
    own when they have one; otherwise the size the imported maps imply
    (``ImportResult.pixel_size_um``), which the inputs then carry (and, when
    only some files carry their own, every section is placed relative to
    that one size, so the placements hold exactly whatever each file says).
    :class:`AmbiguousMatch` when the entries cannot be matched one to one;
    ``ValueError`` when no section could be placed.
    """
    from langslice.core.workspace import Workspace as Plain

    held = source if isinstance(source, RegistrationFile) else read_registration(source)
    result = import_placements(held, workspace)
    host_size = bool(workspace.settings.supplied_pixel_size)
    pixel_size: float | None = None
    overridden: list[str] = []
    if not host_size and result.pixel_size_source == "imported" \
            and result.pixel_size_um is not None:
        pixel_size = float(result.pixel_size_um)
        overridden = [p.section_id for p in result.placements
                      if p.pixel_size_source != "imported"]
        if overridden:  # one pixel size for the job: place every section against it
            # The settings text stays the job's: these frames are keyed by
            # the pixel size, so none is shared with the job's own.
            fixed = replace(
                workspace.settings, supplied_pixel_size=pixel_size,
                stack_pixel_size_um=pixel_size, file_pixel_sizes_um={})
            again = Plain(settings=fixed, image_folder=workspace.image_folder,
                          emit=workspace.emit, atlas_loader=workspace.atlas_loader)
            result = import_placements(held, again)
    if not result.placements:
        reasons = "; ".join(f"{name}: {why}" for name, why in result.refused.items())
        raise ValueError(
            f"No section could be placed from {held.path} ({held.format}): "
            + (reasons or "no entry names a section image here "
               f"(entries: {result.unmatched[:5]})"))

    placements = result.placements
    pitches = [p.pitch_deg for p in placements]
    yaws = [p.yaw_deg for p in placements]
    if max(pitches) - min(pitches) <= ANGLE_EPSILON_DEG \
            and max(yaws) - min(yaws) <= ANGLE_EPSILON_DEG:
        # One plane for the stack (differences below a written file's rounding).
        angles: dict[str, Any] = {"pitch": float(np.median(pitches)),
                                  "yaw": float(np.median(yaws))}
    else:
        angles = {p.section_id: {"pitch": p.pitch_deg, "yaw": p.yaw_deg} for p in placements}
    inputs: dict[str, Any] = {
        "positions": {p.section_id: p.position_mm for p in placements},
        "angles": angles,
        "orientation": {p.section_id: {"flip": p.flip, "rotation_deg": p.rotation_deg}
                        for p in placements},
        "transforms": {p.section_id: p.transform() for p in placements},
    }
    if pixel_size is not None:
        inputs["pixel_size_um"] = pixel_size

    warnings: list[str] = list(held.notes)
    if result.unmatched:
        warnings.append(
            f"{_plural(len(result.unmatched), 'entry', 'entries')} in the file match no "
            f"section image and were ignored: {result.unmatched}")
    if result.missing:
        warnings.append(
            f"{_plural(len(result.missing), 'section has', 'sections have')} no entry in "
            f"the file and no placement: {result.missing}")
    for section_id, why in result.refused.items():
        warnings.append(f"{section_id}: not imported ({why})")
    for placement in placements:
        warnings.extend(f"{placement.section_id}: {problem}" for problem in placement.problems)
    marked = [entry for entry in held.entries if entry.markers]
    if marked:
        warnings.append(MARKERS_NOT_IMPORTED.format(count=len(marked)))
    if pixel_size is not None:
        warnings.append(
            f"No pixel size was given{' for every file' if overridden else ''}: the job "
            f"maps the sections at {pixel_size:.6g} um per pixel, the size the imported "
            "registration implies (the placements hold at it)."
            + (f" The files' own sizes of {overridden} are replaced by it." if overridden
               else ""))
    report = {
        "path": held.path, "format": held.format, "target": held.target or (
            DEEPSLICE_DEFAULT_TARGET if held.format == "deepslice-csv" else None),
        "aligner": held.aligner,
        "sections": [{
            "id": p.section_id, "entry": p.source_filename, "match": p.match,
            "position_mm": p.position_mm, "pitch_deg": p.pitch_deg, "yaw_deg": p.yaw_deg,
            "rotation_deg": p.rotation_deg, "flip": p.flip,
        } for p in placements],
        "unmatched": list(result.unmatched), "missing": list(result.missing),
        "refused": dict(result.refused),
        "pixel_size_um": pixel_size if pixel_size is not None else result.pixel_size_um,
        "pixel_size_source": "imported" if pixel_size is not None else result.pixel_size_source,
        "markers_imported": False,
        "warnings": warnings,
    }
    return RegistrationInputs(inputs=inputs, report=report, result=result)
