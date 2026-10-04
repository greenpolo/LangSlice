"""Linear registrations made elsewhere, read as LangSlice placements.

A user who already registered a stack linearly (QuickNII, VisuAlign,
DeepSlice, or an earlier LangSlice job) brings that registration and runs
only the nonlinear step on top. This module reads the file, matches its
entries to the job's section files and returns each section's placement in
LangSlice's own terms (position, cutting angles, flip, quarter turn, the six
normalized numbers, the pixel size they are relative to), through the exact
inverse of the geometry the job draws with
(:func:`langslice.core.import_geometry.placement_from_pixel_map`). It reads
and returns plain data only: nothing is written to the job or its state.

Formats (``read_registration`` tells them apart by extension and content):

- ``quicknii-json`` / ``visualign-json``: the QUINT JSON QuickNII, VisuAlign
  and DeepSlice write (``target``, ``aligner``, ``slices``: ``filename``,
  ``anchoring`` ``[ox, oy, oz, ux, uy, uz, vx, vy, vz]``, ``width``,
  ``height``, ``nr``, ``markers``); VisuAlign when any slice carries markers.
  The same file :func:`langslice.job.quint.job_export` writes.
- ``quicknii-xml``: QuickNII's XML (``<series><slice filename= nr= width=
  height= anchoring="ox=...&oy=...&..."/>``), also what DeepSlice's
  ``write_QuickNII_XML`` and its older releases wrote (width and height
  ``-999``: not given; ``nr`` as ``4.0``; bare ``&`` in the attribute,
  escaped before parsing as DeepSlice's own reader does).
- ``deepslice-csv``: DeepSlice's ``save_predictions`` CSV (``Filenames``,
  ``ox`` ... ``vz``, and ``width``, ``height``, ``nr`` when present; an
  unnamed pandas index column is ignored). No target: DeepSlice's mouse
  target (``ABA_Mouse_CCFv3_2017_25um.cutlas``) unless the caller says.
- ``langslice-registration``: a job folder's ``registration.json``
  (:mod:`langslice.job.formats`): each placed section's
  ``pixel_to_atlas_um`` on its image size.

Matching the file's entries to the job's section files (:func:`match_sections`),
first rule that finds anything wins, per entry:

1. ``exact`` — the entry's file name (its last path component; ``/`` and
   ``\\`` both separate) equals a section file's name;
2. ``stem`` — the names without their extensions are equal, ignoring case
   (a registration made on PNG copies of TIFF scans);
3. ``section number`` — both names carry exactly one QuickNII/DeepSlice
   section number ``_sNNN`` (``_s`` then digits) and the numbers are equal.

An entry that matches two sections under its rule, or two entries that
match one section, is refused (:class:`AmbiguousMatch`, naming them);
entries matching nothing are listed under ``unmatched``, sections no entry
names under ``missing``.

Geometry. A QuickNII anchoring places the image by fractions of its width
and height from its top-left corner, so it applies to the job's file at any
resolution (a registration made on a downsampled copy carries over); an
image of another aspect ratio is placed by the same fractions and said
(``problems``). QuickNII coordinates are voxel edges on the target's grid:
the Allen CCFv3 target is the 25 um volume whatever the job's Allen
resolution (:func:`langslice.job.quint.from_target_grid`), and the axes are
mapped as the exporter maps them (:func:`langslice.job.quint.quicknii_points_to_atlas_um`).
A registration whose target is not the job atlas's is refused.
``registration.json`` of the same atlas is used in micrometres directly; of
another atlas with the same QuickNII target (another Allen resolution), it
goes through that target.

The pixel size. The six numbers are relative to the pixel size the job will
map each file at: the job's own (``inputs.pixel_size_um`` or the file's
tags, :meth:`langslice.core.workspace.Workspace.calibration`) when it has one;
otherwise the median, over the matched sections, of the size each imported
map implies (:func:`langslice.core.import_geometry.implied_pixel_size_um`),
``pixel_size_source`` ``"imported"``. Such a job must then be given that
pixel size (``ImportResult.pixel_size_um``) for the placements to hold: the
job otherwise estimates one from the tissue.

VisuAlign markers (``[x, y, nx, ny]`` in the pixels of the slice's
``width`` x ``height``: the anchored atlas point ``(x, y)`` moved to the
section point ``(nx, ny)``) are returned raw and rescaled to the job file's
continuous pixel coordinates (top-left corner at 0, :func:`langslice.core.maps.residual_markers`'s
convention). They are not turned into a deformation record: that needs
VisuAlign's own interpolation between markers, which LangSlice does not have.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from langslice.core.import_geometry import (
    RecoveredPlacement,
    implied_pixel_size_um,
    placement_from_pixel_map,
)
from langslice.core.space import Plane
from langslice.job.quint import (
    anchoring_from_pixel_map,
    from_target_grid,
    quicknii_points_to_atlas_um,
    quicknii_target,
    to_target_grid,
)

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace

ANCHORING_KEYS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
#: DeepSlice's CSV has no target; its mouse model's (DeepSlice ``config.json``).
DEEPSLICE_DEFAULT_TARGET = "ABA_Mouse_CCFv3_2017_25um.cutlas"
#: QuickNII / DeepSlice section numbers: ``_s`` then digits (DeepSlice's
#: ``number_sections``, non-legacy).
SECTION_NUMBER = re.compile(r"_s(\d+)")
#: Relative aspect-ratio difference between the registered image and the
#: job's file above which the placement is said to be by fractions only.
ASPECT_TOLERANCE = 0.01
#: The transform ``kind`` an imported placement carries.
IMPORTED_KIND = "imported"


class AmbiguousMatch(ValueError):
    """The file's entries cannot be matched to the job's sections one to one."""


@dataclass(frozen=True)
class RegistrationEntry:
    """One section of a registration file as written there.

    ``anchoring`` (QuickNII formats): the nine numbers in the target's voxel
    space. ``pixel_to_atlas_um`` (``registration.json``): the file pixel ->
    atlas micrometre matrix, for an image of ``size``. ``size`` is the
    registered image's ``(width, height)``, None when the file does not give
    it. ``markers`` are VisuAlign's, as written.
    """

    filename: str
    nr: int | None = None
    size: tuple[int, int] | None = None
    anchoring: tuple[float, ...] | None = None
    pixel_to_atlas_um: np.ndarray | None = None
    markers: list[list[float]] = field(default_factory=list)
    problem: str | None = None


@dataclass(frozen=True)
class RegistrationFile:
    """A registration file as read (:func:`read_registration`)."""

    path: str
    format: str
    target: str | None
    aligner: str | None
    atlas: dict[str, Any] | None
    entries: list[RegistrationEntry]
    notes: list[str] = field(default_factory=list)


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
        """The stored-transform dictionary this placement would be supplied
        as (``kind`` :data:`IMPORTED_KIND`, the six numbers, their reading,
        the calibration they are relative to)."""
        parts = dict(self.placement.in_plane)
        mirrored = bool(parts.pop("mirrored", False))
        parts.pop("translate_x_frac", None)
        parts.pop("translate_y_frac", None)
        return {
            "kind": IMPORTED_KIND,
            "params": list(self.placement.params),
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

    def by_section(self) -> dict[str, ImportedPlacement]:
        return {placement.section_id: placement for placement in self.placements}

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


# --- reading the files -----------------------------------------------------------------


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _size(width: Any, height: Any) -> tuple[int, int] | None:
    w, h = _number(width), _number(height)
    if w is None or h is None or w <= 0 or h <= 0:
        return None
    return int(round(w)), int(round(h))


def _nr(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(round(number))


def _order_nr(order: Any) -> int | None:
    # registration.json's order counts from 0; QuickNII's nr (the exporter's) from 1.
    found = _nr(order)
    return None if found is None else found + 1


def _anchoring(values: Sequence[Any] | None) -> tuple[tuple[float, ...] | None, str | None]:
    if values is None:
        return None, "no anchoring"
    numbers = [_number(value) for value in values]
    if len(numbers) != 9 or any(number is None for number in numbers):
        return None, "the anchoring is not nine finite numbers"
    return tuple(cast(float, number) for number in numbers), None


def _markers(raw: Any) -> list[list[float]]:
    if not isinstance(raw, list) or not raw:
        return []
    # DeepSlice's QUINT writer stores only a slice's first marker, unwrapped.
    rows = [raw] if all(not isinstance(value, list) for value in raw) else raw
    markers = []
    for row in rows:
        if isinstance(row, list) and len(row) >= 4:
            values = [_number(value) for value in row[:4]]
            if all(value is not None for value in values):
                markers.append([cast(float, value) for value in values])
    return markers


def _quint_json(path: str, data: dict[str, Any]) -> RegistrationFile:
    entries = []
    for row in data.get("slices") or []:
        if not isinstance(row, dict):
            continue
        anchoring, problem = _anchoring(row.get("anchoring"))
        entries.append(RegistrationEntry(
            filename=str(row.get("filename", "")), nr=_nr(row.get("nr")),
            size=_size(row.get("width"), row.get("height")), anchoring=anchoring,
            markers=_markers(row.get("markers")), problem=problem))
    kind = "visualign-json" if any(entry.markers for entry in entries) else "quicknii-json"
    return RegistrationFile(path=path, format=kind, target=_text(data.get("target")),
                            aligner=_text(data.get("aligner")), atlas=None, entries=entries)


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _registration_json(path: str, data: dict[str, Any]) -> RegistrationFile:
    entries = []
    for section in data.get("sections") or []:
        image = section.get("image") or {}
        matrix = section.get("pixel_to_atlas_um")
        size = image.get("size")
        problem = section.get("problem") if matrix is None else None
        entries.append(RegistrationEntry(
            filename=str(section.get("id", "")), nr=_order_nr(section.get("order")),
            size=None if not size else (int(size[0]), int(size[1])),
            pixel_to_atlas_um=None if matrix is None else np.asarray(matrix, dtype=np.float64),
            problem=None if matrix is not None else str(problem or "no placement")))
    atlas = data.get("atlas") if isinstance(data.get("atlas"), dict) else None
    return RegistrationFile(path=path, format="langslice-registration", target=None,
                            aligner=_text(data.get("written_by")), atlas=atlas,
                            entries=entries)


def _anchoring_from_text(text: str) -> tuple[tuple[float, ...] | None, str | None]:
    values = dict(part.split("=", 1) for part in text.split("&") if "=" in part)
    return _anchoring([values.get(key) for key in ANCHORING_KEYS]
                      if set(ANCHORING_KEYS) <= set(values) else None)


def _quicknii_xml(path: str, raw: str) -> RegistrationFile:
    import xml.etree.ElementTree as ElementTree

    # As DeepSlice's own reader: escape each bare & (its writer leaves them).
    cleaned = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", raw)
    root = ElementTree.fromstring(cleaned)
    entries = []
    for element in root.iter("slice"):
        anchoring, problem = _anchoring_from_text(element.get("anchoring", ""))
        entries.append(RegistrationEntry(
            filename=element.get("filename", ""), nr=_nr(element.get("nr")),
            size=_size(element.get("width"), element.get("height")), anchoring=anchoring,
            problem=problem))
    return RegistrationFile(path=path, format="quicknii-xml",
                            target=root.get("target") or _declared(raw, "target"),
                            aligner=root.get("aligner") or _declared(raw, "aligner"),
                            atlas=None, entries=entries)


def _declared(raw: str, name: str) -> str | None:
    # DeepSlice 1.2.8's write_QuickNII_XML passes the series attributes as
    # pandas ``namespaces``, so they are written as ``xmlns:aligner="..."``.
    found = re.search(rf'xmlns:{name}="([^"]*)"', raw)
    return found.group(1) if found else None


def _deepslice_csv(path: str, raw: str) -> RegistrationFile:
    reader = csv.DictReader(io.StringIO(raw))
    columns = set(reader.fieldnames or [])
    name_key = next((key for key in ("Filenames", "filename", "Filename") if key in columns),
                    None)
    if name_key is None or not set(ANCHORING_KEYS) <= columns:
        raise ValueError(f"{path}: not a DeepSlice CSV (needs Filenames and "
                         f"{', '.join(ANCHORING_KEYS)})")
    entries = []
    for row in reader:
        anchoring, problem = _anchoring([row.get(key) for key in ANCHORING_KEYS])
        entries.append(RegistrationEntry(
            filename=str(row.get(name_key) or ""), nr=_nr(row.get("nr")),
            size=_size(row.get("width"), row.get("height")), anchoring=anchoring,
            problem=problem))
    return RegistrationFile(
        path=path, format="deepslice-csv", target=None, aligner=None, atlas=None,
        entries=entries,
        notes=["A DeepSlice CSV names no target; read as DeepSlice's mouse target "
               f"{DEEPSLICE_DEFAULT_TARGET} unless another is given"])


def read_registration(path: str | os.PathLike[str]) -> RegistrationFile:
    """Read a registration file (see the module text for the formats).
    ``ValueError`` for a file that is none of them."""
    text_path = str(path)
    raw = Path(text_path).read_text(encoding="utf-8-sig")
    suffix = Path(text_path).suffix.lower()
    if suffix == ".xml" or raw.lstrip().startswith("<"):
        return _quicknii_xml(text_path, raw)
    if suffix == ".csv":
        return _deepslice_csv(text_path, raw)
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{text_path}: not JSON, XML or CSV") from exc
    if isinstance(data, dict) and isinstance(data.get("sections"), list) \
            and "format_version" in data:
        return _registration_json(text_path, data)
    if isinstance(data, dict) and isinstance(data.get("slices"), list):
        return _quint_json(text_path, data)
    raise ValueError(f"{text_path}: neither a QUINT (QuickNII/VisuAlign/DeepSlice) JSON "
                     "nor a LangSlice registration.json")


# --- matching entries to section files -----------------------------------------------


def base_name(filename: str) -> str:
    """The last path component of *filename* (``/`` and ``\\`` both separate)."""
    return re.split(r"[\\/]", filename)[-1]


def section_number(filename: str) -> int | None:
    """The one ``_sNNN`` section number in *filename*'s base name, else None."""
    found = SECTION_NUMBER.findall(os.path.splitext(base_name(filename))[0])
    return int(found[0]) if len(found) == 1 else None


def match_sections(names: Sequence[str], section_ids: Iterable[str],
                   ) -> dict[int, tuple[str, str]]:
    """Entry index -> ``(section id, rule)`` (see the module text for the
    rules). :class:`AmbiguousMatch` when an entry matches two sections under
    its rule, or two entries match one section."""
    ids = list(section_ids)

    def stem(name: str) -> str:
        return os.path.splitext(base_name(name))[0].casefold()

    rules = (
        ("exact", lambda name, sid: base_name(name) == sid),
        ("stem", lambda name, sid: stem(name) == stem(sid)),
        ("section number", lambda name, sid: section_number(name) is not None
         and section_number(name) == section_number(sid)),
    )
    matched: dict[int, tuple[str, str]] = {}
    for index, name in enumerate(names):
        for rule, test in rules:
            hits = [sid for sid in ids if test(name, sid)]
            if len(hits) > 1:
                raise AmbiguousMatch(f"{name!r} matches several sections by {rule}: {hits}")
            if hits:
                matched[index] = (hits[0], rule)
                break
    taken: dict[str, list[str]] = {}
    for index, (sid, _rule) in matched.items():
        taken.setdefault(sid, []).append(names[index])
    doubles = {sid: found for sid, found in taken.items() if len(found) > 1}
    if doubles:
        raise AmbiguousMatch("several entries match one section: " + "; ".join(
            f"{sid}: {found}" for sid, found in sorted(doubles.items())))
    return matched


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
    target: str | None = None,
) -> ImportResult:
    """Every matched section's placement in the job's terms (see the module
    text). *workspace* gives the job's atlas, plane, section files and
    calibration; *section_ids* the sections (default: the folder's images,
    named as the job names them). *orientations* fixes a section's
    ``(rotation_deg, flip)`` (default: chosen per section). *target*
    overrides the file's QuickNII target (a DeepSlice CSV has none).
    A section whose entry cannot be placed is listed under ``refused`` with
    the reason; an ambiguous match raises :class:`AmbiguousMatch`."""
    from langslice.core.discovery import discover_slices
    from langslice.core.layers import atlas_facts
    from langslice.core.maps import render_sizes

    held = source if isinstance(source, RegistrationFile) else read_registration(source)
    if section_ids is None:
        section_ids = [os.path.basename(path)
                       for path in discover_slices(workspace.image_folder)]
    ids = list(section_ids)
    names = [entry.filename for entry in held.entries]
    matched = match_sections(names, ids)
    atlas_object = workspace.atlas
    atlas = atlas_facts(atlas_object)
    plane = cast(Plane, workspace.spec.plane)
    file_target = target or held.target or (
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
        if entry.size is not None:
            registered = entry.size[0] / float(entry.size[1])
            here = sizes.file_size[0] / float(sizes.file_size[1])
            if abs(registered / here - 1.0) > ASPECT_TOLERANCE:
                problems.append(
                    f"registered on a {entry.size[0]} x {entry.size[1]} image, the file is "
                    f"{sizes.file_size[0]} x {sizes.file_size[1]}: placed by the same "
                    "fractions of width and height")
        prepared.append((index, section_id, rule, sizes,
                         pixel_map_on(fractions, sizes.file_size), problems))

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
                orientation=(orientations or {}).get(section_id))
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

