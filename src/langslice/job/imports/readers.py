"""Linear registrations made elsewhere, read as LangSlice placements.

A user who already registered a stack linearly (QuickNII, VisuAlign, DeepSlice,
or an earlier LangSlice job) brings that registration and runs only the
nonlinear step on top. This module reads the file, matches its entries to the
job's section files and returns each section's placement in LangSlice's own
terms (position, cutting angles, flip, quarter turn, the six normalized
numbers, the pixel size they are relative to), through the exact inverse of the
geometry the job draws with
(:func:`langslice.core.maps.imported.placement_from_pixel_map`). It reads and
returns plain data only: nothing is written to the job or its state.

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
  target (``ABA_Mouse_CCFv3_2017_25um.cutlas``).
- ``langslice-registration``: a job folder's ``registration.json``
  (:mod:`langslice.job.public_files`): each placed section's
  ``pixel_to_atlas_um`` on its image size.

Matching the file's entries to the job's section files (:func:`match_sections`),
first rule that finds anything wins, per entry:

1. ``exact`` — the entry's file name (its last path component; ``/`` and
   ``\\`` both separate) equals a section file's name;
2. ``stem`` — the names without their extensions are equal, ignoring case
   (a registration made on PNG copies of TIFF scans);
3. ``export copy`` — the entry's stem, without a final ``_sNNN``, equals the
   section file's stem, ignoring case (the PNG copies a LangSlice export
   names, :mod:`langslice.job.quint_images`);
4. ``section number`` — both names carry exactly one QuickNII/DeepSlice
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

The pixel size. The six numbers are relative to the pixel size the job will map
each file at: the job's own (``inputs.pixel_size_um`` or the file's tags,
:meth:`langslice.core.workspace.Workspace.calibration`) when it has one;
otherwise the median, over the matched sections, of the size each imported map
implies (:func:`langslice.core.maps.imported.implied_pixel_size_um`),
``pixel_size_source`` ``"imported"``. Such a job must then be given that pixel
size (``ImportResult.pixel_size_um``) for the placements to hold: the job is
otherwise uncalibrated.

VisuAlign markers (``[x, y, nx, ny]`` in the pixels of the slice's ``width`` x
``height``: the anchored atlas point ``(x, y)`` moved to the section point
``(nx, ny)``) are returned raw and rescaled to the job file's continuous pixel
coordinates (top-left corner at 0,
:func:`langslice.core.maps.section_maps.residual_markers`'s convention). They
are not turned into a deformation record: that needs VisuAlign's own
interpolation between markers, which LangSlice does not have.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

ANCHORING_KEYS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")
#: DeepSlice's CSV has no target; its mouse model's (DeepSlice ``config.json``).
DEEPSLICE_DEFAULT_TARGET = "ABA_Mouse_CCFv3_2017_25um.cutlas"


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
               f"{DEEPSLICE_DEFAULT_TARGET}"])


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
