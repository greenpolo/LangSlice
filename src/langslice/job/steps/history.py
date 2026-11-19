"""Each section's history: ``sections/<name>/history.jsonl``.

One line per write that changed the section, appended under the job
folder's lock and never rewritten. A line is the section's row of
``state.json`` (its own ``cutting_angles_deg`` included) beside these
fields:

==================  ====================================================
``seq``             the line's number in the file, from 1
``step``            the job's write that appended it (0: the opening)
``by``              the tool whose write it was; ``script`` for a
                    script's own write; ``host`` for a state LangSlice
                    found rather than made (the opening state, or the
                    state before a write when no line held it yet)
``at``              the local time
``registration``    the number of the registration it holds, or null
``digest``          that registration's identity (null: it holds none)
``format_version``  the state format of the row
==================  ====================================================

**Registrations.** A section's registration is its orientation (flip and
quarter turn), its transform and its deformation while that still sits on
this linear placement (:func:`registration_value`). Each distinct one is
numbered per section, from 1, when a write changes the section's
registration to it (the one replaced is numbered first when it was not), or
when ``restore_registration`` lists the one the section holds; a line
holding a numbered one repeats its number. A ``host`` line's registration
is listed as made by what its content says made it (:func:`guessed_maker`).

Undo, redo and ``restore_registration`` put lines back on sections
(:func:`set_head`), each appending a line of its own; the undo steps that
name the lines are ``history/undo.json`` (``undo_steps.py``).

**Record folders.** After each write, a folder under a changed section's
``deformable/`` that neither the section nor a line of its history names,
directly or as a parent, is deleted: a fit saved and never applied.

A line cut short (a write that died mid-line) is skipped when read, and the
next line written starts on a line of its own.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, ValidationError

from langslice.core.fits.deformation import folder_name, linear_key
from langslice.core.fits.frame import digest as content_digest
from langslice.core.jsonable import JsonObject, JsonValue
from langslice.core.stack.affine import StoredAffine
from langslice.core.stack.deformation import DeformationRef
from langslice.core.stack.part import Shape, Whole
from langslice.core.stack.section import FLAT, Angles, Section
from langslice.core.stack.stack import Stack
from langslice.job.folder.layout import JobLayout
from langslice.job.state_file import STATE_FORMAT_VERSION, json_object

logger = logging.getLogger(__name__)

HOST = "host"
SCRIPT = "script"
REGISTRATION_KEYS = ("flip", "rotation_deg", "transform", "deformation")


def now() -> str:
    """Return the local time a line or an undo step is written with.

    Returns
    -------
    str
        ``YYYY-MM-DD HH:MM:SS``.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Row(Shape):
    """Hold one line of a section's history (see the module text).

    The section's row of ``state.json`` is the line's other keys.

    Parameters
    ----------
    seq, step, by, at, registration, digest
        The line's own fields.
    format_version
        The state format of the section's row.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    seq: Whole
    step: Whole
    by: str
    at: str | None = None
    registration: Whole | None = None
    digest: str | None = None
    format_version: Whole = STATE_FORMAT_VERSION

    @classmethod
    def read(cls, data: JsonObject) -> Row | None:
        """Return the line a JSON object holds; None when it holds none.

        Parameters
        ----------
        data
            One line of the file, parsed.

        Returns
        -------
        Row or None
            None for an object that is not a line.
        """
        try:
            return cls.from_json(data)
        except ValidationError:
            return None

    @property
    def section(self) -> dict[str, Any]:
        """Return the section's row of ``state.json`` the line holds."""
        return dict(self.model_extra or {})

    @property
    def flip(self) -> bool:
        """Return the orientation's mirror flag."""
        return bool(self.section.get("flip"))

    @property
    def rotation_deg(self) -> int:
        """Return the orientation's quarter turn, in degrees."""
        rotation = self.section.get("rotation_deg")
        return int(rotation) if isinstance(rotation, (int, float)) else 0

    @property
    def transform(self) -> StoredAffine | None:
        """Return the transform the line holds; None without one."""
        transform = self.section.get("transform")
        return StoredAffine.from_json(transform) if transform else None

    @property
    def deformation(self) -> DeformationRef | None:
        """Return the deformation the line holds; None without one."""
        deformation = self.section.get("deformation")
        return DeformationRef.from_json(deformation) if deformation else None

    @property
    def position_mm(self) -> float | None:
        """Return the atlas position the line holds; None when unplaced."""
        position = self.section.get("position_mm")
        return float(position) if isinstance(position, (int, float)) else None

    @property
    def angles(self) -> Angles:
        """Return the cutting angles the line holds; flat without them."""
        angles = self.section.get("cutting_angles_deg")
        if not isinstance(angles, Mapping):
            return FLAT
        return (angles.get("pitch", 0.0), angles.get("yaw", 0.0))


def registration_value(state: Stack, section: Section) -> dict[str, JsonValue] | None:
    """Return the section's registration in its JSON form.

    Parameters
    ----------
    state
        The stack the section is in (its atlas and plane).
    section
        The section read.

    Returns
    -------
    dict or None
        Its orientation, transform and deformation (null when the
        deformation was fitted on another linear placement); None while
        it has neither a transform nor a current deformation.
    """
    deformation = section.deformation
    if deformation is not None and deformation.linear_key != linear_key(state, section):
        deformation = None
    if section.transform is None and deformation is None:
        return None
    transform = None if section.transform is None else section.transform.to_json()
    applied = None if deformation is None else deformation.to_json()
    return {
        "flip": bool(section.flip),
        "rotation_deg": int(section.rotation_deg),
        "transform": transform,
        "deformation": applied,
    }


def registration_key(state: Stack, section: Section) -> str | None:
    """Return the identity of the registration a section holds.

    Parameters
    ----------
    state
        The stack the section is in.
    section
        The section read.

    Returns
    -------
    str or None
        Its :func:`registration_digest`; None while it holds none.
    """
    value = registration_value(state, section)
    if value is None:
        return None
    return registration_digest(value)


def registration_digest(value: JsonObject) -> str:
    """Return a registration's identity: its orientation, transform and record.

    Parameters
    ----------
    value
        A registration, as :func:`registration_value` gives it.

    Returns
    -------
    str
        Sixteen hex digits.
    """
    deformation = value.get("deformation")
    key = deformation.get("key") if isinstance(deformation, Mapping) else None
    return content_digest(
        {
            "flip": value.get("flip"),
            "rotation_deg": value.get("rotation_deg"),
            "transform": value.get("transform"),
            "deformation": key,
        }
    )[:16]


def guessed_maker(row: Row) -> str:
    """Return what a line's registration says made it.

    Parameters
    ----------
    row
        A line holding a registration.

    Returns
    -------
    str
        The last deformation step's method; else the transform's tool;
        ``supplied`` for a transform no tool of LangSlice made.
    """
    deformation = row.deformation
    if deformation is not None and deformation.steps:
        method = deformation.steps[-1].get("method")
        if method:
            return str(method)
        return "trace_borders" if deformation.steps[-1].get("trace") else "fireants_syn"
    kind = row.section.get("transform")
    kind = kind.get("kind") if isinstance(kind, Mapping) else None
    return {
        "elastix": "elastix_affine",
        "interactive": "interactive_transform",
    }.get(str(kind), "supplied")


def set_head(section: Section, fields: JsonObject) -> None:
    """Put some row fields of a line back on a section, in place.

    Parameters
    ----------
    section
        The live section written.
    fields
        Row keys and their values, as a line holds them.
    """
    restored = section.with_fields(fields)
    for name in Section.model_fields:
        setattr(section, name, getattr(restored, name))


def section_row(section: Section) -> JsonObject:
    """Return a section's row as a line holds it (JSON values only).

    Parameters
    ----------
    section
        The section read.

    Returns
    -------
    JsonObject
        Its row of ``state.json`` with its own cutting angles.
    """
    return section.to_json()


class SectionHistories:
    """Read and append the job folder's section histories (the module text).

    Parameters
    ----------
    layout
        The job folder.
    """

    def __init__(self, layout: JobLayout) -> None:
        self.layout = layout
        self._cache: dict[Path, tuple[tuple[int, int, int], tuple[Row, ...]]] = {}

    def rows(self, section_id: str) -> tuple[Row, ...]:
        """Return a section's lines, oldest first.

        Parameters
        ----------
        section_id
            The section's filename.

        Returns
        -------
        tuple of Row
            Empty for a section without a history; a line that does not
            read (a write cut short) is left out.
        """
        path = self.layout.section_history(section_id)
        try:
            info = path.stat()
        except OSError:
            return ()
        stamp = (info.st_ino, info.st_size, info.st_mtime_ns)
        cached = self._cache.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        text = path.read_text(encoding="utf-8")
        rows = tuple(row for row in map(Row.read, _parsed_lines(text)) if row is not None)
        self._cache[path] = (stamp, rows)
        return rows

    def row(self, section_id: str, seq: int) -> Row | None:
        """Return one line of a section's history.

        Parameters
        ----------
        section_id
            The section's filename.
        seq
            The line's number.

        Returns
        -------
        Row or None
            None without that line.
        """
        return next((row for row in self.rows(section_id) if row.seq == seq), None)

    def number(self, section_id: str, digest: str | None) -> int | None:
        """Return the number a section's history gives a registration.

        Parameters
        ----------
        section_id
            The section's filename.
        digest
            The registration's identity (:func:`registration_digest`).

        Returns
        -------
        int or None
            None while it is unnumbered, and for no registration.
        """
        for row in self.registrations(section_id):
            if digest is not None and row.digest == digest:
                return row.registration
        return None

    def registrations(self, section_id: str) -> list[Row]:
        """Return the first line of each numbered registration of a section.

        Parameters
        ----------
        section_id
            The section's filename.

        Returns
        -------
        list of Row
            By number.
        """
        first: dict[int, Row] = {}
        for row in self.rows(section_id):
            if row.registration is not None:
                first.setdefault(row.registration, row)
        return [first[number] for number in sorted(first)]

    def registration_number(self, state: Stack, section: Section) -> int | None:
        """Return the number of the registration a section holds.

        Parameters
        ----------
        state
            The stack the section is in.
        section
            The section as it stands.

        Returns
        -------
        int or None
            None while it holds none, or one never numbered.
        """
        return self.number(section.id, registration_key(state, section))

    def append(
        self,
        state: Stack,
        section: Section,
        *,
        step: int,
        by: str,
        numbered: bool,
    ) -> Row:
        """Append the section as it stands as a line; call it holding the lock.

        Parameters
        ----------
        state
            The stack the section is in.
        section
            The section written.
        step
            The job's write that appends it.
        by
            The tool whose write it is, or :data:`HOST` or :data:`SCRIPT`.
        numbered
            Number its registration when it has none yet.

        Returns
        -------
        Row
            The line written.
        """
        digest = registration_key(state, section)
        rows = self.rows(section.id)
        number = self.number(section.id, digest)
        if number is None and numbered and digest is not None:
            number = max((row.registration or 0 for row in rows), default=0) + 1
        seq = rows[-1].seq + 1 if rows else 1
        row = Row.from_json(
            {
                "seq": seq,
                "step": step,
                "by": by,
                "at": now(),
                "registration": number,
                "digest": digest,
                **section_row(section),
            }
        )
        path = self.layout.section_history(section.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torn = _ends_mid_line(path)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            if torn:
                stream.write("\n")
            stream.write(json.dumps(row.to_json()) + "\n")
        return row

    def collect_records(self, section: Section) -> list[str]:
        """Delete the section's record folders nothing names; their names.

        A folder of the section's ``deformable/`` stays while the section
        or a line of its history names it, directly or as the parent of
        one named. Nothing is deleted while the section has no
        ``history.jsonl``.

        Parameters
        ----------
        section
            The section as it stands.

        Returns
        -------
        list of str
            The folders deleted.
        """
        folder = self.layout.deformable_dir(section.id)
        if not folder.is_dir() or not self.layout.section_history(section.id).is_file():
            return []
        named = {_record_folder(row.section) for row in self.rows(section.id)}
        named.add(_record_folder(section_row(section)))
        kept = _with_parents(folder, {name for name in named if name})
        removed: list[str] = []
        for path in sorted(folder.iterdir()):
            if path.is_dir() and path.name not in kept:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path.name)
        if removed:
            logger.info(
                "Removed %d deformation record(s) of %s no write applied: %s",
                len(removed),
                section.id,
                ", ".join(removed),
            )
        return removed


def _ends_mid_line(path: Path) -> bool:
    """Return whether a file's last line has no end (a write cut short)."""
    try:
        with path.open("rb") as stream:
            if stream.seek(0, 2) == 0:
                return False
            stream.seek(-1, 2)
            return stream.read(1) != b"\n"
    except FileNotFoundError:
        return False


def _parsed_lines(text: str) -> list[JsonObject]:
    lines: list[JsonObject] = []
    for line in text.splitlines():
        try:
            data = json_object(line)
        except ValueError:
            continue
        if data is not None:
            lines.append(data)
    return lines


def _record_folder(row: JsonObject) -> str | None:
    """Return the record folder a row's deformation names; None for none."""
    deformation = row.get("deformation")
    if not isinstance(deformation, Mapping):
        return None
    record, key = deformation.get("record"), deformation.get("key")
    if isinstance(record, str) and record:
        return Path(record).name
    return folder_name(key) if isinstance(key, str) else None


def _with_parents(folder: Path, names: set[str]) -> set[str]:
    """Return *names* with every parent their ``record.json`` chains name."""
    kept: set[str] = set()
    waiting = sorted(names)
    while waiting:
        name = waiting.pop()
        if name in kept:
            continue
        kept.add(name)
        try:
            meta = json_object((folder / name / "record.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        parent = None if meta is None else meta.get("parent")
        if isinstance(parent, str):
            waiting.append(parent)
    return kept


__all__ = [
    "HOST",
    "REGISTRATION_KEYS",
    "SCRIPT",
    "Row",
    "SectionHistories",
    "guessed_maker",
    "now",
    "registration_digest",
    "registration_key",
    "registration_value",
    "section_row",
    "set_head",
]
