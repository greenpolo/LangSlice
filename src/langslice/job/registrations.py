"""Every registration a section has held, numbered, so one section can go back to one.

A section's registration is what the in-plane tools write: its orientation
(flip and quarter turn), its transform and its deformation (the reference to
its record under ``sections/<stem>/deformable/``; a deformation whose linear
placement has changed is no part of it). Undo is one history for the whole job,
so a section's earlier fit is lost to it once other sections' writes come
between; this store keeps each registration a section has held, whatever comes
after, and ``restore_registration`` (:mod:`langslice.tools.registrations`) puts
one back.

``sections/<stem>/registrations.jsonl``: one line per registration, appended
once under the job folder's lock and never rewritten, numbered from 1 per
section: ``id``, ``by`` (the tool whose write left it, ``script`` for a
script's own write, or, for one held before it was listed, what its content
says made it), ``at`` (local time; null for one held before it was listed),
``digest`` (of its content: two writes that leave the same registration share
its number), the position and cutting angles it was made at, and the
registration itself. Every undo step lists what it leaves
(:meth:`langslice.job.job.Job.list_registrations`, from ``commit``, ``undo``
and ``redo``). Like the deformation records it
names, the store is read back, never derived from the state, and undo leaves
it as it is.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from langslice.core.fits.deformation import linear_key
from langslice.core.fits.frame import digest as content_digest
from langslice.core.jsonable import JsonObject
from langslice.core.stack.affine import StoredAffine
from langslice.core.stack.codec import (
    affine_from_json,
    affine_to_json,
    deformation_from_json,
    deformation_to_json,
)
from langslice.core.stack.deformation import DeformationRef
from langslice.core.stack.json_reads import as_object
from langslice.core.stack.section import Angles, Section
from langslice.core.stack.stack import Stack
from langslice.job.layout import JobLayout

FILE = "registrations.jsonl"
FORMAT = 1
#: Who made a registration a write left when no tool was named (a script's
#: ``job.context``).
SCRIPT = "script"


def value(state: Stack, section: Section) -> dict[str, Any] | None:
    """The section's registration (a copy, in its JSON form): orientation,
    transform and the deformation, when it still sits on this placement;
    None while it has neither a transform nor a deformation."""
    deformation = section.deformation
    placement = linear_key(state, section)
    if deformation is not None and deformation.linear_key != placement:
        deformation = None
    if section.transform is None and deformation is None:
        return None
    transform = None
    if section.transform is not None:
        transform = affine_to_json(section.transform)
    applied = None if deformation is None else deformation_to_json(deformation)
    return {"flip": bool(section.flip), "rotation_deg": int(section.rotation_deg),
            "transform": transform, "deformation": applied}


def entry_transform(entry: JsonObject) -> StoredAffine | None:
    """Return the transform a listed registration holds.

    Parameters
    ----------
    entry
        One registration as its listing holds it.

    Returns
    -------
    StoredAffine or None
        The transform; None for a registration without one.
    """
    transform = entry.get("transform")
    if not transform:
        return None
    return affine_from_json(as_object("transform", transform))


def entry_deformation(entry: JsonObject) -> DeformationRef | None:
    """Return the deformation a listed registration holds.

    Parameters
    ----------
    entry
        One registration as its listing holds it.

    Returns
    -------
    DeformationRef or None
        The deformation reference; None for a registration without one.
    """
    deformation = entry.get("deformation")
    if not deformation:
        return None
    return deformation_from_json(as_object("deformation", deformation))


def digest(held: dict[str, Any]) -> str:
    """The registration's identity: its orientation, transform and the record
    its deformation names."""
    deformation = held.get("deformation") or {}
    return content_digest({"flip": held.get("flip"), "rotation_deg": held.get("rotation_deg"),
                           "transform": held.get("transform"),
                           "deformation": deformation.get("key")})[:16]


def made_by(held: dict[str, Any]) -> str:
    """What a registration held before it was listed says made it: the last
    deformation step's method, else the transform's kind."""
    steps = (held.get("deformation") or {}).get("steps") or []
    if steps:
        last = steps[-1]
        if last.get("method"):
            return str(last["method"])
        return "trace_borders" if last.get("trace") else "fireants_syn"
    kind = (held.get("transform") or {}).get("kind")
    return {"elastix": "elastix_affine", "interactive": "interactive_transform"}.get(
        str(kind), "supplied")


def now() -> str:
    """The local time a registration is listed with."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Registrations:
    """The job folder's registration store (see the module text)."""

    def __init__(self, layout: JobLayout) -> None:
        self.layout = layout
        self._cache: dict[str, tuple[Any, list[dict[str, Any]]]] = {}

    def path(self, section_id: str) -> Path:
        return self.layout.section_dir(section_id) / FILE

    def entries(self, section_id: str) -> list[dict[str, Any]]:
        """The section's registrations, oldest first (a line that does not
        read, a write cut short, is skipped)."""
        path = self.path(section_id)
        try:
            info = os.stat(path)
            stamp = (info.st_ino, info.st_size, info.st_mtime_ns)
            held = self._cache.get(section_id)
            if held is not None and held[0] == stamp:
                return held[1]
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        found: list[dict[str, Any]] = []
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("id"), int):
                found.append(entry)
        self._cache[section_id] = (stamp, found)
        return found

    def get(self, section_id: str, number: int) -> dict[str, Any] | None:
        """The section's registration *number*, or None."""
        return next((entry for entry in self.entries(section_id) if entry["id"] == number),
                    None)

    def find(self, section_id: str, key: str) -> dict[str, Any] | None:
        """The section's registration whose content digest is *key*, or None."""
        return next((entry for entry in self.entries(section_id)
                     if entry.get("digest") == key), None)

    def number_of(self, state: Stack, record: Section) -> int | None:
        """The number of the registration *record* holds now, or None (it holds
        none, or one not listed yet)."""
        held = value(state, record)
        if held is None:
            return None
        entry = self.find(record.id, digest(held))
        return int(entry["id"]) if entry is not None else None

    def add(self, section_id: str, held: dict[str, Any], *, position_mm: float | None,
            angles: Angles, by: str, at: str | None) -> dict[str, Any]:
        """List *held*, a registration of the section, made with it at
        *position_mm* and *angles*: its entry, new or the one it already has.
        Call it holding the job folder's lock."""
        key = digest(held)
        known = self.find(section_id, key)
        if known is not None:
            return known
        entries = self.entries(section_id)
        entry = {"format": FORMAT, "id": max((e["id"] for e in entries), default=0) + 1,
                 "by": by, "at": at, "digest": key, "position_mm": position_mm,
                 "cutting_angles_deg": {"pitch": angles[0], "yaw": angles[1]}, **held}
        path = self.path(section_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, default=str) + "\n")
        self._cache.pop(section_id, None)
        return entry


__all__ = ["FILE", "SCRIPT", "Registrations", "digest", "entry_deformation", "entry_transform",
           "made_by", "now", "value"]
