"""``restore_registration``: sections back to registrations they held before.

Every write's registrations are listed by the job (:mod:`langslice.job.registrations`:
a section's orientation, transform and deformation, numbered per section).
:func:`listing` answers with a section's list (listing the registration it
holds now when it was never listed, which writes no state), each row naming
its transform, the method and settings of its deformation and the pictures
the write that left it saved; :func:`restore` puts one registration of each
section named back, as ONE undo step, the other sections untouched. A
section is refused, as its own row, when it would change a property the job
froze (``FROZEN``), names no registration of the section
(``UNKNOWN_REGISTRATION``), or holds a deformation fitted with the section at
another position or cutting angles (``REGISTRATION_ELSEWHERE``: its linear
placement would not be the one it was fitted on), or one whose record is gone
(``DEFORMATION_UNREADABLE``). :func:`entry_for` is the registration
``look`` draws a section under without writing it.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langslice.core.deformation import linear_key
from langslice.core.state import SliceState, StackState
from langslice.core.status import deformation_entry, transform_entry
from langslice.job import registrations as regs
from langslice.ops.refusal import Refused, check_frozen, unknown_sections

if TYPE_CHECKING:
    from langslice.job.job import Job

UNKNOWN_REGISTRATION = "UNKNOWN_REGISTRATION"
REGISTRATION_ELSEWHERE = "REGISTRATION_ELSEWHERE"
DEFORMATION_UNREADABLE = "DEFORMATION_UNREADABLE"


@dataclass(frozen=True)
class Listed:
    """A section's registrations, oldest first (:func:`row` each), and the
    number of the one it holds now (None when it holds none)."""

    id: str
    held: int | None
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class Restored:
    """What :func:`restore` did for one section: the registration it holds
    now, the one that replaced (None: none, or never listed) and whether
    anything was written (False: it held that one already)."""

    id: str
    registration: int
    replaced: int | None
    written: bool


@dataclass(frozen=True)
class RestoredMany:
    """What :func:`restore` did: one row per section named, in call order
    (``{"id", "status": "ok", "registration", "replaced"?, "written"}``, or
    ``{"id", "status": "error", "error", "message", ...}``), and the
    sections written (one undo step)."""

    rows: list[dict[str, Any]]
    written: list[str]


def _where(position_mm: float | None, angles: dict[str, Any]) -> str:
    place = "no position" if position_mm is None else f"{round(float(position_mm), 3)} mm"
    pitch, yaw = float(angles.get("pitch", 0.0)), float(angles.get("yaw", 0.0))
    return place + (f", pitch {pitch:g} deg, yaw {yaw:g} deg" if pitch or yaw else "")


def _elsewhere(state: StackState, record: SliceState, entry: dict[str, Any]) -> str | None:
    """Why *entry* cannot be put back on *record* as it stands now (its
    deformation was fitted at another position or cutting angles), else None."""
    deformation = entry.get("deformation")
    if not deformation:
        return None
    trial = dataclasses.replace(record, flip=bool(entry.get("flip")),
                                rotation_deg=int(entry.get("rotation_deg") or 0),
                                transform=entry.get("transform"))
    if deformation.get("linear_key") == linear_key(state, trial):
        return None
    return (f"its deformation was fitted with the section at "
            f"{_where(entry.get('position_mm'), entry.get('cutting_angles_deg') or {})}; "
            f"the section is at {_where(record.position_mm, record.cutting_angles_deg)} now")


def row(state: StackState, record: SliceState, entry: dict[str, Any],
        held: int | None, pictures: list[int] | None = None) -> dict[str, Any]:
    """One registration as the listing shows it: its number, the tool that
    made it and when, its transform (kind and knobs,
    :func:`langslice.core.status.transform_entry`), its deformation (method,
    steps and settings, :func:`langslice.core.status.deformation_entry`), the
    *pictures* the write that left it saved, whether the section holds it now,
    and why it cannot be put back as the section stands."""
    transform = transform_entry(entry.get("transform"))
    deformation = deformation_entry(entry.get("deformation"))
    out: dict[str, Any] = {"registration": entry["id"], "by": entry.get("by"),
                           **({"at": entry["at"]} if entry.get("at") else {}),
                           **({"transform": transform} if transform else {}),
                           **({"deformation": deformation} if deformation else {}),
                           **({"pictures": list(pictures)} if pictures else {})}
    if entry["id"] == held:
        out["held"] = True
    elsewhere = _elsewhere(state, record, entry)
    if elsewhere:
        out["not_here"] = elsewhere
    return out


def _resolve(job: Job, section: str) -> SliceState:
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [section])
    return record


def _list_held(job: Job, record: SliceState) -> int | None:
    """The number of the registration *record* holds, listed now (by what its
    content says made it) when it never was; None when it holds none."""
    now = regs.value(job.state, record)
    if now is None:
        return None
    store = job.registrations
    if job.persist and store.find(record.id, regs.digest(now)) is None:
        store.add(record.id, now, position_mm=record.position_mm,
                  angles=dict(record.cutting_angles_deg), by=regs.made_by(now), at=None)
    return store.number_of(job.state, record)


def saved_pictures(job: Job, section: str) -> dict[int, list[int]]:
    """The pictures the writes of *section*'s registrations saved: per
    registration number, the numbers of the pictures drawn by the tool that
    made it showing the section under it (``views.jsonl``'s
    ``registrations``), oldest first."""
    store = job.registrations
    made = {int(entry["id"]): str(entry.get("by") or "") for entry in store.entries(section)}
    found: dict[int, list[int]] = {}
    for picture in job.views.records():
        number = picture.registrations.get(section)
        if number is not None and made.get(number) == picture.tool:
            found.setdefault(int(number), []).append(picture.seq)
    return found


def listing(job: Job, section: str) -> Listed:
    """The section's registrations; the one it holds now listed first when it
    never was. Writes no state."""
    with job.writing():
        record = _resolve(job, section)
        state, store = job.state, job.registrations
        held = _list_held(job, record)
        pictures = saved_pictures(job, record.id)
        rows = [row(state, record, entry, held, pictures.get(int(entry["id"])))
                for entry in store.entries(record.id)]
        return Listed(id=record.id, held=held, rows=rows)


def _unknown(job: Job, record: SliceState, number: Any) -> Refused:
    numbers = [item["id"] for item in job.registrations.entries(record.id)]
    listed = (f"its registrations are {', '.join(str(n) for n in numbers)}" if numbers
              else "it has none listed yet")
    return Refused(UNKNOWN_REGISTRATION, id=record.id, registration=number,
                   registrations=numbers,
                   message=f"{record.id} has no registration {number}; {listed}. "
                           "Call restore_registration with the section alone to list them.")


def entry_for(job: Job, section: Any, number: Any) -> tuple[SliceState, dict[str, Any],
                                                             int | None]:
    """``(record, entry, held)``: registration *number* of *section* as it
    can be drawn on the section as it stands (what ``look`` draws a section
    under without writing it), and the number the section holds. Refused:
    ``UNKNOWN_SLICE_IDS``, ``UNKNOWN_REGISTRATION``,
    ``REGISTRATION_ELSEWHERE``, ``DEFORMATION_UNREADABLE``, ``BAD_ARGS``."""
    record = _resolve(job, str(section))
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise Refused("BAD_ARGS", message=f"{record.id}: a registration is a number from 1; "
                      f"got {number!r}.")
    held = job.registrations.number_of(job.state, record)
    entry = job.registrations.get(record.id, number)
    if entry is None:
        raise _unknown(job, record, number)
    _check_entry(job, record, entry)
    return record, entry, held


def _check_entry(job: Job, record: SliceState, entry: dict[str, Any]) -> None:
    """Refuse *entry* where it cannot sit on *record* as it stands."""
    elsewhere = _elsewhere(job.state, record, entry)
    if elsewhere:
        raise Refused(REGISTRATION_ELSEWHERE, id=record.id, registration=entry["id"],
                      message=f"Registration {entry['id']} of {record.id} cannot go back "
                              f"as the section stands: {elsewhere}. Put the section back "
                              "at that position and those cutting angles first.")
    deformation = entry.get("deformation")
    if deformation and not (job.layout.resolve(str(deformation.get("record", "")))
                            / "record.json").is_file():
        raise Refused(DEFORMATION_UNREADABLE, id=record.id, registration=entry["id"],
                      message=f"The deformation of registration {entry['id']} of "
                              f"{record.id} is no longer in the job folder "
                              f"({deformation.get('record')}).")


def _restore_one(job: Job, record: SliceState, number: Any) -> Restored:
    """Put registration *number* back on *record* (the caller holds the lock
    and commits); refused as :func:`restore` says."""
    store = job.registrations
    held = _list_held(job, record)
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise Refused("BAD_ARGS", message=f"{record.id}: a registration is a number from 1; "
                      f"got {number!r}.")
    entry = store.get(record.id, number)
    if entry is None:
        raise _unknown(job, record, number)
    if held == entry["id"]:
        return Restored(id=record.id, registration=int(entry["id"]), replaced=held,
                        written=False)
    now = regs.value(job.state, record) or {}
    changes: list[str] = []
    if (bool(record.flip), int(record.rotation_deg)) != (bool(entry.get("flip")),
                                                         int(entry.get("rotation_deg") or 0)):
        changes.append("orientation")
    if record.transform != entry.get("transform"):
        changes.append("affine")
    if ((now.get("deformation") or {}).get("key")
            != (entry.get("deformation") or {}).get("key")):
        changes.append("deformation")
    check_frozen(job, [record.id], *changes)
    _check_entry(job, record, entry)
    deformation = entry.get("deformation")
    record.flip = bool(entry.get("flip"))
    record.rotation_deg = int(entry.get("rotation_deg") or 0)
    record.transform = copy.deepcopy(entry.get("transform"))
    record.deformation = copy.deepcopy(deformation) if deformation else None
    return Restored(id=record.id, registration=int(entry["id"]), replaced=held, written=True)


def restore(job: Job, sections: Any) -> RestoredMany:
    """Put back one registration of each section named (*sections*:
    ``[{"id": filename, "registration": number}]``; orientation, transform
    and deformation as the section held them), as ONE undo step (see the
    module text). A section that cannot be restored is its own row and the
    others are written. Refused, nothing written: ``BAD_ARGS`` (no entries,
    an entry that is not ``{"id", "registration"}``),
    ``DUPLICATE_SLICE_IDS``, ``UNKNOWN_SLICE_IDS``."""
    entries = list(sections or [])
    if not entries or not all(isinstance(e, dict) and set(e) == {"id", "registration"}
                              for e in entries):
        raise Refused("BAD_ARGS", message="sections is a list of {\"id\": filename, "
                      "\"registration\": number}, one per section to put back.")
    with job.writing():
        state = job.state
        named: list[SliceState] = []
        unknown: list[str] = []
        for item in entries:
            record = state.resolve(str(item["id"]))
            if record is None:
                unknown.append(str(item["id"]))
            else:
                named.append(record)
        if unknown:
            raise unknown_sections(state, unknown)
        ids = [record.id for record in named]
        doubled = sorted({name for name in ids if ids.count(name) > 1})
        if doubled:
            raise Refused("DUPLICATE_SLICE_IDS", duplicate_ids=doubled,
                          message="Name each section once.")
        before = job.snapshot()
        rows: list[dict[str, Any]] = []
        written: list[str] = []
        for record, item in zip(named, entries, strict=True):
            try:
                done = _restore_one(job, record, item["registration"])
            except Refused as refusal:
                rows.append({"id": record.id, **refusal.payload()})
                continue
            rows.append({"id": done.id, "status": "ok", "registration": done.registration,
                         **({"replaced": done.replaced}
                            if done.replaced not in (None, done.registration) else {}),
                         "written": done.written})
            if done.written:
                written.append(done.id)
        if written:
            job.commit(before)
        return RestoredMany(rows=rows, written=written)


__all__ = ["DEFORMATION_UNREADABLE", "REGISTRATION_ELSEWHERE", "UNKNOWN_REGISTRATION",
           "Listed", "Restored", "RestoredMany", "entry_for", "listing", "restore", "row",
           "saved_pictures"]
