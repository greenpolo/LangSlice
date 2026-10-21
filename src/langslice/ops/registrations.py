"""``restore_registration``: one section back to a registration it held before.

Every write's registrations are listed by the job (:mod:`langslice.job.registrations`:
a section's orientation, transform and deformation, numbered per section).
:func:`listing` answers with a section's list (listing the registration it
holds now when it was never listed, which writes no state); :func:`restore`
puts one back as ONE undo step, the other sections untouched, refused when it
would change a property the job froze (``FROZEN``) or a section background
work is running on (``SECTION_BUSY``), names no registration of the section
(``UNKNOWN_REGISTRATION``), or holds a deformation fitted with the section at
another position or cutting angles (``REGISTRATION_ELSEWHERE``: its linear
placement would not be the one it was fitted on), or one whose record is gone
(``DEFORMATION_UNREADABLE``).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langslice.core.deformation import linear_key
from langslice.core.state import SliceState, StackState
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
    """What :func:`restore` did: the section, the registration it holds now,
    the one that replaced (None: none, or never listed) and whether anything
    was written (False: it held that one already)."""

    id: str
    registration: int
    replaced: int | None
    written: bool


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
        held: int | None) -> dict[str, Any]:
    """One registration as the listing shows it: its number, the tool that
    made it and when, its transform's kind and deformation steps, whether the
    section holds it now, and why it cannot be put back as the section stands."""
    steps = len((entry.get("deformation") or {}).get("steps") or [])
    kind = (entry.get("transform") or {}).get("kind")
    out: dict[str, Any] = {"registration": entry["id"], "by": entry.get("by"),
                           **({"at": entry["at"]} if entry.get("at") else {}),
                           **({"transform": kind} if kind else {}),
                           **({"deformation_steps": steps} if steps else {})}
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


def listing(job: Job, section: str) -> Listed:
    """The section's registrations; the one it holds now listed first when it
    never was. Writes no state."""
    with job.writing():
        record = _resolve(job, section)
        state, store = job.state, job.registrations
        held = _list_held(job, record)
        rows = [row(state, record, entry, held) for entry in store.entries(record.id)]
        return Listed(id=record.id, held=held, rows=rows)


def restore(job: Job, section: str, number: int) -> Restored:
    """Put the section's registration *number* back (orientation, transform
    and deformation as it held them), as one undo step (see the module text)."""
    with job.writing():
        record = _resolve(job, section)
        state, store = job.state, job.registrations
        held = _list_held(job, record)
        entry = store.get(record.id, int(number))
        if entry is None:
            numbers = [item["id"] for item in store.entries(record.id)]
            listed = (f"its registrations are {', '.join(str(n) for n in numbers)}" if numbers
                      else "it has none listed yet")
            raise Refused(UNKNOWN_REGISTRATION, id=record.id, registration=number,
                          registrations=numbers,
                          message=f"{record.id} has no registration {number}; {listed}. "
                                  "Call restore_registration with the section alone to list "
                                  "them.")
        if held == entry["id"]:
            return Restored(id=record.id, registration=int(entry["id"]), replaced=held,
                            written=False)
        now = regs.value(state, record) or {}
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
        elsewhere = _elsewhere(state, record, entry)
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
        before = job.snapshot()
        record.flip = bool(entry.get("flip"))
        record.rotation_deg = int(entry.get("rotation_deg") or 0)
        record.transform = copy.deepcopy(entry.get("transform"))
        record.deformation = copy.deepcopy(deformation) if deformation else None
        job.commit(before)
        return Restored(id=record.id, registration=int(entry["id"]), replaced=held,
                        written=True)


__all__ = ["DEFORMATION_UNREADABLE", "REGISTRATION_ELSEWHERE", "UNKNOWN_REGISTRATION",
           "Listed", "Restored", "listing", "restore", "row"]
