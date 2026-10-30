"""``restore_registration``: sections back to registrations they held before.

Every write's registrations are numbered in each section's history
(:mod:`langslice.job.history`: a section's orientation, transform and
deformation, numbered per section). :func:`listing` answers with a
section's list (numbering the registration it holds now when it never was,
which writes no state), each row naming its transform, the method and
settings of its deformation and the pictures the write that left it saved;
:func:`restore` puts one registration of each section named back, as ONE
undo step, the other sections untouched. A section is refused, as its own
row, when it would change a property the job froze (``FROZEN``), names no
registration of the section (``UNKNOWN_REGISTRATION``), or holds a
deformation fitted with the section at another position or cutting angles
(``REGISTRATION_ELSEWHERE``: its linear placement would not be the one it
was fitted on), or one whose record is gone (``DEFORMATION_UNREADABLE``).
:func:`entry_for` is the registration ``look`` draws a section under
without writing it. The tool's function (``restore_registration_tool``)
answers with a listing, or with a restore's rows and the overlays of the
sections put back.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langslice.core.fits.deformation import linear_key
from langslice.core.jsonable import JsonObject
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.stack.codec import Written
from langslice.core.stack.section import Angles, Section
from langslice.core.stack.stack import Stack
from langslice.core.stack.status import deformation_entry, transform_entry
from langslice.job.history import (
    HOST,
    REGISTRATION_KEYS,
    Row,
    guessed_maker,
    registration_value,
    set_head,
)
from langslice.job.settings.needs import render
from langslice.job.state_file import stack_from_json
from langslice.tools.look import show_result
from langslice.tools.refusal import check_frozen, unknown_sections
from langslice.tools.run import as_list, pictured, spread

if TYPE_CHECKING:
    from langslice.job.job import Job
    from langslice.tools.arguments import SectionRegistration
    from langslice.tools.run import Run


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
    now, the one that replaced (None: none, or never numbered) and whether
    anything was written (False: it held that one already); or why it was
    not restored."""

    id: str
    registration: int = 0
    replaced: int | None = None
    written: bool = False
    #: Why the section was not restored; None when it was.
    refusal: Refused | None = None


@dataclass(frozen=True)
class RestoredMany:
    """What :func:`restore` did: one :class:`Restored` per section named, in
    call order, and the sections written (one undo step)."""

    rows: list[Restored]
    written: list[str]


def _where(position_mm: float | None, angles: Angles) -> str:
    place = (
        "no position"
        if position_mm is None
        else f"{round(float(position_mm), 3)} mm"
    )
    pitch, yaw = float(angles[0]), float(angles[1])
    return place + (
        f", pitch {pitch:g} deg, yaw {yaw:g} deg" if pitch or yaw else ""
    )


def _elsewhere(state: Stack, section: Section, entry: Row) -> str | None:
    """Why *entry* cannot be put back on *section* as it stands now (its
    deformation was fitted at another position or cutting angles), else None."""
    deformation = entry.deformation
    if deformation is None:
        return None
    trial = dataclasses.replace(
        section,
        flip=entry.flip,
        rotation_deg=entry.rotation_deg,
        transform=entry.transform,
    )
    if deformation.linear_key == linear_key(state, trial):
        return None
    return (
        f"its deformation was fitted with the section at "
        f"{_where(entry.position_mm, entry.angles)}; "
        f"the section is at {_where(section.position_mm, section.angles)} now"
    )


def made_by(entry: Row) -> str:
    """Return the tool a registration's listing names.

    Parameters
    ----------
    entry
        The first line of the registration in the section's history.

    Returns
    -------
    str
        The tool whose write numbered it; for a ``host`` line (a state
        LangSlice found), what its content says made it.
    """
    return guessed_maker(entry) if entry.by == HOST else entry.by


def row(
    state: Stack,
    record: Section,
    entry: Row,
    held: int | None,
    pictures: list[int] | None = None,
) -> dict[str, Any]:
    """One registration as the listing shows it: its number, the tool that
    made it and when (:func:`made_by`; no time for one LangSlice found), its
    transform (kind and knobs,
    :func:`langslice.core.stack.status.transform_entry`), its deformation
    (method,
    steps and settings, :func:`langslice.core.stack.status.deformation_entry`),
    the
    *pictures* the write that left it saved, whether the section holds it now,
    and why it cannot be put back as the section stands."""
    transform = transform_entry(entry.transform)
    deformation = deformation_entry(entry.deformation)
    by_host = entry.by == HOST
    out: dict[str, Any] = {
        "registration": entry.registration,
        "by": made_by(entry),
        **({"at": entry.at} if entry.at and not by_host else {}),
        **({"transform": transform} if transform else {}),
        **({"deformation": deformation} if deformation else {}),
        **({"pictures": list(pictures)} if pictures else {}),
    }
    if entry.registration == held:
        out["held"] = True
    elsewhere = _elsewhere(state, record, entry)
    if elsewhere:
        out["not_here"] = elsewhere
    return out


def _resolve(job: Job, section: str) -> Section:
    record = job.state.resolve(section)
    if record is None:
        raise unknown_sections(job.state, [section])
    return record


def saved_pictures(job: Job, section: str) -> dict[int, list[int]]:
    """The pictures the writes of *section*'s registrations saved: per
    registration number, the numbers of the pictures drawn by the tool that
    made it showing the section under it (``views.jsonl``'s
    ``registrations``), oldest first."""
    made = {
        cast("int", entry.registration): made_by(entry)
        for entry in job.histories.registrations(section)
    }
    found: dict[int, list[int]] = {}
    for picture in job.views.records():
        number = picture.registrations.get(section)
        if number is not None and made.get(number) == picture.tool:
            found.setdefault(int(number), []).append(picture.seq)
    return found


def listing(job: Job, section: str) -> Listed:
    """The section's registrations; the one it holds now numbered first when
    it never was. Writes no state."""
    with job.writing():
        record = _resolve(job, section)
        held = job.number_registration(record)
        pictures = saved_pictures(job, record.id)
        rows = [
            row(
                job.state,
                record,
                entry,
                held,
                pictures.get(cast("int", entry.registration)),
            )
            for entry in job.histories.registrations(record.id)
        ]
        return Listed(id=record.id, held=held, rows=rows)


def _unknown(job: Job, record: Section, number: object) -> Refused:
    numbers = [
        entry.registration for entry in job.histories.registrations(record.id)
    ]
    listed = (
        f"its registrations are {', '.join(str(n) for n in numbers)}"
        if numbers
        else "it has none listed yet"
    )
    return Refused(
        RefusalCode.UNKNOWN_REGISTRATION,
        id=record.id,
        registration=number,
        registrations=numbers,
        message=render(
            f"{record.id} has no registration {number}; {listed}.[[?"
            "restore_registration  Call restore_registration with the section "
            "alone to list them.]]",
            job.spec.switched_off,
        ),
    )


def _number(section: Section, number: object) -> int:
    """Return a registration number the model gave; ``BAD_ARGS`` for
    anything but a whole number from 1.
    """
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise Refused(
            RefusalCode.BAD_ARGS,
            message=f"{section.id}: a registration is a number from 1; "
            f"got {number!r}.",
        )
    return number


def entry_for(
    job: Job, section: object, number: object
) -> tuple[Section, Row, int | None]:
    """``(record, entry, held)``: registration *number* of *section* as it
    can be drawn on the section as it stands (what ``look`` draws a section
    under without writing it), and the number the section holds. Refused:
    ``UNKNOWN_SLICE_IDS``, ``UNKNOWN_REGISTRATION``,
    ``REGISTRATION_ELSEWHERE``, ``DEFORMATION_UNREADABLE``, ``BAD_ARGS``."""
    record = _resolve(job, str(section))
    entry = _listed(job, record, number)
    held = job.histories.registration_number(job.state, record)
    if entry is None:
        raise _unknown(job, record, number)
    _check_entry(job, record, entry)
    return record, entry, held


def _listed(job: Job, section: Section, number: object) -> Row | None:
    """Return the first line of registration *number* of a section; None
    without one; ``BAD_ARGS`` for a number that is not one.
    """
    given = _number(section, number)
    for entry in job.histories.registrations(section.id):
        if entry.registration == given:
            return entry
    return None


def _check_entry(job: Job, record: Section, entry: Row) -> None:
    """Refuse *entry* where it cannot sit on *record* as it stands."""
    elsewhere = _elsewhere(job.state, record, entry)
    if elsewhere:
        raise Refused(
            RefusalCode.REGISTRATION_ELSEWHERE,
            id=record.id,
            registration=entry.registration,
            message=f"Registration {entry.registration} of {record.id} cannot go back "
            f"as the section stands: {elsewhere}. Put the section back "
            "at that position and those cutting angles first.",
        )
    deformation = entry.deformation
    if (
        deformation is not None
        and not (
            job.layout.resolve(deformation.record or "") / "record.json"
        ).is_file()
    ):
        raise Refused(
            RefusalCode.DEFORMATION_UNREADABLE,
            id=record.id,
            registration=entry.registration,
            message=f"The deformation of registration {entry.registration} of "
            f"{record.id} is no longer in the job folder "
            f"({deformation.record}).",
        )


def _restore_one(job: Job, record: Section, number: object) -> Restored:
    """Put registration *number* back on *record* (the caller holds the lock
    and commits); refused as :func:`restore` says."""
    held = job.number_registration(record)
    entry = _listed(job, record, number)
    if entry is None:
        raise _unknown(job, record, number)
    restored = cast("int", entry.registration)
    if held == restored:
        return Restored(
            id=record.id, registration=restored, replaced=held, written=False
        )
    changes: list[str] = []
    if (record.flip, record.rotation_deg) != (entry.flip, entry.rotation_deg):
        changes.append("orientation")
    if record.transform != entry.transform:
        changes.append("affine")
    current = registration_value(job.state, record)
    if _record_key(current) != _record_key(entry.section):
        changes.append("deformation")
    check_frozen(job, [record.id], *changes)
    _check_entry(job, record, entry)
    set_head(
        record, {key: entry.section.get(key) for key in REGISTRATION_KEYS}
    )
    return Restored(
        id=record.id, registration=restored, replaced=held, written=True
    )


def _record_key(registration: JsonObject | None) -> str | None:
    """Return the record key of a registration's deformation, or None."""
    deformation = (
        None if registration is None else registration.get("deformation")
    )
    if not isinstance(deformation, Mapping):
        return None
    key = deformation.get("key")
    return key if isinstance(key, str) else None


def restore(job: Job, sections: Any) -> RestoredMany:
    """Put back one registration of each section named (*sections*:
    ``[{"id": filename, "registration": number}]``; orientation, transform
    and deformation as the section held them), as ONE undo step (see the
    module text). A section that cannot be restored is its own row and the
    others are written; any other failure partway puts back the sections
    already changed and raises (nothing written, in memory or on disk).
    Refused, nothing written: ``BAD_ARGS`` (no entries,
    an entry that is not ``{"id", "registration"}``),
    ``DUPLICATE_SLICE_IDS``, ``UNKNOWN_SLICE_IDS``."""
    entries = list(sections or [])
    if not entries or not all(
        isinstance(e, dict) and set(e) == {"id", "registration"}
        for e in entries
    ):
        raise Refused(
            RefusalCode.BAD_ARGS,
            message='sections is a list of {"id": filename, '
            '"registration": number}, one per section to put back.',
        )
    with job.writing():
        state = job.state
        named: list[Section] = []
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
            raise Refused(
                RefusalCode.DUPLICATE_SLICE_IDS,
                duplicate_ids=doubled,
                message="Name each section once.",
            )
        before = job.snapshot()
        rows: list[Restored] = []
        written: list[str] = []
        try:
            for record, item in zip(named, entries, strict=True):
                try:
                    # the number as the call gave it
                    number = cast("object", item["registration"])
                    done = _restore_one(job, record, number)
                except Refused as refusal:
                    rows.append(Restored(record.id, refusal=refusal))
                    continue
                rows.append(done)
                if done.written:
                    written.append(done.id)
        except BaseException:
            # all or nothing: a failure partway puts back the sections already changed
            job.state.replace_with(stack_from_json(before))
            raise
        if written:
            job.commit(before)
        return RestoredMany(rows=rows, written=written)


# --- the tool ----------------------------------------------------------------


def restore_registration_tool(
    run: Run,
    *,
    section: str = "",
    sections: list[SectionRegistration] | None = None,
    view: bool = True,
) -> dict[str, Any]:
    """Run the ``restore_registration`` tool: list, or restore and picture.

    Parameters
    ----------
    run
        The run the tool is called in.
    section, sections, view
        The declared arguments as the model sent them
        (``tools/descriptions/bookkeeping.py``).

    Returns
    -------
    dict
        The listing, or the restore's rows and the rows it changed.
    """
    job = run.job
    entries = as_list(sections)
    if bool(str(section or "").strip()) == bool(entries):
        raise Refused(
            RefusalCode.BAD_ARGS,
            "Give section alone to list its registrations, "
            'or sections ([{"id": filename, "registration": number}]) to '
            "put them back; not both.",
        )
    if not entries:
        listed = listing(job, str(section))
        reply: dict[str, Any] = {
            "status": "ok",
            "id": listed.id,
            "registrations": listed.rows,
        }
        if listed.held is not None:
            reply["held"] = listed.held
        return reply

    done = restore(job, entries)
    ok = [row.id for row in done.rows if row.refusal is None]
    result = _restored_reply(done, restored=bool(ok))
    if done.written:
        result["changed"] = run.changed(done.written)["changed"]
    if not ok or not run.shows(view):
        return result

    put_back = [job.state.by_id(section_id) for section_id in ok]
    in_picture: list[Section] = spread(
        [one for one in put_back if one is not None], job.picture_cap
    )
    looked = show_result(
        job,
        run.workspace,
        "restore_registration",
        "overlay",
        [one.id for one in in_picture],
    )
    return pictured(result, looked)


def _restored_reply(done: RestoredMany, *, restored: bool) -> dict[str, Any]:
    """Return a restore's ``status`` and rows; when every section was refused,
    the first one's code and message lead.
    """
    status = "ok" if restored else "error"
    result: dict[str, Any] = {"status": status}
    first = done.rows[0].refusal
    if not restored and first is not None:  # the first section's reason leads
        result["error"] = first.code
        if first.message:
            result["message"] = first.message
    result["results"] = [_restored_row(row) for row in done.rows]
    return result


def _restored_row(row: Restored) -> Written:
    """Return one section's row: its refusal, or the registration it holds
    now, the one that replaced (when another) and whether it was written.
    """
    if row.refusal is not None:
        return cast("Written", row.refusal.payload(section_id=row.id))
    out: Written = {
        "id": row.id,
        "status": "ok",
        "registration": row.registration,
    }
    if row.replaced not in (None, row.registration):
        out["replaced"] = row.replaced
    out["written"] = row.written
    return out


__all__ = [
    "Listed",
    "Restored",
    "RestoredMany",
    "entry_for",
    "listing",
    "restore",
    "restore_registration_tool",
    "row",
    "saved_pictures",
]
