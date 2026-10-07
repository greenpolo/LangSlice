"""The tools that change a registration by hand, and the bookkeeping: their bodies.

``position_sections`` and ``interactive_transform`` write, then show their
opinionated picture (:func:`langslice.ops.look.show_result`: the
positioning picture, or the section under its new transform with the atlas
borders on it) unless the call's ``view`` is false and the host does not
force it (``JobSpec.force_view``). ``mark_damage`` always shows the marked
regions. ``note``, ``undo``, ``redo`` and ``submit`` answer with rows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langslice.core import layers
from langslice.core.display import default_options
from langslice.core.sizes import picture_edge
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.door import Door, as_list, pictured
from langslice.ops import damage as ops_damage
from langslice.ops import history as ops_history
from langslice.ops import look as ops_look
from langslice.ops import notes as ops_notes
from langslice.ops import positions as ops_positions
from langslice.ops import submit as ops_submit
from langslice.ops import transforms as ops_transforms
from langslice.ops.look import MAX_LOOK_PICTURES
from langslice.ops.refusal import Refused

#: Positions closer than this (mm) are one atlas picture in a positioning picture.
SAME_POSITION_MM = 0.02


def _positions(values: list[float]) -> list[float]:
    """*values* in order, each once (within :data:`SAME_POSITION_MM`)."""
    out: list[float] = []
    for value in values:
        if all(abs(value - held) > SAME_POSITION_MM for held in out):
            out.append(value)
    return out


def bodies(door: Door) -> dict[str, Callable[..., Any]]:
    """The changing and bookkeeping tools' bodies over *door*."""
    job, ctx, spec = door.job, door.ctx, door.spec

    def shows(view: bool) -> bool:
        return bool(view) or bool(spec.force_view)

    def position_sections(sections: list[dict[str, Any]] = [],  # noqa: B006 (read only)
                          cutting_angles: dict[str, Any] | None = None,
                          view: bool = True) -> dict[str, Any]:
        entries = as_list(sections)
        angles = cutting_angles if cutting_angles else None
        rejected: list[dict[str, Any]] = []
        if door.gated and entries:
            kept = []
            for entry in entries:
                record = job.state.resolve(entry.get("id", "")) if isinstance(entry, dict) \
                    else None
                if record is not None and not door.box.compared.get(record.id):
                    # One look is enough: a section is confirmed at one position.
                    rejected.append({"id": record.id, "reason": "not compared since its "
                                     "last write; look at it in mode overlay or positioning "
                                     "first"})
                    continue
                kept.append(entry)
            entries = kept
            if not entries and angles is None:
                return {"status": "refused", "error": "NOT_COMPARED", "rejected": rejected}
        try:
            done = ops_positions.position_sections(job, ctx, entries, angles)
        except Refused as refusal:
            return refusal.payload()
        if not done.written and done.cutting_angles is None:
            return {"status": "error", "error": "NOTHING_WRITTEN",
                    **({"unknown_ids": done.unknown} if done.unknown else {}),
                    **({"rejected": rejected} if rejected else {})}
        door.forget_looks(done.touched)
        if angles is not None:
            door.box.reviewed = False
        result: dict[str, Any] = {
            "written": [{"id": name, "position_mm": round(value, 3)}
                        for name, value in done.written],
            **({"clamped": [{"id": name, "requested_mm": round(asked, 3),
                             "clamped_to_mm": round(value, 3)}
                            for name, asked, value in done.clamped],
                "atlas_range_mm": [round(v, 3) for v in done.position_range]}
               if done.clamped else {}),
            **({"unknown_ids": done.unknown} if done.unknown else {}),
            **({"rejected": rejected} if rejected else {}),
            "order": done.order,
            **({"reordered": done.reordered} if done.reordered else {}),
            **door.answered(*done.touched),
        }
        if not shows(view):
            return result
        pictured_ids = done.touched or [record.id for record in job.state.in_order()]
        records = [job.state.by_id(name) for name in pictured_ids]
        positions = _positions([float(record.position_mm) for record in records
                                if record is not None and record.position_mm is not None])
        shown = ops_look.show_result(job, ctx, "position_sections", "positioning",
                                     pictured_ids, positions_mm=positions)
        return pictured(result, shown)

    def interactive_transform(sections: list[dict[str, Any]],
                              view: bool = True) -> dict[str, Any]:
        entries = as_list(sections)
        if not entries:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "sections must name one or more sections"}
        if len(entries) > MAX_LOOK_PICTURES:
            return {"status": "error", "error": "TOO_MANY_SECTIONS",
                    "max_sections": MAX_LOOK_PICTURES}
        refusal = door.over_cap(len(entries))
        if refusal is not None:
            return refusal
        named = [job.state.resolve(entry.get("id", "")) for entry in entries
                 if isinstance(entry, dict)]
        ids = [record.id for record in named if record is not None]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            return {"status": "error", "error": "DUPLICATE_SLICE_IDS",
                    "duplicate_ids": duplicates,
                    "message": "A call sets each section once; look at the first result "
                    "before a dependent correction."}
        done = ops_transforms.interactive_transform(job, ctx, entries)
        rows: list[dict[str, Any]] = []
        drawn: list[str] = []
        for entry in done.entries:
            if entry.error is not None:
                rows.append(dict(entry.error))
                continue
            row: dict[str, Any] = {"id": entry.id, "status": "ok",
                                   "orientation": entry.orientation,
                                   "written": entry.written}
            if entry.transform is not None:
                row["transform"] = entry.transform.get("physical")
            if entry.message:
                row["message"] = entry.message
            rows.append(row)
            drawn.append(entry.id)
        ok = bool(drawn)
        result: dict[str, Any] = {
            "status": "ok" if ok else "error",
            **({} if ok else {"error": "NOTHING_WRITTEN"}),
            "results": rows,
            **({"changed": door.changed(done.written)["changed"]} if done.written else {}),
        }
        if not ok or not shows(view):
            return result
        shown = ops_look.show_result(job, ctx, "interactive_transform", "overlay", drawn)
        return pictured(result, shown)

    def mark_damage(section: str, regions: list[str], note: str) -> dict[str, Any]:
        options = default_options("overlay", long_edge=picture_edge(ctx))
        try:
            done = ops_damage.mark_damage(job, ctx, section, as_list(regions), note,
                                          options=options)
        except Refused as refusal:
            return refusal.payload()
        result: dict[str, Any] = {
            "status": "ok", "id": done.id, "regions": done.regions, "note": done.note,
            "damaged": done.damaged, "written": done.written,
            **({"render_failed": done.render_failed} if done.render_failed else {}),
            **({"changed": door.changed(done.touched)["changed"]} if done.touched else {}),
        }
        if done.pictures:
            for picture in done.pictures:
                layers.annotate(picture, caption=f"{done.id}: marked damage regions "
                                + ", ".join(done.regions) + " shaded on the section under "
                                "its current registration (left) and on the atlas (right)")
            result[TOOL_MEDIA_PARTS_KEY] = list(done.pictures)
        return result

    def note(text: str) -> dict[str, Any]:
        try:
            return {"status": "ok", "notes": ops_notes.add_note(job, text)}
        except Refused as refusal:
            return refusal.payload()

    def undo() -> dict[str, Any]:
        step = ops_history.undo(job)
        if not step.done:
            return {"status": "error", "error": "NOTHING_TO_UNDO"}
        door.forget_looks(step.moved)
        return {"status": "ok", "undo_depth": step.depth, **door.rows()}

    def redo() -> dict[str, Any]:
        step = ops_history.redo(job)
        if not step.done:
            return {"status": "error", "error": "NOTHING_TO_REDO"}
        door.forget_looks(step.moved)
        return {"status": "ok", "redo_depth": step.depth, **door.rows()}

    def submit(summary: str, notes: list[str], interval_breaks: list[int],
               left_linear: list[dict[str, Any]] = [],  # noqa: B006 (read only)
               tool_context: Any = None) -> dict[str, Any]:
        def look_gate() -> dict[str, Any] | None:
            if spec.has("position") and door.gated and not door.box.reviewed:
                return {"status": "refused", "error": "NOT_REVIEWED",
                        "detail": "look in mode positioning at every section has not run "
                        "since the last position_sections write"}
            return None

        try:
            done = ops_submit.submit(
                job, summary=summary, notes=as_list(notes),
                interval_breaks=as_list(interval_breaks), left_linear=as_list(left_linear),
                traces=door.traces_on, workspace=ctx, gate=look_gate)
        except Refused as refusal:
            return refusal.payload()
        door.box.submission.update({"summary": done.summary, "notes": done.notes,
                                    "interval_breaks": done.interval_breaks})
        if tool_context is not None:
            tool_context.actions.escalate = True
        return {"status": "ok", **({"left_linear": done.left_linear} if done.left_linear
                                   else {}), **door.rows()}

    return {
        "position_sections": position_sections,
        "interactive_transform": interactive_transform, "mark_damage": mark_damage,
        "note": note, "undo": undo, "redo": redo, "submit": submit,
    }
