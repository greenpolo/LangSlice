"""The tools that change a registration by hand, and the bookkeeping: their bodies.

``position_sections`` and ``interactive_transform`` write, then show their
opinionated picture (:func:`langslice.ops.look.show_result`: the
positioning picture, or the section under its new transform with the atlas
borders on it; where the job disables ``look:positioning``, the overlay of
each section ``position_sections`` wrote) unless the call's ``view`` is false
and the host does not force it (``JobSpec.force_view``). ``mark_damage`` always shows the marked
regions. ``restore_registration`` lists a section's registrations, or puts one
back and shows the section's overlay. ``note``, ``undo``, ``redo`` and
``submit`` answer with rows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langslice.core import layers
from langslice.core.atlas.images import nissl_note
from langslice.core.display import default_options
from langslice.core.sizes import picture_edge
from langslice.core.state import unknown_sections
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.door import Door, as_list, pictured
from langslice.ops import damage as ops_damage
from langslice.ops import history as ops_history
from langslice.ops import look as ops_look
from langslice.ops import masks as ops_masks
from langslice.ops import notes as ops_notes
from langslice.ops import positions as ops_positions
from langslice.ops import registrations as ops_registrations
from langslice.ops import submit as ops_submit
from langslice.ops import transforms as ops_transforms
from langslice.ops.refusal import Refused
from langslice.ops.registry import check_section_count

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
        try:
            done = ops_positions.position_sections(job, ctx, entries, angles)
        except Refused as refusal:
            return refusal.payload()
        unknown = unknown_sections(job.state, done.unknown) if done.unknown else None
        if not done.written and done.cutting_angles is None:
            return {"status": "error", "error": "NOTHING_WRITTEN",
                    **({"unknown_ids": done.unknown, "message": unknown["message"],
                        "filenames": unknown["filenames"]} if unknown else {})}
        result: dict[str, Any] = {
            "written": [{"id": name, "position_mm": round(value, 3)}
                        for name, value in done.written],
            **({"clamped": [{"id": name, "requested_mm": round(asked, 3),
                             "clamped_to_mm": round(value, 3)}
                            for name, asked, value in done.clamped],
                "atlas_range_mm": [round(v, 3) for v in done.position_range]}
               if done.clamped else {}),
            **({"unknown_ids": done.unknown, "unknown_message": unknown["message"]}
               if unknown else {}),
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
        # The positioning picture, or, where the job disables that look, the
        # sections' overlays.
        mode = "overlay" if spec.off("look:positioning") else "positioning"
        shown = ops_look.show_result(job, ctx, "position_sections", mode,
                                     pictured_ids, positions_mm=positions)
        return pictured(result, shown)

    def interactive_transform(sections: list[dict[str, Any]],
                              view: bool = True) -> dict[str, Any]:
        entries = as_list(sections)
        if not entries:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "sections must name one or more sections"}
        try:
            check_section_count(job, "interactive_transform", len(entries))
        except Refused as refusal:
            return refusal.payload()
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
        # A deformation this write made stale goes before the picture is
        # drawn, so the picture shows the section as it now stands.
        cleared = job.clear_stale_deformations()
        if cleared:
            result["deformation_cleared"] = cleared
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

    def set_mask(section: str, name: str, source: str = "em",
                 channel: str = "preprocessed", atlas_image: str = "nissl",
                 sorting: dict[str, float] | None = None, delete: bool = False,
                 view: bool = True) -> dict[str, Any]:
        try:
            done = ops_masks.set_mask(
                job, ctx, section, str(name), source=str(source or "em"),
                channel=str(channel or "preprocessed"), atlas_image=str(atlas_image or "nissl"),
                sorting=dict(sorting or {}), delete=bool(delete))
        except Refused as refusal:
            return refusal.payload()
        if done.deleted:
            return {"status": "ok", "id": done.id, "name": done.name, "deleted": True,
                    **door.answered(done.id)}
        entry = done.entry or {}
        result: dict[str, Any] = {
            "status": "ok", "id": done.id, "name": done.name, "source": entry.get("kind"),
            "channel": entry.get("channel"), "atlas_image": entry.get("atlas_image"),
            "params": entry.get("params"), "left_out": round(float(entry.get("left_out", 0)), 4),
            **door.answered(done.id),
        }
        said = nissl_note(ctx.atlas, str(atlas_image or "nissl"))
        if said:
            result["atlas_image_note"] = said
        if not shows(view):
            return result
        shown = ops_look.show_result(job, ctx, "set_mask", "section", [done.id],
                                     masks=[done.name])
        return pictured(result, shown)

    def note(text: str) -> dict[str, Any]:
        try:
            return {"status": "ok", "notes": ops_notes.add_note(job, text)}
        except Refused as refusal:
            return refusal.payload()

    def stepped(word: str, depth_key: str, step: Any) -> dict[str, Any]:
        """The reply to an undo or redo: which step it took back or forward
        (its sections and the fields it changed), the rows of those sections
        and the depth."""
        info = step.step
        what: dict[str, Any] = {"sections": info.get("sections", []),
                                "fields": info.get("fields", [])}
        result: dict[str, Any] = {"status": "ok", word: what, depth_key: step.depth}
        if what["sections"]:
            rows = door.changed(what["sections"])
            result["changed"] = rows["changed"]
            if "placement" in info.get("fields", []):
                result["cutting_angles_deg"] = rows["cutting_angles_deg"]
                result["interval_breaks"] = rows["interval_breaks"]
        return result

    def undo() -> dict[str, Any]:
        step = ops_history.undo(job)
        if not step.done:
            return {"status": "error", "error": "NOTHING_TO_UNDO"}
        return stepped("undone", "undo_depth", step)

    def redo() -> dict[str, Any]:
        step = ops_history.redo(job)
        if not step.done:
            return {"status": "error", "error": "NOTHING_TO_REDO"}
        return stepped("redone", "redo_depth", step)

    def restore_registration(section: str, registration: int = 0,
                             view: bool = True) -> dict[str, Any]:
        try:
            if not registration:
                listed = ops_registrations.listing(job, str(section))
                return {"status": "ok", "id": listed.id, "registrations": listed.rows,
                        **({"held": listed.held} if listed.held is not None else {})}
            done = ops_registrations.restore(job, str(section), int(registration))
        except Refused as refusal:
            return refusal.payload()
        result: dict[str, Any] = {
            "status": "ok", "id": done.id, "registration": done.registration,
            **({"replaced": done.replaced} if done.replaced not in (None, done.registration)
               else {}),
            "written": done.written, **door.answered(done.id),
        }
        if not shows(view):
            return result
        shown = ops_look.show_result(job, ctx, "restore_registration", "overlay", [done.id])
        return pictured(result, shown)

    def submit(summary: str, notes: list[str], interval_breaks: list[str],
               tool_context: Any = None) -> dict[str, Any]:
        try:
            done = ops_submit.submit(
                job, summary=summary, notes=as_list(notes),
                interval_breaks=as_list(interval_breaks),
                traces=door.traces_on, workspace=ctx)
        except Refused as refusal:
            return refusal.payload()
        door.box.submission.update({"summary": done.summary, "notes": done.notes,
                                    "interval_breaks": done.interval_breaks})
        if tool_context is not None:
            tool_context.actions.escalate = True
        return {"status": "ok", **door.rows()}

    return {
        "position_sections": position_sections,
        "interactive_transform": interactive_transform, "mark_damage": mark_damage,
        "set_mask": set_mask, "note": note, "undo": undo, "redo": redo,
        "restore_registration": restore_registration, "submit": submit,
    }
