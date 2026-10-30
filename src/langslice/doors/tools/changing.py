"""The tools that change a registration by hand, and the bookkeeping: their bodies.

``position_sections`` and ``interactive_transform`` write, then show their
opinionated picture (:func:`langslice.tools.look.show_result`: the positioning
picture, or the section under its new transform with the atlas borders on it;
where the job disables ``look:positioning``, the overlay of each section
``position_sections`` wrote) unless the call's ``view`` is false and the host
does not force it (``JobSpec.force_view``). ``mark_damage`` always shows the
marked regions. ``restore_registration`` lists a section's registrations, or
puts one registration of each section named back and shows their overlays.
``note``, ``undo``, ``redo`` and ``submit`` answer with rows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langslice.core.atlas.images import nissl_note
from langslice.core.pictures import drawn as pictures_drawn
from langslice.core.pictures.options import default_options
from langslice.core.pictures.sizes import picture_edge
from langslice.core.refusal import RefusalCode, Refused, ReplyStatus
from langslice.core.stack.codec import physical_to_json
from langslice.core.stack.stack import unknown_sections
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.tools import damage as ops_damage
from langslice.tools import history as ops_history
from langslice.tools import look as ops_look
from langslice.tools import masks as ops_masks
from langslice.tools import notes as ops_notes
from langslice.tools import positions as ops_positions
from langslice.tools import registrations as ops_registrations
from langslice.tools import submit as ops_submit
from langslice.tools import transforms as ops_transforms
from langslice.tools.catalog import check_section_count
from langslice.tools.refusal import unknown_sections as unknown_refusal
from langslice.tools.run import Run, as_list, drawn_image, pictured, spread

#: Positions closer than this (mm) are one atlas picture in a positioning picture.
SAME_POSITION_MM = 0.02


def _positions(values: list[float]) -> list[float]:
    """*values* in order, each once (within :data:`SAME_POSITION_MM`)."""
    out: list[float] = []
    for value in values:
        if all(abs(value - held) > SAME_POSITION_MM for held in out):
            out.append(value)
    return out


def _first_refusal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The top-level error of a call that wrote nothing: the first row's code
    and message (every row keeps its own)."""
    first = rows[0] if rows else {}
    return {"error": first.get("error", RefusalCode.NOTHING_WRITTEN),
            **({"message": first["message"]} if first.get("message") else {})}


def bodies(run: Run) -> dict[str, Callable[..., Any]]:
    """The changing and bookkeeping tools' bodies over *run*."""
    job, ctx, spec = run.job, run.ctx, run.spec

    def shows(view: bool) -> bool:
        return bool(view) or bool(spec.force_view)

    def position_sections(sections: list[dict[str, Any]] = [],  # noqa: B006 (read only)
                          cutting_angles: dict[str, Any] | None = None,
                          view: bool = True) -> dict[str, Any]:
        entries = as_list(sections)
        angles = cutting_angles if cutting_angles else None
        done = ops_positions.position_sections(job, ctx, entries, angles)
        unknown = unknown_sections(job.state, done.unknown) if done.unknown else None
        if not done.written and done.cutting_angles is None and unknown:
            raise Refused(RefusalCode.UNKNOWN_SLICE_IDS, **unknown)  # every name
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
            **run.answered(*done.touched),
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
            raise Refused(RefusalCode.BAD_ARGS,
                          "sections must name one or more sections")
        check_section_count(job, "interactive_transform", len(entries))
        named = [job.state.resolve(entry.get("id", "")) for entry in entries
                 if isinstance(entry, dict)]
        ids = [record.id for record in named if record is not None]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise Refused(RefusalCode.DUPLICATE_SLICE_IDS,
                          duplicate_ids=duplicates,
                          message="A call sets each section once; look at the first "
                          "result before a dependent correction.")
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
                physical = entry.transform.physical
                row["transform"] = None
                if physical is not None:
                    row["transform"] = physical_to_json(physical)
            if entry.message:
                row["message"] = entry.message
            rows.append(row)
            drawn.append(entry.id)
        ok = bool(drawn)
        result: dict[str, Any] = {
            "status": "ok" if ok else "error",
            **({} if ok else {"error": RefusalCode.NOTHING_WRITTEN}),
            "results": rows,
            **({"changed": run.changed(done.written)["changed"]} if done.written else {}),
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
        done = ops_damage.mark_damage(job, ctx, section, as_list(regions), note,
                                      options=options)
        result: dict[str, Any] = {
            "status": "ok", "id": done.id, "regions": done.regions, "note": done.note,
            "damaged": done.damaged, "written": done.written,
            **({"render_failed": done.render_failed} if done.render_failed else {}),
            **({"changed": run.changed(done.touched)["changed"]} if done.touched else {}),
        }
        if done.pictures:
            for picture in done.pictures:
                pictures_drawn.annotate(picture, caption=f"{done.id}: marked damage regions "
                                + ", ".join(done.regions) + " shaded on the section under "
                                "its current registration (left) and on the atlas (right)")
            result[TOOL_MEDIA_PARTS_KEY] = list(done.pictures)
        return result

    def make_mask(name: str, sections: list[str] | None = None, source: str = "em",
                  section_image: str = "preprocessed", atlas_image: str = "nissl",
                  sorting: dict[str, float] | None = None, delete: bool = False,
                  view: bool = True) -> dict[str, Any]:
        named = as_list(sections)
        ids: list[str] | None = None
        if named:
            scope, unknown = run.resolve_many(named)
            if unknown or not scope:
                raise unknown_refusal(job.state, unknown)
            ids = [record.id for record in scope]
        done = ops_masks.make_mask(
            job, ctx, str(name), ids, source=str(source or "em"),
            section_image=str(section_image or "preprocessed"),
            atlas_image=str(atlas_image or "nissl"), sorting=dict(sorting or {}),
            delete=bool(delete))
        rows: list[dict[str, Any]] = []
        for row in done.rows:
            if row["status"] != ReplyStatus.OK:
                rows.append(row)
                continue
            entry = row.get("entry") or {}
            rows.append({"id": row["id"], "status": "ok"} if done.deleted else {
                "id": row["id"], "status": "ok", "source": entry.get("kind"),
                "section_image": entry.get("section_image"),
                "atlas_image": entry.get("atlas_image"), "params": entry.get("params"),
                "left_out": round(float(entry.get("left_out", 0)), 4)})
        result: dict[str, Any] = {"status": "ok", "name": done.name, "rows": rows,
                                  **({"deleted": True} if done.deleted else {}),
                                  "n_sections": len(done.made)}
        said = nissl_note(ctx.atlas, str(atlas_image or "nissl"))
        if said and not done.deleted:
            result["atlas_image_note"] = said
        if done.deleted or not shows(view):
            return result
        made = [record for record in (job.state.by_id(i) for i in done.made) if record is not None]
        shown = spread(made, job.picture_cap)
        sorted_on = drawn_image(section_image) if str(source or "em") == "em" else []
        pictures = ops_look.show_result(job, ctx, "make_mask", "section",
                                        [record.id for record in shown], masks=[done.name],
                                        images=sorted_on)
        return pictured(result, pictures)

    def note(text: str) -> dict[str, Any]:
        return {"status": "ok", "notes": ops_notes.add_note(job, text)}

    def stepped(word: str, depth_key: str, step: Any, asked: int) -> dict[str, Any]:
        """The reply to an undo or redo: which steps it took back or forward
        (their sections and the fields they changed), the rows of those
        sections and the depth; with more than one step asked, how many it
        took."""
        info = step.step
        what: dict[str, Any] = {"sections": info.get("sections", []),
                                "fields": info.get("fields", [])}
        result: dict[str, Any] = {"status": "ok", word: what, depth_key: step.depth}
        if asked > 1:
            result["steps"] = step.steps
            if step.steps < asked:
                result["message"] = (f"{asked} steps were asked and the history held "
                                     f"{step.steps}: all {step.steps} were taken.")
        if what["sections"]:
            rows = run.changed(what["sections"])
            result["changed"] = rows["changed"]
            if "placement" in info.get("fields", []):
                result["cutting_angles_deg"] = rows["cutting_angles_deg"]
                result["interval_breaks"] = rows["interval_breaks"]
        return result

    def step_count(steps: Any) -> int | None:
        try:
            count = int(steps)
        except (TypeError, ValueError):
            return None
        return count if count >= 1 and count == steps else None

    def undo(steps: int = 1) -> dict[str, Any]:
        count = step_count(steps)
        if count is None:
            raise Refused(RefusalCode.BAD_ARGS,
                          "steps is a whole number, 1 or more.")
        step = ops_history.undo(job, count)
        if not step.done:
            raise Refused(RefusalCode.NOTHING_TO_UNDO,
                          "Nothing to undo: the job's history holds no "
                          "earlier step.")
        return stepped("undone", "undo_depth", step, count)

    def redo(steps: int = 1) -> dict[str, Any]:
        count = step_count(steps)
        if count is None:
            raise Refused(RefusalCode.BAD_ARGS,
                          "steps is a whole number, 1 or more.")
        step = ops_history.redo(job, count)
        if not step.done:
            raise Refused(RefusalCode.NOTHING_TO_REDO,
                          "Nothing to redo: redo follows an undo, and a "
                          "write after it clears what was undone.")
        return stepped("redone", "redo_depth", step, count)

    def restore_registration(section: str = "",
                             sections: list[dict[str, Any]] = [],  # noqa: B006 (read only)
                             view: bool = True) -> dict[str, Any]:
        entries = as_list(sections)
        if bool(str(section or "").strip()) == bool(entries):
            raise Refused(RefusalCode.BAD_ARGS,
                          "Give section alone to list its registrations, "
                          "or sections ([{\"id\": filename, \"registration\": number}]) to "
                          "put them back; not both.")
        if not entries:
            listed = ops_registrations.listing(job, str(section))
            return {"status": "ok", "id": listed.id, "registrations": listed.rows,
                    **({"held": listed.held} if listed.held is not None else {})}
        done = ops_registrations.restore(job, entries)
        ok = [row["id"] for row in done.rows
              if row.get("status") == ReplyStatus.OK]
        result: dict[str, Any] = {
            "status": "ok" if ok else "error",
            **({} if ok else _first_refusal(done.rows)),
            "results": done.rows,
            **({"changed": run.changed(done.written)["changed"]} if done.written else {}),
        }
        if not ok or not shows(view):
            return result
        shown = ops_look.show_result(job, ctx, "restore_registration", "overlay",
                                     [record.id for record in spread(
                                         [r for r in (job.state.by_id(i) for i in ok)
                                          if r is not None], job.picture_cap)])
        return pictured(result, shown)

    def submit(summary: str, notes: list[str], interval_breaks: list[str],
               tool_context: Any = None) -> dict[str, Any]:
        done = ops_submit.submit(
            job, summary=summary, notes=as_list(notes),
            interval_breaks=as_list(interval_breaks),
            traces=run.traces_on, workspace=ctx)
        run.box.submission.update({"summary": done.summary, "notes": done.notes,
                                    "interval_breaks": done.interval_breaks})
        if tool_context is not None:
            tool_context.actions.escalate = True
        return {"status": "ok", **run.rows()}

    return {
        "position_sections": position_sections,
        "interactive_transform": interactive_transform, "mark_damage": mark_damage,
        "make_mask": make_mask, "note": note, "undo": undo, "redo": redo,
        "restore_registration": restore_registration, "submit": submit,
    }
