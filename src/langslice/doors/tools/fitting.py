"""The fit tools, one per registration method, and the scripting verbs: their bodies.

``elastix_affine`` and ``ants_syn`` fit, apply, then show each fitted
section under its new registration with the atlas borders on it, zoomed to
the ``restrict_to`` regions when given (:func:`langslice.ops.look.show_result`),
unless the call's ``view`` is false and the host does not force it.
``trace_borders`` starts background work and answers at once; its landing
draws the trace and the fit, and the tool door hands its notice out at the
head of a later reply.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langslice.core.display import available_atlas_channels, canonical_atlas_name, default_options
from langslice.core.sizes import picture_edge
from langslice.doors.tools.door import Door, as_list, pictured
from langslice.ops import deformable as ops_deformable
from langslice.ops import exports as ops_exports
from langslice.ops import look as ops_look
from langslice.ops import traces as ops_traces
from langslice.ops import transforms as ops_transforms
from langslice.ops.refusal import Refused


def bodies(door: Door) -> dict[str, Callable[..., Any]]:
    """The fit tools' bodies over *door*."""
    job, ctx, spec = door.job, door.ctx, door.spec

    def shows(view: bool) -> bool:
        return bool(view) or bool(spec.force_view)

    def atlas_refusal(atlas_image: str) -> dict[str, Any] | None:
        kind = canonical_atlas_name(str(atlas_image or "template").strip().lower())
        offered = [name for name in available_atlas_channels(ctx) if name != "borders"]
        if kind in ("template", "nissl") and kind not in offered:
            return {"status": "error", "error": "FIT_ATLAS_UNAVAILABLE",
                    "message": f"The {kind} atlas image is offered on Allen mouse atlases "
                    f"only; {job.state.atlas} does not cover the CCFv3 grid.",
                    "atlas_image": offered}
        return None

    def elastix_affine(sections: list[str], restrict_to: list[str], atlas_image: str,
                       view: bool = True) -> dict[str, Any]:
        try:
            regions = ops_deformable.region_entries(ctx, job.state, as_list(restrict_to))
        except Refused as refusal:
            return refusal.payload()
        refusal = atlas_refusal(atlas_image)
        if refusal is not None:
            return refusal
        named = as_list(sections)
        count = len(named) if named else len(ops_transforms.fit_targets(job))
        refusal = door.over_cap(count)
        if refusal is not None:
            return refusal
        try:
            done = ops_transforms.elastix_affine(job, ctx, named, regions,
                                                 str(atlas_image or "template"))
        except Refused as refusal:
            return refusal.payload()
        fits = [row for row in done.rows if row.get("status") == "ok"]
        for row in fits:
            row.pop("params", None)  # the six raw numbers stay host-side
            turn = row.pop("turn_deg", None)
            if turn is not None:
                row["warning"] = (
                    f"This fit turns the section {turn:.0f} degrees from its previous "
                    "transform. The overlap compares shapes, not the anatomy inside, so a "
                    "turned or upside-down section can score as high as a correct one.")
        result: dict[str, Any] = {
            "status": "ok" if fits else "error",
            **({} if fits else {"error": "NOTHING_FITTED"}),
            "results": done.rows,
        }
        if not fits or not shows(view):
            return result
        fitted = [str(row["id"]) for row in fits]
        zooms = {str(row["id"]): row["restrict_box"] for row in fits if row.get("restrict_box")}
        shown = ops_look.show_result(job, ctx, "elastix_affine", "overlay", fitted,
                                     zooms=zooms)
        return pictured(result, shown)

    def ants_syn(sections: list[str], restrict_to: list[str], atlas_image: str,
                 stiffness: str, view: bool = True) -> dict[str, Any]:
        try:
            done = ops_deformable.ants_syn(job, ctx, as_list(sections),
                                           restrict_to=as_list(restrict_to),
                                           atlas_image=atlas_image, stiffness=stiffness)
        except Refused as refusal:
            return refusal.payload()
        rows = done.rows
        ok = [row for row in rows if row.get("status") == "ok"]
        result: dict[str, Any] = {
            "status": "ok" if ok else "error",
            **({} if ok else {"error": "NOTHING_FITTED"}),
            "results": rows,
            **({"changed": door.changed(done.written)["changed"]} if done.written else {}),
        }
        if not ok or not shows(view):
            return result
        drawn = [str(row["id"]) for row in ok]
        zooms: dict[str, Any] = {}
        regions = ops_deformable.region_entries(ctx, job.state, as_list(restrict_to))
        if regions:
            for name in drawn:
                record = job.state.by_id(name)
                box = (ops_transforms.region_box(ctx, job.state, record, regions)
                       if record is not None else None)
                if box is not None:
                    zooms[name] = box
        shown = ops_look.show_result(job, ctx, "ants_syn", "overlay", drawn, zooms=zooms)
        return pictured(result, shown)

    def trace_borders(section: str, prompt: str, restrict_to: list[str]) -> dict[str, Any]:
        assert door.image_model is not None  # the tool exists only when traces are on
        options = default_options("borders", long_edge=picture_edge(ctx))
        try:
            done = ops_traces.trace_borders(
                job, ctx, section, image_model=door.image_model, prompt=str(prompt or ""),
                restrict_to=as_list(restrict_to), options=options)
        except Refused as refusal:
            return refusal.payload()
        trace = {key: done.record[key] for key in (
            "status", "error", "message", "cached", "prompt_edited", "attempt",
            "include", "exclude") if key in done.record}
        if done.running:
            return {"status": "running", "id": done.id, "work": done.work,
                    "message": f"This section's trace is already running as work "
                    f"{done.work}; its notice comes when it finishes."}
        if done.landed:
            return {"status": "ok", "id": done.id, "landed": True, "trace": trace,
                    "message": "This trace has already been fitted: the section's "
                    "deformation holds its fit, so nothing was fitted again."}
        return {"status": "started", "id": done.id, "work": done.work, "trace": trace,
                "message": f"Started in the background as work {done.work}"
                + (" (the saved answer at this placement and region choice is reused)"
                   if not done.started else "")
                + ". Carry on; its notice comes at the head of a later reply, and "
                "submit waits for it."}

    def trace_from_atlas(slices: list[str], passes: int) -> dict[str, Any]:
        assert door.image_model is not None  # the verb exists only when traces are on
        try:
            done = ops_traces.trace_from_atlas(job, ctx, as_list(slices),
                                               image_model=door.image_model, passes=passes)
        except Refused as refusal:
            return refusal.payload()
        failed = [row for row in done.rows if row.get("status") == "error"]
        return {
            "status": "error" if len(failed) == len(done.rows) else "ok",
            **({"error": "NOTHING_TRACED"} if len(failed) == len(done.rows) else {}),
            "results": done.rows,
            "message": "Image calls run in the background; submit waits for them.",
        }

    def export_maps(slices: list[str], full_resolution: bool) -> dict[str, Any]:
        try:
            done = ops_exports.export_maps(job, ctx, as_list(slices),
                                           full_resolution=bool(full_resolution))
        except Refused as refusal:
            return refusal.payload()
        return {
            "status": "ok" if done.sections or not done.skipped else "error",
            **({} if done.sections or not done.skipped else {"error": "NOTHING_EXPORTED"}),
            "written": done.sections, "skipped": done.skipped,
            "full_resolution": done.full_resolution, "files_written": done.written,
            "files": [{"path": path, "kind": kind} for path, kind in done.files],
            "seconds": round(done.seconds, 2),
        }

    return {
        "elastix_affine": elastix_affine, "ants_syn": ants_syn,
        "trace_borders": trace_borders, "trace_from_atlas": trace_from_atlas,
        "export_maps": export_maps,
    }
