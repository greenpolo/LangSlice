"""The fit tools, one per registration method, and the scripting verbs: their bodies.

``elastix_affine`` and ``fireants_syn`` fit, apply, then show each fitted
section under its new registration with the atlas borders on it
(:func:`langslice.tools.look.show_result`), drawn on the section image the fit
read, unless the call's ``view`` is false and the host does not force it. With
``restrict_to`` those regions' borders are drawn thick and the others faint,
and the picture is zoomed to the regions' box when that magnifies it at least
:data:`~langslice.tools.affine_fit.MIN_ZOOM_GAIN` times. ``trace_borders``
starts background work and answers at once; its landing draws the trace and the
fit, and the tool door hands its notice out at the head of a later reply.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langslice.core.atlas.images import resolve_image
from langslice.core.fits.flags import FitFlag
from langslice.core.pictures.options import default_options
from langslice.core.pictures.sizes import (
    AUTO_RESOLUTION,
    clamp_resolution,
    picture_edge,
)
from langslice.core.refusal import RefusalCode, Refused, ReplyStatus
from langslice.tools import affine_fit
from langslice.tools import deformable as ops_deformable
from langslice.tools import exports as ops_exports
from langslice.tools import landmarks as ops_landmarks
from langslice.tools import look as ops_look
from langslice.tools import traces as ops_traces
from langslice.tools.catalog import check_section_count
from langslice.tools.run import Run, as_list, drawn_image, pictured


def nothing_fitted(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The top-level error of a fit that fitted nothing. When every section was
    refused for the same reason, that reason's code and sentence lead (the
    section rows keep theirs); otherwise ``NOTHING_FITTED``, each row naming its own."""
    codes = {str(row.get("error")) for row in rows if row.get("error")}
    if rows and len(codes) == 1 and all(row.get("error") for row in rows):
        code = codes.pop()
        message = str(rows[0].get("message") or "")
        if message and len(rows) > 1:
            message = f"Every section was refused for this reason. {message}"
        return {"error": code, **({"message": message} if message else {})}
    return {"error": RefusalCode.NOTHING_FITTED}


def affine_flag_words(flag: dict[str, Any]) -> str:
    """One ``elastix_affine`` plausibility flag
    (:func:`langslice.tools.affine_fit.plausibility_flags`) as a sentence."""
    if flag.get("code") == FitFlag.SCALE_CHANGED:
        scale, held = flag.get("scale") or [], flag.get("previous") or []
        return (f"SCALE_CHANGED: this fit scales the section to {scale} from {held}, more "
                "than a section shrinks or swells; it may have stretched the section over "
                "atlas it does not hold. Compare its picture with the previous one.")
    if flag.get("code") == FitFlag.SHEAR:
        return (f"SHEAR: this fit shears the section by {flag.get('shear')} (it held "
                f"{flag.get('previous')}); a section is rarely sheared that much. Compare its "
                "picture with the previous one.")
    return str(flag.get("code"))


def bodies(run: Run) -> dict[str, Callable[..., Any]]:
    """The fit tools' bodies over *run*."""
    job, ctx, spec = run.job, run.ctx, run.spec

    def shows(view: bool) -> bool:
        return bool(view) or bool(spec.force_view)

    def check_atlas_image(atlas_image: str) -> None:
        kind = str(atlas_image or "nissl").strip()
        try:
            resolve_image(ctx.atlas, kind)
        except ValueError as exc:
            raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from None

    def elastix_affine(sections: list[str], restrict_to: list[str],
                       section_image: str = "preprocessed", atlas_image: str = "nissl",
                       mask: str = "none", edges: bool = True,
                       view: bool = True) -> dict[str, Any]:
        regions = ops_deformable.region_entries(ctx, job.state, as_list(restrict_to))
        check_atlas_image(atlas_image)
        named = as_list(sections)
        if not named:
            raise Refused(RefusalCode.BAD_ARGS,
                          "sections must name one or more sections")
        check_section_count(job, "elastix_affine", len(named))
        done = affine_fit.elastix_affine(
            job, ctx, named, regions, str(section_image or "preprocessed"),
            str(atlas_image or "nissl"), str(mask or "none"), bool(edges))
        fits = [row for row in done.rows
                if row.get("status") == ReplyStatus.OK]
        for row in fits:
            row.pop("params", None)  # the six raw numbers stay host-side
            # the knobs under the one name every reply gives them
            row["transform"] = row.pop("physical", None)
            turn = row.pop("turn_deg", None)
            words = []
            if turn is not None:
                words.append(f"This fit turns the section {turn:.0f} degrees from its "
                             "previous transform.")
            words += [affine_flag_words(flag) for flag in row.get("flags") or []]
            if words:
                row["warning"] = " ".join(words)
        result: dict[str, Any] = {
            "status": "ok" if fits else "error",
            **({} if fits else nothing_fitted(done.rows)),
            "settings": {
                "section_image": str(section_image or "preprocessed"),
                "atlas_image": str(atlas_image or "nissl"), "mask": str(mask or "none"),
                "restrict_to": list(regions), "edges": bool(edges)},
            "results": done.rows,
        }
        if not fits or not shows(view):
            return result
        fitted = [str(row["id"]) for row in fits]
        # A deformation this fit made stale goes before the picture is drawn.
        cleared = job.clear_stale_deformations()
        if cleared:
            result["deformation_cleared"] = cleared
        zooms = {str(row["id"]): window for row in fits
                 if (window := affine_fit.zoom_window(row.get("restrict_box")))}
        shown = ops_look.show_result(job, ctx, "elastix_affine", "overlay", fitted,
                                     zooms=zooms, highlight=regions,
                                     images=drawn_image(section_image))
        return pictured(result, shown)

    def fireants_syn(sections: list[str], restrict_to: list[str],
                     section_image: str = "preprocessed", atlas_image: str = "nissl",
                     mask: str = "none", smoothness_um: float = 80.0,
                     metric: str = "mutual_information", edges: bool = True,
                     detail: str = "standard", view: bool = True) -> dict[str, Any]:
        check_section_count(job, "fireants_syn", len(as_list(sections)))
        done = ops_deformable.fireants_syn(
            job, ctx, as_list(sections), restrict_to=as_list(restrict_to),
            section_image=section_image, atlas_image=atlas_image, mask=mask,
            smoothness_um=smoothness_um, metric=metric, edges=edges, detail=detail)
        rows = done.rows
        ok = [row for row in rows if row.get("status") == ReplyStatus.OK]
        result: dict[str, Any] = {
            "status": "ok" if ok else "error",
            **({} if ok else nothing_fitted(rows)),
            "settings": done.settings,
            "results": rows,
            **({"changed": run.changed(done.written)["changed"]} if done.written else {}),
        }
        if not ok or not shows(view):
            return result
        drawn = [str(row["id"]) for row in ok]
        zooms: dict[str, Any] = {}
        regions = ops_deformable.region_entries(ctx, job.state, as_list(restrict_to))
        if regions:
            for name in drawn:
                record = job.state.by_id(name)
                box = (affine_fit.region_box(ctx, job.state, record, regions)
                       if record is not None else None)
                window = affine_fit.zoom_window(box)
                if window is not None:
                    zooms[name] = window
        shown = ops_look.show_result(job, ctx, "fireants_syn", "overlay", drawn, zooms=zooms,
                                     highlight=regions, images=drawn_image(section_image))
        return pictured(result, shown)

    def trace_borders(section: str, prompt: str, restrict_to: list[str]) -> dict[str, Any]:
        assert run.image_model is not None  # the tool exists only when traces are on
        options = default_options("borders", long_edge=picture_edge(ctx))
        done = ops_traces.trace_borders(
            job, ctx, section, image_model=run.image_model, prompt=str(prompt or ""),
            restrict_to=as_list(restrict_to), options=options)
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

    def export_maps(sections: list[str], full_resolution: bool) -> dict[str, Any]:
        done = ops_exports.export_maps(job, ctx, as_list(sections),
                                       full_resolution=bool(full_resolution))
        return {
            "status": "ok" if done.sections or not done.skipped else "error",
            **({} if done.sections or not done.skipped
               else {"error": RefusalCode.NOTHING_EXPORTED}),
            "written": done.sections, "skipped": done.skipped,
            "full_resolution": done.full_resolution, "files_written": done.written,
            "files": [{"path": path, "kind": kind} for path, kind in done.files],
            "seconds": round(done.seconds, 2),
        }

    def set_landmarks(section: str, pairs: list[dict[str, Any]], delete: list[str],
                      view: bool = True) -> dict[str, Any]:
        if shows(view) and len(pairs) > 4:
            raise Refused(RefusalCode.TOO_MANY_LANDMARK_VIEWS,
                          f"You sent {len(pairs)} pairs with view true, and a call draws one "
                               "picture per pair for at most four. Nothing was written. Send "
                               f"the same {len(pairs)} pairs again with view false (no pair "
                               "pictures; the pairs are listed in the reply), or send them in "
                               "calls of four or fewer.")
        result = ops_landmarks.set_landmarks(job, ctx, section, pairs, delete)
        if not shows(view) or not result["changed"]:
            return result
        from langslice.core.pictures.drawn import collecting
        from langslice.core.pictures.landmarks import pair_picture

        record = job.state.by_id(result["section"])
        assert record is not None
        full_pairs = () if record.landmarks is None else record.landmarks.pairs
        try:
            with collecting() as notes:
                pictures = [pair_picture(ctx, job.state, record, pair, store=job.deformations)
                            for pair in full_pairs if pair.id in result["changed"]]
            entries = ops_look.save_pictures(job, ctx, "set_landmarks", pictures, notes, {})
            return pictured(result, ops_look.Looked(
                pictures=[p.image for p in pictures], entries=entries))
        except Exception as exc:  # noqa: BLE001 - the write stands
            result["render_failed"] = str(exc)
            return result

    def landmark_triangulation(section: str, view: bool = True,
                               resolution: int = 0) -> dict[str, Any]:
        edge: int | None = None
        if resolution and run.level == AUTO_RESOLUTION:
            try:
                edge, note = clamp_resolution(resolution, run.box.max_view_edge)
            except (TypeError, ValueError):
                raise Refused(
                    RefusalCode.BAD_ARGS,
                    "resolution is a whole number of pixels.") from None
        else:
            note = ""
        result = ops_landmarks.landmark_triangulation(job, ctx, section)
        if note:
            result["resolution_note"] = note
        if shows(view):
            shown = ops_look.show_result(job, ctx, "landmark_triangulation",
                                         "overlay", [result["section"]], resolution=edge)
            return pictured(result, shown)
        return result

    return {
        "set_landmarks": set_landmarks, "landmark_triangulation": landmark_triangulation,
        "elastix_affine": elastix_affine, "fireants_syn": fireants_syn,
        "trace_borders": trace_borders,
        "export_maps": export_maps,
    }
