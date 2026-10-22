"""The tools that look, and the image tool: the bodies behind their declarations.

``look``, ``zoom``, ``grep_atlas_view`` and ``make_image`` save their own
pictures (:mod:`langslice.ops.look`): their replies carry ``pictures`` with
the numbers, and the tool door sends the pictures without saving them again.
``status`` adds to the status rows what the run holds besides them (the raw
channels, the made section images, the background work still running, what
this run lets the agent change). The job-folder tools answer with their
``text``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langslice.core import section_images
from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.sizes import AUTO_RESOLUTION
from langslice.core.state import unknown_sections
from langslice.core.toolset import TOOLS, tools_for
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.door import Door, as_list, pictured, spread
from langslice.doors.tools.view_options import clamp_resolution
from langslice.ops import atlas as ops_atlas
from langslice.ops import files as ops_files
from langslice.ops import images as ops_images
from langslice.ops import look as ops_look
from langslice.ops.refusal import Refused


def can_change(door: Door) -> list[str]:
    """What this run lets the agent change, in words (``status``)."""
    spec = door.spec

    def has(name: str) -> bool:
        return TOOLS[name].is_on(spec)

    frozen = spec.frozen_everywhere
    out: list[str] = []
    if spec.has("position") and "position" not in frozen:
        out.append("positions (the order follows them)")
    if has("position_sections") and "angles" not in frozen:
        out.append("cutting angles")
    if spec.has("transform"):
        if has("interactive_transform"):
            by_hand = []
            if "orientation" not in frozen:
                by_hand.append("orientation" + ("" if spec.off("interactive_transform:flip")
                                                else " and mirroring"))
            if "affine" not in frozen:
                by_hand.append("in-plane transforms")
            out.append(" and ".join(by_hand) + " by hand")
        if has("elastix_affine"):
            out.append("in-plane transforms by elastix_affine")
    if has("mark_damage"):
        out.append("damaged regions")
    if spec.has("nonlinear"):
        fits = [name for name in ("fireants_syn", "trace_borders")
                if has(name) and (name != "trace_borders" or door.traces_on)]
        if has("set_landmarks") and has("landmark_triangulation"):
            fits.append("landmark pairs by set_landmarks, applied by landmark_triangulation")
        out.append("deformations" + (f" ({', '.join(fits)})" if fits else ""))
    if has("make_image"):
        out.append("the section images the pictures and fits read (make_image)")
    if has("make_mask"):
        out.append("masks for the fits")
    added = [name for name, tool in tools_for(spec).items()
             if name not in TOOLS and tool.kind == "write" and tool.is_on(spec)]
    if added:  # tools tool plugins added (a job without plugins has none)
        out.append(f"whatever the plugin tools change ({', '.join(added)})")
    return out


def raw_channels(door: Door) -> Any:
    """The raw channels for ``status``: one list when every section shares
    them, else ``{filename: [names]}``."""
    named = {record.id: section_images.raw_names(door.ctx, record.id)
             for record in door.job.state.in_order()}
    distinct = {tuple(names) for names in named.values()}
    return list(next(iter(distinct))) if len(distinct) == 1 else named


def bodies(door: Door) -> dict[str, Callable[..., Any]]:
    """The looking tools' bodies over *door*."""
    job, ctx = door.job, door.ctx

    def look(mode: str, sections: list[str], images: list[str],
             positions_mm: list[float] | None = None, atlas_layers: list[str] | None = None,
             atlas_opacity: float = 0.5, warp: str = "applied",
             resolution: int = 0, layers: list[str] | None = None,
             depth: int = OVERLAY_DEPTH, masks: list[str] | None = None,
             registrations: dict[str, int] | None = None,
             sheet: bool = False) -> dict[str, Any]:
        note = ""
        edge: int | None = None
        if resolution and door.level == AUTO_RESOLUTION:
            try:
                edge, note = clamp_resolution(resolution, door.box.max_view_edge)
            except (TypeError, ValueError):
                return {"status": "error", "error": "BAD_ARGS",
                        "message": "resolution is a whole number of pixels."}
        layers_asked = as_list(atlas_layers)
        try:
            done = ops_look.look(
                job, ctx, str(mode or "").strip().lower(), sections=as_list(sections),
                positions_mm=as_list(positions_mm), images=as_list(images),
                atlas_layers=layers_asked or None, atlas_opacity=atlas_opacity,
                warp=str(warp or "applied").strip().lower(), resolution=edge,
                plugin_layers=as_list(layers), depth=depth, masks=as_list(masks),
                registrations=dict(registrations or {}),
                sheet_edge=int(door.box.max_view_edge) if sheet else None)
        except Refused as refusal:
            return refusal.payload()
        result: dict[str, Any] = {"status": "ok"}
        if note:
            result["resolution_note"] = note
        if done.layers:
            result["layers"] = done.layers
        if getattr(done, "masks", None):
            result["masks"] = done.masks
        result = pictured(result, done)
        if done.sheets:  # the sheets are attached; the pictures they tile are saved
            result["tiled"] = result["pictures"]
            result["pictures"] = [{"id": made["id"], "caption": made["caption"],
                                   "tiles": made["tiles"]} for made in done.sheets]
        return result

    def zoom(box: list[float], picture: int, region: str = "") -> dict[str, Any]:
        try:
            done = ops_look.zoom(job, ctx, as_list(box), picture=int(picture or 0) or None,
                                 region=region)
        except Refused as refusal:
            return refusal.payload()
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "picture is a picture number."}
        result: dict[str, Any] = {"status": "ok", "picture": done.picture,
                                  "redrawn": done.redrawn,
                                  **({"sheet": done.sheet, "box": done.box}
                                     if done.sheet is not None else {})}
        if done.stale:
            result["stale"] = True
        result["pictures"] = list(done.entries)
        result[TOOL_MEDIA_PARTS_KEY] = list(done.pictures)
        return result

    def make_image(
        name: str, sections: list[str], weights: dict[str, float], clahe_clip: float,
        clahe_tiles: int, n4: bool, denoise: bool, contrast: list[float], gamma: float,
        invert: bool, delete: bool, view: bool = True,
    ) -> dict[str, Any]:
        named = as_list(sections)
        ids: list[str] | None = None
        if named:
            scope, unknown = door.resolve_many(named)
            if unknown or not scope:
                return {"status": "error", "error": "UNKNOWN_SLICE_IDS",
                        **unknown_sections(job.state, unknown)}
            ids = [record.id for record in scope]
        try:
            done = ops_images.make_image(
                job, ctx, str(name), ids, weights=dict(weights or {}),
                clahe_clip=clahe_clip, clahe_tiles=clahe_tiles, n4=bool(n4),
                denoise=bool(denoise), contrast=as_list(contrast) or None,
                gamma=float(gamma) if gamma else None, invert=bool(invert),
                delete=bool(delete))
        except Refused as refusal:
            return refusal.payload()
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS", "message": "gamma is a number."}
        result: dict[str, Any] = {
            "status": "ok", "image": done.name, "sections": ids or "every section",
            **({"deleted": True} if done.deleted else {"recipe": done.in_force}),
            "section_images": done.images,
        }
        if done.deleted or door.spec.off("look:section") or not (
                view or door.spec.force_view):
            return result
        records = [record for record in (job.state.by_id(i) for i in done.sections)
                   if record is not None]
        shown = spread(records, job.picture_cap)
        pictures = ops_look.show_result(job, ctx, "make_image", "section",
                                        [record.id for record in shown], images=[done.name])
        return pictured(result, pictures)

    def grep_atlas(query: str, section: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            return {"status": "ok", **ops_atlas.grep_atlas(job, ctx, query, section,
                                                          as_list(points))}
        except Refused as refusal:
            return refusal.payload()

    def grep_atlas_view(regions: list[str], positions_mm: list[float]) -> dict[str, Any]:
        try:
            done = ops_atlas.grep_atlas_view(job, ctx, as_list(regions), as_list(positions_mm))
        except Refused as refusal:
            return refusal.payload()
        result: dict[str, Any] = {"status": "ok", "regions": done.regions}
        if done.regions_not_in_plane:
            result["regions_not_in_plane"] = done.regions_not_in_plane
        if done.clamped:
            result["clamped"] = [{"asked_mm": round(asked, 3), "used_mm": round(used, 3)}
                                 for asked, used in done.clamped]
        result["pictures"] = list(done.entries)
        if done.not_shown:
            result["not_shown"] = list(done.not_shown)
        result[TOOL_MEDIA_PARTS_KEY] = list(done.pictures)
        return result

    def status() -> dict[str, Any]:
        state = job.state
        result: dict[str, Any] = {"status": "ok", **door.rows()}
        result["raw_channels"] = raw_channels(door)
        result["section_images"] = section_images.summary(state, door.spec)
        running = [work.summary() for work in job.background.running()]
        if running:
            result["background_running"] = running
        result["can_change"] = can_change(door)
        return result

    def list_files(path: str, glob: str) -> dict[str, Any]:
        job.views.flush()  # every picture shown so far is on disk and indexed
        try:
            done = ops_files.list_files(job, str(path or "."), str(glob or ""))
        except Refused as refusal:
            return refusal.payload()
        return {"status": "ok", "text": done.text}

    def search_files(query: str, path: str, glob: str) -> dict[str, Any]:
        job.views.flush()
        try:
            done = ops_files.search_files(job, str(query or ""), str(path or "."),
                                          str(glob or ""))
        except Refused as refusal:
            return refusal.payload()
        return {"status": "ok", "text": done.text}

    def read_file(path: str, offset: int, limit: int) -> dict[str, Any]:
        job.views.flush()
        try:
            done = ops_files.read_file(job, str(path or ""), int(offset or 0),
                                       int(limit or 400))
        except Refused as refusal:
            return refusal.payload()
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "offset and limit are whole numbers."}
        result: dict[str, Any] = {"status": "ok", "text": done.text}
        if done.picture is not None:
            result["picture"] = done.picture
        return result

    return {
        "look": look, "zoom": zoom, "make_image": make_image,
        "grep_atlas": grep_atlas, "grep_atlas_view": grep_atlas_view, "status": status,
        "list_files": list_files, "search_files": search_files, "read_file": read_file,
    }

