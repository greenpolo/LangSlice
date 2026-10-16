"""The tools that look, and the two channel tools: the bodies behind their declarations.

``look``, ``zoom``, ``grep_atlas_view`` save their own pictures
(:mod:`langslice.ops.look`): their replies carry ``pictures`` with the
numbers, and the tool door sends the pictures without saving them again.
``set_preprocessed_channel_properties`` returns its BEFORE / AFTER
pictures plain; the door saves them and adds their numbers. ``status`` adds
to the status rows what the run holds besides them (the channel settings,
the preprocessed recipe, the background work still running, what this run
lets the agent change). The job-folder tools answer with their ``text``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from langslice.core import appearance as looks
from langslice.core import layers
from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.captions import caption
from langslice.core.channels import stack_display
from langslice.core.display import default_options
from langslice.core.sizes import AUTO_RESOLUTION, picture_edge
from langslice.core.state import unknown_sections
from langslice.core.toolset import TOOLS, tools_for
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.door import Door, Media, as_list, pictured
from langslice.doors.tools.view_options import clamp_resolution
from langslice.ops import appearance as ops_appearance
from langslice.ops import atlas as ops_atlas
from langslice.ops import files as ops_files
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
    channels = [words for tool, words in (("set_channel_properties", "channel display"),
                                          ("set_preprocessed_channel_properties",
                                           "the preprocessed channel")) if has(tool)]
    if channels:
        out.append(" and ".join(channels))
    added = [name for name, tool in tools_for(spec).items()
             if name not in TOOLS and tool.kind == "write" and tool.is_on(spec)]
    if added:  # tools tool plugins added (a job without plugins has none)
        out.append(f"whatever the plugin tools change ({', '.join(added)})")
    return out


def bodies(door: Door) -> dict[str, Callable[..., Any]]:
    """The looking tools' bodies over *door*."""
    job, ctx = door.job, door.ctx

    def look(mode: str, sections: list[str], channels: list[str],
             positions_mm: list[float] | None = None, atlas_layers: list[str] | None = None,
             atlas_opacity: float = 0.5, warp: str = "applied",
             resolution: int = 0, layers: list[str] | None = None,
             depth: int = OVERLAY_DEPTH) -> dict[str, Any]:
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
                positions_mm=as_list(positions_mm), channels=as_list(channels),
                atlas_layers=layers_asked or None, atlas_opacity=atlas_opacity,
                warp=str(warp or "applied").strip().lower(), resolution=edge,
                plugin_layers=as_list(layers), depth=depth)
        except Refused as refusal:
            return refusal.payload()
        result: dict[str, Any] = {"status": "ok"}
        if note:
            result["resolution_note"] = note
        if done.layers:
            result["layers"] = done.layers
        return pictured(result, done)

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
                                  "redrawn": done.redrawn}
        if done.stale:
            result["stale"] = True
        result["pictures"] = list(done.entries)
        result[TOOL_MEDIA_PARTS_KEY] = list(done.pictures)
        return result

    def set_channel_properties(channel: str, contrast_limits: list[float], gamma: float,
                               colormap: str, reset: bool) -> dict[str, Any]:
        limits = as_list(contrast_limits)
        try:
            done = ops_appearance.set_channel_properties(
                job, ctx, str(channel), contrast_limits=limits or None,
                gamma=float(gamma) if gamma else None,
                colormap=str(colormap).strip() or None, reset=bool(reset))
        except Refused as refusal:
            return refusal.payload()
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS", "message": "gamma is a number."}
        return {"status": "ok", "channel": done.channel, "properties": done.properties,
                "changed": done.changed, "sections": done.sections,
                "intensities": done.intensities}

    def set_preprocessed_channel_properties(
        sections: list[str], channel_weights: list[float], clahe_clip: float,
        clahe_tiles: int, n4: bool, denoise: bool, reset: bool,
    ) -> dict[str, Any]:
        named = as_list(sections)
        if named:
            scope, unknown = door.resolve_many(named)
            if unknown or not scope:
                return {"status": "error", "error": "UNKNOWN_SLICE_IDS",
                        **unknown_sections(job.state, unknown)}
            shown = scope[:job.picture_cap]
            ids: list[str] | None = [record.id for record in scope]
        else:
            scope = job.state.in_order()
            cap = job.picture_cap
            if cap is None:
                shown = list(scope)
            else:
                picks = np.linspace(0, len(scope) - 1, min(len(scope), cap))
                shown = [scope[index] for index in sorted({int(round(v)) for v in picks})]
            ids = None
        options = default_options("section", long_edge=picture_edge(ctx))
        try:
            done = ops_appearance.set_preprocessed(
                job, ctx, ids, channel_weights=as_list(channel_weights) or None,
                clahe_clip=clahe_clip, clahe_tiles=clahe_tiles, n4=bool(n4),
                denoise=bool(denoise), reset=bool(reset), shown=shown, options=options)
        except Refused as refusal:
            return refusal.payload()
        parts: list[Media] = []
        for pair in done.pictures:
            record = pair.record
            label = f"{record.id}  preprocessed channel"
            for when, image, settings in (("BEFORE", pair.before, pair.before_settings),
                                          ("AFTER", pair.after, pair.after_settings)):
                parts.append(layers.note(
                    caption(image, f"{label}  {when} ({looks.describe(settings)})"),
                    sections=(record.id,), mode=when.lower(),
                    caption=f"{record.id} preprocessed channel "
                    f"{when.lower()} this call ({looks.describe(settings)})"))
        names = {record.id: list(ctx.section_channels(record.id)[0]) for record in scope}
        distinct = {tuple(value) for value in names.values()}
        return {
            "status": "ok",
            "scope": ids or "stack",
            "recipe": done.written.in_force,
            "channels": list(next(iter(distinct))) if len(distinct) == 1 else names,
            "shown": [record.id for record in shown],
            TOOL_MEDIA_PARTS_KEY: parts,
        }

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
        result["channel_display"] = stack_display(ctx, state)
        recipe = state.appearance.get(looks.PREPROCESSED)
        result["preprocessed_recipe"] = recipe if recipe else "default"
        running = [work.summary() for work in job.background.running()]
        if running:
            result["background_running"] = running
        result["can_change"] = can_change(door)
        return result

    def list_files(path: str, pattern: str) -> dict[str, Any]:
        job.views.flush()  # every picture shown so far is on disk and indexed
        try:
            done = ops_files.list_files(job, str(path or "."), str(pattern or ""))
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
        "look": look, "zoom": zoom, "set_channel_properties": set_channel_properties,
        "set_preprocessed_channel_properties": set_preprocessed_channel_properties,
        "grep_atlas": grep_atlas, "grep_atlas_view": grep_atlas_view, "status": status,
        "list_files": list_files, "search_files": search_files, "read_file": read_file,
    }

