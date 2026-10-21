"""The tools that look, and the channel tool: the bodies behind their declarations.

``look``, ``zoom``, ``grep_atlas_view`` save their own pictures
(:mod:`langslice.ops.look`): their replies carry ``pictures`` with the
numbers, and the tool door sends the pictures without saving them again.
``set_channel`` returns, for a derived channel, its BEFORE / AFTER
pictures plain; the door saves them and adds their numbers. ``status`` adds
to the status rows what the run holds besides them (the channel settings,
the derived channels' recipes, the background work still running, what this run
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
    if has("set_channel"):
        out.append("channel display and the derived channels' recipes (the preprocessed "
                   "channel among them)")
    if has("set_mask"):
        out.append("the masks fits take")
    added = [name for name, tool in tools_for(spec).items()
             if name not in TOOLS and tool.kind == "write" and tool.is_on(spec)]
    if added:  # tools tool plugins added (a job without plugins has none)
        out.append(f"whatever the plugin tools change ({', '.join(added)})")
    return out


def derived_summary(state: Any) -> dict[str, Any]:
    """Each derived channel's recipe for ``status``: its stack recipe (or ``"default"``)
    and the sections that hold one of their own."""
    out: dict[str, Any] = {}
    for name in looks.derived_names(state):
        out[name] = {"stack": looks.derived_settings(state, name, "") or "default"}
        own = sorted(section for section in (
            (state.appearance.get("derived") or {}).get(name, {}).get("sections") or {}))
        if own:
            out[name]["sections"] = own
    out.setdefault(looks.PREPROCESSED, {"stack": "default"})
    return out


def bodies(door: Door) -> dict[str, Callable[..., Any]]:
    """The looking tools' bodies over *door*."""
    job, ctx = door.job, door.ctx

    def look(mode: str, sections: list[str], channels: list[str],
             positions_mm: list[float] | None = None, atlas_layers: list[str] | None = None,
             atlas_opacity: float = 0.5, warp: str = "applied",
             resolution: int = 0, layers: list[str] | None = None,
             depth: int = OVERLAY_DEPTH, masks: list[str] | None = None) -> dict[str, Any]:
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
                plugin_layers=as_list(layers), depth=depth, masks=as_list(masks))
        except Refused as refusal:
            return refusal.payload()
        result: dict[str, Any] = {"status": "ok"}
        if note:
            result["resolution_note"] = note
        if done.layers:
            result["layers"] = done.layers
        if getattr(done, "masks", None):
            result["masks"] = done.masks
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

    def set_channel(
        name: str, sections: list[str], weights: dict[str, float], clahe_clip: float,
        clahe_tiles: int, n4: bool, denoise: bool, contrast_limits: list[float],
        gamma: float, colormap: str, invert: bool, reset: bool, view: bool = True,
    ) -> dict[str, Any]:
        named = as_list(sections)
        shown: list[Any] = []
        ids: list[str] | None = None
        scope = job.state.in_order()
        if named:
            scope, unknown = door.resolve_many(named)
            if unknown or not scope:
                return {"status": "error", "error": "UNKNOWN_SLICE_IDS",
                        **unknown_sections(job.state, unknown)}
            ids = [record.id for record in scope]
        if view or door.spec.force_view:
            cap = job.picture_cap
            if named:
                shown = scope[:cap] if cap is not None else list(scope)
            elif cap is None:
                shown = list(scope)
            else:
                picks = np.linspace(0, len(scope) - 1, min(len(scope), cap))
                shown = [scope[index] for index in sorted({int(round(v)) for v in picks})]
        options = default_options("section", long_edge=picture_edge(ctx))
        try:
            done = ops_appearance.set_channel(
                job, ctx, str(name), ids, weights=dict(weights or {}),
                clahe_clip=clahe_clip, clahe_tiles=clahe_tiles, n4=bool(n4),
                denoise=bool(denoise), contrast_limits=as_list(contrast_limits) or None,
                gamma=float(gamma) if gamma else None, colormap=str(colormap).strip() or None,
                invert=bool(invert), reset=bool(reset), shown=shown, options=options)
        except Refused as refusal:
            return refusal.payload()
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS", "message": "gamma is a number."}
        if done.kind == "raw":
            return {"status": "ok", "channel": done.name, "properties": done.properties,
                    "changed": done.changed, "sections": done.sections,
                    "intensities": done.intensities, "channels": done.channels}
        parts: list[Media] = []
        for pair in done.pictures:
            record = pair.record
            label = f"{record.id}  {done.name} channel"
            for when, image, settings in (("BEFORE", pair.before, pair.before_settings),
                                          ("AFTER", pair.after, pair.after_settings)):
                parts.append(layers.note(
                    caption(image, f"{label}  {when} ({looks.describe(settings)})"),
                    sections=(record.id,), mode=when.lower(),
                    caption=f"{record.id} {done.name} channel "
                    f"{when.lower()} this call ({looks.describe(settings)})"))
        result: dict[str, Any] = {
            "status": "ok", "channel": done.name, "scope": ids or "stack",
            "recipe": done.in_force, "channels": done.channels,
            "shown": [record.id for record in shown],
        }
        if done.deleted:
            result["deleted"] = True
        result[TOOL_MEDIA_PARTS_KEY] = parts
        return result

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
        result["derived_channels"] = derived_summary(state)
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
        "look": look, "zoom": zoom, "set_channel": set_channel,
        "grep_atlas": grep_atlas, "grep_atlas_view": grep_atlas_view, "status": status,
        "list_files": list_files, "search_files": search_files, "read_file": read_file,
    }

