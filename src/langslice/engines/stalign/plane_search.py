"""The ``stalign_plane_search`` tool's search: each section's plane, tried from a grid of starts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from langslice.engines import _blocks
from langslice.engines.stalign import engine
from langslice.engines.stalign.landing import run_engine, target_to_atlas
from langslice.engines.stalign.prepare import section_frame, section_target
from langslice.engines.stalign.settings import BRIGHTNESS, SAME_LEVEL, STARTS
from langslice.plugin_api import (
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    map_sections,
)


@dataclass(frozen=True)
class _PlaneCall:
    """One ``stalign_plane_search`` call, as every section searches."""

    ctx: ToolContext
    machine: engine.Machine
    folder: Path
    atlas_name: str
    section_image: str
    atlas_image: str
    view: bool


def _search_section(call: _PlaneCall, name: str) -> ToolResult:
    """Search one section's plane: its row and picture; save its record."""
    ctx = call.ctx
    view_of = ctx.reads.section(name)
    sid = view_of.id
    frame, _plane_image, resolved = section_frame(ctx, sid, call.atlas_image, call.section_image)
    inputs = _blocks.inputs_block(
        ctx,
        frame,
        section_image=call.section_image,
        atlas_image=call.atlas_image,
        resolved=resolved,
        mask=None,
        brightness=BRIGHTNESS,
    )
    target = section_target(frame)
    gw, gh = target.size
    um = target.um_per_px
    request = engine.PlaneRequest(
        atlas=call.atlas_name,
        atlas_image=resolved,
        target=target.image,
        target_rows_um=target.rows_um,
        target_cols_um=target.cols_um,
        start=np.linalg.inv(target_to_atlas(frame, um)),
        settings=engine.PlaneSettings(),
        machine=call.machine,
    )
    result = run_engine(partial(engine.run_plane_search, request), call.folder / sid)
    basis = np.linalg.inv(result.affine)
    middle = [(gh - 1) / 2 * um, (gw - 1) / 2 * um]
    centre = basis @ np.array([0.0, *middle, 1.0])
    normal = np.cross(basis[:3, 2], basis[:3, 1])
    plane = ctx.frames.angles_from_plane(normal.tolist(), centre[:3].tolist())
    saved = {"found": plane, "prefit": result.prefit, "device": result.device}
    (call.folder / sid / "plane.json").write_text(
        json.dumps(saved, indent=1, default=float), encoding="utf-8", newline="\n"
    )
    pitch0, yaw0 = (float(v) for v in view_of.angles)
    change = {
        "pitch_deg": round(plane["pitch_deg"] - pitch0, 2),
        "yaw_deg": round(plane["yaw_deg"] - yaw0, 2),
        "position_mm": round(plane["position_mm"] - view_of.position_mm, 3),
    }
    row = {
        "id": sid,
        "status": "ok",
        "inputs": inputs,
        "estimated": None,
        "pitch_deg": round(plane["pitch_deg"], 2),
        "yaw_deg": round(plane["yaw_deg"], 2),
        "position_mm": round(plane["position_mm"], 3),
        "change": change,
        "at_search_edge": _at_search_edge(result.starts, result.best),
    }
    if not call.view:
        return ToolResult(row)
    proposal = {
        "position_mm": plane["position_mm"],
        "angles": (plane["pitch_deg"], plane["yaw_deg"]),
    }
    caption = (
        f"{sid}: atlas at the plane STalign found (pitch "
        f"{row['pitch_deg']}, yaw {row['yaw_deg']}), not written"
    )
    picture = ctx.pictures.overlay_picture(sid, proposal=proposal)
    return ToolResult(row, [ctx.pictures.caption(picture, caption)])


def _at_search_edge(starts: list[list[float]], best: int | None) -> bool:
    """Tell whether the winning start is at the edge of the search's starts.

    Parameters
    ----------
    starts
        Every start's tilts (degrees) and shift (micrometres).
    best
        The winning start's index; None without one.

    Returns
    -------
    bool
        True when one of its tilts or its shift is the search's largest.
    """
    if not starts or best is None:
        return False
    tilt_a, tilt_b, shift = starts[best]
    tilted = (
        abs(abs(tilt_a) - STARTS.tilt_deg) < SAME_LEVEL
        or abs(abs(tilt_b) - STARTS.tilt_deg) < SAME_LEVEL
    )
    return tilted or abs(abs(shift) - STARTS.shift_um) < SAME_LEVEL


def plane_search_tool(
    ctx: ToolContext,
    machine: engine.Machine,
    sections: list[str],
    section_image: str,
    atlas_image: str,
    view: bool,
) -> ToolResult:
    if not sections:
        raise ToolError(RefusalCode.BAD_ARGUMENT, "name one to eight sections")
    call = _PlaneCall(
        ctx,
        machine,
        _blocks.call_folder(ctx, "plane"),
        ctx.reads.atlas().name,
        section_image,
        atlas_image,
        view,
    )
    done = map_sections(partial(_search_section, call), list(sections), machine.threads)
    rows = [one.reply for one in done]
    pictures: list[Image.Image] = []
    for one in done:
        pictures.extend(one.pictures)
    ok = [row for row in rows if row["status"] == "ok"]
    groups = {
        "positive_yaw": [r["id"] for r in ok if r["yaw_deg"] > 0],
        "negative_yaw": [r["id"] for r in ok if r["yaw_deg"] < 0],
        "zero_yaw": [r["id"] for r in ok if r["yaw_deg"] == 0],
    }
    summary = {
        "sections_by_yaw_sign": groups,
        "middle_pitch_deg": round(float(np.median([r["pitch_deg"] for r in ok])), 2)
        if ok
        else None,
        "middle_yaw_magnitude_deg": round(float(np.median(np.abs([r["yaw_deg"] for r in ok]))), 2)
        if ok
        else None,
    }
    return ToolResult(
        {
            "rows": rows,
            "stack": summary,
            "settings": {"section_image": section_image, "atlas_image": atlas_image},
        },
        pictures,
    )
