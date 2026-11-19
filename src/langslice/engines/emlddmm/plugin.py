"""emlddmm as LangSlice tools: the adapter, run in LangSlice's process.

It is a tool of every job (its row of the tool table names
``langslice.engines.emlddmm.plugin:tools``). Options (``plugin_config``):
``device`` (``auto``: the GPU when there is one, else the CPU), ``dtype``
(``float64``), ``section_um`` (58.88), ``section_weight`` (``gap`` or a
thickness in um), ``memory_limit_gb`` (24); any other key is refused. emlddmm
(Apache-2.0, vendored as ``langslice._vendor.emlddmm``) runs in LangSlice's
process: ``engine.py`` imports it when a tool runs. Its declaration is its rows of
the tool table (``langslice.plugins.declared``).

Two tools:

- ``emlddmm_stack``: emlddmm's joint fit of the placed sections against the 3-D atlas, one piece
  of background work over the whole stack, on the stack's current plane: each section's in-plane
  rigid motion, the stack's shared scale and shear, and one 3-D warp read in each section's own
  plane (``out_of_plane=False``, ``lock_slice_axis``), landing as one undo step;
- ``emlddmm_plane``: the same fit with the plane free and no warp, proposing the stack's cutting
  angles and spacing, nothing written.

Frames. The stack frame has z along the planes' normal and (row, column) along the displayed
atlas rows and columns, origin at the most anterior section's plane centre. Section k sits at its
distance from that plane, snapped to a slot grid ``slot_um`` apart along the slicing axis (the
agent's choice, no default; emlddmm needs evenly spaced slices, an empty slot carries no data,
``W0`` 0). Each section enters already placed: its section image (by name), grey, read as it is in
[0, 1], area-averaged to ``section_um``, resampled through the
section's current registration (its
deformation included, ``FitFrame.applied_map``) onto the stack's in-plane grid, which is the union
of the sections' footprints. Marked damage (with the native margin) and everything outside the
image get weight 0. The background mean muB is 1 on a white slide (the median of the image
border over 0.5: brightfield, with muA 0) or 0 on a black one (fluorescence). The atlas
image is read from BrainGlobe at 50 um
(``_atlas_images.py``), divided by its mean.

Files, under ``plugins/<tool>/calls/<n>_<tool>/``: ``settings.json`` (asked, chosen, plugin
options, what emlddmm chose, whether its energy record was captured), ``sections.csv`` (the
stack's per-section results; the plane's are in ``A.json``), ``A.json`` (emlddmm's own matrices),
``energy.csv`` and ``energy.png``, ``images/`` (each section as the engine saw it, its final
sorting and the atlas under the fit), ``weights.npz`` and ``log.txt``
(what emlddmm printed). Replies name them; their numbers never go into the
reply.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

from langslice.engines import _blocks
from langslice.engines._sorting import Sorting
from langslice.engines.emlddmm.declarations import (
    emlddmm_plane,
    emlddmm_stack,
    plane_description,
    stack_description,
)
from langslice.engines.emlddmm.files import FileRecord, write_files
from langslice.engines.emlddmm.landing import StackCall, start_stack
from langslice.engines.emlddmm.run import (
    build_request,
    chosen_settings,
    found_plane,
    lattice_of,
    route_of,
    run_engine,
    schedule_of,
)
from langslice.engines.emlddmm.settings import (
    EXAMPLE,
    FITS,
    OVERLAYS,
    UM_PER_MM,
    Options,
    check_iterations,
    check_sorting,
)
from langslice.engines.emlddmm.stack import Ask, build_stack
from langslice.plugin_api import (
    RefusalCode,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
)
from langslice.plugins.declared import engine_tool


def _stack_tool(
    ctx: ToolContext,
    options: Options,
    ask: Ask,
    kernel_um: float,
    fit: str,
    warp_freedom: float,
    iterations: int,
    sorting: dict[str, Any],
    view: bool,
) -> ToolResult:
    if fit not in FITS:
        raise ToolError(RefusalCode.BAD_ARGUMENT, f"fit is one of {', '.join(FITS)}; got {fit!r}")
    rigid, warp = fit in ("transform", "both"), fit in ("warp", "both")
    steps = check_iterations(iterations)
    sort = check_sorting(sorting)
    options = options.resolved()
    stack = build_stack(ctx, options, ask)
    schedule = schedule_of(stack, steps, kernel_um, warp_freedom, rigid=rigid, warp=warp)
    lattices = [lattice_of(placed) for placed in stack.sections]
    request = build_request(
        options,
        ctx.reads.atlas().name,
        stack,
        schedule,
        sorting=Sorting.from_given(sort),
        mode="stack",
        read_warp=warp,
        points=tuple(lattice.points for lattice in lattices),
    )
    settings = {
        "section_image": ask.section_image,
        "atlas_image": ask.atlas_image,
        "mask": ask.mask,
        "kernel_um": kernel_um,
        "fit": fit,
        "slot_um": ask.slot_um,
        "warp_freedom": warp_freedom,
        "iterations": int(iterations),
        "sorting": sort,
    }
    call = StackCall(
        options=options,
        stack=stack,
        request=request,
        lattices=lattices,
        settings=settings,
        asked={"sections": ask.names, **settings, "view": view},
        chosen=chosen_settings(stack, options, schedule),
        routes={
            placed.id: route_of(placed.section.view, rigid=rigid, warp=warp)
            for placed in stack.sections
        },
        digests={
            placed.id: ctx.reads.section_inputs(placed.id, deformation=True)
            for placed in stack.sections
        },
        rigid=rigid,
        warp=warp,
        view=view,
        folder=_blocks.call_folder(ctx, "stack"),
        note=f"emlddmm stack fit, atlas {stack.resolved}, {len(stack.sections)} sections",
    )
    return start_stack(ctx, call)


def _plane_tool(
    ctx: ToolContext,
    options: Options,
    ask: Ask,
    iterations: int,
    view: bool,
) -> ToolResult:
    steps = check_iterations(iterations)
    options = options.resolved()
    stack = build_stack(ctx, options, ask)
    schedule = schedule_of(
        stack, steps, EXAMPLE.kernel_um, EXAMPLE.warp_freedom, rigid=True, warp=False
    )
    folder = _blocks.call_folder(ctx, "plane")
    request = build_request(
        options,
        ctx.reads.atlas().name,
        stack,
        schedule,
        sorting=Sorting(),
        mode="plane",
        read_warp=False,
        points=tuple(np.zeros((0, 2)) for _ in stack.sections),
    )
    result = run_engine(request, folder)
    plane = found_plane(ctx, stack, result)
    secs = stack.sections
    pitch0, yaw0 = (float(v) for v in secs[0].section.view.angles)
    old = np.array([float(p.section.view.position_mm) for p in secs])
    a2d = result.slice_affines[[p.slot for p in secs]]
    turns = np.degrees(np.arctan2(a2d[:, 1, 0], a2d[:, 0, 0]))
    shifts = a2d[:, :2, 2] / UM_PER_MM
    turns, shifts = turns - np.median(turns), shifts - np.median(shifts, axis=0)
    rows = [
        {
            "id": placed.id,
            "inputs": placed.section.inputs,
            "estimated": None,
            "position_mm": round(f["position_mm"], 3),
            "change_mm": round(f["position_mm"] - float(was), 3),
            "residual": {"turn_deg": round(float(t), 2), "shift_mm": round(float(np.hypot(*s)), 3)},
        }
        for placed, f, was, t, s in zip(secs, plane.positions, old, turns, shifts, strict=True)
    ]
    asked = {
        "sections": ask.names,
        "slot_um": ask.slot_um,
        "section_image": ask.section_image,
        "atlas_image": ask.atlas_image,
        "iterations": int(iterations),
        "view": view,
    }
    proposal = {
        "pitch_deg": plane.pitch_deg,
        "yaw_deg": plane.yaw_deg,
        "spacing_scale": plane.scale,
        "spacing_shift_mm": plane.shift_mm,
    }
    write_files(
        stack,
        result,
        folder,
        FileRecord(
            asked,
            chosen_settings(stack, options, schedule),
            options.settings(),
            {"proposed_plane": proposal, "sections": rows},
        ),
    )
    pictures: list[Image.Image] = []
    if view:
        picks = {int(round(v)) for v in np.linspace(0, len(secs) - 1, min(OVERLAYS, len(secs)))}
        for j in sorted(picks):
            found = plane.positions[j]
            caption = (
                f"{rows[j]['id']}: atlas at the plane emlddmm found (pitch {plane.pitch_deg:.2f}, "
                f"yaw {plane.yaw_deg:.2f}), not written"
            )
            overlay = ctx.pictures.overlay_picture(
                str(rows[j]["id"]),
                proposal={
                    "position_mm": found["position_mm"],
                    "angles": (plane.pitch_deg, plane.yaw_deg),
                },
            )
            pictures.append(ctx.pictures.caption(overlay, caption))
    return ToolResult(
        {
            "pitch_deg": round(plane.pitch_deg, 2),
            "yaw_deg": round(plane.yaw_deg, 2),
            "change": {
                "pitch_deg": round(plane.pitch_deg - pitch0, 2),
                "yaw_deg": round(plane.yaw_deg - yaw0, 2),
            },
            "spacing": {"scale": round(plane.scale, 4), "shift_mm": round(plane.shift_mm, 3)},
            "rows": rows,
            "left_out": stack.left_out,
            **({} if result.energy_captured else {"energy": "not captured"}),
            "files": _blocks.job_path(ctx, folder),
        },
        pictures,
    )


def tools(config: Mapping[str, object] | None = None) -> list[ToolSpec]:
    """Return the plugin's tools.

    Parameters
    ----------
    config
        The plugin's entry of ``plugin_config``; None gives the defaults.

    Returns
    -------
    list of ToolSpec
        ``emlddmm_stack`` and ``emlddmm_plane``.

    Raises
    ------
    PluginError
        For an option the plugin does not take.
    """
    options = Options.from_config(config)
    meta = {
        "method": "emlddmm (Tward et al. 2025, Nat Commun)",
        "commit": "9c4dc80 + e426ece + 6ff62b0",
        "engine": "langslice.engines.emlddmm",
    }

    def stack(
        ctx: ToolContext,
        sections: list[str],
        slot_um: float,
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        mask: str = "none",
        kernel_um: float = 500.0,
        fit: str = "both",
        warp_freedom: float = 10000.0,
        iterations: int = 100,
        sorting: dict[str, Any] | None = None,
        view: bool = True,
    ) -> ToolResult:
        ask = Ask(
            list(dict.fromkeys(sections)),
            float(slot_um),
            section_image,
            atlas_image,
            mask,
            need_transform=True,
        )
        return _stack_tool(
            ctx,
            options,
            ask,
            float(kernel_um),
            str(fit),
            float(warp_freedom),
            iterations,
            dict(sorting or {}),
            view,
        )

    def plane(
        ctx: ToolContext,
        sections: list[str],
        slot_um: float,
        section_image: str = "preprocessed",
        atlas_image: str = "nissl",
        iterations: int = 100,
        view: bool = True,
    ) -> ToolResult:
        ask = Ask(
            list(dict.fromkeys(sections)),
            float(slot_um),
            section_image,
            atlas_image,
            "none",
            need_transform=False,
        )
        return _plane_tool(ctx, options, ask, iterations, view)

    return [
        engine_tool(
            "emlddmm_stack",
            emlddmm_stack,
            stack,
            description=stack_description,
            version="1",
            meta=meta,
        ),
        engine_tool(
            "emlddmm_plane",
            emlddmm_plane,
            plane,
            description=plane_description,
            version="1",
            meta=meta,
        ),
    ]


__all__ = ["tools"]
