"""emlddmm as LangSlice tools: the adapter, run in LangSlice's process.

Load it with ``--tool-plugin langslice.engines.emlddmm.plugin:tools`` or
``create_job(..., plugins=["langslice.engines.emlddmm.plugin:tools"])``;
it is imported from this package (no ``PYTHONPATH``). Options (``plugin_config``): ``python`` (the
``rigid_emlddmm`` interpreter), ``emlddmm_src`` (the fork's folder), ``brainglobe_dir`` (where the
atlases are installed), ``device`` (``auto``: the GPU when there is one, else the CPU),
``dtype`` (``float64``), ``section_um`` (58.88),
``section_weight``
(``gap`` or a thickness in um), ``memory_limit_gb`` (24), ``threads``,
``timeout_s``. emlddmm (Apache-2.0) runs in ``engine.py`` under that interpreter; this module
imports only ``langslice.plugin_api``.

Two tools (design: SliceBench/_local/notes/2026-10-19_comparator_plugin_review/emlddmm_proposal.md):

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
[0, 1] (as the Fig 3 benchmark runner), area-averaged to ``section_um``, resampled through the
section's current registration (its
deformation included, ``FitFrame.applied_map``) onto the stack's in-plane grid, which is the union
of the sections' footprints. Marked damage (with the native margin) and everything outside the
image get weight 0. The background mean muB is 1 on a white slide (the median of the image
border over 0.5: brightfield, with muA 0) or 0 on a black one (fluorescence). The atlas
image is read in the engine at 50 um,
divided by its mean.

Files, under ``plugins/<tool>/calls/<n>_<tool>/``: ``settings.json`` (asked, chosen, plugin
options, what emlddmm chose, whether its energy record was captured), ``sections.csv`` (the
stack's per-section results; the plane's are in ``A.json``), ``A.json`` (emlddmm's own matrices),
``energy.csv`` and ``energy.png``, ``images/`` (each section as the engine saw it, its final
sorting and the atlas under the fit), ``weights.npz``, ``in.npz`` and ``out.npz`` (the engine's
input and output) and ``log.txt``. Replies name them; their numbers never go into the reply.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from langslice.engines import _blocks
from langslice.plugin_api import ToolContext, ToolError, ToolResult, ToolSpec, flag_words

HERE = Path(__file__).resolve().parent
NAME = "emlddmm"
#: Where SliceBench keeps the vendored sources the engines read by path.
COMPARATOR_SRC = Path("~/Software/SliceBench/comparators/src").expanduser()
REF = "langslice.engines.emlddmm.plugin:tools"
DEFAULTS = {
    "python": str(Path("~/.claude-science/conda/envs/rigid_emlddmm/bin/python").expanduser()),
    "emlddmm_src": str(COMPARATOR_SRC / "emlddmm_fix"),
    "brainglobe_dir": str(Path("~/.brainglobe").expanduser()),
    "device": "auto",
    "dtype": "float64",
    "section_um": 58.88,  # the published mouse serial-section example's pixel
    "section_weight": "gap",
    "memory_limit_gb": 24.0,
    "threads": 4,
    "timeout_s": 3600,
}
#: GB of GPU memory per million voxels (slots x rows x columns) in float64, measured on the
#: 2026-10-18 Fig 3 runs (33 M voxels: 12.5 GB; 97 M: more than 27 GB).
GB_PER_MVOX = 0.4
#: The published example's slice thickness, um.
EXAMPLE_SLICE_UM = 20.0
#: examples/mouse_serial_section_example/atlas_to_target_config.json (supplement 7.2.4).
PUBLISHED = {"iterations": [100, 50, 25], "kernel_um": 500.0, "warp_freedom": 10000.0}
SORTING_KEYS = ("match_sd", "background_sd", "artifact_sd", "background_mean", "artifact_mean")
#: What a fit fits (``fit``), as every engine tool names it: the transforms (each
#: section's turn and shift and the stack's scale and shear), the 3-D warp, or both.
FITS = ("transform", "warp", "both")
#: emlddmm's sources as LangSlice runs them (the fork's commits).
ENGINE_VERSION = "emlddmm 9c4dc80 + e426ece + 6ff62b0"


def scale_iterations(iterations: int) -> list[int]:
    """emlddmm's steps at its three scales (200, 100, 50 um) from the first
    scale's: that, half of it and a quarter, as the published example's
    [100, 50, 25] runs."""
    first = int(iterations)
    return [first, max(1, first // 2), max(1, first // 4)]
#: Sections whose planes differ by less than this (an imported placement's rounding) are one stack.
ANGLE_TOLERANCE_DEG = 1.0
LATTICE_PX = 8  # the map is read every 8 fit-grid pixels and interpolated (the warp is smooth)
NO_WARP = 10 ** 6  # emlddmm's v_start past the last iteration: the warp never starts
_DEVICES: dict[str, str] = {}  # the engine interpreter -> its device when asked for ``auto``
OVERLAYS = 4  # the pictures a landing or a plane proposal sends


# --- the agent-facing declarations ----------------------------------------------------------


def emlddmm_stack(
    sections: list[str],
    slot_um: float,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    mask: str = "none",
    kernel_um: float = 500.0,
    fit: Literal["transform", "warp", "both"] = "both",
    warp_freedom: float = 10000.0,
    iterations: int = 100,
    sorting: dict[str, Any] = {},  # noqa: B006 (a declaration's default, as the native stubs)
    view: bool = True,
) -> dict[str, Any]:
    """Fit the whole stack to the atlas at once with emlddmm, on its current plane.

    emlddmm (Tward et al. 2025, Nature Communications) maps whole-brain datasets
    of any modality into the Allen CCF. It models every section as an image of one
    deformed 3-D atlas, with an EM sorting of missing tissue and artifacts and a
    contrast fit in small blocks.

    Use this when you have most of a brain's series, closely and evenly
    spaced, and want placements that agree across the stack and fit the
    brain's overall shape. It bends the brain as a whole, never one section
    alone[[?fireants_syn|landmark_triangulation|trace_borders ; for a local ~
    mismatch on one section (a ventricle, a displaced piece) use a per-section ~
    warp]].

    emlddmm explains every placed section as a picture of the same 3-D atlas:
    one smooth 3-D warp of the atlas, one shared scale and shear for the stack,
    and for each section its own in-plane turn and shift (its mounting). It
    fits all of them together against the sections' images (one section
    image, by name), with a contrast fit in small blocks and an EM sorting of every
    pixel into matching tissue, background and artifact, so holes, debris and
    empty slide are set aside. Neighbouring sections constrain each other: a
    section with little anatomy is bent and shifted the way the stack is. The
    cutting angles and positions stay as they are: put the sections in order
    and within about one interval of their place first. emlddmm has no
    centring step and cannot recover from a start 1 to 3 mm off the tissue,
    so have the atlas outline over each section's tissue, placed and
    roughly aligned, before calling; each section is
    refined from its current registration, its deformation included[[?set_landmarks ~
    , so a landmark move is kept]]. The section's marked damage regions are
    left out automatically[[?mark_damage ; mark them with mark_damage first]].

    emlddmm takes a section mask as soft weights: the mask multiplies each
    pixel's weight in the fit, so a pixel at 0.3 counts at 0.3. Besides the
    mask, emlddmm estimates its own sorting during the fit; the final sorting
    of each section it writes a deformation for is saved with that
    deformation[[?make_mask , and make_mask can take it as a mask for another fit ~
    (source fit)]]. It keeps its own brightness scaling: the section image as it
    is, 0 to 1.

    Writes, per section: with fit "transform" or "both" its transform (turn,
    shift, and the stack's scale and shear) and with fit "warp" or "both" its
    deformation, the section's cut of the 3-D warp. A section that holds a deformation or a
    frozen transform keeps its transform and gets the rigid change folded
    into its deformation. A section whose result cannot be written still
    shapes the stack. The whole landing is one undo step.

    The sections are put on a grid of slots slot_um apart along the slicing
    axis, each moved to its nearest slot for the fit (at most half a slot),
    and the memory and time grow as the grid gets finer. The call is refused
    when the stack would not fit the memory, and says which spacing would.

    Starts one piece of background work over the stack and returns at once
    with its id. Every section in the stack refuses writes until it lands;
    you can look, zoom and work on sections you left out. When it lands, the
    next tool reply starts with its notice: how many sections were written,
    those fitted for the stack and not written with the reason, the sections
    whose warp has plausibility flags with their codes, and the overlays of
    up to four sections (flagged ones first, then those that moved most) with
    how far each moved. Judge the rest with look. emlddmm's own energy record,
    the images it saw and the per-section results (movement, warp
    displacement, fold fraction, flags, tissue set aside) are files in the
    job[[?read_file , which read_file opens]].

    Args:
        sections: Filenames; empty means every section with a position and a
            transform. At least two.
        slot_um: The slot spacing in micrometres; you set it, there is no
            default. Get the order right and each section within about one
            interval of its place first. Then set slot_um to the section
            thickness, or to about half the interval between neighbouring
            sections. Finer costs memory that many machines lack; coarser
            moves sections further from where you placed them and limits what
            the fit can resolve along the slicing axis. Two sections may not
            share a slot.
        section_image: The section image to fit, by name: "preprocessed", a raw channel~
            [[?make_image , or an image made with make_image]]. Read as grey, as it
            is, 0 to 1. Default "preprocessed".
        atlas_image: One of the atlas's images (the job statement lists
            them), or "nissl", the atlas's Nissl (its template when it ships
            none). Default "nissl"; the reply names the image used.
        mask: "none", or the name of a mask that every section in the stack
            has[[?make_mask  (made with make_mask)]]; its weights multiply the
            fit's weight on that section. Default "none".
        kernel_um: The width of the kernel that smooths the 3-D warp's
            velocity, in micrometres, across sections too (emlddmm's a; not a
            Gaussian sigma on the displacement, as smoothness_um is
            elsewhere): tissue closer than this bends together. Default 500.
        fit: "transform" refits each section's in-plane turn and shift and
            the stack's scale and shear and writes them; "warp" keeps every
            transform and fits the 3-D warp alone; "both" fits both. Default
            "both".
        warp_freedom: How freely the warp may bend; larger bends more. Start
            large and reduce if the warps look unrealistic. Default 10000.
        iterations: Steps at the 200 um scale; the 100 and 50 um scales take
            half and a quarter as many. Default 100.
        sorting: Any of "match_sd", "background_sd", "artifact_sd",
            "background_mean", "artifact_mean" (brightness from 0, black, to 1,
            white); keys left out take emlddmm's values. emlddmm's sorting is
            its own EM with an artifact class: it takes no "method" or
            "artifact".
        view: True sends the overlays with the notice; false sends none.

    Returns:
        The work id, the number of sections in the stack, those left out with
        why, the sections that will not get a transform and what they get, the
        furthest a section moves to its slot, the device, and per section what
        the fit uses (inputs: the section image, the atlas image as asked and as
        used, the mask, the damage, the brightness scaling); or why it was not
        started. The notice that follows brings, per section, what emlddmm
        estimated (its sorting and the share of the tissue it left out).
    """
    ...


def emlddmm_plane(
    sections: list[str],
    slot_um: float,
    section_image: str = "preprocessed",
    atlas_image: str = "nissl",
    iterations: int = 100,
    view: bool = True,
) -> dict[str, Any]:
    """Estimate the stack's cutting angles and spacing with emlddmm. Writes nothing.

    Fits emlddmm's 3-D affine to the whole placed stack at once, with the
    plane free and no warp: every section votes on one pitch and yaw, and
    the spacing may stretch and shift as a whole (a section cannot move on
    its own). Starts from the current angles and positions, so place and
    orient every section first, with the atlas outline over its tissue (emlddmm
    cannot recover from a start 1 to 3 mm off); one call is one descent and lands short
    from far away, so write the angles[[?position_sections  with ~
    position_sections]] and run it again until two passes agree within a
    degree or two. emlddmm cannot flip a section: a section mounted mirrored
    and not yet flipped pulls the yaw toward zero and stands out from the
    others in its residual turn or shift here; flip it[[?interactive_transform  with ~
    interactive_transform]] and run again before writing. Keep the
    positions you placed unless the spacing change looks right in the
    pictures. emlddmm's energy record and the images it saw are files in the
    job[[?read_file , which read_file opens]].

    Args:
        sections: Filenames; empty means every section with a position. At
            least two.
        slot_um: The slot spacing in micrometres, as for emlddmm_stack: you
            set it, there is no default. With the order right and each section
            within about one interval of its place, set it to the section
            thickness, or to about half the interval between neighbouring
            sections.
        section_image: The section image to fit, as for emlddmm_stack.
        atlas_image: One of the atlas's images or "nissl", as for the fit.
        iterations: Steps at the coarsest scale, as for emlddmm_stack. Default
            100.
        view: True returns the pictures described below; false returns none.

    Returns:
        pitch_deg, yaw_deg and their change from the stack's angles; the
        spacing scale and shift_mm; per section what the fit used (inputs), its
        proposed position_mm, the
        change, and the residual turn (deg) and shift (mm) it needed at the
        found plane beyond what the stack as a whole needed; and the overlays
        of up to four sections drawn at the found plane, not written.
    """
    ...


# --- frames -----------------------------------------------------------------------------------


def _unit(v: Any) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def _down(image: np.ndarray, factor: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Area-average by *factor* (never up): the image and (sx, sy), down pixel k sitting at
    original pixel (k + 0.5) * s - 0.5."""
    h, w = image.shape[:2]
    if factor <= 1.0:
        return image, (1.0, 1.0)
    wd, hd = max(1, round(w / factor)), max(1, round(h / factor))
    return cv2.resize(image, (wd, hd), interpolation=cv2.INTER_AREA), (w / wd, h / hd)


def _placed(frame: Any) -> np.ndarray:
    placed = frame.placed_atlas_to_section
    return np.asarray(frame.atlas_to_section if placed is None else placed, np.float64)


def _sample(array: np.ndarray, pts: np.ndarray, mode: int = cv2.INTER_LINEAR) -> np.ndarray:
    """*array* (H, W[, C]) read at grid pixels *pts* (..., 2) as (x, y); edges replicated."""
    shape = pts.shape[:-1]
    flat = np.asarray(pts, np.float32).reshape(-1, 2)
    n = len(flat)
    pad = np.zeros((-n % 1024, 2), np.float32)  # remap wants maps under 32767 per side
    flat = np.concatenate([flat, pad]).reshape(-1, 1024, 2)
    channels = [array] if array.ndim == 2 else [array[..., c] for c in range(array.shape[2])]
    got = [cv2.remap(np.ascontiguousarray(c, dtype=np.float32), flat[..., 0], flat[..., 1], mode,
                     borderMode=cv2.BORDER_REPLICATE).reshape(-1)[:n] for c in channels]
    return np.stack(got, -1).reshape(*shape, len(got)) if array.ndim == 3 else got[0].reshape(shape)




def _section(ctx: ToolContext, sid: str, section_image: str, atlas_image: str, mask: str,
             config: dict[str, Any]) -> dict[str, Any]:
    """Everything one section contributes, taken from its fit frame (which is then dropped):
    the registration (placement and applied map), its plane, the section's image in [0, 1]
    area-averaged to the section pixel, the tissue outline and the damage."""
    section_um = config["section_um"]
    frame = _blocks.frame_for(ctx, sid, atlas_image, section_image)
    _plane_image, resolved = _blocks.atlas_plane(frame, atlas_image)
    held = _blocks.section_mask(ctx, sid, mask)
    placed = _placed(frame)
    grey = _blocks.section_pixels(frame).astype(np.float64).mean(axis=2)
    gh, gw = grey.shape
    border = np.concatenate([grey[0], grey[-1], grey[:, 0], grey[:, -1]])
    slide = float(np.median(border))
    outline = cv2.warpAffine((frame.atlas["annotation"] > 0).astype(np.uint8), placed[:2],
                             (gw, gh), flags=cv2.INTER_NEAREST) > 0
    unit = 255.0 if grey.max() > 1.5 else 1.0
    scaled, back = grey / unit, float(slide / unit > 0.5)  # white slide: brightfield
    plane = frame.plane
    # micrometres per fit-grid pixel at the section's linear scale (atlas micrometres)
    px_um = float(np.mean(plane["px_um"])) / float(np.sqrt(abs(np.linalg.det(placed[:2, :2]))))
    down, (sx, sy) = _down(scaled.astype(np.float32), section_um / px_um)
    soft = None if held is None else _down(np.asarray(held.weights, np.float32),
                                           section_um / px_um)[0]
    m = np.asarray(plane["native_to_um"], np.float64)
    c, dx, dy = (np.asarray(plane[k], np.float64) for k in ("center_um", "dx", "dy"))
    Q = np.array([[dx @ m[:, 0], dx @ m[:, 1]], [dy @ m[:, 0], dy @ m[:, 1]]])
    return {
        "id": sid, "placed": placed, "has_transform": frame.placed_atlas_to_section is not None,
        "applied": None if frame.applied_map is None else np.asarray(frame.applied_map,
                                                                      np.float32),
        "grid": (gw, gh), "back": back, "soft": soft, "resolved": resolved,
        "inputs": _blocks.inputs_block(ctx, frame, section_image=section_image,
                                       atlas_image=atlas_image, resolved=resolved, mask=held,
                                       brightness="as it is, 0-1"),
        "px_um": px_um, "down": down, "down_s": (sx, sy),
        "centre": c, "dx": dx, "dy": dy, "Q": Q, "Qi": np.linalg.inv(Q),
        "q": np.array([dx @ (m[:, 2] - c), dy @ (m[:, 2] - c)]),
        "native_mm": float(np.mean(plane["px_um"])) / 1000.0,
        "tissue_native": frame.atlas["annotation"] > 0,
        "outline_grid": outline,
        "damage": None if frame.damage_zone is None else np.asarray(frame.damage_zone, bool)}


def _reg(sec: dict[str, Any], pts: np.ndarray) -> np.ndarray:
    """The section's current registration: fit-grid pixels (..., 2) -> native atlas pixels."""
    if sec["applied"] is not None:
        return _sample(sec["applied"], pts).astype(np.float64)
    inv = np.linalg.inv(sec["placed"])
    return np.asarray(pts, np.float64) @ inv[:2, :2].T + inv[:2, 2]


def _unreg(sec: dict[str, Any], p: np.ndarray) -> np.ndarray:
    """Native atlas pixels -> the fit-grid pixels the registration sends there. Linear: the
    placement. With a deformation: a fixed-point iteration from the placement."""
    lin = sec["placed"][:2, :2]
    g = p @ lin.T + sec["placed"][:2, 2]
    if sec["applied"] is None:
        return g
    for _ in range(20):
        g = g - (_reg(sec, g) - p) @ lin.T
    return g


def _plane_um(sec: dict[str, Any], p: np.ndarray) -> np.ndarray:
    """Native atlas pixels -> in-plane micrometres (column, row) about the plane's centre."""
    return (sec["Q"] @ np.asarray(p, np.float64).reshape(-1, 2).T).T.reshape(p.shape) + sec["q"]


# --- the stack --------------------------------------------------------------------------------


def _slots(z: np.ndarray, slot_um: float, cos: float) -> tuple[np.ndarray, float]:
    dz = slot_um * cos
    return np.rint(z / dz).astype(int), dz


def _memory_gb(n_slots: int, ny: int, nx: int, dtype: str) -> float:
    return n_slots * ny * nx * GB_PER_MVOX / 1e6 / (2.0 if dtype == "float32" else 1.0)


def _stack(ctx: ToolContext, config: dict[str, Any], names: list[str], slot_um: float,
           section_image: str, atlas_image: str, mask: str,
           need_transform: bool) -> dict[str, Any]:
    """The stack of the placed sections: geometry, slots, memory check, images on the stack grid.
    A section that cannot enter (no placement) is left out with its reason."""
    if not slot_um > 0:
        raise ToolError("BAD_ARGS", "slot_um is a positive number of micrometres")
    left_out: list[dict[str, Any]] = []
    secs = []
    for view in ([ctx.section(name) for name in names] if names else ctx.sections()):
        what = "a position and a transform" if need_transform else "a position"
        try:
            if view.position_mm is None or (need_transform and view.affine is None):
                raise ToolError("INVALID_LINEAR_PLACEMENT", f"needs {what} first")
            sec = _section(ctx, view.id, section_image, atlas_image, mask, config)
            if need_transform and not sec["has_transform"]:
                raise ToolError("INVALID_LINEAR_PLACEMENT", f"needs {what} first")
        except ToolError as exc:
            if exc.code == "BAD_ARGS":
                raise
            left_out.append({"id": view.id, "why": exc.code, "message": exc.message})
            continue
        sec["view"] = view
        secs.append(sec)
    if len(secs) < 2:
        raise ToolError("NEEDS_STACK", "emlddmm fits a stack: at least two sections with "
                        "positions" + (" and transforms" if need_transform else ""),
                        left_out=left_out)
    dxs, dys = (np.array([_unit(s[k]) for s in secs]) for k in ("dx", "dy"))
    dx, dy = _unit(dxs.mean(0)), _unit(dys.mean(0))
    dy = _unit(dy - (dy @ dx) * dx)
    tilt = max(math.degrees(math.acos(float(np.clip(min((dxs @ dx).min(), (dys @ dy).min()), -1,
                                                    1)))), 0.0)
    if tilt > ANGLE_TOLERANCE_DEG:  # sections of one job share its cutting angles
        raise ToolError("NOT_A_STACK", f"the sections' planes differ by {tilt:.1f} degrees: "
                        "emlddmm needs one cutting angle for the stack")
    n = _unit(np.cross(dy, dx))  # rows (n, dy, dx): a proper rotation
    centres = np.array([s["centre"] for s in secs])
    order = np.argsort(centres @ n)
    secs = [secs[i] for i in order]
    centres = centres[order]
    c0 = centres[0]
    rel = centres - c0
    far = rel[-1]
    cos = abs(float(_unit(far) @ n)) if np.linalg.norm(far) > 0 else 1.0
    z = rel @ n
    slot, dz = _slots(z, slot_um, cos)
    section_um = config["section_um"]
    # the stack's in-plane grid: the union of the sections' footprints (the image border read
    # through each registration), a margin of two pixels
    corners = []
    off_xy = np.stack([rel @ dx, rel @ dy], 1)
    for sec, off in zip(secs, off_xy, strict=True):
        w, h = sec["grid"]
        edge = np.concatenate([np.stack([np.linspace(0, w - 1, 24), np.zeros(24)], 1),
                               np.stack([np.linspace(0, w - 1, 24), np.full(24, h - 1.0)], 1),
                               np.stack([np.zeros(24), np.linspace(0, h - 1, 24)], 1),
                               np.stack([np.full(24, w - 1.0), np.linspace(0, h - 1, 24)], 1)])
        corners.append(_plane_um(sec, _reg(sec, edge)) + off)
    corners = np.concatenate(corners)
    lo = np.floor(corners.min(0) / section_um) * section_um - 2 * section_um
    hi = np.ceil(corners.max(0) / section_um) * section_um + 2 * section_um
    gx = np.arange(lo[0], hi[0] + 1e-6, section_um)
    gy = np.arange(lo[1], hi[1] + 1e-6, section_um)
    n_slots = int(slot.max()) + 1
    memory = _memory_gb(n_slots, len(gy), len(gx), config["dtype"])
    taken: dict[int, str] = {}
    for k, sec in zip(slot.tolist(), secs, strict=True):
        if k in taken:
            raise ToolError("BAD_ARGS", f"{taken[k]} and {sec['id']} share a {slot_um:g} um "
                            "slot; use a finer slot_um (or leave one of them out)")
        taken[k] = sec["id"]
    limit = float(config["memory_limit_gb"])
    if memory > limit:
        fits = next((float(s) for s in np.arange(slot_um, 5000.0, 5.0)
                     if _memory_gb(int(_slots(z, float(s), cos)[0].max()) + 1, len(gy), len(gx),
                                   config["dtype"]) <= limit), None)
        advice = "fewer sections or a smaller field"
        if fits is not None:
            shared = len(set(_slots(z, fits, cos)[0].tolist())) != len(secs)
            advice = (f"a slot_um of {fits:g} or more would fit"
                      + (", but then two sections share a slot: leave some sections out"
                         if shared else ""))
        raise ToolError("STACK_TOO_LARGE", f"{n_slots} slots of {len(gy)} x {len(gx)} pixels "
                        f"need about {memory:.0f} GB, over the {limit:g} GB limit; {advice}",
                        estimate_gb=round(memory, 1))
    gap = float(slot.max() - slot.min()) * dz / max(len(secs) - 1, 1)
    weight = config["section_weight"]
    thick = gap if weight == "gap" else float(weight)
    J = np.zeros((n_slots, len(gy), len(gx)), np.float32)
    W0 = np.zeros((n_slots, len(gy), len(gx)), np.float32)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    for i, (sec, off, k) in enumerate(zip(secs, off_xy, slot, strict=True)):
        uv = np.stack([mesh_x - off[0], mesh_y - off[1]], -1) - sec["q"]
        p = uv @ sec["Qi"].T  # native pixels of the plane under every stack pixel
        g = _unreg(sec, p)
        sx, sy = sec["down_s"]
        at = (g + 0.5) / np.array([sx, sy]) - 0.5
        w, h = sec["grid"]
        inside = ((g[..., 0] >= -0.5) & (g[..., 0] <= w - 0.5)
                  & (g[..., 1] >= -0.5) & (g[..., 1] <= h - 0.5))
        J[k] = np.where(inside, _sample(sec["down"], at), sec["back"])
        keep = inside.copy()
        if sec["damage"] is not None:
            keep &= ~(_sample(sec["damage"].astype(np.uint8), p, cv2.INTER_NEAREST) > 0)
        W0[k] = keep
        if sec["soft"] is not None:  # the section mask: emlddmm takes soft weights, W0 *= weights
            W0[k] *= np.clip(_sample(sec["soft"], at), 0.0, 1.0)
        tissue = _sample(sec["tissue_native"].astype(np.uint8), p, cv2.INTER_NEAREST) > 0
        sec["tissue_stack"] = tissue & keep
        sec["slot"] = int(k)
        sec["snap_um"] = float((k * dz - z[i]) / cos)
    return {"secs": secs, "left_out": left_out, "n": n, "dx": dx, "dy": dy, "c0": c0, "dz": dz,
            "z": z, "slot": slot, "slot_um": float(slot_um), "n_slots": n_slots, "tilt_deg": tilt,
            "gx": gx, "gy": gy, "off_xy": off_xy, "J": J, "W0": W0, "memory_gb": memory,
            "weight_um": thick, "back": secs[0]["back"], "R": np.stack([n, dy, dx]),
            "zs": np.arange(n_slots) * dz,
            "section_um": section_um, "resolved": secs[0]["resolved"]}


def _cfg(st: dict[str, Any], iterations: list[int], kernel_um: float, warp_freedom: float,
         rigid: bool, warp: bool) -> dict[str, Any]:
    """emlddmm's keyword arguments per scale: the published mouse serial-section example, with
    each section weighed as the gap it stands for (the Fig 3 runner's scaling: sigmaR x
    sqrt(k), ev and eA x k with k = thickness / slot spacing, eA2d x 20 / slot spacing)."""
    k = st["weight_um"] / st["dz"]
    return {"downI": [[4, 4, 4], [2, 2, 2], [1, 1, 1]], "downJ": [[1, 4, 4], [1, 2, 2], [1, 1, 1]],
            "n_iter": [int(i) for i in iterations], "a": [float(kernel_um)], "dv": [1000.0],
            "v_start": [0 if warp else NO_WARP], "ev": [1e-2 * k],
            "eA": [1e7 * k if rigid else 0.0],
            "eA2d": [1e5 * EXAMPLE_SLICE_UM / st["dz"] if rigid else 0.0],
            "local_contrast": [[1, 16, 16]], "sigmaR": [float(warp_freedom) * math.sqrt(k)]}


def _mu(st: dict[str, Any]) -> dict[str, Any]:
    """The background (and, on a white slide, artifact) mean: 1 on a white slide, else 0, as the
    benchmark runner."""
    return {"muB": [[st["back"]]], **({"muA": [[0.0]]} if st["back"] else {})}


def _check_iterations(iterations: Any) -> list[int]:
    """The three scales' steps of *iterations*, the first scale's (:func:`scale_iterations`)."""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 4:
        raise ToolError("BAD_ARGS", "iterations is a whole number of steps at the coarsest "
                        "scale, 4 or more (the finer scales take half and a quarter)")
    return scale_iterations(iterations)


def _check_sorting(sorting: dict[str, Any]) -> dict[str, float]:
    held = [key for key in sorting if key in ("method", "artifact")]
    if held:
        raise ToolError("BAD_ARGS", f"emlddmm's sorting is its own EM with an artifact class: "
                        f"it takes no {' or '.join(held)}; its keys are "
                        f"{', '.join(SORTING_KEYS)}")
    unknown = [key for key in sorting if key not in SORTING_KEYS]
    if unknown:
        raise ToolError("BAD_ARGS", f"unknown sorting keys {unknown}; the keys are "
                        f"{', '.join(SORTING_KEYS)}")
    return {key: float(value) for key, value in sorting.items()}


def _require(config: dict[str, Any], *keys: str) -> None:
    """Refuse at call time when an environment or source folder the engine needs is missing."""
    for key in keys:
        path = Path(str(config[key])).expanduser()
        if not path.exists():
            raise ToolError("ENGINE_UNAVAILABLE",
                            f"{NAME} cannot run here: {key} {str(path)!r} does not exist (set "
                            f"it in the plugin_config of {REF})")


# --- the engine -------------------------------------------------------------------------------


def _engine(config: dict[str, Any], folder: Path, arrays: dict[str, Any],
            params: dict[str, Any]) -> dict[str, Any]:
    """Run engine.py on *arrays* in *folder* (inputs, outputs and log kept there)."""
    folder.mkdir(parents=True, exist_ok=True)
    src, dst = folder / "in.npz", folder / "out.npz"
    np.savez_compressed(src, params=np.array(json.dumps(
        {**params, "emlddmm_src": config["emlddmm_src"],
         "brainglobe_dir": config["brainglobe_dir"], "device": config["device"],
         "dtype": config["dtype"]})), **arrays)
    env = {**os.environ, "OMP_NUM_THREADS": str(config["threads"])}
    done = subprocess.run([config["python"], "-I", str(HERE / "engine.py"), str(src), str(dst)],
                          capture_output=True, text=True, env=env, cwd=str(folder),
                          timeout=float(config["timeout_s"]))
    (folder / "log.txt").write_text(done.stdout + done.stderr)
    if done.returncode != 0 or not dst.exists():
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        raise ToolError("ENGINE_FAILED", f"emlddmm did not run: {tail[0]}")
    out = dict(np.load(dst, allow_pickle=False))
    if "error" in out:
        raise ToolError("ENGINE_FAILED", f"emlddmm failed: {out['error']}")
    return out


def _relative(path: Path, ctx: ToolContext) -> str:
    return str(path.relative_to(ctx.job_dir)) if path.is_relative_to(ctx.job_dir) else str(path)


def _arrays(st: dict[str, Any], queries: list[np.ndarray]) -> dict[str, Any]:
    return {"J": st["J"], "W0": st["W0"], "zs": st["zs"], "gy": st["gy"], "gx": st["gx"],
            "R": st["R"], "c0": st["c0"], "slot": np.asarray(st["slot"], int),
            "q": np.concatenate(queries, 0) if queries else np.zeros((0, 2)),
            "q_n": np.array([len(q) for q in queries], int)}


# --- pictures ---------------------------------------------------------------------------------


def _font(size: int = 11) -> Any:
    try:
        return ImageFont.load_default(size)
    except TypeError:  # an older Pillow
        return ImageFont.load_default()


def _sheet_picture(st: dict[str, Any], i: int, sec: dict[str, Any], out: dict[str, Any],
                   path: Path) -> None:
    """One section as the engine saw it, its final sorting (white compared, black ignored) and
    the atlas under the fit; a file in the job, not sent."""
    k = sec["slot"]
    seen = np.clip(st["J"][k] / 1.6, 0, 1)
    sort = np.clip(np.asarray(out["WM"][k], np.float32), 0, 1) * (st["W0"][k] > 0)
    atl = np.asarray(out["Acut"][i], np.float32)
    lo, hi = np.percentile(atl[st["W0"][k] > 0], [1, 99]) if (st["W0"][k] > 0).any() else (0, 1)
    atl = np.clip((atl - lo) / max(hi - lo, 1e-9), 0, 1)
    tiles = [(t * 255).astype(np.uint8) for t in (seen, sort, atl)]
    gap = np.full((tiles[0].shape[0], 4), 255, np.uint8)
    row = np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], 1)
    image = Image.fromarray(row).convert("RGB")
    image = image.resize((image.width * 3, image.height * 3), Image.Resampling.LANCZOS)
    band = Image.new("RGB", (image.width, image.height + 22), "white")
    band.paste(image, (0, 22))
    ImageDraw.Draw(band).text((6, 5), f"{sec['id']}  slot {k}: the section as sent | the final "
                              "sorting (white compared) | the atlas under the fit", fill="black",
                              font=_font(11))
    band.save(path)


def _energy_files(energy: np.ndarray, folder: Path) -> None:
    """emlddmm's own energy record as ``energy.csv`` and a small line plot (files in the job)."""
    with open(folder / "energy.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scale", "iteration", "total", "mismatch", "regularization"])
        for row in energy:
            writer.writerow([int(row[0]), int(row[1]), *[f"{v:.6g}" for v in row[2:]]])
    h, w, pad = 260, 720, 34
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    if len(energy) > 1:
        for col, shade, name in ((2, (0, 0, 0), "total"), (3, (110, 110, 110), "mismatch"),
                                 (4, (190, 190, 190), "regularization")):
            y = energy[:, col]
            lo, hi = float(y.min()), float(y.max())
            span = max(hi - lo, 1e-12)
            pts = [(pad + (w - 2 * pad) * j / (len(y) - 1),
                    h - pad - (h - 2 * pad) * (v - lo) / span) for j, v in enumerate(y)]
            draw.line(pts, fill=shade, width=1)
            draw.text((pad, 4 + 10 * (col - 2)), f"{name}: min {lo:.4g}, max {hi:.4g} "
                      "(each scaled to its own range)", fill=shade, font=_font(9))
    image.save(folder / "energy.png")


# --- landing helpers --------------------------------------------------------------------------


def _expand(native: np.ndarray, w: int, h: int) -> np.ndarray:
    """A map read on the lattice (nh, nw, 2) as a full fit-grid map (h, w, 2)."""
    nh, nw = native.shape[:2]
    xs = (np.arange(w) * (nw - 1) / max(w - 1, 1)).astype(np.float32)
    ys = (np.arange(h) * (nh - 1) / max(h - 1, 1)).astype(np.float32)
    mx, my = np.meshgrid(xs, ys)
    return np.stack([cv2.remap(np.ascontiguousarray(native[..., c], np.float32), mx, my,
                               cv2.INTER_LINEAR) for c in range(2)], -1).astype(np.float64)


def _lattice(sec: dict[str, Any], off: np.ndarray) -> dict[str, Any]:
    """Where the map is read on one section: a lattice of fit-grid pixels, each carried through
    the section's current registration to its place in the stack frame (row, column um)."""
    w, h = sec["grid"]
    nw, nh = int(math.ceil((w - 1) / LATTICE_PX)) + 1, int(math.ceil((h - 1) / LATTICE_PX)) + 1
    xs, ys = np.linspace(0, w - 1, nw), np.linspace(0, h - 1, nh)
    gxy = np.stack(np.meshgrid(xs, ys), -1)  # (nh, nw, 2) as (x, y)
    p_old = _reg(sec, gxy)
    uv = _plane_um(sec, p_old) + off  # (column, row) um in the stack frame
    return {"shape": (nh, nw), "gxy": gxy, "p_old": p_old,
            "q": np.stack([uv[..., 1].ravel(), uv[..., 0].ravel()], -1)}


def _native_from_atlas(sec: dict[str, Any], X: np.ndarray, shape: tuple[int, int]
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Atlas um (N, 3) on the stack's slot -> native pixels of the section's own plane (nh, nw, 2)
    and the distance off that plane, um."""
    d = np.asarray(X, np.float64) - sec["centre"]
    uv = np.stack([d @ sec["dx"], d @ sec["dy"]], -1)
    normal = _unit(np.cross(sec["dy"], sec["dx"]))
    return ((uv - sec["q"]) @ sec["Qi"].T).reshape(*shape, 2), d @ normal


def _flags(summary: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(str(flag.get("code")) for flag in summary.get("flags") or []))


def _set_aside(out: dict[str, Any], k: int, sec: dict[str, Any]) -> dict[str, float]:
    """The share of the section's tissue (the placed atlas outline, where it holds data) the
    EM sorting set aside as background (WB) and as artifact (WA)."""
    inside = sec["tissue_stack"]
    return {key: round(float((np.asarray(out[key][k], np.float32) > 0.5)[inside].mean()), 3)
            if inside.any() else 0.0 for key in ("WB", "WA")}


def _write_files(st: dict[str, Any], out: dict[str, Any], folder: Path, secs: list[dict[str, Any]],
                 A: np.ndarray, A0: np.ndarray, A2d: np.ndarray, extra: dict[str, Any],
                 asked: dict[str, Any], chosen: dict[str, Any], config: dict[str, Any]) -> None:
    """The call's readable files (see the module text); nothing here goes into a reply."""
    _energy_files(np.asarray(out["energy"]), folder)
    (folder / "images").mkdir(exist_ok=True)
    for i, sec in enumerate(secs):
        _sheet_picture(st, i, sec, out, folder / "images" / f"{i:02d}_{Path(sec['id']).stem}.png")
    np.savez_compressed(folder / "weights.npz", WM=out["WM"], WB=out["WB"], WA=out["WA"],
                        slot=np.asarray(st["slot"]))
    (folder / "A.json").write_text(json.dumps({"A": A.tolist(), "A_start": A0.tolist(),
                                               "A2d": A2d.tolist(), **extra}, indent=1,
                                              default=float))
    (folder / "settings.json").write_text(json.dumps(
        {"asked": asked, "chosen": chosen, "plugin": config,
         "emlddmm": json.loads(str(out["settings"])), "device": str(out["device"]),
         "energy": "captured" if _captured(out) else "not captured",
         "peak_gpu_gb": float(out["peak_gpu_gb"]), "seconds": float(out["seconds"])},
        indent=1, default=str))


def _captured(out: dict[str, Any]) -> bool:
    return bool(out["energy_captured"])


def _chosen(st: dict[str, Any], config: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """What the plugin picked from the stack, kept in settings.json."""
    return {"slot_um": st["slot_um"], "slot_spacing_along_normal_um": round(st["dz"], 3),
            "n_slots": st["n_slots"], "stack_shape": list(st["J"].shape),
            "largest_plane_tilt_deg": round(st["tilt_deg"], 3), "section_px_um": st["section_um"],
            "section_stands_for_um": round(st["weight_um"], 2), "device": config["device"],
            "dtype": config["dtype"], "memory_estimate_gb": round(st["memory_gb"], 2),
            "memory_limit_gb": config["memory_limit_gb"], "emlddmm": cfg}


def _device(config: dict[str, Any]) -> str:
    """The device the engine will run on: ``auto`` is asked of the engine's own torch, once."""
    if config["device"] != "auto":
        return str(config["device"])
    if config["python"] not in _DEVICES:
        done = subprocess.run([config["python"], "-I", "-c",
                               "import torch; print(torch.cuda.is_available())"],
                              capture_output=True, text=True, timeout=120)
        _DEVICES[config["python"]] = "cuda" if done.stdout.strip().endswith("True") else "cpu"
    return _DEVICES[config["python"]]


# --- emlddmm_stack ----------------------------------------------------------------------------


def _meta(rec: dict[str, Any]) -> dict[str, Any]:
    """What the deformation record keeps of the call besides its settings: its
    inputs and estimates."""
    return {"inputs": rec["sec"]["inputs"], "estimated": rec["estimated"]}


def _route(view: Any, rigid: bool, warp: bool) -> str:
    """How one section's result is written: ``transform`` (its transform, then the warp on top),
    ``deformation`` (the rigid change folded into its deformation) or ``none``."""
    free_a, free_d = "affine" not in view.frozen, "deformation" not in view.frozen
    if rigid and free_a and not view.deformation:
        return "transform" if (not warp or free_d) else "none"
    return "deformation" if (warp or rigid) and free_d else "none"


GETS = {"deformation": "its transform kept, the rigid change folded into its deformation",
        "none": "nothing (its transform or deformation is frozen)"}


def _weights_on_grid(st: dict[str, Any], sec: dict[str, Any], off: np.ndarray,
                     stack_weights: np.ndarray) -> np.ndarray:
    """emlddmm's final sorting of one section (its slot of ``WM``, on the stack grid) as a
    float16 weight map on the section's fit grid, read through its current registration."""
    w, h = sec["grid"]
    gxy = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float64)
    uv = _plane_um(sec, _reg(sec, gxy)) + off  # (column, row) um in the stack frame
    ix = ((uv[..., 0] - st["gx"][0]) / st["section_um"]).astype(np.float32)
    iy = ((uv[..., 1] - st["gy"][0]) / st["section_um"]).astype(np.float32)
    got = cv2.remap(np.ascontiguousarray(stack_weights, np.float32), ix, iy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return np.clip(got, 0.0, 1.0).astype(np.float16)


def _stack_tool(ctx: ToolContext, config: dict[str, Any], names: list[str], slot_um: float,
                section_image: str, atlas_image: str, mask: str, kernel_um: float,
                fit: str, warp_freedom: float, iterations: int,
                sorting: dict[str, Any], view: bool) -> ToolResult:
    if fit not in FITS:
        raise ToolError("BAD_ARGS", f"fit is one of {', '.join(FITS)}; got {fit!r}")
    rigid, warp = fit in ("transform", "both"), fit in ("warp", "both")
    steps = _check_iterations(iterations)
    sort = _check_sorting(sorting)
    config = {**config, "device": _device(config)}
    st = _stack(ctx, config, names, slot_um, section_image, atlas_image, mask,
                need_transform=True)
    cfg = _cfg(st, steps, kernel_um, warp_freedom, rigid, warp)
    atlas = ctx.atlas()
    secs = st["secs"]
    lattices = [_lattice(sec, off) for sec, off in zip(secs, st["off_xy"], strict=True)]
    params = {"mode": "stack", "atlas": atlas.name, "image": st["resolved"], "cfg": cfg,
              "sorting": sort, "warp": warp, "mu": _mu(st)}
    folder = _blocks.call_folder(ctx, "stack")
    arrays = _arrays(st, [lat["q"] for lat in lattices])
    digests = {sec["id"]: ctx.section_inputs(sec["id"], deformation=True) for sec in secs}
    settings = {"section_image": section_image, "atlas_image": atlas_image, "mask": mask,
                "kernel_um": kernel_um, "fit": fit, "slot_um": slot_um,
                "warp_freedom": warp_freedom, "iterations": int(iterations), "sorting": sort}
    asked = {"sections": names, **settings, "view": view}
    chosen = _chosen(st, config, cfg)
    routes = {sec["id"]: _route(sec["view"], rigid, warp) for sec in secs}
    note = f"emlddmm stack fit, atlas {st['resolved']}, {len(secs)} sections"

    def land(lctx: ToolContext) -> ToolResult:
        out = _engine(config, folder, arrays, params)
        A, A0, A2d = (np.asarray(out[k], np.float64) for k in ("A", "A0", "A2d"))
        records, at = [], 0
        for sec, lat in zip(secs, lattices, strict=True):
            n_pts = len(lat["q"])
            X, L = out["X"][at:at + n_pts], out["L"][at:at + n_pts]
            at += n_pts
            nat_full, off_plane = _native_from_atlas(sec, X, lat["shape"])
            nat_lin, _ = _native_from_atlas(sec, L, lat["shape"])
            gxy = np.concatenate([lat["gxy"].reshape(-1, 2), np.ones((n_pts, 1))], 1)
            matrix = np.linalg.lstsq(gxy, nat_lin.reshape(-1, 2), rcond=None)[0].T
            inside = sec["outline_grid"][np.clip(lat["gxy"][..., 1].round().astype(int), 0,
                                                 sec["grid"][1] - 1),
                                         np.clip(lat["gxy"][..., 0].round().astype(int), 0,
                                                 sec["grid"][0] - 1)]
            used = nat_full if warp else nat_lin
            moved = np.hypot(*(used - lat["p_old"]).transpose(2, 0, 1)) * sec["native_mm"]
            moved = moved[inside] if inside.any() else moved.ravel()
            weights = (_weights_on_grid(st, sec, st["off_xy"][len(records)], out["WM"][sec["slot"]])
                       if warp else None)
            records.append({"sec": sec, "native": used, "matrix": matrix, "weights": weights,
                            "estimated": _blocks.estimated_block(weights, sec["outline_grid"]),
                            "off_plane_um": float(np.abs(off_plane).max()),
                            "moved_mm": (float(np.median(moved)), float(moved.max())),
                            "set_aside": _set_aside(out, sec["slot"], sec)})
        _write_files(st, out, folder, secs, A, A0, A2d, {}, asked, chosen, config)
        # every conversion that can run before the write lock does; a deformation on a section
        # whose transform is just rewritten composes with the new placement, so it waits
        t_rows = {rec["sec"]["id"]: lctx.transform_from_matrix(rec["sec"]["id"], rec["matrix"],
                                                               note=note).row()
                  for rec in records if routes[rec["sec"]["id"]] == "transform"}
        proposals = {}
        for rec in records:
            sec = rec["sec"]
            if routes[sec["id"]] == "deformation":
                proposals[sec["id"]] = lctx.deformation_from_map(
                    sec["id"], _expand(rec["native"], *sec["grid"]), note=note,
                    meta=_meta(rec), weights=rec["weights"], settings=settings,
                    # read through its deformation: built on it
                    start="current" if sec["applied"] is not None else "linear",
                    engine_settings=chosen, engine_version=ENGINE_VERSION)
        written: dict[str, str] = {}
        failed: dict[str, dict[str, Any]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        with lctx.writing():
            if t_rows:
                done = lctx.set_transforms(list(t_rows.values()),
                                           expect={sid: digests[sid] for sid in t_rows})
                for row in done.rows:
                    if row.get("status") == "ok":
                        written[row["id"]] = "transform"
                    else:
                        failed[row["id"]] = row
            for rec in records:
                sec = rec["sec"]
                if routes[sec["id"]] == "transform" and warp and sec["id"] not in failed:
                    proposals[sec["id"]] = lctx.deformation_from_map(
                        sec["id"], _expand(rec["native"], *sec["grid"]), note=note,
                        meta=_meta(rec), weights=rec["weights"], settings=settings,
                        engine_settings=chosen, engine_version=ENGINE_VERSION)
            if proposals:
                done = lctx.set_deformations(
                    [p.row() for p in proposals.values()],
                    expect={sid: digests[sid] for sid in proposals if routes[sid] == "deformation"})
                for row in done.rows:
                    if row.get("status") == "ok":
                        summaries[row["id"]] = proposals[row["id"]].summary
                        written[row["id"]] = ("transform and deformation"
                                              if written.get(row["id"]) == "transform"
                                              else "deformation (rigid folded in)"
                                              if rigid else "deformation")
                    else:
                        failed[row["id"]] = row
        rows = []
        for rec in records:
            sid = rec["sec"]["id"]
            code = failed.get(sid, {}).get("error")
            if sid in written and sid not in failed:
                problem, wrote = "", written[sid]
            elif sid in written:  # the transform stands, the warp did not land
                problem, wrote = f"transform only, deformation {code}", "transform only"
            else:
                problem, wrote = code or "FROZEN", "not written"
            rows.append({"id": sid, "slot": rec["sec"]["slot"], "snap_um": rec["sec"]["snap_um"],
                         "moved": rec["moved_mm"], "off_plane_um": rec["off_plane_um"],
                         "summary": summaries.get(sid, {}), "set_aside": rec["set_aside"],
                         "inputs": rec["sec"]["inputs"], "estimated": rec["estimated"],
                         "written": wrote, "problem": problem})
        with open(folder / "sections.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["section", "slot", "snap_um", "moved_median_mm", "moved_max_mm",
                             "off_plane_max_um", "displacement_median_mm", "displacement_max_mm",
                             "fold_fraction", "flags",
                             "set_aside_background", "set_aside_artifact", "written", "problem"])
            for r in rows:
                s = r["summary"]
                disp = s.get("displacement_mm", {})
                writer.writerow([r["id"], r["slot"], round(r["snap_um"], 2),
                                 round(r["moved"][0], 4), round(r["moved"][1], 4),
                                 round(r["off_plane_um"], 1), disp.get("median", ""),
                                 disp.get("max", ""), s.get("fold_fraction", ""),
                                 " ".join(_flags(s)),
                                 r["set_aside"]["WB"], r["set_aside"]["WA"], r["written"],
                                 r["problem"]])
        problems = [f"{r['id']} ({r['problem']})" for r in rows if r["problem"]]
        flagged = [f"{r['id']} ({flag_words(r['summary'])})" for r in rows
                   if _flags(r["summary"])]
        shown = sorted((r for r in rows if r["id"] in written),
                       key=lambda r: (not _flags(r["summary"]), -r["moved"][0]))[:OVERLAYS]
        drawn = ", ".join(f"{r['id']} ({r['moved'][0]:.2f})" for r in shown)
        relative = _relative(folder, lctx)
        left = [f"{r['id']} ({r['estimated']['left_out_fraction']:.0%})" for r in rows
                if r["estimated"] and r["estimated"]["left_out_fraction"] is not None]
        energy = (f"emlddmm's energy record: {relative}/energy.csv (plot: {relative}/energy.png)"
                  if _captured(out) else "emlddmm's energy record was not captured")
        waited = float(out.get("gpu_wait_s", 0.0))
        text = (f"{note}: written for {len(written)} of {len(secs)} sections"
                + (f"; it waited {round(waited)} s for the machine's GPU queue"
                   if waited >= 1.0 else "")
                + (f"; not (fully) written: {', '.join(problems)}" if problems else "")
                + (f"; warp flags: {', '.join(flagged)}" if flagged else "")
                + (f"; the sorting emlddmm estimated left out, of the tissue, {', '.join(left)}, "
                   "saved with each deformation" if left else "")
                + (f". Overlays, with how far each moved (median, mm): {drawn}"
                   if view and shown else "")
                + f". {energy}; per-section results: {relative}/sections.csv; "
                f"the images it saw: {relative}/images/.")
        pictures = [lctx.caption(lctx.overlay_picture(r["id"]), f"{r['id']}: after emlddmm")
                    for r in shown] if view else []
        return ToolResult({"notice": text, "files": relative, "settings": settings,
                           **({"gpu_wait_s": round(waited)} if waited >= 1.0 else {}),
                           "rows": [{"id": r["id"], "inputs": r["inputs"],
                                     "estimated": r["estimated"]} for r in rows]}, pictures)

    try:
        work = ctx.start_work([sec["id"] for sec in secs], land)
    except Exception as exc:  # SECTION_BUSY and the like
        as_payload = getattr(exc, "payload", None)
        payload = as_payload() if as_payload else {"error": "NOT_STARTED",
                                                   "message": str(exc)}
        details = {k: v for k, v in payload.items() if k not in ("status", "error", "message")}
        raise ToolError(str(payload.get("error")), str(payload.get("message") or exc),
                        **details) from exc
    other = [{"id": sid, "gets": GETS[route]} for sid, route in routes.items()
             if route != "transform"]
    return ToolResult({"work": work, "stack": len(secs), "left_out": st["left_out"],
                       "settings": settings,
                       "rows": [{"id": sec["id"], "inputs": sec["inputs"], "estimated": None}
                                for sec in secs],
                       **({"other_routes": other} if other else {}),
                       "largest_snap_um": round(max(abs(s["snap_um"]) for s in secs), 1),
                       "device": config["device"]
                       + (" (the fit takes tens of minutes)" if config["device"] == "cpu" else ""),
                       "files": _relative(folder, ctx)})


# --- emlddmm_plane ----------------------------------------------------------------------------


def _plane_tool(ctx: ToolContext, config: dict[str, Any], names: list[str], slot_um: float,
                section_image: str, atlas_image: str, iterations: int,
                view: bool) -> ToolResult:
    steps = _check_iterations(iterations)
    config = {**config, "device": _device(config)}
    st = _stack(ctx, config, names, slot_um, section_image, atlas_image, "none",
                need_transform=False)
    cfg = _cfg(st, steps, PUBLISHED["kernel_um"], PUBLISHED["warp_freedom"],
               rigid=True, warp=False)
    atlas = ctx.atlas()
    secs = st["secs"]
    params = {"mode": "plane", "atlas": atlas.name, "image": st["resolved"], "cfg": cfg,
              "sorting": {}, "warp": False, "mu": _mu(st), "up_vector": (-st["dy"]).tolist()}
    folder = _blocks.call_folder(ctx, "plane")
    out = _engine(config, folder, _arrays(st, [np.zeros((0, 2)) for _ in secs]), params)
    A, A0, A2d = (np.asarray(out[k], np.float64) for k in ("A", "A0", "A2d"))
    # The slot planes' common normal is A's first row (the gradient of the slot coordinate), the
    # atlas point on a plane is A^-1 [z, 0, 0, 1] about the volume's centre, which the engine
    # put at the origin.
    centre = (np.asarray(atlas.shape, float) - 1) / 2.0 * np.asarray(atlas.resolution_um)
    Ai = np.linalg.inv(A)
    found = [ctx.angles_from_plane(A[0, :3].tolist(),
                              ((Ai @ np.array([z, 0.0, 0.0, 1.0]))[:3] + centre).tolist())
             for z in st["z"]]
    pitch, yaw = found[0]["pitch_deg"], found[0]["yaw_deg"]
    pitch0, yaw0 = (float(v) for v in secs[0]["view"].angles)
    old = np.array([float(sec["view"].position_mm) for sec in secs])
    new = np.array([f["position_mm"] for f in found])
    scale, shift_mm = (np.polyfit(old, new, 1) if np.ptp(old) > 0
                       else (1.0, float(np.mean(new - old))))
    # emlddmm moves a shared in-plane turn and shift into every section's A2d (and the start
    # carries one when the stack is tilted): a section's own residual is what it has beyond the
    # stack's median
    a2d = A2d[[sec["slot"] for sec in secs]]
    turns = np.degrees(np.arctan2(a2d[:, 1, 0], a2d[:, 0, 0]))
    shifts = a2d[:, :2, 2] / 1000.0
    turns, shifts = turns - np.median(turns), shifts - np.median(shifts, axis=0)
    rows = [{"id": sec["id"], "inputs": sec["inputs"], "estimated": None,
             "position_mm": round(f["position_mm"], 3),
             "change_mm": round(f["position_mm"] - float(was), 3),
             "residual": {"turn_deg": round(float(t), 2),
                          "shift_mm": round(float(np.hypot(*s)), 3)}}
            for sec, f, was, t, s in zip(secs, found, old, turns, shifts, strict=True)]
    asked = {"sections": names, "slot_um": slot_um, "section_image": section_image,
             "atlas_image": atlas_image, "iterations": int(iterations), "view": view}
    proposal = {"pitch_deg": pitch, "yaw_deg": yaw, "spacing_scale": float(scale),
                "spacing_shift_mm": float(shift_mm)}
    _write_files(st, out, folder, secs, A, A0, A2d, {"proposed_plane": proposal, "sections": rows},
                 asked, _chosen(st, config, cfg), config)
    pictures = []
    if view:
        for j in sorted({int(round(v)) for v in np.linspace(0, len(secs) - 1,
                                                            min(OVERLAYS, len(secs)))}):
            pictures.append(ctx.caption(
                ctx.overlay_picture(rows[j]["id"], proposal={"position_mm": found[j]["position_mm"],
                                                             "angles": (pitch, yaw)}),
                f"{rows[j]['id']}: atlas at the plane emlddmm found (pitch {pitch:.2f}, yaw "
                f"{yaw:.2f}), not written"))
    return ToolResult({"pitch_deg": round(pitch, 2), "yaw_deg": round(yaw, 2),
                       "change": {"pitch_deg": round(pitch - pitch0, 2),
                                  "yaw_deg": round(yaw - yaw0, 2)},
                       "spacing": {"scale": round(float(scale), 4),
                                   "shift_mm": round(float(shift_mm), 3)},
                       "rows": rows, "left_out": st["left_out"],
                       **({} if _captured(out) else {"energy": "not captured"}),
                       "files": _relative(folder, ctx)}, pictures)


# --- the factory ------------------------------------------------------------------------------


def tools(config: dict[str, Any] | None = None) -> list[ToolSpec]:
    options = {**DEFAULTS, **(config or {})}
    meta = {"method": "emlddmm (Tward et al. 2025, Nat Commun)",
            "commit": "9c4dc80 + e426ece + 6ff62b0", "engine": "langslice.engines.emlddmm"}

    def stack(ctx: ToolContext, sections: list[str], slot_um: float,
              section_image: str = "preprocessed", atlas_image: str = "nissl",
              mask: str = "none", kernel_um: float = 500.0, fit: str = "both",
              warp_freedom: float = 10000.0, iterations: int = 100,
              sorting: dict[str, Any] | None = None, view: bool = True) -> ToolResult:
        _require(options, "python", "emlddmm_src", "brainglobe_dir")
        return _stack_tool(ctx, options, list(dict.fromkeys(sections)), float(slot_um),
                           section_image, atlas_image, mask, float(kernel_um), str(fit),
                           float(warp_freedom), iterations, dict(sorting or {}), view)

    def plane(ctx: ToolContext, sections: list[str], slot_um: float,
              section_image: str = "preprocessed", atlas_image: str = "nissl",
              iterations: int = 100, view: bool = True) -> ToolResult:
        _require(options, "python", "emlddmm_src", "brainglobe_dir")
        return _plane_tool(ctx, options, list(dict.fromkeys(sections)), float(slot_um),
                           section_image, atlas_image, iterations, view)

    return [
        ToolSpec("emlddmm_stack", emlddmm_stack, stack, kind="write", group="Nonlinear",
                 task="nonlinear", writes=("affine", "deformation"), long=True,
                 limits={"pictures": OVERLAYS}, version="1", meta=meta),
        ToolSpec("emlddmm_plane", emlddmm_plane, plane, kind="read", group="Positioning",
                 task="angles", limits={"pictures": OVERLAYS}, version="1", meta=meta),
    ]


__all__ = ["tools"]
