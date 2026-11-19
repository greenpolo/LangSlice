"""Running emlddmm on a stack and reading its answer onto each section."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from langslice.core.failures import ENGINE_FAILURES
from langslice.engines._pixels import (
    sample_pixels,
)
from langslice.engines._sorting import Sorting
from langslice.engines.emlddmm import engine
from langslice.engines.emlddmm.settings import EXAMPLE, LATTICE_PX, Array, Options
from langslice.engines.emlddmm.stack import Placed, Stack
from langslice.plugin_api import (
    RefusalCode,
    SectionView,
    ToolContext,
    ToolError,
)


def schedule_of(
    stack: Stack,
    iterations: list[int],
    kernel_um: float,
    warp_freedom: float,
    *,
    rigid: bool,
    warp: bool,
) -> engine.Schedule:
    """emlddmm's arguments per scale: the published mouse serial-section example, with each
    section weighed as the gap it stands for."""
    k = stack.weight_um / stack.spacing_um
    return engine.Schedule(
        iterations=tuple(int(i) for i in iterations),
        kernel_um=float(kernel_um),
        velocity_size_um=EXAMPLE.velocity_size_um,
        warp_start=0 if warp else EXAMPLE.no_warp,
        warp_step=EXAMPLE.warp_step * k,
        affine_step=EXAMPLE.affine_step * k if rigid else 0.0,
        slice_step=EXAMPLE.slice_step * EXAMPLE.slice_um / stack.spacing_um if rigid else 0.0,
        warp_sigma=float(warp_freedom) * math.sqrt(k),
    )


def run_engine(request: engine.FitRequest, folder: Path) -> engine.FitResult:
    """Run emlddmm in this process; what it printed is kept in *folder*'s ``log.txt``."""
    folder.mkdir(parents=True, exist_ok=True)
    try:
        result = engine.run(request)
    except ENGINE_FAILURES as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise ToolError(RefusalCode.FIT_FAILED, f"emlddmm failed: {failure}") from exc
    (folder / "log.txt").write_text(result.log, encoding="utf-8", newline="\n")
    return result


def build_request(
    options: Options,
    atlas: str,
    stack: Stack,
    schedule: engine.Schedule,
    *,
    sorting: Sorting,
    mode: engine.Mode,
    read_warp: bool,
    points: tuple[Array, ...],
) -> engine.FitRequest:
    """Build the engine's request for a stack."""
    return engine.FitRequest(
        atlas=atlas,
        atlas_image=stack.resolved,
        stack=stack.images_for_engine(),
        frame=engine.StackFrame(
            rotation=stack.rotation,
            origin_um=stack.origin_um,
            slots=tuple(placed.slot for placed in stack.sections),
        ),
        schedule=schedule,
        sorting=sorting,
        white_slide=stack.white_slide,
        mode=mode,
        read_warp=read_warp,
        points=points,
        device=options.device,
        single_precision=options.dtype == "float32",
        up_vector=tuple((-stack.dy).tolist()) if mode == "plane" else None,
    )


def expand(native: Array, w: int, h: int) -> Array:
    """A map read on the lattice (nh, nw, 2) as a full fit-grid map (h, w, 2)."""
    nh, nw = native.shape[:2]
    xs = (np.arange(w) * (nw - 1) / max(w - 1, 1)).astype(np.float32)
    ys = (np.arange(h) * (nh - 1) / max(h - 1, 1)).astype(np.float32)
    mx, my = np.meshgrid(xs, ys)
    return sample_pixels(native.astype(np.float32), np.stack([mx, my], -1)).astype(np.float64)


@dataclass(frozen=True)
class Lattice:
    """Where the map is read on one section: a lattice of fit-grid pixels, each carried through
    the section's current registration to its place in the stack frame.

    Parameters
    ----------
    shape
        The lattice, ``(rows, cols)``.
    pixels
        Its fit-grid pixels ``(rows, cols, 2)`` as (x, y).
    native
        The native atlas pixels the current registration sends them to.
    points
        Their places in the stack frame, ``(n, 2)`` (row, col) um.
    """

    shape: tuple[int, int]
    pixels: Array
    native: Array
    points: Array


def lattice_of(placed: Placed) -> Lattice:
    """Lay the lattice the map is read on over one section."""
    section = placed.section
    w, h = section.size
    nw, nh = int(math.ceil((w - 1) / LATTICE_PX)) + 1, int(math.ceil((h - 1) / LATTICE_PX)) + 1
    xs, ys = np.linspace(0, w - 1, nw), np.linspace(0, h - 1, nh)
    pixels = np.stack(np.meshgrid(xs, ys), -1)
    native = section.registered(pixels)
    uv = section.plane_um(native) + placed.offset_um
    return Lattice((nh, nw), pixels, native, np.stack([uv[..., 1].ravel(), uv[..., 0].ravel()], -1))


def summary_flags(summary: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(str(flag.get("code")) for flag in summary.get("flags") or []))


def share_set_aside(result: engine.FitResult, placed: Placed) -> dict[str, float]:
    """The share of the section's tissue (the placed atlas outline, where it holds data) the
    EM sorting set aside as background (WB) and as artifact (WA)."""
    inside = placed.tissue
    maps = {"WB": result.background_weights, "WA": result.artifact_weights}
    return {
        key: round(float((np.asarray(weights[placed.slot], np.float32) > 0.5)[inside].mean()), 3)
        if inside.any()
        else 0.0
        for key, weights in maps.items()
    }


def weights_on_grid(stack: Stack, placed: Placed, stack_weights: np.ndarray) -> np.ndarray:
    """emlddmm's final sorting of one section (its slot of ``WM``, on the stack grid) as a
    float16 weight map on the section's fit grid, read through its current registration."""
    section = placed.section
    w, h = section.size
    gxy = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float64)
    uv = section.plane_um(section.registered(gxy)) + placed.offset_um
    ix = ((uv[..., 0] - stack.cols_um[0]) / stack.section_um).astype(np.float32)
    iy = ((uv[..., 1] - stack.rows_um[0]) / stack.section_um).astype(np.float32)
    got = sample_pixels(stack_weights.astype(np.float32), np.stack([ix, iy], -1))
    return np.clip(got, 0.0, 1.0).astype(np.float16)


def route_of(view: SectionView, *, rigid: bool, warp: bool) -> str:
    """How one section's result is written: ``transform`` (its transform, then the warp on top),
    ``deformation`` (the rigid change folded into its deformation) or ``none``."""
    free_a, free_d = "affine" not in view.frozen, "deformation" not in view.frozen
    if rigid and free_a and not view.deformation:
        return "transform" if (not warp or free_d) else "none"
    return "deformation" if (warp or rigid) and free_d else "none"


def chosen_settings(stack: Stack, options: Options, schedule: engine.Schedule) -> dict[str, Any]:
    """What the plugin picked from the stack, kept in settings.json."""
    return {
        "slot_um": stack.slot_um,
        "slot_spacing_along_normal_um": round(stack.spacing_um, 3),
        "n_slots": stack.n_slots,
        "stack_shape": list(stack.images.shape),
        "largest_plane_tilt_deg": round(stack.tilt_deg, 3),
        "section_px_um": stack.section_um,
        "section_stands_for_um": round(stack.weight_um, 2),
        "device": options.device,
        "dtype": options.dtype,
        "memory_estimate_gb": round(stack.memory_gb, 2),
        "memory_limit_gb": options.memory_limit_gb,
        "emlddmm": schedule.keywords(),
    }


@dataclass(frozen=True)
class _FoundPlane:
    """Hold the plane emlddmm found for the stack.

    Parameters
    ----------
    pitch_deg, yaw_deg
        The stack's cutting angles.
    positions
        Each section's plane at the found affine (position, angles).
    scale, shift_mm
        The spacing's stretch and shift as a whole.
    """

    pitch_deg: float
    yaw_deg: float
    positions: list[dict[str, float]]
    scale: float
    shift_mm: float


def found_plane(ctx: ToolContext, stack: Stack, result: engine.FitResult) -> _FoundPlane:
    """Read the found plane of every section from the fit's affine."""
    atlas = ctx.reads.atlas()
    centre = (np.asarray(atlas.shape, float) - 1) / 2.0 * np.asarray(atlas.resolution_um)
    inverse = np.linalg.inv(result.affine)
    found = [
        ctx.frames.angles_from_plane(
            result.affine[0, :3].tolist(),
            ((inverse @ np.array([z, 0.0, 0.0, 1.0]))[:3] + centre).tolist(),
        )
        for z in stack.depths_um
    ]
    old = np.array([float(p.section.view.position_mm) for p in stack.sections])
    new = np.array([f["position_mm"] for f in found])
    scale, shift_mm = (
        np.polyfit(old, new, 1) if np.ptp(old) > 0 else (1.0, float(np.mean(new - old)))
    )
    return _FoundPlane(
        found[0]["pitch_deg"], found[0]["yaw_deg"], found, float(scale), float(shift_mm)
    )
