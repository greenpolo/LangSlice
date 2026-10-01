"""In-plane transforms: the closed-form fit and the interactive tools' core.

Two routes to the same field. :func:`fit_silhouette` is plain code — the shared
moments fit (:func:`langslice.affine.silhouette_affine`) of a section's
silhouette onto its atlas section. The interactive route is the main agent's
own hand: `adjust_transforms` in :mod:`langslice.linear.toolbox`, whose
arithmetic (calibration and the decomposition it reports) lives here. Shared
point-fit geometry helpers are retained for analysis and historical results.

Both draw ONE picture, :func:`langslice.linear.render.physical_overlay`: the
section under its transform with the atlas family outlines on top at true
physical scale. Parameters are ABBA's — rotation about the canvas centre (or a
chosen pivot), per-axis scales, translations in MILLIMETRES — which only mean
anything once the canvas is calibrated, so every payload carries the
calibration and where it came from.

Everything here is a PROPOSAL: six normalized numbers recorded on the state,
never applied to the user's images (see :mod:`langslice.affine` for the
convention).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.affine import (
    decompose_affine,
    normalized_affine,
    silhouette_affine,
)
from langslice.atlas.render import atlas_um_per_px
from langslice.linear.atlas_fetch import atlas_mask
from langslice.linear.render import (
    OVERLAY_LONG_EDGE,
    PREVIEW_LONG_EDGE,
    canvas_geometry,
    canvas_um_per_px,
    estimate_um_per_px,
    physical_overlay,
    render_slice,
    rescale_section_matrix,
    shown_scale,
    shown_section,
)
from langslice.linear.state import SliceState, StackState
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

logger = logging.getLogger(__name__)

# --- calibration ---------------------------------------------------------


def calibrate(
    state: StackState, ctx: EngineContext, record: SliceState, section: Image.Image
) -> tuple[float, str]:
    """``(canvas micrometres per pixel, source)``, never failing.

    The file's tags or the host answer first (``"file"`` / ``"host"``). With
    neither, the section's tissue width against the atlas anatomy's gives a
    scale, reported as ``"estimated"`` — a shape fit, not a calibration, and
    the payloads say so. If even that is degenerate the atlas's own voxel size
    stands in, still ``"estimated"``: an alignment loop with an honest guess
    beats one that crashes.
    """
    known, source = canvas_um_per_px(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    if known is not None:
        return known, source
    estimated = estimate_um_per_px(
        section,
        ctx.atlas,
        record.position_mm or 0.0,
        cast(Plane, state.plane),
        state.pitch_deg,
        state.yaw_deg,
    )
    return (estimated or atlas_um_per_px(ctx.atlas)), "estimated"


# --- the closed-form fit -------------------------------------------------


def _fit_matrix_in_section_frame(
    fit_matrix: np.ndarray,
    fit_size: tuple[int, int],
    section: Image.Image,
    geometry: Any,
) -> np.ndarray:
    """The silhouette fit's 2x3, re-expressed on the physical overlay's canvas.

    The fit works in its own frame: the section resized to
    :data:`~langslice.affine.AFFINE_LONG_EDGE` and the atlas silhouette
    STRETCHED to that same frame. To draw it truthfully the whole chain has to
    be composed — section render -> fit frame -> atlas pixels -> canvas —
    otherwise the picture shows the fit against an atlas that is the wrong
    size, which is the very error physical calibration exists to remove.
    """
    fit_w, fit_h = fit_size
    rows, cols = geometry.annotation.shape[:2]
    to_section = fit_w / float(section.width)  # uniform: the fit keeps aspect
    kx = cols * geometry.atlas_scale / fit_w
    ky = rows * geometry.atlas_scale / fit_h
    ox = geometry.atlas_offset[0] - geometry.section_offset[0]
    oy = geometry.atlas_offset[1] - geometry.section_offset[1]
    chain = (
        np.array([[kx, 0.0, ox], [0.0, ky, oy], [0.0, 0.0, 1.0]])
        @ np.vstack([np.asarray(fit_matrix, dtype=np.float64), [0.0, 0.0, 1.0]])
        @ np.diag([to_section, to_section, 1.0])
    )
    return chain[:2]


def _conjugate(matrix: np.ndarray, offset: tuple[float, float]) -> np.ndarray:
    """The same map read on a frame shifted by *offset*.

    The section sits at ``geometry.section_offset`` on the canvas, so a 2x3
    written on one frame becomes the other's by conjugating with that shift —
    which is all that separates the fit's canvas answer from the six numbers,
    which live on the section.
    """
    ox, oy = float(offset[0]), float(offset[1])
    shift = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
    inverse = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
    square = np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])
    return (shift @ square @ inverse)[:2]


def fit_silhouette(
    state: StackState,
    ctx: EngineContext,
    record: SliceState,
    *,
    draw: Callable[[Image.Image, float, np.ndarray], list[Image.Image]] | None = None,
) -> dict[str, Any]:
    """Fit the silhouette affine for one positioned section.

    The whole tissue outline is matched against the whole atlas outline
    (:func:`langslice.affine.silhouette_affine`, in its own frame). Damaged
    sections are refused before this runs (see `toolbox.fit_affine`).

    Returns the tool-shaped payload: on success ``params`` (six normalized
    numbers on the section's frame), ``iou``, the ``physical`` knobs about the
    canvas centre, the ``calibration`` the panel was drawn with, and
    ``panels`` (the overlay labelled with the section id, or what *draw*
    returned for the working frame and fitted matrix). The fit measures against
    the atlas plane at the stack's cutting angles, the same plane every
    picture in the run shows.
    """
    if record.position_mm is None:
        return {"status": "error", "error": "NO_POSITION", "id": record.id}
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    um_per_px, source = calibrate(state, ctx, record, section)
    try:
        geometry = canvas_geometry(
            section.size,
            um_per_px,
            ctx.atlas,
            record.position_mm,
            cast(Plane, state.plane),
            state.pitch_deg,
            state.yaw_deg,
        )
        position = record.position_mm
        fit = silhouette_affine(
            section,
            atlas=ctx.atlas,
            position_mm=position,
            plane=cast(Plane, state.plane),
            atlas_mask_at=lambda size: atlas_mask(ctx, state, position, size),
        )
        iou = float(fit.iou)
        in_section = _fit_matrix_in_section_frame(
            fit.matrix, fit.size, section, geometry
        )
        on_canvas = _conjugate(in_section, geometry.section_offset)
    except Exception as exc:
        logger.warning("fit_affine: silhouette fit failed for %s: %s", record.id, exc)
        return {
            "status": "error",
            "error": "FIT_FAILED",
            "id": record.id,
            "message": str(exc),
        }

    # The panel may be drawn from a larger render (image_resolution); the fit
    # above and the numbers below stay on the working frame. *draw* (the
    # toolbox's display options) receives the working frame and the fitted
    # matrix on it and carries both onto whatever it draws from.
    if draw is not None:
        panels = draw(section, um_per_px, in_section)
    else:
        shown, shown_um, (fx, fy) = shown_section(ctx, record, section, um_per_px)
        panels = [physical_overlay(
            shown,
            shown_um,
            ctx.atlas,
            record.position_mm,
            cast(Plane, state.plane),
            state.pitch_deg,
            state.yaw_deg,
            in_section if shown is section else rescale_section_matrix(in_section, fx, fy),
            label=record.id,
            long_edge=OVERLAY_LONG_EDGE,
            scale=shown_scale(ctx),
        )]
    params = normalized_affine(in_section, section.size)
    width, height = geometry.size
    payload: dict[str, Any] = {
        "status": "ok",
        "id": record.id,
        "position_mm": round(record.position_mm, 3),
        "iou": round(float(iou), 3),
        "params": params,
        # The same five knobs the interactive tools take, about the canvas
        # centre: one representation for every transform this run records.
        "physical": {
            **physical_params(
                on_canvas, pivot=(width / 2.0, height / 2.0), um_per_px=um_per_px
            ),
            "pivot": [0.5, 0.5],
        },
        "mirrored": bool(decompose_affine(params)["mirrored"]),
        "calibration": {
            "section_um_per_px": round(um_per_px, 4),
            "source": source,
        },
        "panels": panels,
    }
    return payload


# --- what the interactive tools share ------------------------------------


def physical_decomposition(params: Any, size: tuple[int, int]) -> dict[str, Any]:
    """decompose_affine without its translation fractions.

    In the alignment loop the shift IS the entered millimetres; the matrix's
    fractional offsets also absorb the pivot-based scale and rotation, so they
    read as a contradiction (four agents flagged "+0.04 mm entered, negative
    fraction reported"). Rotation, scales, shear and mirrored stay.
    """
    out = decompose_affine(params, size)
    out.pop("translate_x_frac", None)
    out.pop("translate_y_frac", None)
    return out


def similarity_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 2x3 similarity (rotation, one scale, shift) mapping *src* to *dst*.

    Umeyama's closed form, so two points give the exact answer and more give
    the least-squares one. A similarity is what two landmarks can support: a
    per-axis scale needs three.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    x, y = src - mu_src, dst - mu_dst
    cov = (y.T @ x) / len(src)
    u, singular, vt = np.linalg.svd(cov)
    correction = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # never mirror the section
        correction[1, 1] = -1.0
    rotation = u @ correction @ vt
    variance = float((x**2).sum() / len(src))
    scale = float((singular * np.diag(correction)).sum() / variance) if variance > 0 else 1.0
    linear = scale * rotation
    return np.column_stack([linear, mu_dst - linear @ mu_src])


def affine_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The least-squares 2x3 affine mapping *src* to *dst*; needs 3+ points."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    design = np.column_stack([src, np.ones(len(src))])
    solution, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return np.asarray(solution.T, dtype=np.float64)


def physical_params(
    matrix: np.ndarray, *, pivot: tuple[float, float], um_per_px: float
) -> dict[str, float]:
    """A 2x3 in canvas pixels as the five knobs, about *pivot*.

    The inverse of :func:`langslice.affine.physical_affine_matrix`: rotation
    and scales come out of the linear part, and the shift is what is left of
    the translation once the pivot's own displacement is taken out. ``shear``
    rides along because a three-point affine can have some and the five knobs
    cannot express it.
    """
    values = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    linear, offset = values[:, :2], values[:, 2]
    centre = np.asarray(pivot, dtype=np.float64)
    shift = offset - centre + linear @ centre
    parts = decompose_affine(values)
    return {
        "rotation_deg": round(float(parts["rotation_deg"]), 3),
        "scale_x": round(float(parts["scale_x"]), 4),
        "scale_y": round(float(parts["scale_y"]), 4),
        "translate_x_mm": round(float(shift[0]) * um_per_px / 1000.0, 4),
        "translate_y_mm": round(float(shift[1]) * um_per_px / 1000.0, 4),
        "shear": round(float(parts["shear"]), 4),
    }
