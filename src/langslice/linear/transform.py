"""In-plane transforms: the closed-form fit and the interactive tools' core.

Two routes to the same field. :func:`fit_silhouette` is plain code — the shared
moments fit (:func:`langslice.affine.silhouette_affine`) of a section's
silhouette onto its atlas section. The interactive route is the main agent's
own hand: `preview_transform` / `landmarks` / `set_transform` in
:mod:`langslice.linear.toolbox`, whose arithmetic (calibration, the
decomposition it reports, the landmark fits) lives here.

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
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
from PIL import Image

from langslice.affine import (
    decompose_affine,
    extract_slice_silhouette,
    mask_affine,
    normalized_affine,
    silhouette_affine,
)
from langslice.atlas.render import atlas_um_per_px
from langslice.linear.render import (
    PREVIEW_LONG_EDGE,
    atlas_mask_canvas,
    canvas_geometry,
    canvas_um_per_px,
    estimate_um_per_px,
    physical_overlay,
    render_slice,
    template_canvas,
    zoom_box,
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


def _resolve_half_turn(
    matrix: np.ndarray,
    section: Image.Image,
    target_mask: np.ndarray,
    box: tuple[int, int, int, int],
    template: np.ndarray,
    section_offset: tuple[int, int],
) -> np.ndarray:
    """Pick between a moments fit and its 180-degree twin by template match.

    A moments fit knows the principal axes but not which way along them the
    tissue points, so two candidates differ by a half turn about the target
    centroid and tie on silhouette overlap — on an intact section the correct
    one wins by 0.04-0.09 IoU, inside an ROI on a damaged section by 0.003
    (measured on M05 D_08, 2026-09-06). Correlating the warped section with
    the atlas template inside the box breaks the tie with what the silhouette
    cannot see: the anatomy's brightness pattern.
    """
    ys, xs = np.nonzero(target_mask)
    if ys.size == 0:
        return matrix
    cx, cy = float(xs.mean()), float(ys.mean())
    half_turn = np.array([[-1.0, 0.0, 2.0 * cx], [0.0, -1.0, 2.0 * cy], [0.0, 0.0, 1.0]])
    square = np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])
    twin = (half_turn @ square)[:2]

    height, width = template.shape[:2]
    rgb = np.asarray(section.convert("RGB"), dtype=np.uint8)
    gray_section = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    placed = np.zeros((height, width), dtype=np.uint8)
    ox, oy = section_offset
    placed[oy : oy + gray_section.shape[0], ox : ox + gray_section.shape[1]] = gray_section
    reference = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY).astype(np.float32)
    x0, y0, x1, y1 = box

    def score(candidate: np.ndarray) -> float:
        warped = cv2.warpAffine(placed, candidate, (width, height), flags=cv2.INTER_LINEAR)
        a = warped[y0:y1, x0:x1].astype(np.float32)
        b = reference[y0:y1, x0:x1]
        a = a - a.mean()
        b = b - b.mean()
        denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denominator) if denominator > 0 else 0.0

    return twin if score(twin) > score(matrix) else matrix


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


def roi_masks(
    section: Image.Image, geometry: Any, roi: list[float]
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Section tissue and atlas anatomy on the canvas, both cut to *roi*.

    Both masks are built at IDENTITY on the physical canvas — the very frame
    ``preview_transform``'s zoom box names — so a box drawn on a preview is
    the box the fit measures in. Everything outside it is zeroed, which is the
    point: a section missing its cortex can still be fitted on the anatomy it
    kept.
    """
    width, height = geometry.size
    box = zoom_box(list(roi), geometry.size)
    tissue = np.zeros((height, width), dtype=np.uint8)
    gray = cv2.cvtColor(np.asarray(section.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    silhouette = extract_slice_silhouette(gray)
    ox, oy = geometry.section_offset
    tissue[oy : oy + silhouette.shape[0], ox : ox + silhouette.shape[1]] = silhouette
    keep = np.zeros((height, width), dtype=np.uint8)
    keep[box[1] : box[3], box[0] : box[2]] = 1
    return tissue * keep, atlas_mask_canvas(geometry) * keep, box


def fit_silhouette(
    state: StackState,
    ctx: EngineContext,
    record: SliceState,
    roi: list[float] | None = None,
) -> dict[str, Any]:
    """Fit the silhouette affine for one positioned section.

    Without *roi* the whole tissue outline is matched against the whole atlas
    outline (:func:`langslice.affine.silhouette_affine`, in its own frame).
    With *roi* — ``[fx0, fy0, fx1, fy1]`` of the CANVAS — only what lies
    inside that box is matched, on the physical canvas, which is what makes a
    damaged section fittable at all.

    Returns the tool-shaped payload: on success ``params`` (six normalized
    numbers on the section's frame), ``iou``, the ``physical`` knobs about the
    canvas centre, the ``calibration`` the panel was drawn with, and a
    ``panel`` image labelled with the section id. ``flat_atlas_fit`` is
    reported when the stack carries cutting angles and no roi: the moments fit
    measures against the flat atlas section, because it builds its own atlas
    silhouette from the voxel grid.
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
        if roi:
            source_mask, target_mask, box = roi_masks(section, geometry, roi)
            on_canvas, iou, _pattern = mask_affine(source_mask, target_mask)
            on_canvas = _resolve_half_turn(
                on_canvas, section, target_mask, box,
                template_canvas(
                    ctx.atlas, record.position_mm, cast(Plane, state.plane),
                    state.pitch_deg, state.yaw_deg, geometry,
                ),
                geometry.section_offset,
            )
            in_section = _conjugate(
                on_canvas, (-geometry.section_offset[0], -geometry.section_offset[1])
            )
        else:
            fit = silhouette_affine(
                section,
                atlas=ctx.atlas,
                position_mm=record.position_mm,
                plane=cast(Plane, state.plane),
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

    panel = physical_overlay(
        section,
        um_per_px,
        ctx.atlas,
        record.position_mm,
        cast(Plane, state.plane),
        state.pitch_deg,
        state.yaw_deg,
        in_section,
        label=record.id,
    )
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
        "panel": panel,
    }
    if roi:
        payload["roi"] = [
            round(box[0] / width, 4),
            round(box[1] / height, 4),
            round(box[2] / width, 4),
            round(box[3] / height, 4),
        ]
    elif state.is_oblique:
        payload["flat_atlas_fit"] = True
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
