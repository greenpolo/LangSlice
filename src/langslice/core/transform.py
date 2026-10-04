"""In-plane transforms: the closed-form fit and the interactive tools' core.

Three routes to the same field. :func:`fit_elastix` (``fit_affine``'s default)
refines the section's current placement with an Elastix intensity affine on
the deformable package's prepared stain and atlas images.
:func:`fit_silhouette` is plain code — the shared moments fit
(:func:`langslice.core.affine.silhouette_affine`) of a section's silhouette onto
its atlas section. The interactive route is the main agent's
own hand: `adjust_transforms` in :mod:`langslice.doors.tools.toolbox`, whose
arithmetic (calibration and the decomposition it reports) lives here. Shared
point-fit geometry helpers are retained for analysis and historical results.

The fits return numbers, never pictures: each ok payload carries its
:class:`FitFrame` (the working frame, its calibration and the fitted matrix
on it) under :data:`FIT_FRAME_KEY`, and the caller draws whatever picture it
wants from that. Parameters are ABBA's — rotation about the canvas centre (or a
chosen pivot), per-axis scales, translations in MILLIMETRES — which only mean
anything once the canvas is calibrated, so every payload carries the
calibration and where it came from.

Everything here is a PROPOSAL: six normalized numbers recorded on the state,
never applied to the user's images (see :mod:`langslice.core.affine` for the
convention).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from langslice.core.affine import (
    AFFINE_LONG_EDGE,
    axis_ratio,
    decompose_affine,
    denormalized_affine,
    mask_affine,
    normalized_affine,
    silhouette_affine,
    tissue_silhouette,
)
from langslice.core.atlas.render import atlas_um_per_px
from langslice.core.canvas import canvas_geometry, estimate_um_per_px
from langslice.core.sections import PREVIEW_LONG_EDGE, canvas_um_per_px, render_slice
from langslice.core.space import Plane
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

# --- calibration ---------------------------------------------------------


def calibrate(
    state: StackState, ctx: Workspace, record: SliceState, section: Image.Image
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

#: Key of the :class:`FitFrame` in an ok fit payload. A caller that draws
#: pops it; one that records the payload as JSON must drop it.
FIT_FRAME_KEY = "frame"


@dataclass(frozen=True)
class FitFrame:
    """What a picture of one fit is drawn from.

    ``section`` is the working frame the fit ran on (the oriented section at
    ``PREVIEW_LONG_EDGE``), ``um_per_px`` its calibration and ``matrix`` the
    fitted 2x3 on it (the six stored numbers, in pixels).
    """

    section: Image.Image
    um_per_px: float
    matrix: np.ndarray
    #: Native atlas-plane pixels on the section's displayed left, as the fit
    #: resolved them (from the placement it started from) when an include or
    #: exclude entry named a side; None otherwise. A picture of the fit
    #: highlights one-sided regions with this same split, so the picture
    #: shows the sides the fit used, however far the fit turned the section.
    left: np.ndarray | None = None

#: Below this long/short axis ratio an outline has no defined long axis, and a
#: region-restricted fit's reply says so with the rotation it chose (M04_D_08
#: with both hemispheres missing: tissue 1.14, kept atlas 1.59 -> turned 83 deg).
ROUND_AXIS_RATIO = 1.2


class RegionRefusal(ValueError):
    """A region restriction the silhouette method cannot honour, with its code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class RegionFit:
    """A silhouette fit restricted to atlas regions.

    ``matrix`` is on the section's frame, like the six stored numbers
    (section pixels -> the frame the atlas is drawn on, shifted by the
    section's canvas offset).
    """

    matrix: np.ndarray
    iou: float
    report: dict[str, Any] = field(default_factory=dict)
    #: The section's displayed left on the native plane, as the fit resolved
    #: it (None when no entry named a side).
    left: np.ndarray | None = None


def _place(mask: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """*mask* drawn through *matrix* onto a ``(w, h)`` frame, 0/255 (nearest)."""
    return cv2.warpAffine(mask.astype(np.uint8) * 255, np.asarray(matrix, dtype=np.float64),
                          size, flags=cv2.INTER_NEAREST, borderValue=0)


def _square(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(matrix, dtype=np.float64)[:2], [0.0, 0.0, 1.0]])


def region_silhouette_fit(
    section: Image.Image,
    geometry: Any,
    atlas: Any,
    current: np.ndarray,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    long_edge: int = AFFINE_LONG_EDGE,
    plane_at: tuple[float, str, float, float] = (0.0, "coronal", 0.0, 0.0),
) -> RegionFit:
    """The moments fit with atlas regions removed or singled out.

    Regions mean what they mean in ``fit_deformable`` and resolve the same
    way (acronyms or ids, descendants included, :mod:`langslice.core.deformable`):
    *exclude* leaves those regions out of the atlas footprint; *include*
    keeps only the footprint within ``DEFAULT_NEIGHBOURHOOD_UM`` of them. The
    section side is the tissue that corresponds under the section's
    *current* placement (its stored transform on the section frame, or
    identity): minus what that placement lays on excluded regions, and only
    what it lays within the included zone. One pass, no hidden iteration
    (iterating moved D_08 by ten degrees in eight passes without settling):
    a second call starts from the first call's answer.

    The fit runs on the TRUE-SCALE canvas (*geometry*), not the whole-outline
    fit's frame with the atlas stretched to the section's aspect, so its
    rotation and scales are physical. A silhouette carries only its
    outline, so an included zone that never reaches the atlas outline gives
    the fit nothing to measure (``REGIONS_INSIDE_OUTLINE``).

    An entry may name one side (``"CTX:left"``, :mod:`langslice.core.atlas.sides`):
    the section's side, carried onto the atlas plane through *current*;
    *plane_at* is the plane *geometry* was drawn at (position mm, plane,
    pitch, yaw), which the sides are derived from.
    """
    from langslice.core.atlas.sides import SideError, has_sides, native_left
    from langslice.core.deformable.atlas_images import regions_mask
    from langslice.core.deformable.masks import dilate, outline
    from langslice.core.deformable.settings import DEFAULT_NEIGHBOURHOOD_UM

    labels = np.asarray(geometry.annotation)
    footprint = labels != 0
    left: np.ndarray | None = None
    if has_sides([*include, *exclude]):
        # Sides are the section's: carry the native plane onto the section
        # frame through the CURRENT placement (native -> canvas is the atlas
        # scale; the section reaches the canvas through *current*).
        native_to_section = (np.linalg.inv(_square(current))[:2, :2]
                             * float(geometry.atlas_scale))
        try:
            left = native_left(atlas, *plane_at, native_to_section)
        except SideError as error:
            raise RegionRefusal(error.code, str(error)) from error
    dropped = (regions_mask(atlas, labels, exclude, left) & footprint
               if exclude else np.zeros(labels.shape, dtype=bool))
    kept = footprint & ~dropped
    if not kept.any():
        raise RegionRefusal("REGIONS_LEAVE_NOTHING",
                            "Excluding these regions leaves no atlas at this position.")
    report: dict[str, Any] = {
        "include": list(include), "exclude": list(exclude),
        "atlas_kept_fraction": round(float(kept.sum()) / float(footprint.sum()), 3),
    }
    notes: list[str] = []
    near: np.ndarray | None = None
    zone = kept
    if include:
        wanted = regions_mask(atlas, labels, include, left) & kept
        if not wanted.any():
            raise RegionRefusal("REGIONS_ABSENT", "None of the included regions is in the "
                                "atlas plane at this position.")
        near = dilate(wanted, DEFAULT_NEIGHBOURHOOD_UM / atlas_um_per_px(atlas))
        zone = kept & near
        rim = outline(kept)
        share = float((rim & near).sum()) / float(max(1, rim.sum()))
        report["outline_share"] = round(share, 3)
        if share == 0.0:
            raise RegionRefusal(
                "REGIONS_INSIDE_OUTLINE",
                "The included regions (plus 300 um) do not reach the atlas outline at this "
                "position. A silhouette fit compares outlines only, so it has nothing to "
                "fit them by; use adjust_transforms or fit_deformable.")
        notes.append(f"Only the outline counts: the included regions reach {share:.0%} of "
                     "the atlas outline, and only that part steers this fit.")
    if exclude and not (outline(footprint) & dropped).any():
        notes.append("The excluded regions do not reach the atlas outline here, so they "
                     "barely change a silhouette fit.")

    # One true-scale frame: the canvas, shrunk so its long edge is *long_edge*.
    scale = long_edge / float(max(geometry.size))
    size = (max(1, round(geometry.size[0] * scale)), max(1, round(geometry.size[1] * scale)))
    shrink = np.diag([scale, scale, 1.0])
    ox, oy = (float(v) for v in geometry.section_offset)
    to_canvas = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
    atlas_to_frame = shrink @ np.array([
        [geometry.atlas_scale, 0.0, geometry.atlas_offset[0]],
        [0.0, geometry.atlas_scale, geometry.atlas_offset[1]],
        [0.0, 0.0, 1.0],
    ])
    tissue_small, _rgb = tissue_silhouette(
        section, max(1, round(max(section.size) * scale)))
    section_to_frame = shrink @ to_canvas @ np.diag(
        [section.width / tissue_small.shape[1], section.height / tissue_small.shape[0], 1.0])
    tissue = _place(tissue_small > 0, section_to_frame[:2], size) > 0
    zone_f = _place(zone, atlas_to_frame[:2], size)

    # The current placement on this frame: frame px -> where it lays the atlas.
    placed = shrink @ to_canvas @ _square(current) @ np.linalg.inv(to_canvas) @ np.linalg.inv(
        shrink)
    used = tissue.copy()
    h, w = tissue.shape
    for mask, keep in ((dropped if exclude else None, False), (near, True)):
        if mask is None:
            continue
        on_mask = cv2.warpAffine(
            _place(mask, atlas_to_frame[:2], size), placed[:2], (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0) > 0
        used &= on_mask if keep else ~on_mask
    if not used.any():
        raise RegionRefusal("NO_TISSUE_IN_REGIONS", "The current placement lays no tissue on "
                            "the kept regions; the restriction leaves nothing to fit.")
    fitted, iou, _pattern = mask_affine(used.astype(np.uint8) * 255, zone_f)
    ratios = (axis_ratio(used.astype(np.uint8)), axis_ratio(zone_f))
    report["axis_ratio"] = {"tissue": round(ratios[0], 2), "atlas": round(ratios[1], 2)}
    turn = abs((decompose_affine(fitted)["rotation_deg"]
                - decompose_affine(placed[:2])["rotation_deg"] + 180.0) % 360.0 - 180.0)
    if min(ratios) < ROUND_AXIS_RATIO or turn > 45.0:
        notes.append(
            f"The fit turns the section {turn:.0f} degrees from its current placement: a "
            "moments fit turns the tissue's long axis onto the kept atlas's, and their axis "
            f"ratios are {ratios[0]:.2f} (tissue) and {ratios[1]:.2f} (atlas); an outline "
            "with a ratio near 1.0 has no defined long axis.")
    # Back from the frame to the section's own frame (the six stored numbers').
    on_canvas = np.linalg.inv(shrink) @ _square(fitted) @ shrink
    in_section = np.linalg.inv(to_canvas) @ on_canvas @ to_canvas
    report["tissue_used_fraction"] = round(float(used.sum()) / float(tissue.sum()), 3)
    if notes:
        report["note"] = " ".join(notes)
    return RegionFit(matrix=in_section[:2], iou=float(iou), report=report, left=left)


def _fit_matrix_in_section_frame(
    fit_matrix: np.ndarray,
    fit_size: tuple[int, int],
    section: Image.Image,
    geometry: Any,
) -> np.ndarray:
    """The silhouette fit's 2x3, re-expressed on the physical overlay's canvas.

    The fit works in its own frame: the section resized to
    :data:`~langslice.core.affine.AFFINE_LONG_EDGE` and the atlas silhouette
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
    ctx: Workspace,
    record: SliceState,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """Fit the silhouette affine for one positioned section.

    With *include* / *exclude* the fit is :func:`region_silhouette_fit`'s
    (the payload adds its ``regions`` report, and ``iou`` is the overlap of
    the kept atlas footprint with the tissue that corresponds); without them
    it is the whole-outline fit, unchanged.

    The whole tissue outline is matched against the whole atlas outline
    (:func:`langslice.core.affine.silhouette_affine`, in its own frame). Damaged
    sections are refused before this runs (see `ops.transforms.fit_affine`).

    Returns the tool-shaped payload: on success ``params`` (six normalized
    numbers on the section's frame), ``iou``, the ``physical`` knobs about the
    canvas centre, the ``calibration`` of the working frame, and the
    :class:`FitFrame` to draw it from (:data:`FIT_FRAME_KEY`). The fit measures against
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
        regions: dict[str, Any] | None = None
        left: np.ndarray | None = None
        if include or exclude:
            stored = (record.transform or {}).get("params")
            current = (denormalized_affine(stored, section.size)
                       if stored is not None and len(stored) == 6 else np.eye(3)[:2])
            restricted = region_silhouette_fit(section, geometry, ctx.atlas, current,
                                               plane_at=(position, str(state.plane),
                                                         state.pitch_deg, state.yaw_deg),
                                               include=include, exclude=exclude)
            regions = restricted.report
            left = restricted.left
            iou = restricted.iou
            in_section = restricted.matrix
        else:
            fit = silhouette_affine(
                section,
                atlas=ctx.atlas,
                position_mm=position,
                plane=cast(Plane, state.plane),
                pitch_deg=state.pitch_deg,
                yaw_deg=state.yaw_deg,
            )
            iou = float(fit.iou)
            in_section = _fit_matrix_in_section_frame(
                fit.matrix, fit.size, section, geometry
            )
    except RegionRefusal as refusal:
        return {"status": "error", "error": refusal.code, "id": record.id,
                "message": str(refusal)}
    except Exception as exc:
        logger.warning("fit_affine: silhouette fit failed for %s: %s", record.id, exc)
        return {
            "status": "error",
            "error": "FIT_FAILED",
            "id": record.id,
            "message": str(exc),
        }

    return _fit_payload(record, section, um_per_px, source, geometry, in_section, iou,
                        regions=regions, left=left)


def _fit_payload(
    record: SliceState,
    section: Image.Image,
    um_per_px: float,
    source: str,
    geometry: Any,
    in_section: np.ndarray,
    iou: float,
    *,
    regions: dict[str, Any] | None,
    left: np.ndarray | None = None,
) -> dict[str, Any]:
    """The tool-shaped payload of one fit, whichever method made it.

    *in_section* is the fitted 2x3 on the working frame *section*; the
    numbers stay on that frame, and the :class:`FitFrame` lets a caller
    carry the matrix onto whatever render it draws from.
    """
    assert record.position_mm is not None
    on_canvas = _conjugate(in_section, geometry.section_offset)
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
        FIT_FRAME_KEY: FitFrame(section=section, um_per_px=um_per_px, matrix=in_section,
                                left=left),
    }
    if regions is not None:
        payload["regions"] = regions
    return payload


# --- the Elastix affine ----------------------------------------------------

#: The atlas image the Elastix affine fits the section to. The 2026-10-02
#: stain ceiling test (deformable ``CLAUDE.md``) found the ARA template's pial
#: outline on the fluorescent sections' bright rim, where ABBA's Nissl sat
#: 40-80 um inside it.
ELASTIX_ATLAS_IMAGE = "ara"
#: Working grid of the Elastix affine (``core.deformable.settings.DETAIL``: 20 um).
ELASTIX_DETAIL = "standard"
#: The identity transform's six normalized numbers.
IDENTITY_PARAMS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def elastix_settings(
    include: Sequence[str] = (), exclude: Sequence[str] = (),
    atlas_image: str = ELASTIX_ATLAS_IMAGE,
) -> Any:
    """The deformable package's stain-fit inputs the Elastix affine reads.

    The section's ``fit`` appearance against *atlas_image* (``ara``, the
    default :data:`ELASTIX_ATLAS_IMAGE`, or ``nissl`` where ABBA's atlas is
    installed: ``fit_affine``'s ``fit_atlas``),
    mutual information plus the edge channel (the Elastix stain fit's own
    pairing). *include* becomes the restricted ``structures`` (the regions
    plus 300 um), *exclude* the excluded regions; one-sided entries such as
    ``"CTX:left"`` resolve as in ``fit_deformable``.
    """
    from langslice.core.deformable import FitSettings

    return FitSettings(
        engine="elastix", detail=ELASTIX_DETAIL,  # type: ignore[arg-type]
        atlas_image=atlas_image,  # type: ignore[arg-type]
        section_image="stain", stain_metric="mutual_information", stain_edges=True,
        exclude=tuple(exclude), structures=tuple(include),
    )


@dataclass
class ElastixFit:
    """One Elastix affine on the section's fit grid.

    ``params`` and ``start_params`` are the six normalized numbers (the
    section's frame, as stored); the placements map native atlas-plane pixels
    onto the fit grid (``deformable.Placement``) before and after.
    """

    params: list[float]
    start_params: list[float]
    iou: float
    start_iou: float
    report: dict[str, Any]
    placement: Any
    start_placement: Any
    grid_image: Image.Image
    fit_image: Image.Image
    engine: dict[str, Any]
    #: The section's displayed left on the native plane, resolved once from
    #: the start placement (None when no entry named a side).
    left: np.ndarray | None = None


def _overlap(prepared: Any, atlas: Any, atlas_to_section: np.ndarray,
             include: Sequence[str]) -> float:
    """Tissue against the kept atlas footprint under *atlas_to_section*.

    Mirrors the silhouette fit's region overlap: excluded regions leave the
    atlas side, the tissue laid on them leaves the section side, and with
    *include* both are limited to those regions plus 300 um.
    """
    from langslice.core.deformable.atlas_images import placement_left, regions_mask
    from langslice.core.deformable.geometry import warp_affine
    from langslice.core.deformable.masks import dilate
    from langslice.core.deformable.settings import DEFAULT_NEIGHBOURHOOD_UM

    native = prepared.native
    dropped = prepared.excluded_mask
    kept_native = native != 0
    if dropped is not None:
        kept_native &= ~dropped
    zone = None
    if include:
        left = placement_left(atlas, prepared.placement, include)
        chosen = regions_mask(atlas, native, include, left) & kept_native
        zone = dilate(chosen, DEFAULT_NEIGHBOURHOOD_UM / atlas_um_per_px(atlas))
    height, width = prepared.tissue.shape

    def placed(mask: np.ndarray) -> np.ndarray:
        return warp_affine(mask.astype(np.uint8), atlas_to_section, (width, height),
                           nearest=True) > 0

    kept = placed(kept_native)
    tissue = prepared.tissue.copy()
    if dropped is not None:
        tissue &= ~placed(dropped)
    if zone is not None:
        near = placed(zone)
        kept &= near
        tissue &= near
    union = float((kept | tissue).sum())
    return float((kept & tissue).sum()) / union if union else 0.0


def elastix_affine(
    state: StackState,
    ctx: Workspace,
    record: SliceState,
    start_params: Sequence[float],
    calibration: dict[str, Any],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    atlas_image: str = ELASTIX_ATLAS_IMAGE,
) -> ElastixFit:
    """Refine one section's placement with an Elastix intensity affine.

    Starts from *start_params* (the section's current six numbers, drawn
    with *calibration*) and never searches from scratch: the atlas plane is
    placed there on the deformable fit grid (``deformation.fit_grid``), the
    section's ``fit`` appearance and the placed ARA template are prepared
    exactly as a deformable stain fit prepares them (tissue and atlas masks,
    excluded regions blanked, the edge channel; ``deformable.prepare_fit``),
    and :func:`langslice.core.deformable.engines.run_elastix_affine` fits an affine
    from the identity. An intact section gets no torn-edge band (its whole
    outline is real, and the band's rule would also mark outline the start
    placement merely overhangs); a damaged one gets the deformable fit's
    automatic band. Raises on a placement, region or engine failure.
    """
    from dataclasses import replace

    from langslice.core import deformation
    from langslice.core.deformable import prepare_fit
    from langslice.core.deformable.atlas_images import placement_left
    from langslice.core.deformable.engines import run_elastix_affine

    start = [float(v) for v in start_params]
    grid = deformation.fit_grid(state, ctx, record,
                                transform={"params": start, "calibration": dict(calibration)})
    image, look = deformation.stain_image(ctx, state, grid, deformation.FIT_LOOK)
    width, height = grid.image.size
    torn = None if record.damaged else np.zeros((height, width), dtype=bool)
    prepared = prepare_fit(image, ctx.atlas, grid.placement,
                           elastix_settings(include, exclude, atlas_image), torn_band=torn,
                           abba=ctx.abba_atlas if atlas_image == "nissl" else None)
    result = run_elastix_affine(prepared.inputs)
    # Millimetres on the fit grid are pixel index x mm/px (geometry.py), so
    # the engine's map becomes a map of grid pixels: section -> placed atlas.
    mm = grid.placement.section_mm_per_px
    to_mm = np.diag([mm, mm, 1.0])
    step = np.linalg.inv(to_mm) @ result.matrix_mm @ to_mm
    # The stored matrix M draws the section onto the atlas's frame; a section
    # point p now meets the atlas where p's step lands, so the new matrix is
    # M @ step and the atlas reaches the section through inv(step).
    matrix = np.vstack([denormalized_affine(start, (width, height)), [0.0, 0.0, 1.0]]) @ step
    params = normalized_affine(matrix[:2], (width, height))
    fitted = replace(grid.placement,
                     atlas_to_section=np.linalg.inv(step) @ grid.placement.atlas_to_section)
    footprint = float((prepared.native != 0).sum())
    kept = footprint - (float((prepared.excluded_mask & (prepared.native != 0)).sum())
                        if prepared.excluded_mask is not None else 0.0)
    report: dict[str, Any] = {
        "include": list(include), "exclude": list(exclude),
        "atlas_kept_fraction": round(kept / footprint, 3) if footprint else 0.0,
    }
    return ElastixFit(
        params=params, start_params=start,
        iou=_overlap(prepared, ctx.atlas, fitted.atlas_to_section, include),
        start_iou=_overlap(prepared, ctx.atlas, grid.placement.atlas_to_section, include),
        report=report, placement=fitted, start_placement=grid.placement,
        grid_image=grid.image, fit_image=image,
        engine={"version": result.engine_version, "runtime_s": round(result.runtime_s, 2),
                "fit_look": look, "native_parameters": result.native_parameters},
        left=placement_left(ctx.atlas, grid.placement, [*include, *exclude]),
    )


def _start_calibration(
    state: StackState, ctx: Workspace, record: SliceState, section: Image.Image,
    stored: bool,
) -> tuple[float, str]:
    """The calibration the section's current placement was drawn with.

    The file's or host's answer when there is one; otherwise the stored
    transform's own (its six numbers mean that placement only at that
    scale), and only without either a fresh :func:`calibrate`.
    """
    known, source = canvas_um_per_px(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    if known is not None:
        return known, source
    if stored:
        held = (record.transform or {}).get("calibration") or {}
        try:
            value = float(held["section_um_per_px"])
        except (KeyError, TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0:
            return value, str(held.get("source") or "stored")
    return calibrate(state, ctx, record, section)


def fit_elastix(
    state: StackState,
    ctx: Workspace,
    record: SliceState,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    atlas_image: str = ELASTIX_ATLAS_IMAGE,
) -> dict[str, Any]:
    """`fit_affine`'s Elastix method for one positioned section.

    A local refinement of the section's CURRENT placement (its stored six
    numbers, or the identity without a transform) by :func:`elastix_affine`;
    the payload is :func:`fit_silhouette`'s, with ``iou`` the overlap of the
    tissue and the kept atlas footprint under the fitted placement, and a
    ``regions`` report when regions were given.
    """
    from langslice.core.atlas.sides import SideError

    if record.position_mm is None:
        return {"status": "error", "error": "NO_POSITION", "id": record.id}
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    stored = (record.transform or {}).get("params")
    start = (list(stored) if stored is not None and len(stored) == 6
             else list(IDENTITY_PARAMS))
    has_stored = stored is not None and len(stored) == 6
    um_per_px, source = _start_calibration(state, ctx, record, section, has_stored)
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
        fit = elastix_affine(state, ctx, record, start,
                             {"section_um_per_px": um_per_px, "source": source},
                             include=include, exclude=exclude, atlas_image=atlas_image)
    except SideError as error:
        return {"status": "error", "error": error.code, "id": record.id, "message": str(error)}
    except Exception as exc:
        logger.warning("fit_affine: elastix fit failed for %s: %s", record.id, exc)
        return {
            "status": "error",
            "error": "FIT_FAILED",
            "id": record.id,
            "message": f"The Elastix affine failed: {exc}",
        }
    in_section = denormalized_affine(fit.params, section.size)
    return _fit_payload(record, section, um_per_px, source, geometry, in_section, fit.iou,
                        regions=fit.report if (include or exclude) else None, left=fit.left)


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
    """A 2x3 in canvas pixels as the knobs, about *pivot*.

    The inverse of :func:`langslice.core.affine.physical_affine_matrix`: rotation,
    scales and ``shear`` come out of the linear part
    (:func:`langslice.core.affine.decompose_affine`'s convention, the one the
    ``shear`` knob of ``adjust_transforms`` takes), and the shift is what is
    left of the translation once the pivot's own displacement is taken out.
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
