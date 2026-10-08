"""In-plane transforms: the Elastix affine fit and the interactive tool's core.

Two routes to the same field. :func:`fit_elastix` (``elastix_affine``)
refines the section's current placement with an Elastix intensity affine on
the deformable package's prepared stain and atlas images. The interactive
route is the agent's own hand (``interactive_transform``,
:mod:`langslice.ops.transforms`), whose arithmetic (calibration and the
decomposition it reports) lives here.

The fit returns numbers, never pictures. Parameters are ABBA's — rotation
about the canvas centre (or a chosen pivot), per-axis scales, translations in
MILLIMETRES — which only mean anything once the canvas is calibrated, so
every payload carries the calibration and where it came from.

Everything here is a PROPOSAL: six normalized numbers recorded on the state,
never applied to the user's images (see :mod:`langslice.core.affine` for the
convention).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from PIL import Image

from langslice.core.affine import (
    IDENTITY_PARAMS,
    decompose_affine,
    denormalized_affine,
    normalized_affine,
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
        record.pitch_deg,
        record.yaw_deg,
    )
    return (estimated or atlas_um_per_px(ctx.atlas)), "estimated"


# --- the closed-form fit -------------------------------------------------

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
) -> dict[str, Any]:
    """The tool-shaped payload of one fit.

    *in_section* is the fitted 2x3 on the working frame *section*; the
    numbers stay on that frame.
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
    }
    if regions is not None:
        payload["regions"] = regions
    return payload


# --- the Elastix affine ----------------------------------------------------

#: The atlas image the Elastix affine fits the section to: the ARA
#: template's pial outline lies on a fluorescent section's bright rim.
ELASTIX_ATLAS_IMAGE = "template"
#: Working grid of the Elastix affine (``core.deformable.settings.DETAIL``: 20 um).
ELASTIX_DETAIL = "standard"


#: A restricted region must have support in both atlas-plane dimensions.
#: Three-voxel regions have crashed Elastix; this is a conservative preflight,
#: not a registration-quality score. Check before adding the fit's halo.
ELASTIX_MIN_REGION_SPAN = 4


class RegionTooSmall(ValueError):
    """An Elastix fit whose selected anatomy is too small to fit safely."""


def _check_elastix_region(prepared: Any, atlas: Any, include: Sequence[str]) -> None:
    if not include:
        return
    from langslice.core.deformable.atlas_images import placement_left, regions_mask

    left = placement_left(atlas, prepared.placement, include)
    mask = regions_mask(atlas, prepared.native, include, left)
    if prepared.excluded_mask is not None:
        mask &= ~prepared.excluded_mask
    ys, xs = np.nonzero(mask)
    span = (int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1) if xs.size else (0, 0)
    if min(span) < ELASTIX_MIN_REGION_SPAN or xs.size < ELASTIX_MIN_REGION_SPAN ** 2:
        raise RegionTooSmall(
            "Region is too small for elastix. Select a larger region or use "
            "interactive_transform; no fit was run. "
            f"Selected atlas support: {span[0]} x {span[1]} pixels, {xs.size} pixels total.")


def elastix_settings(
    include: Sequence[str] = (), exclude: Sequence[str] = (),
    atlas_image: str = ELASTIX_ATLAS_IMAGE,
) -> Any:
    """The deformable package's stain-fit inputs the Elastix affine reads.

    The section's preprocessed channel against *atlas_image* (``template``, the
    default :data:`ELASTIX_ATLAS_IMAGE`, or ``nissl`` on an Allen mouse atlas:
    ``elastix_affine``'s ``atlas_image``),
    mutual information plus the edge channel (the Elastix stain fit's own
    pairing). *include* becomes the restricted ``structures`` (the regions
    plus 300 um), *exclude* the excluded regions; one-sided entries such as
    ``"CTX:left"`` resolve as in every deformable fit.
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


def _overlap(prepared: Any, atlas: Any, atlas_to_section: np.ndarray,
             include: Sequence[str]) -> float:
    """Tissue against the kept atlas footprint under *atlas_to_section*.

    Excluded regions leave the atlas side, the tissue laid on them leaves the section side, and with
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
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    atlas_image: str = ELASTIX_ATLAS_IMAGE,
) -> ElastixFit:
    """Refine one section's placement with an Elastix intensity affine.

    Starts from *start_params* (the section's current six numbers, at the
    scale its pictures draw it, :func:`calibrate`) and never searches from
    scratch: the atlas plane is placed there on the deformable fit grid
    (``deformation.fit_grid``), the
    section's preprocessed channel (:func:`langslice.core.deformation.stain_image`)
    and the placed atlas template are prepared
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
    from langslice.core.deformable.engines import run_elastix_affine

    start = [float(v) for v in start_params]
    grid = deformation.fit_grid(state, ctx, record, transform={"params": start})
    image, look = deformation.stain_image(ctx, state, grid, deformation.FIT_LOOK)
    width, height = grid.image.size
    torn = None if record.damaged else np.zeros((height, width), dtype=bool)
    prepared = prepare_fit(image, ctx.atlas, grid.placement,
                           elastix_settings(include, exclude, atlas_image), torn_band=torn,
                           nissl=ctx.nissl_atlas if atlas_image == "nissl" else None)
    _check_elastix_region(prepared, ctx.atlas, include)
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
    )


def fit_elastix(
    state: StackState,
    ctx: Workspace,
    record: SliceState,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    atlas_image: str = ELASTIX_ATLAS_IMAGE,
) -> dict[str, Any]:
    """``elastix_affine`` for one positioned section.

    A local refinement of the section's CURRENT placement (its stored six
    numbers, or the identity without a transform) by :func:`elastix_affine`.
    The payload: ``params`` (six normalized numbers on the section's frame),
    ``iou`` (the overlap of the tissue and the kept atlas footprint under the
    fitted placement), the ``physical`` knobs about the canvas centre, the
    ``calibration`` of the working frame, and a ``regions`` report when
    regions were given.
    """
    from langslice.core.atlas.sides import SideError

    if record.position_mm is None:
        return {"status": "error", "error": "NO_POSITION", "id": record.id}
    section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
    stored = (record.transform or {}).get("params")
    start = (list(stored) if stored is not None and len(stored) == 6
             else list(IDENTITY_PARAMS))
    um_per_px, source = calibrate(state, ctx, record, section)
    try:
        geometry = canvas_geometry(
            section.size,
            um_per_px,
            ctx.atlas,
            record.position_mm,
            cast(Plane, state.plane),
            record.pitch_deg,
            record.yaw_deg,
        )
        fit = elastix_affine(state, ctx, record, start,
                             include=include, exclude=exclude, atlas_image=atlas_image)
    except RegionTooSmall as error:
        return {"status": "error", "error": "REGION_TOO_SMALL", "id": record.id,
                "message": str(error)}
    except SideError as error:
        return {"status": "error", "error": error.code, "id": record.id, "message": str(error)}
    except Exception as exc:
        logger.warning("elastix_affine failed for %s: %s", record.id, exc)
        return {
            "status": "error",
            "error": "FIT_FAILED",
            "id": record.id,
            "message": f"The Elastix affine failed: {exc}",
        }
    in_section = denormalized_affine(fit.params, section.size)
    return _fit_payload(record, section, um_per_px, source, geometry, in_section, fit.iou,
                        regions=fit.report if (include or exclude) else None)


# --- what the interactive tools share ------------------------------------


def physical_decomposition(params: Any, size: tuple[int, int]) -> dict[str, Any]:
    """decompose_affine without its translation fractions.

    In the alignment loop the shift IS the entered millimetres; the matrix's
    fractional offsets also absorb the pivot-based scale and rotation, so
    they would read as a contradiction ("+0.04 mm entered, negative fraction
    reported"). Rotation, scales, shear and mirrored stay.
    """
    out = decompose_affine(params, size)
    out.pop("translate_x_frac", None)
    out.pop("translate_y_frac", None)
    return out


def physical_params(
    matrix: np.ndarray, *, pivot: tuple[float, float], um_per_px: float
) -> dict[str, float]:
    """A 2x3 in canvas pixels as the knobs, about *pivot*.

    The inverse of :func:`langslice.core.affine.physical_affine_matrix`: rotation,
    scales and ``shear`` come out of the linear part
    (:func:`langslice.core.affine.decompose_affine`'s convention, the one the
    ``shear`` knob of ``interactive_transform`` takes), and the shift is what is
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
