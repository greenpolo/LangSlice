"""One deformable fit of a placed atlas plane onto a section: prepare, run, record.

Route A (no image model): the section's stain image against an atlas image
(one of the atlas's images; in practice mostly an outline fit), optionally in
sequential steps each restricted to the neighbourhood of chosen structures.
Route B (image model): the model's extracted lines against atlas borders
rendered from the same merged region set the model was shown, or
(``labels="model"``) the lines turned into named regions against the placed
atlas regions.

Everything that needs the atlas object happens in :func:`prepare_fit`; the
engine call works on plain arrays, so :func:`fit_prepared` runs several
fits in a process pool and builds the records back in the caller.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from langslice.core.atlas.levels import MODEL_DEPTH, labels_at
from langslice.core.atlas.sides import split_side
from langslice.core.deformable.atlas_images import (
    native_labels,
    placed_atlas_image,
    placement_left,
    regions_mask,
    resolve_structures,
    soft_lines,
    structure_acronyms,
    ventricle_ids,
    whole_region_ids,
)
from langslice.core.deformable.engines import (
    FIT_THREADS,
    EngineInputs,
    EngineResult,
    engine_worker,
    fit_device,
    run_engine,
)
from langslice.core.deformable.geometry import Placement, WorkingGrid, sample_native
from langslice.core.deformable.masks import (
    EM_PARAMETERS,
    MASK_MARGIN_MM,
    Masks,
    dilate,
    em_masks,
    footprint_masks,
)
from langslice.core.deformable.record import DeformableRecord, diagnose
from langslice.core.deformable.regions import named_regions
from langslice.core.deformable.settings import (
    CORRELATION_RADIUS_UM,
    DETAIL,
    EDGE_CHANNEL_WEIGHT,
    EDGE_SIGMA_UM,
    LINE_SOFTENING_UM,
    FitSettings,
)
from langslice.core.oblique import plane_index_coordinates
from langslice.core.space import atlas_space_context

#: Metric weight of each label-map channel; ANTs' own label registration
#: weights every label pair 1.0, next to 1.0 for the image pair.
LABEL_CHANNEL_WEIGHT = 1.0
#: Stain normalization percentiles inside the fixed mask.
STAIN_PERCENTILES = (1.0, 99.0)
#: A region-restricted step keeps its full displacement inside the region's
#: fit neighbourhood and fades it (raised cosine) to none over this distance
#: beyond: ANTs' masks only limit where the metric is measured, so its smooth
#: field otherwise moves tissue far from the region.
RESTRICTED_TAPER_MM = 0.3
#: Fixed-point iterations (and relaxation) that invert a tapered step.
TAPER_INVERSE_ITERATIONS = 200
TAPER_INVERSE_RELAXATION = 0.5


@dataclass
class PreparedFit:
    """Everything a fit needs, with the engine's part as plain arrays. A warp no
    engine computes (:func:`prepare_warp`) has no *settings* and no *inputs*."""

    settings: FitSettings | None
    placement: Placement
    grid: WorkingGrid
    inputs: EngineInputs | None
    native: np.ndarray
    #: The placed atlas footprint the fixed mask starts from, on the section grid.
    footprint: np.ndarray
    #: Ids excluded on both sides (whole regions; diagnostics skip them).
    excluded: frozenset[int]
    ventricles: frozenset[int]
    atlas_meta: dict[str, Any]
    previous: DeformableRecord | None = None
    details: dict[str, Any] = field(default_factory=dict)
    #: Native-plane pixels excluded from the fit, one-sided entries included.
    excluded_mask: np.ndarray | None = None
    #: A restricted step's weight on the section grid (1 in the region's fit
    #: neighbourhood, fading to 0 over :data:`RESTRICTED_TAPER_MM`); None
    #: for a fit over the whole section.
    step_weight: np.ndarray | None = None
    #: A whole-section stain fit's damage zone on the working grid: the fit
    #: runs twice, the second time with the EM section mask
    #: (:func:`run_prepared`, ``masks.em_masks``). None: one run with the
    #: footprint masks (the Elastix affine, restricted steps, traced lines).
    em_damage_zone: np.ndarray | None = None


@dataclass
class CandidateFailure:
    """A candidate whose engine raised; the other candidates still return."""

    settings: FitSettings
    error: str


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def _normalize_stain(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = gray[mask] if mask.any() else gray.ravel()
    lo, hi = np.percentile(values, STAIN_PERCENTILES)
    return np.clip((gray - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def edge_image(image: np.ndarray, region: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gradient magnitude after a light Gaussian, scaled to [0, 1] inside *region*.

    The edge channel of a stain fit. The magnitude ignores which side of an
    edge is brighter, so a fluorescent section (bright tissue), a brightfield
    one (dark tissue) and the atlas image all show the same pial surface,
    ventricle walls, fibre-tract and layer boundaries, whatever their
    intensities do region by region.
    """
    smooth = cv2.GaussianBlur(np.asarray(image, dtype=np.float32), (0, 0), max(sigma_px, 0.5))
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    values = magnitude[region] if region.any() else magnitude.ravel()
    top = float(np.percentile(values, STAIN_PERCENTILES[1])) if values.size else 0.0
    if top <= 0:
        return np.zeros(magnitude.shape, dtype=np.float32)
    return np.clip(magnitude / top, 0.0, 1.0).astype(np.float32)


def _fill_from_surroundings(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """*image* with *hole* filled by OpenCV's Telea inpainting (no step at its rim)."""
    top = float(image.max()) or 1.0
    scaled = np.clip(image / top * 255.0, 0, 255).astype(np.uint8)
    filled = cv2.inpaint(scaled, hole.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return np.where(hole, filled.astype(np.float32) / 255.0 * top, image).astype(np.float32)


def _remap(image: np.ndarray, offset_px: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    """Sample *image* at q + offset_px(q) (pixels), outside = 0."""
    height, width = image.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    return cv2.remap(
        image.astype(np.float64 if nearest else np.float32),
        (xx + offset_px[..., 0]).astype(np.float32), (yy + offset_px[..., 1]).astype(np.float32),
        cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(image.dtype)


def _sample_field(field_mm: np.ndarray, coordinates_px: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, 2) field at (H, W, 2) [x, y] pixel coordinates."""
    rows, cols = coordinates_px[..., 1], coordinates_px[..., 0]
    return np.stack([
        ndi.map_coordinates(field_mm[..., k], [rows, cols], order=1, mode="nearest")
        for k in range(2)
    ], axis=-1).astype(np.float32)


def _native_coords(matrix: np.ndarray, offset_px: np.ndarray | None,
                   shape: tuple[int, int]) -> np.ndarray:
    """Native atlas coords of every pixel q (+ offset) through inverse(*matrix*)."""
    yy, xx = np.indices(shape, dtype=np.float64)
    if offset_px is not None:
        xx = xx + offset_px[..., 0]
        yy = yy + offset_px[..., 1]
    inverse = np.linalg.inv(matrix)
    return np.stack([inverse[k, 0] * xx + inverse[k, 1] * yy + inverse[k, 2]
                     for k in range(2)], axis=-1)


def _atlas_meta(atlas: Any, placement: Placement, native: np.ndarray) -> dict[str, Any]:
    coords = plane_index_coordinates(
        atlas, placement.position_mm, placement.plane, placement.pitch_deg, placement.yaw_deg,
    )
    origin = coords[:, 0, 0]
    if coords.shape[1] < 2 or coords.shape[2] < 2:
        raise ValueError("Atlas plane is too small to fit")
    matrix = np.stack([coords[:, 0, 1] - origin, coords[:, 1, 0] - origin, origin], axis=1)
    context = atlas_space_context(atlas)
    return {
        "atlas_name": str(getattr(atlas, "atlas_name", placement.atlas_name)),
        "orientation": context.orientation,
        "resolution_um": list(context.resolution_um),
        "native_size": [int(native.shape[1]), int(native.shape[0])],
        "native_to_volume_index": matrix.tolist(),
        "native_to_volume_note": (
            "rows = atlas volume axes in the atlas's own order; volume_index = "
            "M[:, 0] * native_x + M[:, 1] * native_y + M[:, 2]"
        ),
    }


def prepare_fit(
    section_image: Image.Image,
    atlas: Any,
    placement: Placement,
    settings: FitSettings,
    *,
    lines: np.ndarray | None = None,
    previous: DeformableRecord | None = None,
) -> PreparedFit:
    """Build the working-grid images and masks for one candidate.

    *section_image* is the section on the grid *placement* refers to; *lines*
    (route B) is the model's line mask on that same grid. With *previous*,
    this is the next step of a sequential fit: the atlas starts where
    *previous* left it and the result composes onto it.
    """
    width, height = section_image.size
    if settings.section_image == "lines" or settings.labels == "model":
        if lines is None or np.shape(lines) != (height, width):
            raise ValueError("This setting needs the model's line mask on the section grid")

    grid, native, working_coords, current, footprint = _placed_atlas(
        atlas, placement, (width, height), settings.detail, previous)
    spacing = float(np.mean(grid.spacing_mm))
    atlas_to_working = grid.section_to_working @ placement.atlas_to_section
    # One-sided entries ("CTX:left") need the placement's displayed left.
    left = placement_left(atlas, placement, (*settings.exclude, *settings.structures))
    excluded = whole_region_ids(atlas, settings.exclude)
    excluded_native = (regions_mask(atlas, native, settings.exclude, left)
                       if settings.exclude else None)
    ventricles = ventricle_ids(atlas)
    offset = _previous_offset(grid, previous)
    excluded_w = (sample_native(excluded_native, working_coords) if excluded_native is not None
                  else np.zeros(current.shape, dtype=bool))
    softening_px = LINE_SOFTENING_UM / 1000.0 / spacing

    moving = placed_atlas_image(
        settings.atlas_image, native, atlas, placement, atlas_to_working, grid.size,
        excluded_native, softening_px=softening_px,
    )
    if offset is not None:
        moving = _remap(moving, offset)

    lines_w = None
    if lines is not None:
        lines_w = grid.to_working(np.asarray(lines, dtype=np.float32)) > 0
    if settings.section_image == "lines":
        assert lines_w is not None
        fixed = soft_lines(lines_w, softening_px)
    else:
        fixed = grid.to_working(_gray(section_image))
        fixed = _normalize_stain(fixed, footprint)

    kept = (current != 0) & ~excluded_w
    masks = footprint_masks(current, excluded_w, spacing)
    fixed_mask, moving_mask = masks.fixed.copy(), masks.moving.copy()

    details: dict[str, Any] = {}
    region: np.ndarray | None = None
    step_weight: np.ndarray | None = None
    if settings.structures:
        chosen = regions_mask(atlas, native, settings.structures, left)
        in_reach = sample_native(chosen, working_coords) & kept
        if not in_reach.any():
            raise ValueError("None of the chosen structures is present at this placement")
        region = in_reach
        neighbourhood = dilate(in_reach, settings.neighbourhood_um / 1000.0 / spacing)
        fixed_mask &= neighbourhood
        moving_mask &= neighbourhood
        details["structures"] = sorted(resolve_structures(
            atlas, [split_side(entry)[0] for entry in settings.structures]))
        details["neighbourhood_um"] = settings.neighbourhood_um
        step_weight = _taper_weight(in_reach, spacing, settings.neighbourhood_um / 1000.0,
                                    (width, height))
        details["taper_um"] = [settings.neighbourhood_um,
                               settings.neighbourhood_um + RESTRICTED_TAPER_MM * 1000.0]

    fixed_labels: list[np.ndarray] = []
    moving_labels: list[np.ndarray] = []
    channel_names: list[str] = []

    def add_channel(name: str, section_side: np.ndarray, atlas_side: np.ndarray) -> None:
        blur = max(softening_px, 0.5)
        fixed_labels.append(cv2.GaussianBlur(section_side.astype(np.float32), (0, 0), blur))
        moving_labels.append(cv2.GaussianBlur(atlas_side.astype(np.float32), (0, 0), blur))
        channel_names.append(name)

    if settings.labels == "model":
        assert lines_w is not None
        merged = labels_at(current, atlas, MODEL_DEPTH)
        merged = np.where(kept, merged, 0)
        section_regions, unnamed, report = named_regions(lines_w, merged, footprint, spacing)
        fixed_mask &= ~unnamed
        allowed = set(np.unique(section_regions)) & set(np.unique(merged))
        allowed.discard(0)
        if region is not None:
            allowed &= {int(i) for i in np.unique(merged[region])}
        for uid in sorted(int(i) for i in allowed):
            add_channel(str(uid), section_regions == uid, merged == uid)
        details["model_regions"] = {
            "named": len([r for r in report if not r["dropped"]]),
            "dropped": [r for r in report if r["dropped"]],
        }
    details["label_channels"] = channel_names
    if not fixed_mask.any() or not moving_mask.any():
        raise ValueError("The fit masks are empty")

    fixed_edges = moving_edges = None
    if settings.section_image == "stain" and settings.stain_edges:
        sigma_px = EDGE_SIGMA_UM / 1000.0 / spacing
        fixed_edges = edge_image(fixed, fixed_mask, sigma_px)
        source = moving
        if excluded_w.any():
            # Blanking an excluded region draws a false edge around it, which
            # pulled the tissue in (Elastix); fill the region from its
            # surroundings first so only real edges remain.
            source = _fill_from_surroundings(moving, dilate(excluded_w, 1.0))
        moving_edges = edge_image(source, moving_mask, sigma_px)
        details["edge_channel"] = {"sigma_um": EDGE_SIGMA_UM, "weight": EDGE_CHANNEL_WEIGHT}

    inputs = EngineInputs(
        fixed=np.ascontiguousarray(fixed, dtype=np.float32),
        moving=np.ascontiguousarray(moving, dtype=np.float32),
        fixed_mask=fixed_mask, moving_mask=moving_mask,
        spacing_mm=grid.spacing_mm, origin_mm=grid.origin_mm,
        fixed_labels=fixed_labels, moving_labels=moving_labels,
        label_weights=[LABEL_CHANNEL_WEIGHT] * len(fixed_labels),
        fixed_edges=fixed_edges, moving_edges=moving_edges,
        edge_weight=EDGE_CHANNEL_WEIGHT if fixed_edges is not None else 0.0,
        correlation_radius_px=max(1, round(CORRELATION_RADIUS_UM / 1000.0 / spacing)),
    )
    whole_stain = (settings.engine == "ants" and settings.section_image == "stain"
                   and not settings.structures)
    if whole_stain:
        details["section_mask"] = "em"
    return PreparedFit(
        settings=settings, placement=placement, grid=grid, inputs=inputs, native=native,
        footprint=_on_section(footprint, (width, height)),
        excluded=excluded, ventricles=ventricles,
        atlas_meta=_atlas_record(atlas, placement, native),
        previous=previous, details=details, excluded_mask=excluded_native,
        step_weight=step_weight, em_damage_zone=masks.damage_zone if whole_stain else None,
    )


def _taper_weight(region: np.ndarray, spacing_mm: float, neighbourhood_mm: float,
                  section_size: tuple[int, int]) -> np.ndarray:
    """1 within *neighbourhood_mm* of *region* (working grid), a raised cosine to 0
    over :data:`RESTRICTED_TAPER_MM` beyond, on the section grid."""
    distance = np.asarray(ndi.distance_transform_edt(~region), dtype=np.float64) * spacing_mm
    t = np.clip((distance - neighbourhood_mm) / RESTRICTED_TAPER_MM, 0.0, 1.0)
    weight = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
    return cv2.resize(weight, section_size, interpolation=cv2.INTER_LINEAR)


def _invert_step(field_px: np.ndarray, guess_px: np.ndarray) -> np.ndarray:
    """v with v(q) = -w(q + v(q)) for a step w (section pixels): damped fixed point
    from *guess_px*."""
    height, width = field_px.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    channels = [np.ascontiguousarray(field_px[..., k], dtype=np.float32) for k in range(2)]
    inverse = guess_px.astype(np.float32).copy()
    for _ in range(TAPER_INVERSE_ITERATIONS):
        mx, my = xx + inverse[..., 0], yy + inverse[..., 1]
        sampled = np.stack([cv2.remap(c, mx, my, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE) for c in channels], -1)
        inverse += TAPER_INVERSE_RELAXATION * (-sampled - inverse)
    return inverse


def prepare_warp(
    atlas: Any, placement: Placement, size: tuple[int, int], detail: str, *,
    previous: DeformableRecord | None = None,
) -> PreparedFit:
    """What :func:`finish_fit` reads for a warp no engine computes (the
    landmark warp): the native plane, the placed atlas footprint (as a
    :func:`prepare_fit` at *detail* has it), the ventricles and the atlas
    facts. No images, masks or settings: nothing excluded, no details."""
    grid, native, _coords, _current, footprint = _placed_atlas(
        atlas, placement, size, detail, previous)
    return PreparedFit(
        settings=None, placement=placement, grid=grid, inputs=None, native=native,
        footprint=_on_section(footprint, size), excluded=frozenset(),
        ventricles=ventricle_ids(atlas), atlas_meta=_atlas_record(atlas, placement, native),
        previous=previous,
    )


def _previous_offset(grid: WorkingGrid, previous: DeformableRecord | None) -> np.ndarray | None:
    """Where *previous* left the atlas on the working grid, in pixels (None: placed)."""
    if previous is None:
        return None
    prev_w = cv2.resize(previous.field_mm, grid.size, interpolation=cv2.INTER_LINEAR)
    return np.stack([prev_w[..., 0] / grid.spacing_mm[0],
                     prev_w[..., 1] / grid.spacing_mm[1]], axis=-1)


def _placed_atlas(
    atlas: Any, placement: Placement, size: tuple[int, int], detail: str,
    previous: DeformableRecord | None,
) -> tuple[WorkingGrid, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(grid, native labels, working-grid native coordinates, labels there,
    footprint)``: the atlas's current frame on the working grid of *detail*
    (placed, or where *previous* left it) and its footprint widened by
    :data:`MASK_MARGIN_MM`, on that grid."""
    if previous is not None and (tuple(previous.section_size) != tuple(size)
                                 or previous.placement.to_dict() != placement.to_dict()):
        raise ValueError("A sequential step must use its previous record's section and placement")
    grid = WorkingGrid.for_section(size, placement.section_mm_per_px,
                                   DETAIL[detail].working_um / 1000)
    native = native_labels(atlas, placement)
    atlas_to_working = grid.section_to_working @ placement.atlas_to_section
    working_coords = _native_coords(atlas_to_working, _previous_offset(grid, previous),
                                    (grid.size[1], grid.size[0]))
    current = sample_native(native, working_coords)
    footprint = dilate(current != 0, MASK_MARGIN_MM / float(np.mean(grid.spacing_mm)))
    return grid, native, working_coords, current, footprint


def _on_section(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0


def _atlas_record(atlas: Any, placement: Placement, native: np.ndarray) -> dict[str, Any]:
    return {**_atlas_meta(atlas, placement, native),
            "acronyms": {str(k): v for k, v in structure_acronyms(
                atlas, np.unique(native)).items()}}


def finish_fit(prepared: PreparedFit, result: EngineResult) -> DeformableRecord:
    """Interpolate the engine's fields to the section, compose, label and diagnose."""
    grid = prepared.grid
    mm = prepared.placement.section_mm_per_px
    width, height = grid.section_size
    step_field = grid.field_to_section(result.field_mm)
    step_inverse = grid.field_to_section(result.inverse_field_mm)
    if prepared.step_weight is not None:
        weight = prepared.step_weight[..., None]
        step_field = (step_field * weight).astype(np.float32)
        step_inverse = (_invert_step(step_field / mm, step_inverse * weight / mm)
                        * mm).astype(np.float32)
    yy, xx = np.indices((height, width), dtype=np.float64)
    previous = prepared.previous
    if previous is None:
        total, inverse = step_field, step_inverse
    else:
        # Section p -> p + w(p) in the previous frame -> + u_prev(p + w(p)).
        moved = np.stack([xx + step_field[..., 0] / mm, yy + step_field[..., 1] / mm], axis=-1)
        total = step_field + _sample_field(previous.field_mm, moved)
        # Placed q -> q + v_prev(q) -> + v_w(q + v_prev(q)).
        assert previous.inverse_field_mm is not None
        back = np.stack([xx + previous.inverse_field_mm[..., 0] / mm,
                         yy + previous.inverse_field_mm[..., 1] / mm], axis=-1)
        inverse = previous.inverse_field_mm + _sample_field(step_inverse, back)
    offset = np.stack([total[..., 0] / mm, total[..., 1] / mm], axis=-1)
    warped = sample_native(prepared.native,
                           _native_coords(prepared.placement.atlas_to_section, offset,
                                          (height, width)))
    placed = sample_native(prepared.native,
                           _native_coords(prepared.placement.atlas_to_section, None,
                                          (height, width)))
    acronyms = {int(k): v for k, v in prepared.atlas_meta["acronyms"].items()}
    diag_warped, diag_placed = warped, placed
    if prepared.excluded_mask is not None and prepared.excluded_mask.any():
        # Area diagnostics leave excluded pixels out, so a region excluded on
        # one side is still judged on the side the fit used.
        kept_native = np.where(prepared.excluded_mask, 0, prepared.native)
        diag_warped = sample_native(kept_native, _native_coords(
            prepared.placement.atlas_to_section, offset, (height, width)))
        diag_placed = sample_native(kept_native, _native_coords(
            prepared.placement.atlas_to_section, None, (height, width)))
    diagnostics = diagnose(
        total, mm, diag_warped, diag_placed, prepared.footprint, ventricles=prepared.ventricles,
        excluded=prepared.excluded, acronyms=acronyms,
    )
    back = _sample_field(inverse, np.stack([xx + offset[..., 0], yy + offset[..., 1]], -1))
    error = np.linalg.norm(total + back, axis=-1)[prepared.footprint]
    diagnostics["inverse_consistency_mm"] = {
        "p99": float(np.percentile(error, 99)) if error.size else 0.0,
        "max": float(error.max()) if error.size else 0.0,
    }
    inverse_source = result.inverse_source
    if previous is not None:
        earlier = previous.inverse_source.removeprefix("composed: ")
        inverse_source = f"composed: {earlier} then {result.inverse_source}"
    engine = {
        "name": prepared.settings.engine if prepared.settings is not None else None,
        "version": result.engine_version,
        "runtime_s": result.runtime_s, "native_parameters": result.native_parameters,
        "notes": result.notes,
        "working_grid": {"size": list(grid.size), "spacing_mm": list(grid.spacing_mm),
                         "origin_mm": list(grid.origin_mm)},
        "step_details": prepared.details,
    }
    left_out = None
    if result.left_out is not None:
        left_out = _on_section(result.left_out, (width, height))
    return DeformableRecord(
        placement=prepared.placement, section_size=(width, height),
        settings=prepared.settings, field_mm=total.astype(np.float32),
        inverse_field_mm=inverse.astype(np.float32),
        inverse_source=inverse_source,
        labels=warped.astype(prepared.native.dtype),
        footprint=prepared.footprint,
        excluded_ids=sorted(prepared.excluded), engine=engine, diagnostics=diagnostics,
        atlas={k: v for k, v in prepared.atlas_meta.items() if k != "acronyms"},
        step=0 if previous is None else previous.step + 1, parent=previous,
        left_out=left_out,
    )


def run_fit(
    inputs: EngineInputs, settings: FitSettings, em_damage_zone: np.ndarray | None,
) -> EngineResult:
    """The engine work of one prepared fit, on plain arrays (a pool worker runs it).

    With *em_damage_zone* (a whole-section stain fit): a first run with the
    footprint masks, the EM weights of the section against the atlas image it
    warped (``masks.em_masks``), and a second run from the same start with
    the EM masks, whose result is the fit's. Its native parameters record the
    EM's (``section_mask``) and ``left_out`` the section pixels it dropped.
    """
    first = run_engine(inputs, settings)
    if em_damage_zone is None:
        return first
    spacing = np.asarray(inputs.spacing_mm, dtype=np.float32)
    warped = _remap(inputs.moving, first.field_mm / spacing)
    masks, facts = em_masks(inputs.fixed, warped,
                            Masks(inputs.fixed_mask, inputs.moving_mask, em_damage_zone),
                            float(np.mean(inputs.spacing_mm)))
    second = run_engine(replace(inputs, fixed_mask=masks.fixed, moving_mask=masks.moving),
                        settings)
    second.native_parameters["section_mask"] = {
        "kind": "em", **EM_PARAMETERS, "fitted": facts["params"],
        "left_out_in_footprint": facts["left_out_in_footprint"],
        "first_fit_runtime_s": first.runtime_s,
    }
    second.runtime_s += first.runtime_s
    second.left_out = facts["left_out"]
    return second


def run_prepared(prepared: PreparedFit) -> DeformableRecord:
    """Run one prepared fit in this process and build its record."""
    return finish_fit(prepared, run_fit(*_engine_args(prepared)))


def fit_prepared(
    prepared: Sequence[PreparedFit], *, max_workers: int | None = None,
) -> list[DeformableRecord | CandidateFailure]:
    """Run already prepared fits, in order.

    The prepared fits may come from different sections, section images or
    settings. A failing fit comes back as a :class:`CandidateFailure`. On a
    GPU the fits run one after another in this process (each takes about a
    second there; workers would each pay for their own CUDA start). On the
    CPU they run concurrently, one process each: every worker fits on
    :data:`~langslice.core.deformable.engines.FIT_THREADS` threads (identical
    inputs give identical fields), so by default there are as many workers as
    that leaves room for, at most one per fit.
    """
    if not prepared:
        return []
    results: list[DeformableRecord | CandidateFailure] = []
    if fit_device() == "cuda":
        for item in prepared:
            try:
                results.append(run_prepared(item))
            except Exception as exc:  # noqa: BLE001 - one candidate must not sink the rest
                results.append(CandidateFailure(_engine_args(item)[1],
                                                f"{type(exc).__name__}: {exc}"))
        return results
    workers = max_workers or min(len(prepared), max(1, (os.cpu_count() or 2) // FIT_THREADS))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=engine_worker) as pool:
        futures = [pool.submit(run_fit, *_engine_args(p)) for p in prepared]
        for item, future in zip(prepared, futures, strict=True):
            try:
                results.append(finish_fit(item, future.result()))
            except Exception as exc:  # noqa: BLE001 - one candidate must not sink the rest
                results.append(CandidateFailure(_engine_args(item)[1],
                                                f"{type(exc).__name__}: {exc}"))
    return results


def _engine_args(
    prepared: PreparedFit,
) -> tuple[EngineInputs, FitSettings, np.ndarray | None]:
    """The engine's inputs, settings and EM damage zone of a fit; a
    :func:`prepare_warp` has none."""
    if prepared.inputs is None or prepared.settings is None:
        raise ValueError("A prepared warp has no engine fit to run")
    return prepared.inputs, prepared.settings, prepared.em_damage_zone
