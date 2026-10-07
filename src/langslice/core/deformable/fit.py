"""One deformable fit of a placed atlas plane onto a section: prepare, run, record.

Route A (no image model): the section's stain image against an atlas image
(``template`` or ``nissl``; in practice mostly an outline fit), optionally in
sequential steps each restricted to the neighbourhood of chosen structures,
optionally with the automatic tissue/ventricle label channels. Route B (image
model): the model's extracted lines against atlas borders rendered from the
same merged region set the model was shown, or (``labels="model"``) the
lines turned into named regions against the placed atlas regions.

Everything that needs the atlas object happens in :func:`prepare_fit`; the
engine call works on plain arrays, so :func:`fit_prepared` runs several
fits in a process pool and builds the records back in the caller.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from langslice.core.atlas.render import family_labels
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
    ants_preprocess,
    ants_worker,
    run_engine,
)
from langslice.core.deformable.geometry import Placement, WorkingGrid, sample_native
from langslice.core.deformable.masks import (
    EXCLUSION_MARGIN_MM,
    MASK_MARGIN_MM,
    dilate,
    tissue_masks,
    torn_edge_band,
    ventricle_holes,
)
from langslice.core.deformable.nissl import NisslAtlas
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
#: Stain normalization percentiles inside the tissue.
STAIN_PERCENTILES = (1.0, 99.0)


@dataclass
class PreparedFit:
    """Everything a fit needs, with the engine's part as plain arrays."""

    settings: FitSettings
    placement: Placement
    grid: WorkingGrid
    inputs: EngineInputs
    native: np.ndarray
    tissue: np.ndarray
    torn_band: np.ndarray
    #: Ids excluded on both sides (whole regions; diagnostics skip them).
    excluded: frozenset[int]
    ventricles: frozenset[int]
    atlas_meta: dict[str, Any]
    previous: DeformableRecord | None = None
    details: dict[str, Any] = field(default_factory=dict)
    #: Native-plane pixels excluded from the fit, one-sided entries included.
    excluded_mask: np.ndarray | None = None


@dataclass
class CandidateFailure:
    """A candidate whose engine raised; the other candidates still return."""

    settings: FitSettings
    error: str


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def _normalize_stain(gray: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    values = gray[tissue] if tissue.any() else gray.ravel()
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
    torn_band: np.ndarray | None = None,
    previous: DeformableRecord | None = None,
    nissl: NisslAtlas | None = None,
) -> PreparedFit:
    """Build the working-grid images and masks for one candidate.

    *section_image* is the section on the grid *placement* refers to; *lines*
    (route B) is the model's line mask on that same grid. *torn_band*
    overrides the automatic torn-edge band (a boolean mask on the section
    grid). With *previous*, this is the next step of a sequential fit: the
    atlas starts where *previous* left it and the result composes onto it.
    """
    width, height = section_image.size
    mm = placement.section_mm_per_px
    if settings.section_image == "lines" or settings.labels == "model":
        if lines is None or np.shape(lines) != (height, width):
            raise ValueError("This setting needs the model's line mask on the section grid")
    if torn_band is not None and np.shape(torn_band) != (height, width):
        raise ValueError("torn_band must be a mask on the section grid")
    if previous is not None and (tuple(previous.section_size) != (width, height)
                                 or previous.placement.to_dict() != placement.to_dict()):
        raise ValueError("A sequential step must use its previous record's section and placement")

    tissue, raw_tissue = tissue_masks(section_image)
    grid = WorkingGrid.for_section((width, height), mm, DETAIL[settings.detail].working_um / 1000)
    spacing = float(np.mean(grid.spacing_mm))
    atlas_to_working = grid.section_to_working @ placement.atlas_to_section
    native = native_labels(atlas, placement)
    # One-sided entries ("CTX:left") need the placement's displayed left.
    left = placement_left(atlas, placement, (*settings.exclude, *settings.structures))
    excluded = whole_region_ids(atlas, settings.exclude)
    excluded_native = (regions_mask(atlas, native, settings.exclude, left)
                       if settings.exclude else None)
    ventricles = ventricle_ids(atlas)

    # The atlas's current frame on the working grid: placed, or where the
    # previous step left it.
    offset = None
    if previous is not None:
        prev_w = cv2.resize(previous.field_mm, grid.size, interpolation=cv2.INTER_LINEAR)
        offset = np.stack([prev_w[..., 0] / grid.spacing_mm[0],
                           prev_w[..., 1] / grid.spacing_mm[1]], axis=-1)
    working_coords = _native_coords(atlas_to_working, offset, (grid.size[1], grid.size[0]))
    current = sample_native(native, working_coords)
    excluded_w = (sample_native(excluded_native, working_coords) if excluded_native is not None
                  else np.zeros(current.shape, dtype=bool))
    softening_px = LINE_SOFTENING_UM / 1000.0 / spacing

    moving = placed_atlas_image(
        settings.atlas_image, native, atlas, placement, atlas_to_working, grid.size,
        excluded_native, softening_px=softening_px, nissl=nissl,
    )
    if offset is not None:
        moving = _remap(moving, offset)

    tissue_w = grid.to_working(tissue.astype(np.uint8), nearest=True) > 0
    raw_w = grid.to_working(raw_tissue.astype(np.uint8), nearest=True) > 0
    lines_w = None
    if lines is not None:
        lines_w = grid.to_working(np.asarray(lines, dtype=np.float32)) > 0
    if settings.section_image == "lines":
        assert lines_w is not None
        fixed = soft_lines(lines_w, softening_px)
    else:
        fixed = grid.to_working(_gray(section_image))
        fixed = ants_preprocess(fixed, tissue_w, settings.preprocess, grid.spacing_mm)
        fixed = _normalize_stain(fixed, tissue_w)

    kept = (current != 0) & ~excluded_w
    if torn_band is not None:
        torn_w = grid.to_working(torn_band.astype(np.uint8), nearest=True) > 0
        torn_section = torn_band.astype(bool)
    else:
        torn_w = torn_edge_band(tissue_w, kept, spacing)
        torn_section = cv2.resize(torn_w.astype(np.uint8), (width, height),
                                  interpolation=cv2.INTER_NEAREST) > 0
    margin_px = MASK_MARGIN_MM / spacing
    fixed_mask = dilate(tissue_w, margin_px) & ~torn_w
    # Blanking leaves an edge where an excluded region was; a margin around
    # it stays out of the moving mask so that edge cannot attract the tissue.
    excluded_zone = dilate(excluded_w, EXCLUSION_MARGIN_MM / spacing)
    moving_mask = dilate(current != 0, margin_px) & ~excluded_zone

    details: dict[str, Any] = {}
    region: np.ndarray | None = None
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

    fixed_labels: list[np.ndarray] = []
    moving_labels: list[np.ndarray] = []
    channel_names: list[str] = []

    def add_channel(name: str, section_side: np.ndarray, atlas_side: np.ndarray) -> None:
        blur = max(softening_px, 0.5)
        fixed_labels.append(cv2.GaussianBlur(section_side.astype(np.float32), (0, 0), blur))
        moving_labels.append(cv2.GaussianBlur(atlas_side.astype(np.float32), (0, 0), blur))
        channel_names.append(name)

    if settings.labels == "auto":
        add_channel("tissue", tissue_w, kept)
        atlas_ventricles = np.isin(current, list(ventricles)) & kept
        holes = ventricle_holes(raw_w, tissue_w, atlas_ventricles, spacing)
        details["detected_ventricle_area_mm2"] = float(holes.sum() * spacing ** 2)
        # A missed detection is not evidence of a collapsed ventricle, so the
        # ventricle pair is only added when the section shows one.
        if holes.any():
            add_channel("ventricles", holes, atlas_ventricles)
    elif settings.labels == "model":
        assert lines_w is not None
        merged = family_labels(current, atlas)
        merged = np.where(kept, merged, 0)
        section_regions, unnamed, report = named_regions(lines_w, merged, tissue_w, spacing)
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
        raise ValueError("The fit masks are empty: no tissue overlaps the placed atlas")

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
    return PreparedFit(
        settings=settings, placement=placement, grid=grid, inputs=inputs, native=native,
        tissue=tissue, torn_band=torn_section, excluded=excluded, ventricles=ventricles,
        atlas_meta={**_atlas_meta(atlas, placement, native),
                    "acronyms": {str(k): v for k, v in structure_acronyms(
                        atlas, np.unique(native)).items()}},
        previous=previous, details=details, excluded_mask=excluded_native,
    )


def finish_fit(prepared: PreparedFit, result: EngineResult) -> DeformableRecord:
    """Interpolate the engine's fields to the section, compose, label and diagnose."""
    grid = prepared.grid
    mm = prepared.placement.section_mm_per_px
    width, height = grid.section_size
    step_field = grid.field_to_section(result.field_mm)
    step_inverse = grid.field_to_section(result.inverse_field_mm)
    yy, xx = np.indices((height, width), dtype=np.float64)
    previous = prepared.previous
    if previous is None:
        total, inverse = step_field, step_inverse
    else:
        # Section p -> p + w(p) in the previous frame -> + u_prev(p + w(p)).
        moved = np.stack([xx + step_field[..., 0] / mm, yy + step_field[..., 1] / mm], axis=-1)
        total = step_field + _sample_field(previous.field_mm, moved)
        inverse = None
        if previous.inverse_field_mm is not None:
            # Placed q -> q + v_prev(q) -> + v_w(q + v_prev(q)).
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
        total, mm, diag_warped, diag_placed, prepared.tissue, ventricles=prepared.ventricles,
        excluded=prepared.excluded, acronyms=acronyms,
    )
    if inverse is not None:
        back = _sample_field(inverse, np.stack([xx + offset[..., 0], yy + offset[..., 1]], -1))
        error = np.linalg.norm(total + back, axis=-1)[prepared.tissue]
        diagnostics["inverse_consistency_mm"] = {
            "p99": float(np.percentile(error, 99)) if error.size else 0.0,
            "max": float(error.max()) if error.size else 0.0,
        }
    inverse_source = result.inverse_source
    if previous is not None:
        earlier = previous.inverse_source.removeprefix("composed: ")
        inverse_source = f"composed: {earlier} then {result.inverse_source}"
        if inverse is None:
            inverse_source = "unavailable: an earlier step has no inverse"
    engine = {
        "name": prepared.settings.engine, "version": result.engine_version,
        "runtime_s": result.runtime_s, "native_parameters": result.native_parameters,
        "notes": result.notes,
        "working_grid": {"size": list(grid.size), "spacing_mm": list(grid.spacing_mm),
                         "origin_mm": list(grid.origin_mm)},
        "step_details": prepared.details,
    }
    return DeformableRecord(
        placement=prepared.placement, section_size=(width, height),
        settings=prepared.settings, field_mm=total.astype(np.float32),
        inverse_field_mm=None if inverse is None else inverse.astype(np.float32),
        inverse_source=inverse_source,
        labels=np.where(prepared.tissue, warped, 0).astype(prepared.native.dtype),
        tissue=prepared.tissue, torn_band=prepared.torn_band,
        excluded_ids=sorted(prepared.excluded), engine=engine, diagnostics=diagnostics,
        atlas={k: v for k, v in prepared.atlas_meta.items() if k != "acronyms"},
        step=0 if previous is None else previous.step + 1, parent=previous,
    )


def fit_section(
    section_image: Image.Image,
    atlas: Any,
    placement: Placement,
    settings: FitSettings | None = None,
    **options: Any,
) -> DeformableRecord:
    """Prepare, run and record one fit (options as :func:`prepare_fit`)."""
    prepared = prepare_fit(section_image, atlas, placement, settings or FitSettings(), **options)
    return finish_fit(prepared, run_engine(prepared.inputs, prepared.settings))


def _run_candidate(inputs: EngineInputs, settings: FitSettings) -> EngineResult:
    return run_engine(inputs, settings)


def fit_prepared(
    prepared: Sequence[PreparedFit], *, max_workers: int | None = None,
) -> list[DeformableRecord | CandidateFailure]:
    """Run already prepared fits concurrently (one process each), in order.

    The prepared fits may come from different sections, section images or
    settings. A failing fit comes
    back as a :class:`CandidateFailure`. Every worker fits on
    :data:`~langslice.core.deformable.engines.FIT_THREADS` threads (identical
    inputs give identical fields), so by default there are as many workers as
    that leaves room for, at most one per fit.
    """
    if not prepared:
        return []
    workers = max_workers or min(len(prepared), max(1, (os.cpu_count() or 2) // FIT_THREADS))
    context = multiprocessing.get_context("spawn")
    results: list[DeformableRecord | CandidateFailure] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=ants_worker) as pool:
        futures = [pool.submit(_run_candidate, p.inputs, p.settings) for p in prepared]
        for item, future in zip(prepared, futures, strict=True):
            try:
                results.append(finish_fit(item, future.result()))
            except Exception as exc:  # noqa: BLE001 - one candidate must not sink the rest
                results.append(CandidateFailure(item.settings, f"{type(exc).__name__}: {exc}"))
    return results
