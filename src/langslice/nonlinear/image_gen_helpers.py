"""Deterministic helpers for image-gen registration."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from langslice.atlas.core import get_root_mask
from langslice.atlas.recolor import color_lut
from langslice.atlas.render import annotation_slice
from langslice.atlas.render import family_mapping as _family_mapping
from langslice.nonlinear.types import Deformation
from langslice.space import Plane

logger = logging.getLogger(__name__)


#: Width of an ARA-style leaf delineation line, at a 2048px canvas.
_LEAF_BORDER_PX = 2.0


def line_width_px(long_edge: int) -> int:
    """Delineation-line width for a render whose long edge is *long_edge* px."""
    return max(1, round(_LEAF_BORDER_PX * long_edge / 2048))


#: Ventricle ids, for callers that want the ventricular system dropped to
#: background. The registration lineup does NOT: it asks the model to deform
#: an atlas plate onto the tissue, where a ventricle is a region to place
#: like any other (and a useful landmark for the fit). Blacking them out was
#: for the retired lineup, where the model painted ON the section and a
#: ventricle was a hole with no tissue to paint.
#: "cerebral aqueduct" and "ventricular systems" are spelled out in full:
#: a bare "aqueduct"/"ventric" would swallow periaqueductal gray and the
#: periventricular nuclei, which are real tissue.
_VENTRICLE_KEYWORDS = (
    "ventricle",
    "central canal",
    "choroid",
    "subependymal",
    "cerebral aqueduct",
    "ventricular systems",
)
_BLACKOUT_PLANES = {"coronal"}
_ventricle_ids_cache: dict[str, frozenset[int]] = {}


def _ventricle_ids(atlas: Any) -> frozenset[int]:
    name = str(getattr(atlas, "atlas_name", id(atlas)))
    cached = _ventricle_ids_cache.get(name)
    if cached is not None:
        return cached
    ids: set[int] = set()
    structures = getattr(atlas, "structures", None)
    try:
        records = list(structures.values()) if structures is not None else []
    except Exception:  # noqa: BLE001 - structure table without .values()
        records = []
    for record in records:
        try:
            if any(k in str(record["name"]).lower() for k in _VENTRICLE_KEYWORDS):
                ids.add(int(record["id"]))
        except Exception:  # noqa: BLE001 - malformed structure records
            continue
    result = frozenset(ids)
    _ventricle_ids_cache[name] = result
    return result


def _annotation_slice(
    atlas: Any,
    position_mm: float,
    *,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    blackout: bool = False,
) -> np.ndarray:
    """:func:`~langslice.atlas.render.annotation_slice`, ventricles kept.

    With ``blackout=True`` the ventricular system is dropped to background
    on the planes in :data:`_BLACKOUT_PLANES` before anything downstream
    sees it. The registration path never asks for that: the model deforms
    an atlas plate rather than painting the section, so a ventricle is a
    region to place, and the render, the classifier palette, the Elastix
    side and the ledger all read the same plane.
    """
    sliced = annotation_slice(atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg)
    if blackout and plane in _BLACKOUT_PLANES:
        vids = _ventricle_ids(atlas)
        if vids:
            # np.where, not in-place: the oblique sampler may hand back cached data
            sliced = np.where(np.isin(sliced, list(vids)), 0, sliced)
    return sliced


def _classified_to_rgb(classified_2d: np.ndarray, atlas: Any) -> np.ndarray:
    """Rebuild a clean RGB map from classified ids: exact palette colors on black.

    Elastix registers this cleaned map against the atlas render — the
    generated image's preserved background (white slide, anything) and any
    color drift would otherwise poison the per-channel metric.
    """
    lut = color_lut(atlas)
    rgb = np.zeros((*classified_2d.shape, 3), dtype=np.uint8)
    for uid in np.unique(classified_2d):
        uid_int = int(uid)
        if uid_int == 0:
            continue
        rgb[classified_2d == uid_int] = lut.get(uid_int, (128, 128, 128))
    return rgb


def _merge_classified(classified_2d: np.ndarray, atlas: Any, merge_eps: float = 40.0) -> np.ndarray:
    """Map region ids onto one representative id per merged family color.

    An image model paints one flat shade per area, so the atlas render's thin
    per-layer shade bands have no counterpart in the generated map; register
    them raw and Elastix folds those bands into nothing (measured 10-36%%
    negative-Jacobian area). Registration inputs and border overlays share
    this granularity; classification, markers, and the ledger keep the full
    palette.
    """
    mapping = _family_mapping((int(u) for u in np.unique(classified_2d)), atlas, merge_eps)
    merged = np.zeros_like(classified_2d)
    for uid, rep_id in mapping.items():
        merged[classified_2d == uid] = rep_id
    return merged


def _warp_classified_labels(classified_2d: np.ndarray, result_transform: Any) -> np.ndarray:
    """Warp an integer label map with nearest-neighbor interpolation.

    Linear interpolation blends labels/colors at region boundaries into
    values belonging to neither side, which then classify to arbitrary third
    regions — speckle bands along every border. Order-0 keeps labels exact.
    The transform's interpolation order is patched for the call and restored.
    """
    import itk

    # Large Allen structure ids do not survive a float32 round-trip, so warp
    # compact indices and map back; out-of-domain pixels fill with 0, which
    # lands on index 0 = background (np.unique sorts, and 0 is present).
    ids = np.unique(np.concatenate([[0], classified_2d.ravel()]))
    compact = np.searchsorted(ids, classified_2d)

    n_maps = int(result_transform.GetNumberOfParameterMaps())
    previous: list[tuple[str, ...]] = []
    for idx in range(n_maps):
        try:
            prev = tuple(result_transform.GetParameter(idx, "FinalBSplineInterpolationOrder"))
        except Exception:
            prev = ("3",)
        previous.append(prev)
        result_transform.SetParameter(idx, "FinalBSplineInterpolationOrder", "0")
    try:
        image = itk.image_from_array(compact.astype(np.float32))
        warped = itk.transformix_filter(image, result_transform)  # type: ignore[attr-defined]
        warped_idx = np.clip(
            np.asarray(itk.array_from_image(warped)).round().astype(np.int64),
            0,
            len(ids) - 1,
        )
        return ids[warped_idx].astype(classified_2d.dtype)
    finally:
        for idx, values in enumerate(previous):
            result_transform.SetParameter(idx, "FinalBSplineInterpolationOrder", list(values))


def _extract_borders_from_classified(classified_2d: np.ndarray) -> np.ndarray:
    """Single-pixel region borders from classified ids, by neighbor difference.

    Interior borders mark one side of each id change; the outer silhouette
    marks the foreground pixels touching background. No contours, no
    smoothing — each boundary is one crisp line.
    """
    interior = np.zeros(classified_2d.shape, dtype=bool)
    interior[:, 1:] |= classified_2d[:, 1:] != classified_2d[:, :-1]
    interior[1:, :] |= classified_2d[1:, :] != classified_2d[:-1, :]

    fg = classified_2d != 0
    bg_padded = np.pad(classified_2d == 0, 1, constant_values=True)
    touches_bg = (
        bg_padded[2:, 1:-1] | bg_padded[:-2, 1:-1] | bg_padded[1:-1, 2:] | bg_padded[1:-1, :-2]
    )
    borders = ((interior & fg) | (fg & touches_bg)).astype(np.uint8) * 255
    return borders


def _grid_spacing_px(image_shape: tuple[int, ...]) -> int:
    """B-spline control-grid spacing scaled to the working resolution.

    /36 (~370um at a 2048 canvas): the coarser /18 grid could not bend at
    folia scale and visibly under-fit the model's map (Nash: the warped
    borders "don't adhere to the model's output"); the grid ablation showed
    /36 tracks better AND optimizes faster, with the masked metric guarding
    against noise-chasing.
    """
    return max(32, round(max(image_shape) / 36))


def _run_elastix_april_borders(
    fixed_borders: np.ndarray, moving_borders: np.ndarray
) -> tuple[Any, float]:
    """The April 2026 Elastix stage (commit 6077972), on border images.

    One B-spline map, no affine stage: FinalGridSpacingInPhysicalUnits 64,
    AdvancedMeanSquares + TransformBendingEnergyPenalty (weight 10), 512
    iterations, 4 resolutions. Fixed = the model's borders (slice space),
    moving = the atlas plate's borders — the direction the rest of the
    pipeline warps in. (April's code passed the atlas as FIXED and then
    resampled the atlas through that transform, i.e. warped it by the
    inverse of the fit; that is not reproduced here.)
    """
    import itk

    start = time.perf_counter()
    fixed_image = itk.image_from_array(fixed_borders.astype(np.float32))
    moving_image = itk.image_from_array(moving_borders.astype(np.float32))
    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]
    parameter_map = parameter_object.GetDefaultParameterMap("bspline")
    parameter_map["FinalGridSpacingInPhysicalUnits"] = ("64",)
    parameter_map["Metric"] = ("AdvancedMeanSquares", "TransformBendingEnergyPenalty")
    parameter_map["Metric0Weight"] = ("1.0",)
    parameter_map["Metric1Weight"] = ("10.0",)
    parameter_map["MaximumNumberOfIterations"] = ("512",)
    parameter_map["NumberOfResolutions"] = ("4",)
    parameter_object.AddParameterMap(parameter_map)
    _result_image, result_transform = itk.elastix_registration_method(  # type: ignore[attr-defined]
        fixed_image, moving_image, parameter_object=parameter_object, log_to_console=False
    )
    elapsed = time.perf_counter() - start
    logger.info("Elastix (April borders B-spline) completed in %.1fs", elapsed)
    return result_transform, elapsed


def _build_multichannel_parameter_object(
    grid_spacing: int = 32, n_channels: int = 3, deformation: Deformation = "bspline"
) -> Any:
    """Affine+B-spline ParameterObject registering *n_channels* jointly.

    Both images share one palette, so the identity intensity mapping —
    AdvancedMeanSquares per channel — is the right metric, not correlation.
    Elastix's multi-metric machinery requires images == pyramids == metrics,
    and the bending-energy penalty counts as a metric without consuming an
    image, so a duplicate of channel 0 fills its slot (weight 0 in the affine
    stage, where the penalty does not apply). The duplicate slot stays in the
    affine-only object too, so the caller's channel list is built the same way
    either way.

    With ``deformation="affine"`` only the affine map is emitted.
    """
    import itk

    n_slots = n_channels + 1  # channels + the penalty's dummy slot
    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]

    stages = ("affine",) if deformation == "affine" else ("affine", "bspline")
    for kind in stages:
        param_map = parameter_object.GetDefaultParameterMap(kind)
        param_map["Registration"] = ("MultiMetricMultiResolutionRegistration",)
        if kind == "affine":
            metrics: tuple[str, ...] = ("AdvancedMeanSquares",) * n_slots
            weights = ("1.0",) * n_channels + ("0.0",)
            param_map["MaximumNumberOfIterations"] = ("512",)
            param_map["AutomaticTransformInitialization"] = ("true",)
            param_map["AutomaticTransformInitializationMethod"] = ("CenterOfGravity",)
        else:
            metrics = ("AdvancedMeanSquares",) * n_channels + ("TransformBendingEnergyPenalty",)
            # The data term is a SUM over channels, so the penalty must scale
            # with channel count to keep the same regularization strength the
            # 3-channel RGB setup was calibrated at (penalty 3.0 per 3
            # channels = 1.0 per channel); a fixed weight silently dilutes it
            # N/3-fold and the warp folds.
            weights = ("1.0",) * n_channels + (str(float(n_channels)),)
            param_map["FinalGridSpacingInPhysicalUnits"] = (str(grid_spacing),)
            param_map["MaximumNumberOfIterations"] = ("1024",)
        param_map["Metric"] = metrics
        for i, weight in enumerate(weights):
            param_map[f"Metric{i}Weight"] = (weight,)
        param_map["FixedImagePyramid"] = ("FixedSmoothingImagePyramid",) * n_slots
        param_map["MovingImagePyramid"] = ("MovingSmoothingImagePyramid",) * n_slots
        param_map["Interpolator"] = ("LinearInterpolator",) * n_slots
        param_map["ImageSampler"] = ("RandomCoordinate",) * n_slots
        param_map["NumberOfResolutions"] = ("4",)
        if n_channels > 8:
            # Every metric pays the full random-sample budget per iteration;
            # at one-channel-per-family counts, trim it so runtime does not
            # scale with the family count. (3-channel callers keep the
            # elastix default they were tuned at.)
            param_map["NumberOfSpatialSamples"] = ("2048",)
        parameter_object.AddParameterMap(param_map)

    return parameter_object


def _register_channel_stacks(
    fixed_channels: list[np.ndarray],
    moving_channels: list[np.ndarray],
    fixed_mask: np.ndarray | None = None,
    deformation: Deformation = "bspline",
) -> tuple[Any, float]:
    """Elastix registration of N fixed/moving channel pairs under one transform.

    The returned transform maps fixed pixels to moving pixels. With
    ``fixed_mask``, the metric is sampled only inside the mask — the model's
    segmented tissue — so omitted (missing) tissue never constrains the fit
    and the regularized transform extrapolates smoothly across it.
    """
    import itk

    start = time.perf_counter()

    def _image(channel: np.ndarray) -> Any:
        return itk.image_from_array(np.ascontiguousarray(channel, dtype=np.float32))

    elastix = itk.ElastixRegistrationMethod.New(  # type: ignore[attr-defined]
        _image(fixed_channels[0]), _image(moving_channels[0])
    )
    for fixed_c, moving_c in zip(fixed_channels[1:], moving_channels[1:], strict=True):
        elastix.AddFixedImage(_image(fixed_c))
        elastix.AddMovingImage(_image(moving_c))
    # Duplicate of channel 0 fills the penalty metric's image slot.
    elastix.AddFixedImage(_image(fixed_channels[0]))
    elastix.AddMovingImage(_image(moving_channels[0]))
    if fixed_mask is not None and fixed_mask.any():
        # uint8 array -> itk.Image[UC, 2], the mask type Elastix expects.
        elastix.SetFixedMask(
            itk.image_from_array(np.ascontiguousarray(fixed_mask.astype(np.uint8)))
        )
    elastix.SetParameterObject(
        _build_multichannel_parameter_object(
            _grid_spacing_px(fixed_channels[0].shape),
            n_channels=len(fixed_channels),
            deformation=deformation,
        )
    )
    elastix.SetLogToConsole(False)
    elastix.UpdateLargestPossibleRegion()
    result_transform = elastix.GetTransformParameterObject()

    elapsed = time.perf_counter() - start
    logger.info(
        "Elastix %d-channel %s registration completed in %.1fs",
        len(fixed_channels),
        "affine" if deformation == "affine" else "affine+B-spline",
        elapsed,
    )
    return result_transform, elapsed


def _compute_deformation_field(result_transform: Any, moving_gray: np.ndarray) -> np.ndarray | None:
    """Dense (H, W, 2) displacement field for an Elastix transform, or None on failure."""
    import itk

    try:
        moving_image = itk.image_from_array(moving_gray.astype(np.float32))
        field = itk.transformix_deformation_field(  # type: ignore[attr-defined]
            moving_image, result_transform
        )
        return np.asarray(itk.array_from_image(field), dtype=np.float64)
    except Exception as exc:
        logger.warning("Deformation field unavailable: %s", exc)
        return None


#: Elastix error-code thresholds. A code is only emitted when the warp is
#: physically implausible, so the confirm gate can refuse on any code present.
_MAJOR_REGION_SHARE = 0.01  # region matters if >= 1% of the atlas foreground
_COLLAPSE_RATIO = 0.2  # major region shrank below 20% of its expected share
_FOLD_FRACTION = 0.005  # > 0.5% of foreground pixels fold through themselves
_UNCOVERED_FRACTION = 0.25  # > 25% of real tissue left outside the warped atlas


def _division_of(structures: Any):
    """id -> coarse-division ancestor id (depth-2 in the structure tree).

    Falls back to the id itself when the tree carries no usable path (test
    fakes, foreign atlases without paths).
    """

    def division(uid: int) -> int:
        try:
            path = structures[uid]["structure_id_path"]
            return int(path[min(2, len(path) - 1)])
        except Exception:
            return uid

    return division


def _region_label(structures: Any, uid: int) -> str:
    try:
        return str(structures[uid].get("acronym") or structures[uid].get("name") or uid)
    except Exception:
        return str(uid)


def _elastix_report(
    *,
    atlas_classified: np.ndarray,
    warped_classified: np.ndarray,
    structures: Any = None,
    deformation_field: np.ndarray | None = None,
    tissue_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Post-Elastix sanity check, reported as error codes on the Elastix step.

    There is one real failure in this pipeline — Elastix not working on the
    generated image — and each code names a measured, physically implausible
    consequence of it. No codes means the registration succeeded mechanically;
    anatomical quality is still the agent's visual call.
    """
    codes: list[dict[str, Any]] = []

    atlas_fg = atlas_classified != 0
    warped_fg = warped_classified != 0
    atlas_total = int(atlas_fg.sum())
    warped_total = int(warped_fg.sum())

    if atlas_total and warped_total:
        # Aggregate to coarse DIVISIONS before judging: an image model will
        # never reproduce layer-level shading reliably, so a thin cortical
        # layer thinning under the warp is not an Elastix failure — a whole
        # division vanishing is. The ledger keeps full granularity.
        division = _division_of(structures)
        expected: dict[int, int] = {}
        ids, counts = np.unique(atlas_classified[atlas_fg], return_counts=True)
        for uid, count in zip(ids, counts, strict=False):
            expected[division(int(uid))] = expected.get(division(int(uid)), 0) + int(count)
        warped: dict[int, int] = {}
        ids, counts = np.unique(warped_classified[warped_fg], return_counts=True)
        for uid, count in zip(ids, counts, strict=False):
            warped[division(int(uid))] = warped.get(division(int(uid)), 0) + int(count)
        for div, count in expected.items():
            share = count / atlas_total
            if share < _MAJOR_REGION_SHARE:
                continue
            wshare = warped.get(div, 0) / warped_total
            if wshare == 0.0:
                codes.append(
                    {
                        "code": "REGION_MISSING",
                        "region": _region_label(structures, div),
                        "expected_share": round(float(share), 4),
                    }
                )
            elif wshare < _COLLAPSE_RATIO * share:
                codes.append(
                    {
                        "code": "REGION_COLLAPSED",
                        "region": _region_label(structures, div),
                        "expected_share": round(float(share), 4),
                        "warped_share": round(float(wshare), 4),
                    }
                )

    if deformation_field is not None and warped_total:
        u = deformation_field[..., 0]
        v = deformation_field[..., 1]
        du_dy, du_dx = np.gradient(u)
        dv_dy, dv_dx = np.gradient(v)
        jacobian_det = (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx
        folded = float(np.mean(jacobian_det[warped_fg] <= 0.0))
        if folded > _FOLD_FRACTION:
            codes.append({"code": "WARP_FOLDS", "folded_fraction": round(folded, 4)})

    if tissue_mask is not None:
        tissue_total = int(tissue_mask.sum())
        if tissue_total:
            uncovered = float(np.mean(~warped_fg[tissue_mask]))
            if uncovered > _UNCOVERED_FRACTION:
                codes.append(
                    {
                        "code": "TISSUE_UNCOVERED",
                        "uncovered_fraction": round(uncovered, 4),
                    }
                )

    return {
        "codes": codes,
        "atlas_foreground_px": atlas_total,
        "warped_foreground_px": warped_total,
        "jacobian_available": deformation_field is not None,
        "tissue_mask_available": tissue_mask is not None,
    }


#: The atlas silhouette mask now lives with the other atlas slice accessors.
#: Kept under its old name because the registration modules — and their
#: monkeypatching tests — import it from here.
_build_atlas_root_mask = get_root_mask

