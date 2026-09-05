"""Deterministic helpers for image-gen registration."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from langslice.atlas import position_mm_to_index
from langslice.atlas.core import get_root_mask, orient_slice_for_display
from langslice.atlas.recolor import active_palette, color_lut
from langslice.space import (
    Plane,
    atlas_space_context,
    slice_axis_index,
)

logger = logging.getLogger(__name__)


#: A generated pixel farther than this from every palette color is background,
#: not tissue. Must stay below the white-to-fiber-tract-gray distance (~88).
_BG_COLOR_DISTANCE = 70.0

#: Long edge the model-facing region map is drawn at when no size is asked
#: for. Matches the working canvas (``image_gen_registration._MAX_LONG_EDGE``)
#: so the atlas reference reaches the model at the histology's own
#: resolution instead of as upscaled voxel blocks.
_RENDER_LONG_EDGE = 2048

#: Width of an ARA-style leaf delineation line, at a 2048px canvas.
_LEAF_BORDER_PX = 2.0

#: Ventricle blackout (re-instated 2026-09-01; originally deleted in the
#: palette rework despite Nash's instruction to keep it for coronal only).
#: In CORONAL sections the ventricular system is a hole in the tissue —
#: matching histology holes, these ids become background everywhere the
#: pipeline reads the annotation (render, classifier palette, Elastix side,
#: ledger stay consistent). Sagittal/horizontal keep their ventricles.
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
) -> np.ndarray:
    """The atlas annotation at *position_mm*, oriented for display.

    With a non-zero cutting angle the plane is resliced obliquely instead of
    taken flat off the voxel grid. Measured on the LSD_910 hand
    registrations, whose block was cut at 4 degrees: matching the plane is
    worth far more than any fit tuning (fit-only family dice 0.93 -> 0.96,
    boundary p95 34px -> 9px over 33 slices).

    On the planes in :data:`_BLACKOUT_PLANES` the ventricular system is
    blacked out to background before anything downstream sees it.
    """
    if pitch_deg or yaw_deg:
        from langslice.oblique import sample_oblique_annotation

        sliced = sample_oblique_annotation(atlas, position_mm, plane, pitch_deg, yaw_deg)
    else:
        idx = position_mm_to_index(atlas, position_mm, plane=plane)
        axis = slice_axis_index(atlas_space_context(atlas), plane)
        sliced = orient_slice_for_display(
            np.asarray(np.take(atlas.annotation, idx, axis=axis)), plane
        )
    if plane in _BLACKOUT_PLANES:
        vids = _ventricle_ids(atlas)
        if vids:
            # np.where, not in-place: the oblique sampler may hand back cached data
            sliced = np.where(np.isin(sliced, list(vids)), 0, sliced)
    return sliced


def _generate_colored_region_slice(
    atlas: Any,
    position_mm: float,
    target_size: tuple[int, int] | None = None,
    *,
    plane: Plane = "coronal",
    smooth: bool = True,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> Image.Image:
    """Render an atlas annotation slice as a color-preserving RGB region map.

    ``smooth`` (the default) renders at ``_RENDER_LONG_EDGE`` with EXACT
    voxel-true geometry — NEAREST-upscaled ids, boundaries exactly where the
    atlas puts them, no contour smoothing or rounding of any kind (Nash:
    rounded borders corrupt the geometry the model is told to trust). Pass
    ``smooth=False`` for the native-size pixel-exact render the Elastix side
    registers against and classifies back.

    Every model-facing render delineates its painted colors: wherever the
    fill color changes, a line in a darker shade of each side's own color
    (neighbor difference, so lines follow the true boundary). Under
    ``palette="leaf-borders"`` the finer leaf boundaries are added on top,
    color lines heavier — the full Allen-Reference-Atlas plate treatment.
    MODEL-FACING decoration only: it never touches the ``smooth=False``
    render, so the Elastix pair and everything classified from it are
    identical either way.
    Under ``palette="family-flat"`` the leaves are collapsed to their
    registration families FIRST (:func:`_plane_families`), so the paint is
    one flat color per family and the fill-change lines fall at family
    boundaries.
    """
    annotation_slice = _annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    height, width = annotation_slice.shape
    lut = color_lut(atlas)
    style = active_palette()

    if smooth:
        from scipy import ndimage

        if target_size is None:
            scale = max(1.0, _RENDER_LONG_EDGE / max(height, width))
            target_size = (round(width * scale), round(height * scale))
        ids = np.asarray(
            Image.fromarray(annotation_slice.astype(np.int32), mode="I").resize(
                target_size, Image.Resampling.NEAREST
            ),
            dtype=np.int64,
        )
        if style == "family-flat":
            families = _plane_families(annotation_slice, atlas)
            flat = np.zeros_like(ids)
            for uid in np.unique(ids):
                if int(uid):
                    flat[ids == uid] = families.get(int(uid), int(uid))
            ids = flat
        rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
        for uid in np.unique(ids):
            if int(uid):
                rgb[ids == uid] = lut.get(int(uid), (128, 128, 128))
        # Every model-facing render delineates its painted colors — a line in
        # a darker shade of the fill wherever the fill changes, ARA-plate
        # style (Nash 2026-09-01: borders around the colored regions). One
        # label per COLOR, not per registration family: the family clustering
        # was tried here first and left visibly different shades (cerebellar
        # lobules, cortical areas, striatum vs tubercle) inside one family
        # with no line between them. Under "leaf-borders" the finer leaf
        # hairlines are added on top.
        from langslice.nonlinear.render import darker

        units = (rgb.astype(np.int64) * np.array([65536, 256, 1])).sum(axis=2)

        def _boundary(labels: np.ndarray) -> np.ndarray:
            b = np.zeros(labels.shape, dtype=bool)
            b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
            b[1:, :] |= labels[1:, :] != labels[:-1, :]
            return b & (labels != 0)

        leaf_w = max(1, round(_LEAF_BORDER_PX * max(target_size) / 2048))
        leaf_b = np.zeros(ids.shape, dtype=bool)
        if active_palette() == "leaf-borders":
            leaf_b = _boundary(ids)
            if leaf_w > 1:
                leaf_b = np.asarray(
                    ndimage.binary_dilation(leaf_b, iterations=leaf_w - 1), dtype=bool
                )
        unit_b = np.asarray(
            ndimage.binary_dilation(_boundary(units), iterations=2 * leaf_w - 1),
            dtype=bool,
        )
        border = (leaf_b | unit_b) & (ids != 0)
        dark_lut = {
            int(u): darker(lut.get(int(u), (128, 128, 128)))
            for u in np.unique(ids)
            if int(u)
        }
        for uid, dcol in dark_lut.items():
            rgb[border & (ids == uid)] = dcol
        return Image.fromarray(rgb, mode="RGB")

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for uid in np.unique(annotation_slice):
        uid_int = int(uid)
        if uid_int == 0:
            continue
        rgb[annotation_slice == uid_int] = lut.get(uid_int, (128, 128, 128))

    image = Image.fromarray(rgb, mode="RGB")
    if target_size is not None:
        image = image.resize(target_size, resample=Image.Resampling.NEAREST)
    return image


def _classify_pixels_to_region_ids(
    model_output_rgb: np.ndarray,
    atlas: Any,
    position_mm: float,
    *,
    plane: Plane = "coronal",
    off_palette_background: bool = True,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Classify RGB pixels to the nearest atlas region color at *position_mm*.

    ``off_palette_background`` sends far-from-palette pixels to background —
    right for MODEL output (its preserved background can be any color), wrong
    for our own renders: warping blends colors at region boundaries, and the
    cutoff would erase thin regions there (pass False for those).

    The cutting angles must match the render the pixels came from: the
    palette is built from the ids that plane actually contains.
    """
    annotation_slice = _annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    lut = color_lut(atlas)

    color_to_id: dict[tuple[int, int, int], int] = {}
    for uid in np.unique(annotation_slice):
        uid_int = int(uid)
        if uid_int == 0:
            continue
        color = lut.get(uid_int)
        if color is not None:
            color_to_id[color] = uid_int

    if active_palette() == "family-flat":
        # The model was shown one flat color per family, so classify straight
        # to that family's representative id — _merge_classified then keeps
        # what comes out (representatives sit farther apart than the merge
        # radius). Every LEAF color stays in the palette, keyed to the same
        # representative: the Elastix-side render is drawn per leaf whatever
        # the style, and nearest-FAMILY-color is not the family a leaf
        # belongs to (measured on Allen coronal 3.9mm: 10 of 64 leaf colors
        # sit nearest a family that is not their own, 15% of the pixels).
        families = _plane_families(annotation_slice, atlas)
        color_to_id = {color: families[uid] for color, uid in color_to_id.items()}
    # Every model-facing render carries darker(color) delineation lines the
    # model may paint back (unit borders always; leaf hairlines too under
    # "leaf-borders"). Measured: at 2px they are ~7% of the foreground, and
    # far enough off-palette that the background cutoff would punch them
    # straight through the regions they delineate — so the line color is
    # part of the palette, mapping to the region it belongs to.
    from langslice.nonlinear.render import darker

    color_to_id = {darker(c): uid for c, uid in color_to_id.items()} | color_to_id

    if not color_to_id:
        logger.warning("No atlas structure colors found; returning all-zero classification")
        return np.zeros(model_output_rgb.shape[:2], dtype=np.int32)

    palette_colors = np.array(list(color_to_id.keys()), dtype=np.float32)
    palette_ids = np.array(list(color_to_id.values()), dtype=np.int32)

    height, width = model_output_rgb.shape[:2]
    pixels = model_output_rgb.reshape(-1, 3).astype(np.float32)
    # Background is anything far from every palette color — the model keeps
    # the original background (white slide, dark fluorescence, black margin)
    # untouched in edit mode, so it can be any color, not just black.
    background_mask = np.max(pixels, axis=1) < 20.0

    diff = pixels[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]
    distances_sq = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(distances_sq, axis=1)
    classified = palette_ids[nearest_idx]
    if off_palette_background:
        off = distances_sq[np.arange(len(pixels)), nearest_idx] > _BG_COLOR_DISTANCE**2
        background_mask = background_mask | off
    classified[background_mask] = 0
    return classified.reshape(height, width)


def _despeckle_classified(classified_2d: np.ndarray, min_px: int = 16) -> np.ndarray:
    """Reassign connected components smaller than *min_px* to their surroundings.

    Classification speckle — antialiased edge pixels landing off-palette or
    on a third region's color — shows up as confetti in the borders and noise
    in the region accounting. Each tiny component takes the modal id of its
    one-pixel ring.
    """
    from scipy import ndimage

    out = classified_2d.copy()
    for uid in np.unique(classified_2d):
        labels, n = ndimage.label(out == uid)  # type: ignore[misc]
        if not n:
            continue
        sizes = np.bincount(labels.ravel())
        boxes = ndimage.find_objects(labels)
        for comp in np.nonzero(sizes[1:] < min_px)[0] + 1:
            # Inside the speck's own bounding box (grown by the 1px ring):
            # a real painting classifies into thousands of specks, and
            # dilating the whole 3-megapixel frame for each one took minutes.
            box = tuple(
                slice(max(0, s.start - 1), min(dim, s.stop + 1))
                for s, dim in zip(boxes[comp - 1], out.shape, strict=True)
            )
            comp_mask = labels[box] == comp
            ring = ndimage.binary_dilation(comp_mask) & ~comp_mask
            vals = out[box][ring]
            if vals.size:
                ids, counts = np.unique(vals, return_counts=True)
                out[box][comp_mask] = int(ids[counts.argmax()])
    return out


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


def _family_mapping(
    uids: Iterable[int], atlas: Any, merge_eps: float = 40.0
) -> dict[int, int]:
    """region id -> representative id of its merged color family.

    Deterministic in the id set it is given: pass the ids of every map that
    has to share one vocabulary (e.g. a generated map and the atlas render),
    or the two get different representatives for the same family.
    """
    lut = color_lut(atlas)
    reps: list[tuple[tuple[int, int, int], int]] = []
    mapping: dict[int, int] = {}
    for uid in sorted(int(u) for u in uids if int(u) != 0):
        color = lut.get(uid, (128, 128, 128))
        for rep_color, rep_id in reps:
            if sum((a - b) ** 2 for a, b in zip(color, rep_color, strict=False)) <= merge_eps**2:
                mapping[uid] = rep_id
                break
        else:
            reps.append((color, uid))
            mapping[uid] = uid
    return mapping


def _plane_families(annotation_slice: np.ndarray, atlas: Any) -> dict[int, int]:
    """region id -> family representative, for every id one plane contains.

    :func:`_family_mapping` is deterministic only in the id set it is given,
    and a CLASSIFIED map carries one id per distinct COLOR (nearest-color
    classification cannot tell two structures the palette paints alike
    apart), not the raw annotation's ids. So the families are taken over that
    reduced set — the classifier's own key set — and the representatives come
    out as the ones ``_merge_classified`` picks downstream, which is what lets
    a family-flat render and a family render merge to the same map.
    """
    lut = color_lut(atlas)
    uids = [int(u) for u in np.unique(annotation_slice) if int(u) and int(u) in lut]
    by_color = {lut[uid]: uid for uid in uids}
    families = _family_mapping(by_color.values(), atlas)
    return {uid: families[by_color[lut[uid]]] for uid in uids}


def _merge_classified(
    classified_2d: np.ndarray, atlas: Any, merge_eps: float = 40.0
) -> np.ndarray:
    """Map region ids onto one representative id per merged family color.

    An image model paints one flat shade per area, so the atlas render's thin
    per-layer shade bands have no counterpart in the generated map; register
    them raw and Elastix folds those bands into nothing (measured 10-36%%
    negative-Jacobian area). Registration inputs and border overlays share
    this granularity; classification, markers, and the ledger keep the full
    palette.
    """
    mapping = _family_mapping(
        (int(u) for u in np.unique(classified_2d)), atlas, merge_eps
    )
    merged = np.zeros_like(classified_2d)
    for uid, rep_id in mapping.items():
        merged[classified_2d == uid] = rep_id
    return merged


def _registration_rgb(
    classified_2d: np.ndarray, atlas: Any, merge_eps: float = 40.0
) -> np.ndarray:
    """RGB for the REGISTRATION pair: exact palette colors at merged granularity."""
    return _classified_to_rgb(_merge_classified(classified_2d, atlas, merge_eps), atlas)


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


def _build_elastix_parameter_object(grid_spacing: int = 32) -> Any:
    """Build the standard affine+B-spline ParameterObject used for colored registration."""
    import itk

    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]

    affine_map = parameter_object.GetDefaultParameterMap("affine")
    affine_map["Metric"] = ("AdvancedNormalizedCorrelation",)
    affine_map["MaximumNumberOfIterations"] = ("512",)
    affine_map["NumberOfResolutions"] = ("4",)
    affine_map["AutomaticTransformInitialization"] = ("true",)
    affine_map["AutomaticTransformInitializationMethod"] = ("CenterOfGravity",)
    parameter_object.AddParameterMap(affine_map)

    bspline_map = parameter_object.GetDefaultParameterMap("bspline")
    bspline_map["FinalGridSpacingInPhysicalUnits"] = (str(grid_spacing),)
    bspline_map["Metric"] = ("AdvancedNormalizedCorrelation", "TransformBendingEnergyPenalty")
    bspline_map["Metric0Weight"] = ("1.0",)
    bspline_map["Metric1Weight"] = ("3.0",)
    bspline_map["MaximumNumberOfIterations"] = ("1024",)
    bspline_map["NumberOfResolutions"] = ("4",)
    parameter_object.AddParameterMap(bspline_map)

    return parameter_object


def _run_elastix_registration(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
) -> tuple[Any, float]:
    """Run two-stage affine plus B-spline Elastix registration."""
    import itk

    start = time.perf_counter()
    fixed_image = itk.image_from_array(fixed_gray.astype(np.float32))
    moving_image = itk.image_from_array(moving_gray.astype(np.float32))

    parameter_object = _build_elastix_parameter_object()

    _result_image, result_transform = itk.elastix_registration_method(  # type: ignore[attr-defined]
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    elapsed = time.perf_counter() - start
    logger.info("Elastix affine+B-spline registration completed in %.1fs", elapsed)
    return result_transform, elapsed


def _build_multichannel_parameter_object(
    grid_spacing: int = 32, n_channels: int = 3
) -> Any:
    """Affine+B-spline ParameterObject registering *n_channels* jointly.

    Both images share one palette, so the identity intensity mapping —
    AdvancedMeanSquares per channel — is the right metric, not correlation.
    Elastix's multi-metric machinery requires images == pyramids == metrics,
    and the bending-energy penalty counts as a metric without consuming an
    image, so a duplicate of channel 0 fills its slot (weight 0 in the affine
    stage, where the penalty does not apply).
    """
    import itk

    n_slots = n_channels + 1  # channels + the penalty's dummy slot
    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]

    for kind in ("affine", "bspline"):
        param_map = parameter_object.GetDefaultParameterMap(kind)
        param_map["Registration"] = ("MultiMetricMultiResolutionRegistration",)
        if kind == "affine":
            metrics: tuple[str, ...] = ("AdvancedMeanSquares",) * n_slots
            weights = ("1.0",) * n_channels + ("0.0",)
            param_map["MaximumNumberOfIterations"] = ("512",)
            param_map["AutomaticTransformInitialization"] = ("true",)
            param_map["AutomaticTransformInitializationMethod"] = ("CenterOfGravity",)
        else:
            metrics = ("AdvancedMeanSquares",) * n_channels + (
                "TransformBendingEnergyPenalty",
            )
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
        )
    )
    elastix.SetLogToConsole(False)
    elastix.UpdateLargestPossibleRegion()
    result_transform = elastix.GetTransformParameterObject()

    elapsed = time.perf_counter() - start
    logger.info(
        "Elastix %d-channel affine+B-spline registration completed in %.1fs",
        len(fixed_channels),
        elapsed,
    )
    return result_transform, elapsed


def _register_rgb_pair(
    fixed_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    fixed_mask: np.ndarray | None = None,
) -> tuple[Any, float]:
    """Joint-RGB registration of two same-palette region maps (ABBA adapter path)."""
    return _register_channel_stacks(
        [fixed_rgb[:, :, c] for c in range(3)],
        [moving_rgb[:, :, c] for c in range(3)],
        fixed_mask,
    )


#: Dilation (px) of the segmented-tissue mask before it gates the Elastix
#: metric: a little boundary context sharpens the fit at region edges.
_FIXED_MASK_DILATE_PX = 8


def _register_region_maps(
    atlas_classified: np.ndarray,
    generated_classified: np.ndarray,
    atlas: Any,
    fixed_mask: np.ndarray | None = None,
    merge_eps: float = 40.0,
) -> tuple[Any, float]:
    """Register the atlas label map to the model's label map as joint RGB at
    merged-family granularity.

    Fixed = model output (slice space), moving = atlas. Both maps are merged
    through ONE family mapping built on their union of ids, so a family
    renders the same color on both sides (per-map mappings can pick
    different representatives for the same family and shift its color by up
    to the merge radius). The RGB encoding is a measured choice, not a
    default: benchmarked on the sag140 debris case against one binary
    channel per family (the label-registration literature's standard, with
    engaged bending penalties, larger sample budgets, and an RGB-affine
    initialization) and against clamped per-family signed distance maps —
    every label-channel variant came out WORSE than not deforming at all
    (family agreement 0.46-0.54 vs 0.675 identity vs 0.80 RGB, folds 2-21%
    vs 0), because Elastix's sampled ASGD optimizer starves on channels
    whose gradient lives only in thin boundary shells. RGB's known flaw
    stays: a mismatched label earns partial credit whenever its color sits
    nearer than black, so junk (preserved debris classified to far-away
    regions' shades) can pull boundaries slightly. A dense-evaluation
    engine (NiftyReg 4-D SSD, ANTs label registration) is the candidate fix
    if that flaw matters more later. ``fixed_mask`` restricts the metric to
    the model's segmented tissue.
    """
    mapping = _family_mapping(
        (
            int(u)
            for u in np.unique(
                np.concatenate(
                    [atlas_classified.ravel(), generated_classified.ravel()]
                )
            )
        ),
        atlas,
        merge_eps,
    )

    def _rgb_channels(classified: np.ndarray) -> list[np.ndarray]:
        merged = np.zeros_like(classified)
        for uid, rep in mapping.items():
            merged[classified == uid] = rep
        rgb = _classified_to_rgb(merged, atlas)
        return [rgb[:, :, c].astype(np.float32) for c in range(3)]

    if fixed_mask is not None:
        from scipy import ndimage

        fixed_mask = ndimage.binary_dilation(
            fixed_mask.astype(bool), iterations=_FIXED_MASK_DILATE_PX
        )
    return _register_channel_stacks(
        _rgb_channels(generated_classified), _rgb_channels(atlas_classified), fixed_mask
    )


def _warp_atlas_rgb(atlas_rgb: np.ndarray, result_transform: Any) -> np.ndarray:
    """Warp an RGB atlas image through an Elastix transform."""
    import itk

    warped_channels = []
    for channel in range(3):
        channel_image = itk.image_from_array(atlas_rgb[:, :, channel].astype(np.float32))
        warped = itk.transformix_filter(channel_image, result_transform)  # type: ignore[attr-defined]
        warped_channels.append(itk.array_from_image(warped).astype(np.uint8))
    return np.stack(warped_channels, axis=-1)


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


def _mm2_per_pixel(
    atlas: Any, target_size: tuple[int, int], *, plane: Plane = "coronal"
) -> float | None:
    """Physical area of one working-canvas pixel, from the atlas voxel size.

    The atlas render fills the whole canvas, so canvas area maps to the full
    in-plane physical extent; total area is invariant to the display
    transpose. None when the atlas cannot supply shape/resolution.
    """
    try:
        axis = slice_axis_index(atlas_space_context(atlas), plane)
        shape = [s for i, s in enumerate(atlas.annotation.shape) if i != axis]
        res_um = [float(r) for i, r in enumerate(atlas.resolution) if i != axis]
        area_mm2 = (shape[0] * res_um[0] / 1000.0) * (shape[1] * res_um[1] / 1000.0)
        return area_mm2 / float(target_size[0] * target_size[1])
    except Exception:
        return None


def _region_ledger(
    *,
    atlas_classified: np.ndarray,
    warped_classified: np.ndarray,
    structures: Any = None,
    mm2_per_px: float | None = None,
) -> list[dict[str, Any]]:
    """Per-region area accounting: expected (atlas render) vs warped result.

    Every region present in either image gets a row; the Elastix error codes
    are thresholded views of exactly these numbers. Granularity follows the
    color palette — regions sharing a family color classify as one id.
    """
    atlas_fg = atlas_classified != 0
    warped_fg = warped_classified != 0
    atlas_total = int(atlas_fg.sum())
    warped_total = int(warped_fg.sum())
    expected: dict[int, int] = {}
    warped: dict[int, int] = {}
    if atlas_total:
        ids, counts = np.unique(atlas_classified[atlas_fg], return_counts=True)
        expected = {int(u): int(c) for u, c in zip(ids, counts, strict=False)}
    if warped_total:
        ids, counts = np.unique(warped_classified[warped_fg], return_counts=True)
        warped = {int(u): int(c) for u, c in zip(ids, counts, strict=False)}

    rows: list[dict[str, Any]] = []
    for uid in sorted(set(expected) | set(warped), key=lambda u: -expected.get(u, 0)):
        row: dict[str, Any] = {
            "region": _region_label(structures, uid),
            "id": uid,
            "expected_px": expected.get(uid, 0),
            "warped_px": warped.get(uid, 0),
            "expected_share": round(expected.get(uid, 0) / atlas_total, 5) if atlas_total else None,
            "warped_share": round(warped.get(uid, 0) / warped_total, 5) if warped_total else None,
        }
        if mm2_per_px is not None:
            row["expected_mm2"] = round(expected.get(uid, 0) * mm2_per_px, 4)
            row["warped_mm2"] = round(warped.get(uid, 0) * mm2_per_px, 4)
        rows.append(row)
    return rows


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
            codes.append(
                {"code": "WARP_FOLDS", "folded_fraction": round(folded, 4)}
            )

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


def _format_elastix_param_value(value: str) -> str:
    """Quote string-typed Elastix parameter values; leave numeric values bare."""
    try:
        float(value)
        return value
    except (ValueError, TypeError):
        return '"' + value + '"'


def _write_param_map_to_disk(param_map: dict[str, Any], path: Path) -> None:
    """Serialize an Elastix parameter map (dict) to text format on disk.

    Sidesteps ``itk.ParameterObject.WriteParameterFile(dict, path)`` — the
    SWIG binding only exposes the single-arg form usefully from Python, so we
    emit the well-defined Elastix parameter-file format ourselves.
    """
    lines = ["// LangSlice Elastix parameter file"]
    for key, values in param_map.items():
        formatted = " ".join(_format_elastix_param_value(v) for v in values)
        lines.append(f"({key} {formatted})")
    path.write_text("\n".join(lines) + "\n")


def _write_forward_transform_to_disk(
    result_transform: Any,
    out_dir: Path,
) -> str:
    """Write the forward Elastix parameter maps to disk and return the final-stage path.

    The fixed-to-fixed inverse pattern (re-register with an initial transform)
    needs the forward transform on disk because itk-elastix's Python binding
    doesn't accept an in-memory transform object via
    ``initial_transform_parameter_object``.

    Two issues this function works around:

    1. The forward registration is run without ``output_directory``, so the
       in-memory B-spline parameter map's ``InitialTransformParameterFileName``
       (singular ``Parameter``, despite older itk docs mentioning the plural
       form) is empty — Elastix can't chain back to stage 0 without it. We
       explicitly patch the chain to point at the per-stage file we're about to
       write, using forward slashes so the parser is happy on Windows.

    2. ``ParameterObject.WriteParameterFile(map_dict, path)`` is not callable
       from Python (SWIG binding limitation), so we write the file ourselves.

    Returns the absolute path to ``TransformParameters.<last>.txt``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_maps = int(result_transform.GetNumberOfParameterMaps())
    if n_maps == 0:
        raise ValueError("result_transform has no parameter maps to write")

    written: list[Path] = []
    for idx in range(n_maps):
        path = out_dir / f"TransformParameters.{idx}.txt"
        pmap = dict(result_transform.GetParameterMap(idx))
        # Patch the chained-initial reference so Elastix can resolve the prior
        # stage when it reads this file back during the inverse run.
        if idx == 0:
            pmap["InitialTransformParameterFileName"] = ["NoInitialTransform"]
        else:
            pmap["InitialTransformParameterFileName"] = [
                str(written[-1]).replace("\\", "/")
            ]
        _write_param_map_to_disk(pmap, path)
        written.append(path)
    return str(written[-1])


def _warp_slice_to_atlas(
    slice_rgb: np.ndarray,
    parameter_object: Any,
    forward_transform_params_path: str,
    fixed_image_itk: Any,
) -> tuple[np.ndarray, Any]:
    """Compute the inverse Elastix transform and warp the histology slice into atlas space.

    Uses the canonical itk-elastix fixed-to-fixed inverse pattern (Example 11):
    re-register the forward "fixed" image to itself with the forward transform
    parameters as the initial guess. The resulting transform is the inverse and
    can be applied to the slice (which lives in the same coordinate frame as
    the forward fixed image) to warp it into atlas space.

    Args:
        slice_rgb: RGB histology slice at the forward run's target size.
        parameter_object: Same ``itk.ParameterObject`` used for the forward run.
        forward_transform_params_path: Absolute path to the forward run's final
            ``TransformParameters.N.txt`` on disk.
        fixed_image_itk: The float32 ``itk.Image`` used as the *fixed* image in
            the forward Elastix run (shares slice-space coordinates with the
            histology slice at target_size).

    Returns:
        ``(warped_slice_rgb_uint8, inverse_transform_parameters)``.
    """
    import itk

    _inverse_image, inverse_transform_parameters = (
        itk.elastix_registration_method(  # type: ignore[attr-defined]
            fixed_image_itk,
            fixed_image_itk,
            parameter_object=parameter_object,
            initial_transform_parameter_file_name=forward_transform_params_path,
            log_to_console=False,
        )
    )

    warped_channels: list[np.ndarray] = []
    for channel in range(3):
        channel_image = itk.image_from_array(slice_rgb[:, :, channel].astype(np.float32))
        warped = itk.transformix_filter(  # type: ignore[attr-defined]
            channel_image, inverse_transform_parameters
        )
        warped_channels.append(
            np.clip(itk.array_from_image(warped), 0, 255).astype(np.uint8)
        )
    return np.stack(warped_channels, axis=-1), inverse_transform_parameters


def _run_inverse_warp_for_slice(
    slice_rgb: np.ndarray,
    *,
    forward_fixed_gray: np.ndarray,
    forward_result_transform: Any,
    scratch_dir: Path | None = None,
) -> tuple[np.ndarray, Any]:
    """Convenience: run the inverse warp end-to-end from in-memory forward outputs.

    Handles the small ceremony around the fixed-to-fixed inverse pattern:
    rebuilds the forward parameter object, writes the forward transform to a
    temp file, re-creates the forward fixed itk.Image, and delegates to
    :func:`_warp_slice_to_atlas`.

    Returns ``(warped_slice_rgb_uint8, inverse_transform_parameters)``.
    """
    import itk

    parameter_object = _build_elastix_parameter_object(
        _grid_spacing_px(forward_fixed_gray.shape)
    )
    fixed_image_itk = itk.image_from_array(forward_fixed_gray.astype(np.float32))

    use_tempdir = scratch_dir is None
    if use_tempdir:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="langslice_inverse_warp_")
        scratch_dir = Path(tmp_ctx.name)
    else:
        tmp_ctx = None
        scratch_dir = Path(scratch_dir)

    try:
        forward_params_path = _write_forward_transform_to_disk(
            forward_result_transform, scratch_dir
        )
        warped_slice_rgb, inverse_params = _warp_slice_to_atlas(
            slice_rgb=slice_rgb,
            parameter_object=parameter_object,
            forward_transform_params_path=forward_params_path,
            fixed_image_itk=fixed_image_itk,
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    return warped_slice_rgb, inverse_params


def _extract_visualign_markers(
    deformation_field: np.ndarray | None,
    scale_to_slice: float,
    origin_px: tuple[float, float] = (0.0, 0.0),
) -> list[list[float]]:
    """Sample the composed Elastix deformation field on a grid as VisuAlign markers.

    The field comes from transformix, so it composes ALL registration stages
    (affine + B-spline) and holds exact displacements. Reading the B-spline
    parameter map instead — as this function once did — silently dropped the
    affine stage and mistook B-spline coefficients for displacements
    (overstating an isolated node's displacement by up to ~2.25x).

    ``origin_px`` is where the original slice image starts on the (possibly
    padded) working canvas; markers are expressed in original-image pixels and
    may legitimately fall OUTSIDE the image bounds — with a fragment or
    hemibrain the complete atlas exceeds the picture. No clipping happens
    here; consumers clip to their own needs in the integration adapters.
    """
    if deformation_field is None:
        logger.warning("No deformation field available; returning empty markers")
        return []

    # Sample at the transform's own control-grid spacing — the field carries
    # no information between control points; consumer-specific densities
    # belong in the integration adapters.
    height, width = deformation_field.shape[:2]
    spacing = _grid_spacing_px((height, width))
    markers: list[list[float]] = []
    ys = np.unique(np.append(np.arange(0.0, height, spacing), height - 1.0))
    xs = np.unique(np.append(np.arange(0.0, width, spacing), width - 1.0))
    for y in ys:
        for x in xs:
            dx, dy = deformation_field[int(round(y)), int(round(x))][:2]
            ox = (x - origin_px[0]) * scale_to_slice
            oy = (y - origin_px[1]) * scale_to_slice
            nx_pos = (x + float(dx) - origin_px[0]) * scale_to_slice
            ny_pos = (y + float(dy) - origin_px[1]) * scale_to_slice
            markers.append([ox, oy, nx_pos, ny_pos])

    logger.info(
        "Extracted %d VisuAlign markers at %dpx field spacing", len(markers), spacing
    )
    return markers


def _segmentation_prompt_for_plane(plane: Plane = "coronal") -> str:
    """Deprecated alias — prompt text now lives in ``nonlinear.model_prompts``."""
    from langslice.nonlinear.model_prompts import base_segmentation_prompt

    return base_segmentation_prompt(plane)


# --- generation report (human/benchmark diagnostics; never reaches the model) -
#: Normalized analysis grid. Both maps are resampled onto it over their own
#: foreground bbox, which is the only alignment this report assumes.
_GEN_GRID = 512
#: A painted component is judged only above this share of the painted
#: foreground — only "big no-nos" flag, never dust or edge speckle.
_GEN_MIN_COMPONENT_SHARE = 0.01
#: A generated pixel counts as ectopic when it lies farther than this
#: fraction of the brain extent from ANY atlas pixel of the same region.
#: Deformation moves a region's paint; it does not teleport it out of its
#: own neighbourhood.
_GEN_ECTOPIC_RADIUS = 0.02
#: A component flags only when nearly all of it is ectopic — a region that
#: merely spilled past its atlas outline keeps a foot inside.
_GEN_ECTOPIC_FRACTION = 0.9
#: The invented-subdivision test adds up small parcels instead of judging one
#: component, so it uses a wider halo: at 2% every region in a shifted
#: brainstem reads as foreign inside its neighbours, at 5% only paint from
#: genuinely elsewhere does.
_GEN_INVENTED_RADIUS = 0.05
#: An atlas region hosts the invented-subdivision test above this share of
#: the atlas foreground.
_GEN_HOST_MIN_SHARE = 0.02
#: Per invented parcel: share of the host it must cover to count.
_GEN_INVENTED_LABEL_SHARE = 0.01
#: Distinct invented parcels inside one host, and their combined share of it,
#: before the host flags. One foreign parcel is a neighbour that drifted (and
#: gets an ECTOPIC_REGION flag of its own if it is big enough); several
#: unrelated ones are a parcellation the model made up.
_GEN_INVENTED_MIN_LABELS = 2
_GEN_INVENTED_TOTAL_SHARE = 0.15


def _normalized_masks(classified_2d: np.ndarray) -> dict[int, np.ndarray]:
    """Per-region boolean masks resampled onto a unit grid over the fg bbox."""
    import cv2

    fg = classified_2d != 0
    if not fg.any():
        return {}
    ys, xs = np.nonzero(fg)
    crop = classified_2d[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    resized = cv2.resize(
        crop.astype(np.float32), (_GEN_GRID, _GEN_GRID), interpolation=cv2.INTER_NEAREST
    ).astype(classified_2d.dtype)
    return {
        int(uid): resized == int(uid) for uid in np.unique(resized) if int(uid) != 0
    }


def _ectopic_masks(
    gen_masks: dict[int, np.ndarray],
    ref_masks: dict[int, np.ndarray],
    radius_fraction: float = _GEN_ECTOPIC_RADIUS,
) -> dict[int, np.ndarray]:
    """Per region, the generated pixels far from that region's atlas footprint.

    "Far" is a fixed fraction of the brain extent, so the test asks a
    topological question (is this paint anywhere near where the region
    lives?) rather than a metric one (did it move?). A region absent from
    the atlas render is ectopic wherever it was painted.
    """
    from scipy import ndimage

    radius = radius_fraction * _GEN_GRID
    out: dict[int, np.ndarray] = {}
    for uid, mask in gen_masks.items():
        ref_mask = ref_masks.get(uid)
        if ref_mask is None or not ref_mask.any():
            out[uid] = mask.copy()
            continue
        distance = np.asarray(ndimage.distance_transform_edt(~ref_mask))
        out[uid] = mask & (distance > radius)
    return out


def generation_report(
    generated_classified: np.ndarray,
    atlas_classified: np.ndarray,
    atlas: Any,
    *,
    tissue_mask: np.ndarray | None = None,
    preserved_fraction: float | None = None,
    structures: Any = None,
) -> dict[str, Any]:
    """Diagnose the GENERATED map against the atlas map — for humans only.

    Two flags, both built on the same primitive: paint that sits nowhere
    near its own region's atlas footprint. ECTOPIC_REGION reports a
    substantial component of it (a region painted where that region is not,
    e.g. ventricle grey filling the cerebellum); INVENTED_SUBDIVISION
    reports several such parcels inside ONE atlas region (the model
    inventing a parcellation, e.g. glomeruli-like islands in the olfactory
    bulb). Neither can be tripped by deformation: regions are expected to
    shift, stretch and grow, and both tests ask only whether the paint is in
    the region's own neighbourhood at all. Silhouette and edit-anchor
    numbers are reported without flags. This report is never sent to any
    model.
    """
    from scipy import ndimage

    family = _family_mapping(
        [*np.unique(generated_classified), *np.unique(atlas_classified)], atlas
    )
    gen = np.zeros_like(generated_classified)
    ref = np.zeros_like(atlas_classified)
    for uid, rep in family.items():
        gen[generated_classified == uid] = rep
        ref[atlas_classified == uid] = rep

    gen_masks = _normalized_masks(gen)
    ref_masks = _normalized_masks(ref)
    ectopic = _ectopic_masks(gen_masks, ref_masks)
    far_ectopic = _ectopic_masks(gen_masks, ref_masks, _GEN_INVENTED_RADIUS)
    flags: list[dict[str, Any]] = []

    gen_total = sum(int(m.sum()) for m in gen_masks.values()) or 1
    ref_total = sum(int(m.sum()) for m in ref_masks.values()) or 1

    def _atlas_under(mask: np.ndarray) -> int | None:
        hits = [(int((mask & m).sum()), uid) for uid, m in ref_masks.items()]
        hits = [h for h in hits if h[0] > 0]
        return max(hits)[1] if hits else None

    for uid, mask in gen_masks.items():
        ect = ectopic[uid]
        labels, count = ndimage.label(mask)  # type: ignore[misc]
        for comp in range(1, count + 1):
            comp_mask = labels == comp
            share = comp_mask.sum() / gen_total
            if share < _GEN_MIN_COMPONENT_SHARE:
                continue
            ectopic_share = (comp_mask & ect).sum() / comp_mask.sum()
            if ectopic_share < _GEN_ECTOPIC_FRACTION:
                continue
            host = _atlas_under(comp_mask)
            host_mask = ref_masks.get(host) if host is not None else None
            ys, xs = np.nonzero(comp_mask)
            flags.append(
                {
                    "code": "ECTOPIC_REGION",
                    "region": _region_label(structures, uid),
                    "share": round(float(share), 4),
                    "ectopic_fraction": round(float(ectopic_share), 3),
                    "at": [
                        round(float(xs.mean()) / _GEN_GRID, 2),
                        round(float(ys.mean()) / _GEN_GRID, 2),
                    ],
                    "atlas_has_there": _region_label(structures, host)
                    if host is not None
                    else None,
                    "covers_of_that_region": round(
                        float((comp_mask & host_mask).sum() / int(host_mask.sum())), 3
                    )
                    if host_mask is not None and host_mask.any()
                    else None,
                }
            )

    for host_uid, host_mask in ref_masks.items():
        host_area = int(host_mask.sum())
        if host_area / ref_total < _GEN_HOST_MIN_SHARE:
            continue
        parcels = []
        for uid, ect in far_ectopic.items():
            if uid == host_uid:
                continue
            inside = int((ect & host_mask).sum())
            if inside / host_area >= _GEN_INVENTED_LABEL_SHARE:
                parcels.append((inside / host_area, uid))
        total = sum(share for share, _ in parcels)
        if len(parcels) >= _GEN_INVENTED_MIN_LABELS and total >= _GEN_INVENTED_TOTAL_SHARE:
            flags.append(
                {
                    "code": "INVENTED_SUBDIVISION",
                    "region": _region_label(structures, host_uid),
                    "invented_share_of_region": round(float(total), 3),
                    "parcels": [
                        {
                            "region": _region_label(structures, uid),
                            "share_of_host": round(float(share), 3),
                        }
                        for share, uid in sorted(parcels, reverse=True)
                    ],
                }
            )

    report: dict[str, Any] = {"flags": flags}
    # Families the atlas has on this slice (>=1% of its foreground) that the
    # generated map lacks entirely: a completeness number for candidate
    # ranking — under the segment-only-existing contract a large region CAN
    # legitimately be absent, so this is a ranking signal, never a flag.
    ref_total = sum(int(m.sum()) for m in ref_masks.values()) or 1
    missing = [
        uid
        for uid, m in ref_masks.items()
        if m.sum() / ref_total >= 0.01 and uid not in gen_masks
    ]
    report["families_missing"] = len(missing)
    if tissue_mask is not None and tissue_mask.any():
        painted = generated_classified != 0
        inter = int((painted & tissue_mask).sum())
        union = int((painted | tissue_mask).sum()) or 1
        report["silhouette_iou"] = round(inter / union, 4)
        report["map_beyond_tissue"] = round(
            float((painted & ~tissue_mask).sum() / max(int(painted.sum()), 1)), 4
        )
        report["tissue_unpainted"] = round(
            float((tissue_mask & ~painted).sum() / max(int(tissue_mask.sum()), 1)), 4
        )
    if preserved_fraction is not None:
        report["preserved_background_fraction"] = round(float(preserved_fraction), 4)
    return report
