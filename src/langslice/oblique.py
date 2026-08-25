"""Oblique-plane sampling and (pitch, yaw) fitting against a BrainGlobe atlas.

Adapted from `brainglobe-registration <https://github.com/brainglobe/brainglobe-registration>`_,
copyright the BrainGlobe developers, BSD-3-Clause. Specifically:

* the Euler-angle rotation-matrix construction of
  ``brainglobe_registration.utils.transforms.create_rotation_matrix``,
* the MI / NCC / SSIM similarity metrics of
  ``brainglobe_registration.similarity_metrics``,
* the coarse-to-fine angle search of
  ``brainglobe_registration.automated_target_selection``.

Three deliberate departures from upstream:

1. **Sample the plane, don't rotate the volume.** Upstream rotates the whole
   atlas volume (dask ``affine_transform``) and then takes a z slice. We sample
   a single arbitrary plane with :func:`scipy.ndimage.map_coordinates`, which is
   three orders of magnitude cheaper and lets a per-slice fit run in ~1 s.
2. **No new dependencies.** Upstream pulls scikit-image (SSIM),
   scikit-learn (kNN mutual information) and bayes-opt (which itself pulls
   scikit-learn). MI here is the standard joint-histogram estimator, SSIM is
   the Wang et al. formulation over OpenCV Gaussians, and the optimiser is a
   coarse-to-fine grid. Over two or three bounded parameters that benches as
   well as Gaussian-process Bayesian search, costs nothing, and — unlike the
   Nelder-Mead simplex tried first — does not latch onto a local ridge when the
   metric surface is bumpy at the sub-degree scale.
3. **The mirror hypothesis.** :func:`fit_oblique` can fit the section both as
   given and left-right mirrored, and reports both scores. An obliquely
   resliced atlas is *not* left-right symmetric when the yaw is non-zero, so
   the two hypotheses are distinguishable in a way that flat coronal planes
   provably are not.

Angle convention (defined on the raw atlas axes, before any display swap):

* ``pitch_deg`` rotates about the plane's **column** axis. For a coronal plane
  in an ``asr``-style atlas that is the ML axis, i.e. the usual "cutting angle"
  that makes the section tilt anterior-dorsal / posterior-ventral. A pitched
  plane is still left-right symmetric.
* ``yaw_deg`` rotates about the plane's **row** axis (DV for coronal), i.e. the
  section samples a more anterior AP level on one side than the other. This is
  the *only* component that breaks left-right symmetry, and therefore the only
  one that can carry a hemisphere-flip signal.

Both are right-handed rotations in the atlas index frame, applied pitch first.
"""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from langslice.affine import (
    _affine_from_pose,  # pyright: ignore[reportPrivateUsage]
    _moments_pose,  # pyright: ignore[reportPrivateUsage]
    extract_slice_silhouette,
    silhouette_iou,
)
from langslice.atlas.core import orient_slice_for_display
from langslice.space import Plane, atlas_space_context, slice_axis_index

Metric = Literal["mi", "ncc", "ssim", "combined"]

#: Weights for ``metric="combined"``, matching brainglobe-registration's default
#: ordering (MI, NCC, SSIM).
DEFAULT_WEIGHTS: tuple[float, float, float] = (0.7, 0.15, 0.15)

#: Gaussian blur applied to both images before scoring, as a fraction of the
#: atlas plane's long edge. Upstream leans on an Elastix affine per candidate to
#: soak up residual pose error; a blur buys most of that robustness for none of
#: the runtime, and a fraction (rather than a pixel count) keeps it meaningful
#: across atlas resolutions.
DEFAULT_BLUR_FRACTION = 0.01

#: Only the two det>0 sign patterns. brainglobe's :func:`silhouette_affine`
#: also tries the two reflections, which would silently un-mirror a flipped
#: section and destroy exactly the signal this module exists to measure.
_ROTATION_SIGNS: tuple[tuple[int, int], ...] = ((1, 1), (-1, -1))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def plane_axes(atlas: Any, plane: Plane = "coronal") -> tuple[int, int, int]:
    """``(normal_axis, row_axis, col_axis)`` of *plane* in atlas array order.

    Row/col are the two non-normal axes in ascending index order, which is the
    layout :func:`numpy.take` produces and therefore what
    :func:`langslice.atlas.core.get_reference_slice` starts from.
    """
    context = atlas_space_context(atlas)
    normal = slice_axis_index(context, plane)
    row, col = (a for a in range(3) if a != normal)
    return normal, row, col


def build_rotation_matrix(
    pitch_deg: float, yaw_deg: float, *, row_axis: int, col_axis: int
) -> np.ndarray:
    """3x3 rotation: *pitch_deg* about *col_axis*, then *yaw_deg* about *row_axis*.

    Extrinsic (fixed-frame) composition, so the returned matrix is
    ``R_yaw @ R_pitch``. Axes are atlas array axes, not anatomical names.
    """
    letters = "".join("xyz"[axis] for axis in (col_axis, row_axis))
    return np.asarray(
        Rotation.from_euler(letters, [pitch_deg, yaw_deg], degrees=True).as_matrix(),
        dtype=np.float64,
    )


_volume_cache: dict[tuple[str, str, int, int], tuple[np.ndarray, tuple[float, float, float]]] = {}


def _scaled_volume(
    atlas: Any, kind: str, downsample: int
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Contiguous float32 volume strided by *downsample*, plus its mm resolution.

    Cached per (atlas, kind, downsample): a 25 um mouse template is 300 MB as
    float32 and the fit re-samples it hundreds of times per section.
    """
    key = (str(getattr(atlas, "atlas_name", "?")), kind, int(downsample), id(atlas))
    cache = _volume_cache
    hit = cache.get(key)
    if hit is not None:
        return hit
    raw = np.asarray(getattr(atlas, kind))
    step = max(1, int(downsample))
    volume = np.ascontiguousarray(raw[::step, ::step, ::step], dtype=np.float32)
    context = atlas_space_context(atlas)
    res_mm = cast(
        "tuple[float, float, float]",
        tuple(r / 1000.0 * step for r in context.resolution_um),
    )
    if len(cache) > 8:  # ponytail: tiny manual LRU; atlases are few and huge
        cache.clear()
    cache[key] = (volume, res_mm)
    return volume, res_mm


def sample_oblique_plane(
    atlas: Any,
    position_mm: float,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    *,
    volume: str | np.ndarray = "template",
    downsample: int = 1,
    order: int = 1,
) -> np.ndarray:
    """Sample one arbitrary oblique plane out of an atlas volume.

    The plane passes through the volume's in-plane centre at *position_mm*
    along the slice-normal axis (atlas-native millimetres from the anterior
    edge for a coronal plane), tilted by *pitch_deg* and *yaw_deg*.

    At ``pitch_deg == yaw_deg == 0`` and ``downsample=1`` this reproduces
    :func:`langslice.atlas.core.get_reference_slice`'s array exactly (same
    shape, same display orientation, same voxels) for any *position_mm* that
    lands on the voxel grid.

    Args:
        atlas: BrainGlobe-style atlas exposing ``template``/``annotation``.
        position_mm: Position along the slice-normal axis.
        plane: Slicing plane.
        pitch_deg: Rotation about the plane's column axis (degrees).
        yaw_deg: Rotation about the plane's row axis (degrees).
        volume: ``"template"``, ``"annotation"``, or a full-resolution array
            with the same shape as the template.
        downsample: Integer stride applied to all three volume axes. The
            returned plane shrinks by the same factor.
        order: ``map_coordinates`` spline order. 1 (linear) is right for a
            search loop; 3 for a final render.

    Returns:
        2D float32 array, oriented like the rest of ``langslice.atlas.core``.
    """
    normal_axis, row_axis, col_axis = plane_axes(atlas, plane)
    if isinstance(volume, str):
        vol, res_mm = _scaled_volume(atlas, volume, downsample)
    else:
        step = max(1, int(downsample))
        vol = np.ascontiguousarray(np.asarray(volume)[::step, ::step, ::step], dtype=np.float32)
        context = atlas_space_context(atlas)
        res_mm = cast(
            "tuple[float, float, float]",
            tuple(r / 1000.0 * step for r in context.resolution_um),
        )

    height, width = vol.shape[row_axis], vol.shape[col_axis]
    rotation = build_rotation_matrix(pitch_deg, yaw_deg, row_axis=row_axis, col_axis=col_axis)

    # In-plane unit directions in physical space, expressed on the atlas axes,
    # converted to index steps one voxel wide along their own axis. Dividing
    # component-wise by the per-axis resolution keeps anisotropic atlases square.
    res = np.asarray(res_mm, dtype=np.float64)
    basis_row = np.zeros(3)
    basis_row[row_axis] = 1.0
    basis_col = np.zeros(3)
    basis_col[col_axis] = 1.0
    step_row = (rotation @ basis_row) * res[row_axis] / res
    step_col = (rotation @ basis_col) * res[col_axis] / res

    centre = np.zeros(3)
    centre[normal_axis] = position_mm / res[normal_axis]
    centre[row_axis] = (height - 1) / 2.0
    centre[col_axis] = (width - 1) / 2.0

    rows = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0)[:, None]
    cols = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0)[None, :]
    coords = np.empty((3, height, width), dtype=np.float64)
    for axis in range(3):
        coords[axis] = centre[axis] + rows * step_row[axis] + cols * step_col[axis]

    sampled = map_coordinates(vol, coords, order=order, mode="constant", cval=0.0)
    return np.asarray(orient_slice_for_display(sampled, plane), dtype=np.float32)


# --------------------------------------------------------------------------
# similarity metrics (adapted from brainglobe_registration.similarity_metrics)
# --------------------------------------------------------------------------


def normalise_image(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Scale to [0, 1] using the min/max inside *mask* (or the whole frame)."""
    arr = np.asarray(image, dtype=np.float32)
    values = arr[mask] if mask is not None else arr
    if values.size == 0:
        return np.zeros_like(arr)
    lo, hi = float(values.min()), float(values.max())
    return (arr - lo) / (hi - lo + 1e-8)


def mutual_information(
    moving: np.ndarray, fixed: np.ndarray, *, bins: int = 64, mask: np.ndarray | None = None
) -> float:
    """Normalised mutual information in [0, 1] from a joint histogram.

    Upstream uses scikit-learn's kNN ``mutual_info_regression``; the histogram
    estimator is the textbook choice for images, has no dependency, and is
    ~100x faster. Normalising by the marginal entropies (``2I / (H_a + H_b)``)
    keeps scores comparable across masks of different size.
    """
    a = np.asarray(moving, dtype=np.float32)
    b = np.asarray(fixed, dtype=np.float32)
    if mask is not None:
        a, b = a[mask], b[mask]
    else:
        a, b = a.ravel(), b.ravel()
    if a.size == 0:
        return 0.0

    # A joint histogram needs samples per CELL, not per axis: 64x64 bins over a
    # 900-pixel mask is almost all zeros and the estimate turns to noise. Cap
    # the bin count so the joint histogram averages ~8 samples per cell.
    bins = min(bins, max(8, int(math.sqrt(a.size / 8.0))))
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    joint = joint / max(joint.sum(), 1.0)
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)

    def entropy(p: np.ndarray) -> float:
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum())

    h_a, h_b, h_ab = entropy(p_a), entropy(p_b), entropy(joint)
    denominator = h_a + h_b
    if denominator <= 0:
        return 0.0
    return float(2.0 * (h_a + h_b - h_ab) / denominator)


def safe_ncc(
    moving: np.ndarray, fixed: np.ndarray, *, mask: np.ndarray | None = None
) -> float:
    """Normalised cross-correlation in [-1, 1]; 0.0 if either input is constant."""
    a = np.asarray(moving, dtype=np.float64)
    b = np.asarray(fixed, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Input shapes must match. Got {a.shape} and {b.shape}")
    if mask is not None:
        a, b = a[mask], b[mask]
    else:
        a, b = a.ravel(), b.ravel()
    if a.size < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def structural_similarity(
    moving: np.ndarray, fixed: np.ndarray, *, data_range: float = 1.0, sigma: float = 1.5
) -> float:
    """Global mean SSIM (Wang et al. 2004) over OpenCV Gaussians.

    Same formulation and default Gaussian width as
    ``skimage.metrics.structural_similarity(gaussian_weights=True)``.
    """
    a = np.asarray(moving, dtype=np.float32)
    b = np.asarray(fixed, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Input shapes must match. Got {a.shape} and {b.shape}")
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def blur(img: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT))

    mu_a, mu_b = blur(a), blur(b)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = blur(a * a) - mu_aa
    var_b = blur(b * b) - mu_bb
    cov = blur(a * b) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * cov + c2)) / (
        (mu_aa + mu_bb + c1) * (var_a + var_b + c2)
    )
    return float(np.mean(ssim_map))


def compute_similarity_metric(
    moving: np.ndarray,
    fixed: np.ndarray,
    metric: Metric = "mi",
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    *,
    mask: np.ndarray | None = None,
    blur_sigma: float = 0.0,
) -> float:
    """Similarity between two same-shape 2D images. Higher is better.

    *mask* (a boolean array) restricts MI and NCC to a region of interest;
    SSIM is always computed over the whole frame because it is spatial.

    *blur_sigma* Gaussian-blurs both images first, in pixels. This is not
    cosmetic: a closed-form moments pose leaves a couple of percent of residual
    scale and rotation error, and on sharp images that alone costs more
    similarity than a 15-degree change of cutting plane does, which flattens the
    score surface into noise.
    """
    if moving.shape != fixed.shape:
        raise ValueError(f"Input shapes must match. Got {moving.shape} and {fixed.shape}")
    if blur_sigma > 0:
        moving = cast(np.ndarray, cv2.GaussianBlur(moving.astype(np.float32), (0, 0), blur_sigma))
        fixed = cast(np.ndarray, cv2.GaussianBlur(fixed.astype(np.float32), (0, 0), blur_sigma))
    a = normalise_image(moving, mask)
    b = normalise_image(fixed, mask)

    if metric == "mi":
        return mutual_information(a, b, mask=mask)
    if metric == "ncc":
        return safe_ncc(a, b, mask=mask)
    if metric == "ssim":
        return structural_similarity(a, b)
    if metric == "combined":
        if len(weights) != 3:
            raise ValueError(f"Invalid weights: {weights!r}. Must be a 3-tuple of floats.")
        return (
            weights[0] * mutual_information(a, b, mask=mask)
            + weights[1] * safe_ncc(a, b, mask=mask)
            + weights[2] * structural_similarity(a, b)
        )
    raise ValueError(
        f"Unsupported metric {metric!r}. Choose from 'mi', 'ncc', 'ssim', 'combined'."
    )


# --------------------------------------------------------------------------
# section pose normalisation
# --------------------------------------------------------------------------


def _as_gray(image: Any) -> np.ndarray:
    """Coerce a PIL image / RGB array / gray array to a 2D float32 array."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(
            arr.astype(np.uint8) if arr.dtype != np.uint8 else arr, cv2.COLOR_RGB2GRAY
        )
    return arr.astype(np.float32)


def section_silhouette(section_gray: np.ndarray) -> np.ndarray:
    """Filled uint8 tissue silhouette of a grayscale section. Hoistable — the
    fit computes it once and reuses it for every candidate plane."""
    gray_u8 = cv2.normalize(section_gray, None, 0, 255, cv2.NORM_MINMAX).astype(  # type: ignore[call-overload]
        np.uint8
    )
    mask = extract_slice_silhouette(gray_u8)
    if not mask.any():
        raise ValueError("Section silhouette is empty — cannot align.")
    return mask


def align_section_to_plane(
    section_gray: np.ndarray,
    target_mask: np.ndarray,
    *,
    section_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Warp a section into the frame of one atlas plane, by silhouette moments.

    Rotations only — the reflection candidates that
    :func:`langslice.affine.silhouette_affine` also tries would undo a
    hemisphere flip, which is exactly the thing under test here.

    Re-fitting against *each candidate plane's own* silhouette (rather than
    once against the flat plane) matters: an oblique cut has a larger, differently
    shaped cross-section, and normalising every candidate onto the flat
    silhouette throws away that cue and biases the fit toward large |pitch|.

    Returns ``(warped_section, warped_mask, iou)``.
    """
    mask = section_silhouette(section_gray) if section_mask is None else section_mask

    height, width = target_mask.shape
    dst_mask = (np.asarray(target_mask) > 0).astype(np.uint8) * 255
    src_c, src_eigvals, src_v = _moments_pose(mask)
    dst_c, dst_eigvals, dst_v = _moments_pose(dst_mask)

    best_iou = -1.0
    best_matrix: np.ndarray | None = None
    best_warped_mask: np.ndarray | None = None
    for signs in _ROTATION_SIGNS:
        candidate = _affine_from_pose(
            src_c, src_eigvals, src_v, dst_c, dst_eigvals, dst_v, signs
        )
        warped_mask = cv2.warpAffine(
            mask, candidate, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
        )
        iou = silhouette_iou(warped_mask, dst_mask)
        if iou > best_iou:
            best_iou, best_matrix, best_warped_mask = iou, candidate, warped_mask

    if best_matrix is None or best_warped_mask is None:
        raise ValueError("No rotation candidate could be computed.")
    warped = cv2.warpAffine(
        section_gray.astype(np.float32),
        best_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    return np.asarray(warped, dtype=np.float32), np.asarray(best_warped_mask) > 0, best_iou


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------


def _grid_points(bounds: tuple[float, float], n: int) -> np.ndarray:
    lo, hi = float(bounds[0]), float(bounds[1])
    if math.isclose(lo, hi) or n <= 1:
        return np.array([(lo + hi) / 2.0])
    return np.linspace(lo, hi, n)


def _fit_one_branch(
    atlas: Any,
    section_gray: np.ndarray,
    position_mm: float,
    plane: Plane,
    pitch_bounds: tuple[float, float],
    yaw_bounds: tuple[float, float],
    position_window_mm: float,
    metric: Metric,
    weights: tuple[float, float, float],
    downsample: int,
    grid: int,
    refine: bool,
    blur_fraction: float,
) -> dict[str, Any]:
    silhouette = section_silhouette(section_gray)
    evals = 0
    last_iou = 0.0

    def score(pitch: float, yaw: float, pos: float) -> float:
        nonlocal evals, last_iou
        evals += 1
        atlas_plane = sample_oblique_plane(
            atlas, pos, plane, pitch, yaw, volume="template", downsample=downsample
        )
        # The atlas mask comes from the ANNOTATION, never from Otsu. Otsu on a
        # tilted template plane is discontinuous in the angle — on the Allen
        # 25 um template the largest-component step drops a fifth of the
        # silhouette between pitch 12 and pitch 13 — which puts a cliff in the
        # score surface and makes the whole fit chase an artefact. order=0
        # because interpolating structure IDs is meaningless, and because a
        # linear sample of them dilates the mask by half a voxel, which then
        # asks the pose fit to scale the section up by a few percent.
        target_mask = (
            sample_oblique_plane(
                atlas,
                pos,
                plane,
                pitch,
                yaw,
                volume="annotation",
                downsample=downsample,
                order=0,
            )
            > 0
        )
        warped, warped_mask, last_iou = align_section_to_plane(
            section_gray, target_mask, section_mask=silhouette
        )
        return compute_similarity_metric(
            warped,
            atlas_plane * target_mask,  # the Allen template's background is not 0
            metric,
            weights,
            mask=warped_mask,
            blur_sigma=blur_fraction * max(atlas_plane.shape),
        )

    cache: dict[tuple[int, int, int], float] = {}

    def cached_score(pitch: float, yaw: float, pos: float) -> float:
        key = (round(pitch * 1000), round(yaw * 1000), round(pos * 100000))
        hit = cache.get(key)
        if hit is None:
            hit = score(pitch, yaw, pos)
            cache[key] = hit
        return hit

    pos_bounds = (position_mm - position_window_mm, position_mm + position_window_mm)
    free_position = position_window_mm > 0
    bounds = [pitch_bounds, yaw_bounds, pos_bounds]
    counts = [grid, grid, 5 if free_position else 1]

    # Coarse-to-fine grid. Nelder-Mead was the first thing tried here and it is
    # the wrong tool: on a sub-degree scale the metric surface is bumpy enough
    # that a simplex latches onto a local ridge and never finds the true peak,
    # while halving the grid spacing three times is both robust and cheap.
    axes = [_grid_points(b, n) for b, n in zip(bounds, counts, strict=True)]
    best_score, best = -np.inf, [0.0, 0.0, position_mm]
    spans = [
        float(axis[1] - axis[0]) if axis.size > 1 else 0.0 for axis in axes
    ]

    for round_index in range(4 if refine else 1):
        if round_index:
            axes = [
                (
                    np.clip(np.linspace(centre - span, centre + span, 5), b[0], b[1])
                    if span > 0
                    else np.array([centre])
                )
                for centre, span, b in zip(best, spans, bounds, strict=True)
            ]
            spans = [span / 2.0 for span in spans]
        for pitch in axes[0]:
            for yaw in axes[1]:
                for pos in axes[2]:
                    value = cached_score(float(pitch), float(yaw), float(pos))
                    if value > best_score:
                        best_score = value
                        best = [float(pitch), float(yaw), float(pos)]

    best_pitch, best_yaw, best_pos = best
    if not free_position:
        best_pos = position_mm
    score(best_pitch, best_yaw, best_pos)  # re-run the winner so last_iou is its IoU
    iou = last_iou

    return {
        "pitch_deg": best_pitch,
        "yaw_deg": best_yaw,
        "position_mm": best_pos,
        "score": best_score,
        "iou": iou,
        "n_evals": evals,
    }


def fit_oblique(
    atlas: Any,
    section_image: Any,
    position_mm: float,
    plane: Plane = "coronal",
    *,
    pitch_bounds: tuple[float, float] = (-15.0, 15.0),
    yaw_bounds: tuple[float, float] = (-15.0, 15.0),
    position_window_mm: float = 0.0,
    metric: Metric = "mi",
    allow_mirror: bool = True,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    downsample: int = 2,
    grid: int = 7,
    refine: bool = True,
    blur_fraction: float = DEFAULT_BLUR_FRACTION,
) -> dict[str, Any]:
    """Fit signed oblique angles for one section at a known atlas position.

    The section is pose-normalised onto the atlas plane by silhouette moments
    (rotations only), then ``(pitch, yaw)`` — and optionally position within a
    window — are searched by a coarse grid followed by three halving refines.

    With ``allow_mirror=True`` the whole fit runs twice, once on the section as
    given and once on its left-right mirror, and both results are returned. A
    pitched-only plane is left-right symmetric, so the two branches will tie;
    any separation comes from yaw.

    Args:
        atlas: BrainGlobe-style atlas.
        section_image: PIL image or array (RGB or grayscale) of the section,
            already contrast-normalised and downscaled.
        position_mm: Known position along the slice-normal axis.
        plane: Slicing plane.
        pitch_bounds: ``(low, high)`` degrees about the plane's column axis.
        yaw_bounds: ``(low, high)`` degrees about the plane's row axis.
        position_window_mm: If > 0, also search position in
            ``position_mm ± window``.
        metric: ``"mi"``, ``"ncc"``, ``"ssim"`` or ``"combined"``.
        allow_mirror: Fit the mirrored section as a competing hypothesis.
        weights: MI/NCC/SSIM weights for ``metric="combined"``.
        downsample: Volume stride. 2 (50 um on the Allen 25 um atlas) is
            plenty for an angle fit and keeps a section under a second.
        grid: Points per angle axis in the coarse grid.
        refine: Run the three halving refine rounds after the coarse grid.
        blur_fraction: Gaussian blur applied to both images before scoring, as
            a fraction of the atlas plane's long edge, to absorb residual pose
            error.

    Returns:
        Dict with the winning ``pitch_deg``, ``yaw_deg``, ``position_mm``,
        ``score`` and ``mirrored`` flag, the per-branch results under
        ``as_is`` / ``mirror``, the convenience keys ``score_as_is`` /
        ``score_mirrored`` / ``mirror_margin`` (mirrored minus as-is), and
        bookkeeping (``metric``, ``downsample``, ``n_evals``).
    """
    gray = _as_gray(section_image)

    def run(image: np.ndarray) -> dict[str, Any]:
        return _fit_one_branch(
            atlas,
            image,
            position_mm,
            plane,
            pitch_bounds,
            yaw_bounds,
            position_window_mm,
            metric,
            weights,
            downsample,
            grid,
            refine,
            blur_fraction,
        )

    as_is = run(gray)
    mirror = run(np.ascontiguousarray(gray[:, ::-1])) if allow_mirror else None

    mirrored = mirror is not None and mirror["score"] > as_is["score"]
    winner = mirror if mirrored else as_is
    assert winner is not None

    return {
        "pitch_deg": winner["pitch_deg"],
        "yaw_deg": winner["yaw_deg"],
        "position_mm": winner["position_mm"],
        "score": winner["score"],
        "mirrored": mirrored,
        "as_is": as_is,
        "mirror": mirror,
        "score_as_is": as_is["score"],
        "score_mirrored": mirror["score"] if mirror else None,
        "mirror_margin": (mirror["score"] - as_is["score"]) if mirror else None,
        "iou": winner["iou"],
        "metric": metric,
        "downsample": downsample,
        "n_evals": as_is["n_evals"] + (mirror["n_evals"] if mirror else 0),
    }


def score_oblique(
    atlas: Any,
    section_image: Any,
    position_mm: float,
    plane: Plane = "coronal",
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    metric: Metric = "mi",
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    downsample: int = 2,
    blur_fraction: float = DEFAULT_BLUR_FRACTION,
    mirror: bool = False,
) -> float:
    """Score one section against one *fixed* oblique plane.

    The constrained counterpart to :func:`fit_oblique`: when the brain's
    cutting angle is already known, a per-section flip decision is just this
    score with and without ``mirror``.
    """
    gray = _as_gray(section_image)
    if mirror:
        gray = np.ascontiguousarray(gray[:, ::-1])
    atlas_plane = sample_oblique_plane(
        atlas, position_mm, plane, pitch_deg, yaw_deg, volume="template", downsample=downsample
    )
    target_mask = (
        sample_oblique_plane(
            atlas,
            position_mm,
            plane,
            pitch_deg,
            yaw_deg,
            volume="annotation",
            downsample=downsample,
            order=0,
        )
        > 0
    )
    warped, warped_mask, _ = align_section_to_plane(gray, target_mask)
    return compute_similarity_metric(
        warped,
        atlas_plane * target_mask,
        metric,
        weights,
        mask=warped_mask,
        blur_sigma=blur_fraction * max(atlas_plane.shape),
    )
