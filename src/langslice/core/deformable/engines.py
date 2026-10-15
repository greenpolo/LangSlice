"""The deformable engine (FireANTs SyN) and the Elastix affine, behind plain calls.

Both see the same thing: a fixed image (the section), a moving image (the
placed atlas, already on the section's working grid), a fixed and a moving
mask, and the physical spacing and origin of that grid. :func:`run_engine`
(FireANTs' SyN, on the GPU when there is one) returns a displacement field in
millimetres on the working grid that points from a section point to the
matching point of the placed atlas image — the convention of
brainglobe-registration and of ANTs' resampling fields:
``atlas_point = section_point + field(section_point)``, and the inverse.
:func:`run_elastix_affine` returns an affine on the same millimetres.

No registration solver lives here; the libraries do the fitting. Pure arrays
in, pure arrays out, so the adapters run unchanged in a worker process.

Determinism: identical inputs give identical results. FireANTs runs with
torch's deterministic algorithms (:func:`_deterministic`); on the CPU torch
and Elastix both run on :data:`FIT_THREADS` threads, because a multithreaded
sum's last bits depend on how the work is split. Elastix's random coordinate
sampler gets a fixed seed.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from langslice.core.deformable.settings import (
    DETAIL,
    ELASTIX_STAIN_METRIC,
    STIFFNESS,
    FitSettings,
    Metric,
)

#: Histogram bins for mutual information, both engines.
MUTUAL_INFORMATION_BINS = 32
#: Random samples per Elastix iteration.
ELASTIX_SPATIAL_SAMPLES = 4096
#: Threads every CPU engine call uses, whatever the machine or pool size: the
#: thread count changes a multithreaded fit's last bits (module docstring).
FIT_THREADS = 8
#: Elastix's RandomSeed, and torch's seed around a FireANTs fit.
RANDOM_SEED = 20261002
#: The largest move of one gradient step, in FireANTs' half-resolutions (half a
#: pixel of the level), fixed. Tuned on Carey, PL_Projections and LSD sections:
#: at 0.8 close-ups of the cerebellar granular layers and the hippocampal cell
#: layers fit at least as tightly as ANTs SyN did and fluorescent brainstem
#: kept its shape; 3.2 tore folia across the white matter. The optimizer is
#: FireANTs' SGD with ANTs' step rule (:func:`_global_step_sgd`), not its
#: default Adam: Adam scales each pixel's step by its own gradient history, so
#: pixels the masks leave out (an excluded region) moved as far as any, driven
#: by the faint smoothed gradient around them.
FIREANTS_LEARNING_RATE = 0.8
#: A fit of traced lines (``section_image="lines"``: the image model's lines
#: and the regions they enclose against the atlas borders and regions) is run
#: as the antspyx SyN it replaced ran it, which these three settings restore;
#: tuned on Carey (seven brains, the image model's saved replies, scored
#: against the raters), where FireANTs at the stain fit's settings followed
#: hallucinated lines, folded and labelled worse than ANTs. The step is ANTs'
#: own 0.2 voxel (0.4 half-pixels), half the stain fit's.
TRACED_LEARNING_RATE = 0.4
#: The whole warp of a traced fit is smoothed this many times wider than its
#: stiffness says (medium: 0.04 mm instead of 0.02), so a stray or invented
#: line bends the atlas less locally.
TRACED_TOTAL_SIGMA_FACTOR = 2.0
#: Each pyramid level halves the grid (shrink factors 2^(n-1)..1 for n levels).
FIREANTS_SHRINK = 2
#: Fixed-point iterations (and relaxation) that invert each half of the SyN
#: warp (:func:`_invert`) and a tapered restricted step (``fit._invert_step``, the
#: same fixed point on the section grid in numpy).
INVERSE_ITERATIONS = 200
INVERSE_RELAXATION = 0.5
#: The release built from github.com/rohitrango/FireANTs at f8a9cc2; part of
#: the deformable cache key (``core.deformation.cache_key``).
FIREANTS_VERSION = "1.5.1"
#: CUBLAS needs this before its first use for deterministic matrix products.
_CUBLAS_ENV = ("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class EngineInputs:
    """One fit problem on the working grid (all arrays H x W)."""

    fixed: np.ndarray
    moving: np.ndarray
    fixed_mask: np.ndarray
    moving_mask: np.ndarray
    spacing_mm: tuple[float, float]
    origin_mm: tuple[float, float]
    #: Label-map channels: paired region indicators (float, softened) and
    #: their metric weights. FireANTs only.
    fixed_labels: list[np.ndarray] = field(default_factory=list)
    moving_labels: list[np.ndarray] = field(default_factory=list)
    label_weights: list[float] = field(default_factory=list)
    #: Edge channel (stain fits): smoothed gradient magnitudes of both images
    #: and its metric weight; None without one.
    fixed_edges: np.ndarray | None = None
    moving_edges: np.ndarray | None = None
    edge_weight: float = 0.0
    #: Local correlation window radius in working-grid pixels.
    correlation_radius_px: int = 4


@dataclass
class EngineResult:
    """Fields are (H, W, 2) [dx, dy] millimetres on the working grid."""

    field_mm: np.ndarray
    inverse_field_mm: np.ndarray
    inverse_source: str
    native_parameters: dict[str, Any]
    runtime_s: float
    engine_version: str
    notes: list[str] = field(default_factory=list)
    #: Section pixels the fit's metric left out (an EM section mask,
    #: ``masks.em_masks``), on the working grid; None for a footprint mask.
    left_out: np.ndarray | None = None


def run_engine(inputs: EngineInputs, settings: FitSettings) -> EngineResult:
    """Fit one deformation with FireANTs' SyN. Raises when FireANTs is not
    installed or fails, and for any other engine (Elastix fits affines only:
    :func:`run_elastix_affine`)."""
    if settings.engine != "fireants":
        raise ValueError(f"No deformable fit with {settings.engine!r}: the deformable "
                         "engine is SyN (FireANTs)")
    return _run_fireants(inputs, settings, settings.metric)


_TORCH_LOCK = threading.Lock()


def import_fireants() -> Any:
    """Import torch and FireANTs' SyN (``RuntimeError`` with the install hint).

    Sets ``CUBLAS_WORKSPACE_CONFIG`` first when nothing set it, so the
    deterministic algorithms can run on the GPU.
    """
    os.environ.setdefault(*_CUBLAS_ENV)
    try:
        import torch
        # FireANTs warns once per module that its optional fused CUDA kernels
        # are absent; the plain torch path is what LangSlice runs.
        for name in ("fireants", "fireants.registration.optimizers.adam",
                     "fireants.interpolator", "fireants.registration.abstract",
                     "fireants.io.image"):
            logging.getLogger(name).setLevel(logging.ERROR)
        with contextlib.redirect_stdout(io.StringIO()):  # "Using FFO: False"
            from fireants.registration.syn import SyNRegistration  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The deformable engine needs FireANTs and torch, LangSlice dependencies "
            "(reinstall LangSlice)"
        ) from exc
    return torch


def fit_device() -> str:
    """``cuda`` when torch sees a GPU, else ``cpu``."""
    torch = import_fireants()
    return "cuda" if torch.cuda.is_available() else "cpu"


def engine_worker() -> None:
    """Process-pool initializer for CPU fits: torch on :data:`FIT_THREADS` threads."""
    torch = import_fireants()
    torch.set_num_threads(FIT_THREADS)


@contextlib.contextmanager
def _deterministic(torch: Any, device: str) -> Iterator[None]:
    """Torch's deterministic algorithms, seed and (CPU) thread count for one
    fit; the process's own settings come back afterwards."""
    with _TORCH_LOCK:
        held = (torch.are_deterministic_algorithms_enabled(),
                torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                torch.get_num_threads())
        # Torch flags its grid sampler's backward as nondeterministic for the
        # image's gradient (atomic adds), which no fit asks for; the warp's
        # gradient is one sum per output pixel. warn_only keeps that op, and
        # the deformable tests check that identical inputs give identical
        # fields. cuDNN's grid sampler is off: its backward is not one.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        if device == "cpu":
            torch.set_num_threads(FIT_THREADS)
        torch.manual_seed(RANDOM_SEED)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*does not have a deterministic")
                yield
        finally:
            torch.use_deterministic_algorithms(held[0])
            torch.backends.cudnn.enabled = held[1]
            torch.backends.cudnn.benchmark = held[2]
            torch.set_num_threads(held[3])


def _channel_loss(
    torch: Any, terms: list[tuple[str, float]], kernel_px: int, *, balanced: bool = False,
) -> Any:
    """ANTs' multivariate metric on FireANTs' images: one metric per channel
    (``terms``: metric name and weight, in channel order), weights normalized,
    every term measured under one mask, the last channel of both images
    (fixed x warped moving). FireANTs' own losses apply one metric to every
    channel (label-map channels need mean squares beside the image pair's),
    and its masked local correlation sums each window over masked pixels
    too, so a blanked excluded region still pulled the tissue around it; here,
    as in ITK's neighbourhood correlation, a window counts only the pixels
    inside the mask.

    *balanced* (mean-squares terms only, the traced fit): ITK's multi-metric
    rule (``ObjectToObjectMultiMetricv4``), which ANTs applied: each term's
    gradient is scaled to unit length before its weight applies, so a small
    region's channel pulls as much as a large one and as the line channel. A
    plain sum lets the channels with the most contrast (large regions) drive
    the warp. The length is estimated at each evaluation from the warped
    images (the term's derivative times each image's spatial gradient, both
    halves) and held constant for that step."""
    from fireants.losses.cc import separable_filtering
    from fireants.losses.mi import GlobalMutualInformationLoss

    information = GlobalMutualInformationLoss(num_bins=MUTUAL_INFORMATION_BINS, masked=True)
    total_weight = float(sum(weight for _metric, weight in terms))

    def box(x: Any) -> Any:
        ones = torch.ones(kernel_px, dtype=x.dtype, device=x.device)
        return separable_filtering(x, [ones, ones])

    def correlation(p: Any, t: Any, mask: Any) -> Any:
        n = box(mask)
        valid = n > 0.5 * kernel_px
        n = n.clamp(min=1.0)
        sp, st = box(mask * p), box(mask * t)
        cross = box(mask * p * t) - sp * st / n
        var_p = (box(mask * p * p) - sp * sp / n).clamp(min=1e-5)
        var_t = (box(mask * t * t) - st * st / n).clamp(min=1e-5)
        ncc = (cross * cross + 1e-5) / (var_p * var_t + 1e-5)
        return -(ncc.clamp(max=1.0) * mask * valid).sum() / ((mask * valid).sum() + 1e-8)

    if balanced and any(metric != "mean_squares" for metric, _w in terms):
        raise ValueError("A balanced channel loss takes mean-squares terms only")
    weights = torch.tensor([weight / total_weight for _metric, weight in terms])

    class BalancedLoss(torch.nn.Module):
        def forward(self, pred: Any, target: Any) -> Any:
            mask = (pred[:, -1:] * target[:, -1:]).detach()
            area = mask.sum() + 1e-8
            n = len(terms)
            p, t = pred[:, :n], target[:, :n]
            values = ((p - t) ** 2 * mask).sum(dim=(0, 2, 3)) / area
            with torch.no_grad():
                d = 2.0 * (p - t) * mask / area
                length = torch.zeros(n, dtype=p.dtype, device=p.device)
                for image in (p, t):
                    gy, gx = torch.gradient(image, dim=(2, 3))
                    length += ((d * gx) ** 2 + (d * gy) ** 2).sum(dim=(0, 2, 3))
                length = length.sqrt()
                scale = torch.where(length > 1e-12, 1.0 / length.clamp(min=1e-12),
                                    torch.zeros_like(length))
            return (weights.to(p.device, p.dtype) * scale * values).sum()

    if balanced:
        return BalancedLoss()

    class ChannelLoss(torch.nn.Module):
        def forward(self, pred: Any, target: Any) -> Any:
            # The masks move with their images but carry no gradient, as in
            # ANTs: through them the warp would be rewarded for sliding an
            # excluded zone onto tissue that matches badly.
            mask = (pred[:, -1:] * target[:, -1:]).detach()
            area = mask.sum() + 1e-8
            loss = pred.new_zeros(())
            for k, (metric, weight) in enumerate(terms):
                p, t = pred[:, k:k + 1], target[:, k:k + 1]
                if metric == "local_correlation":
                    value = correlation(p, t, mask)
                elif metric == "mean_squares":
                    value = ((p - t) ** 2 * mask).sum() / area
                else:
                    value = information(torch.cat([p, mask], 1),
                                        torch.cat([t, torch.ones_like(mask)], 1))
                loss = loss + (weight / total_weight) * value
            return loss

    return ChannelLoss()


def _global_step_sgd() -> type:
    from fireants.registration.optimizers.sgd import WarpSGD

    class GlobalStepSGD(WarpSGD):
        """ANTs' gradient step: the gradient scaled so its largest vector is one,
        so FireANTs' SGD (one composition per step, ``compose_n`` 1) moves that
        pixel lr half-resolutions and every other pixel in proportion."""

        def step(self, loss: Any = None) -> None:
            grad = self.warp.grad
            grad.data.div_(grad.data.norm(p=2, dim=-1).max() + 1e-12)
            super().step(loss)

    return GlobalStepSGD


def _invert(torch: Any, displacement: Any) -> Any:
    """v with v(x) = -u(x + v(x)) for a (1, H, W, 2) normalized displacement u:
    damped fixed point, by sampling only. FireANTs' own inversion
    (``compositive_warp_inverse``) is an optimization whose gradient scatters
    into the field it samples, which differs run to run on the GPU."""
    import torch.nn.functional as F

    u = displacement.permute(0, 3, 1, 2)
    eye = torch.eye(2, 3, dtype=u.dtype, device=u.device)[None]
    base = F.affine_grid(eye, list(u.shape), align_corners=True)
    v = -displacement
    for _ in range(INVERSE_ITERATIONS):
        sampled = F.grid_sample(u, base + v, mode="bilinear", padding_mode="border",
                                align_corners=True).permute(0, 2, 3, 1)
        v = v + INVERSE_RELAXATION * (-sampled - v)
    return v


def _run_fireants(inputs: EngineInputs, settings: FitSettings, metric: Metric) -> EngineResult:
    torch = import_fireants()
    import SimpleITK as sitk
    from fireants.io.image import BatchedImages, Image
    from fireants.registration.syn import SyNRegistration
    from fireants.utils.util import compose_warp

    if metric not in ("local_correlation", "mutual_information", "mean_squares"):
        raise ValueError(f"FireANTs has no {metric!r} metric here")
    start = time.perf_counter()
    device = fit_device()
    level = DETAIL[settings.detail]
    stiffness = STIFFNESS[settings.stiffness]
    voxel_mm = float(np.mean(inputs.spacing_mm))
    traced = settings.section_image == "lines"
    learning_rate = TRACED_LEARNING_RATE if traced else FIREANTS_LEARNING_RATE
    total_sigma_mm = stiffness.total_sigma_mm * (TRACED_TOTAL_SIGMA_FACTOR if traced else 1.0)
    # FireANTs takes Gaussian sigmas in voxels of each pyramid level, as ANTs
    # took its variances.
    grad_sigma = stiffness.update_sigma_mm / voxel_mm
    warp_sigma = total_sigma_mm / voxel_mm
    radius = max(1, int(inputs.correlation_radius_px))
    edges = inputs.fixed_edges is not None and inputs.moving_edges is not None
    terms: list[tuple[str, float]] = [(metric, 1.0)]
    fixed_channels = [inputs.fixed]
    moving_channels = [inputs.moving]
    if edges:
        assert inputs.fixed_edges is not None and inputs.moving_edges is not None
        # The edge channel uses the intensity pair's metric.
        terms.append((metric, float(inputs.edge_weight)))
        fixed_channels.append(inputs.fixed_edges)
        moving_channels.append(inputs.moving_edges)
    # Each label pair is a mean-squares term on region indicators, the
    # construction of ants.label_image_registration.
    for f, m, w in zip(inputs.fixed_labels, inputs.moving_labels, inputs.label_weights,
                       strict=True):
        terms.append(("mean_squares", float(w)))
        fixed_channels.append(f)
        moving_channels.append(m)

    def image(channels: list[np.ndarray], mask: np.ndarray) -> Any:
        # The mask rides as the last channel (FireANTs' masked mode), so it is
        # resized, never blurred, with the pyramid and warped with the atlas.
        stack = np.stack([*channels, mask.astype(np.float32)], axis=-1).astype(np.float64)
        itk = sitk.GetImageFromArray(stack, isVector=True)
        itk.SetSpacing([float(v) for v in inputs.spacing_mm])
        itk.SetOrigin([float(v) for v in inputs.origin_mm])
        return BatchedImages([Image(itk, device=device, dtype=torch.float32)])

    levels = len(level.iterations)
    scales = [FIREANTS_SHRINK ** k for k in range(levels - 1, -1, -1)]
    height, width = inputs.fixed.shape
    with _deterministic(torch, device):
        fixed = image(fixed_channels, inputs.fixed_mask)
        moving = image(moving_channels, inputs.moving_mask)
        registration: Any = SyNRegistration(
            scales=scales, iterations=list(level.iterations),
            fixed_images=fixed, moving_images=moving,
            loss_type="masked_custom",
            custom_loss=_channel_loss(torch, terms, 2 * radius + 1, balanced=traced),
            optimizer="SGD", optimizer_lr=learning_rate,
            optimizer_params={"compose_n": 1},
            smooth_grad_sigma=grad_sigma, smooth_warp_sigma=warp_sigma,
            progress_bar=False, dtype=torch.float32,
        )
        # FireANTs builds its optimizers by name; the subclass changes only the step.
        for half in (registration.fwd_warp, registration.rev_warp):
            half.optimizer.__class__ = _global_step_sgd()
        registration.optimize()
        # Section -> atlas through the midpoint: rev^-1, then fwd; atlas ->
        # section: fwd^-1, then rev. Both images share one grid, so the
        # affine part is the identity.
        with torch.no_grad():
            forward_warp = registration.fwd_warp.get_warp().detach()
            reverse_warp = registration.rev_warp.get_warp().detach()
            forward = compose_warp(forward_warp, _invert(torch, reverse_warp))
            inverse = compose_warp(reverse_warp, _invert(torch, forward_warp))
        # Normalized [-1, 1] displacements (x, y) -> millimetres.
        to_mm = np.array([(width - 1) / 2.0 * inputs.spacing_mm[0],
                          (height - 1) / 2.0 * inputs.spacing_mm[1]], dtype=np.float64)
        forward_mm = forward[0].detach().cpu().double().numpy() * to_mm
        inverse_mm = inverse[0].detach().cpu().double().numpy() * to_mm
        del registration, fixed, moving
        if device == "cuda":
            torch.cuda.empty_cache()
    options: dict[str, Any] = {
        "metric": metric, "loss_terms": [list(t) for t in terms],
        "scales": scales, "iterations": list(level.iterations),
        "optimizer": "SGD, ANTs' step rule", "optimizer_lr": learning_rate,
        "smooth_grad_sigma_px": grad_sigma, "smooth_warp_sigma_px": warp_sigma,
        "update_sigma_mm": stiffness.update_sigma_mm, "total_sigma_mm": total_sigma_mm,
        "balanced_channels": traced,
        "edge_channel": edges,
        "edge_weight": float(inputs.edge_weight) if edges else 0.0,
        "label_channels": len(inputs.fixed_labels),
        "label_weights": [float(w) for w in inputs.label_weights],
        "device": device, "random_seed": RANDOM_SEED, "deterministic": True,
    }
    if metric == "local_correlation":
        options["cc_kernel_px"] = 2 * radius + 1
        options["correlation_radius_mm"] = radius * voxel_mm
    if device == "cpu":
        options["threads"] = FIT_THREADS
    return EngineResult(
        field_mm=np.ascontiguousarray(forward_mm, dtype=np.float32),
        inverse_field_mm=np.ascontiguousarray(inverse_mm, dtype=np.float32),
        inverse_source="engine",
        native_parameters={"SyNRegistration": options},
        runtime_s=time.perf_counter() - start,
        engine_version=f"fireants {FIREANTS_VERSION} (torch {torch.__version__}, {device})",
    )


#: Elastix's name for the affine's metric (:data:`~.settings.ELASTIX_STAIN_METRIC`).
ELASTIX_METRIC = "AdvancedMattesMutualInformation"


def _elastix_image(array: np.ndarray, inputs: EngineInputs, pixel: type = np.float32) -> Any:
    """An ITK image on the working grid (x = columns, y = rows, millimetres)."""
    import itk

    result = itk.image_from_array(np.ascontiguousarray(array, dtype=pixel))
    result.SetSpacing([float(v) for v in inputs.spacing_mm])
    result.SetOrigin([float(v) for v in inputs.origin_mm])
    return result


def _elastix_register(
    inputs: EngineInputs, parameter_object: Any, extra_pairs: list[tuple[Any, Any]],
) -> tuple[Any, Any]:
    """Run one Elastix registration of the intensity pair plus *extra_pairs*.

    Every pair shares the fixed and moving masks. Returns the moving image
    and the transform parameter object.
    """
    import itk

    fixed = _elastix_image(inputs.fixed, inputs)
    moving = _elastix_image(inputs.moving, inputs)
    fixed_mask = _elastix_image(inputs.fixed_mask.astype(np.uint8), inputs, np.uint8)
    moving_mask = _elastix_image(inputs.moving_mask.astype(np.uint8), inputs, np.uint8)
    # The method's own SetNumberOfThreads crashes this itk-elastix build
    # (segfault in the optimizer's parameter estimation); every component
    # takes ITK's global default when it is built, so that is set instead,
    # for this call only.
    threader = itk.MultiThreaderBase  # type: ignore[attr-defined]
    held_threads = threader.GetGlobalDefaultNumberOfThreads()
    threader.SetGlobalDefaultNumberOfThreads(FIT_THREADS)
    try:
        method = itk.ElastixRegistrationMethod.New(fixed, moving)  # type: ignore[attr-defined]
        method.SetFixedMask(fixed_mask)
        method.SetMovingMask(moving_mask)
        for extra_fixed, extra_moving in extra_pairs:
            method.AddFixedImage(extra_fixed)
            method.AddMovingImage(extra_moving)
            method.AddFixedMask(fixed_mask)
            method.AddMovingMask(moving_mask)
        method.SetParameterObject(parameter_object)
        method.SetLogToConsole(False)
        method.UpdateLargestPossibleRegion()
    finally:
        threader.SetGlobalDefaultNumberOfThreads(held_threads)
    return moving, method.GetTransformParameterObject()


def _parameter_maps(transform: Any) -> list[dict[str, list[str]]]:
    return [
        {key: list(transform.GetParameterMap(i)[key])
         for key in transform.GetParameterMap(i).keys()}
        for i in range(transform.GetNumberOfParameterMaps())
    ]


#: Pyramid levels and iterations per level of the Elastix affine refinement
#: (``elastix_affine``), which starts from a placement that is already close.
ELASTIX_AFFINE_RESOLUTIONS = 3
ELASTIX_AFFINE_ITERATIONS = 500


@dataclass
class AffineResult:
    """An affine on the working grid's millimetres, section -> placed atlas.

    ``matrix_mm`` (3x3) maps a section point (fixed) to the placed-atlas point
    it corresponds to (moving), the direction of a deformable field:
    ``atlas_point = matrix_mm @ section_point``.
    """

    matrix_mm: np.ndarray
    native_parameters: dict[str, Any]
    runtime_s: float
    engine_version: str


def run_elastix_affine(inputs: EngineInputs) -> AffineResult:
    """An Elastix affine (rotation, scales, shear, shift) starting at the identity.

    The identity is the placement the moving image was drawn at, so this is a
    local refinement of that placement, never a search from scratch
    (``AutomaticTransformInitialization`` off). The images, masks and edge
    channel are a deformable stain fit's; a fixed seed and thread count.
    Raises when Elastix
    fails or returns a non-finite or singular matrix.
    """
    import itk

    if inputs.fixed_labels:
        raise ValueError("Label-map channels are SyN-only")
    start = time.perf_counter()
    edges = inputs.fixed_edges is not None and inputs.moving_edges is not None
    metrics = 2 if edges else 1
    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]
    pm = parameter_object.GetDefaultParameterMap("affine")
    pm["Transform"] = ("AffineTransform",)
    pm["AutomaticTransformInitialization"] = ("false",)
    pm["AutomaticScalesEstimation"] = ("true",)
    pm["Metric"] = (ELASTIX_METRIC,) * metrics
    if edges:
        # Metric k reads image pair k; pyramids, interpolators and samplers
        # number one per metric (Elastix's multi-image rules).
        pm["Registration"] = ("MultiMetricMultiResolutionRegistration",)
        pm["Metric0Weight"] = ("1.0",)
        pm["Metric1Weight"] = (repr(float(inputs.edge_weight)),)
    for key in ("FixedImagePyramid", "MovingImagePyramid", "Interpolator"):
        pm[key] = (pm[key][0],) * metrics
    pm["ImageSampler"] = ("RandomCoordinate",) * metrics
    pm["NumberOfHistogramBins"] = (str(MUTUAL_INFORMATION_BINS),)
    pm["NumberOfResolutions"] = (str(ELASTIX_AFFINE_RESOLUTIONS),)
    pm["MaximumNumberOfIterations"] = (str(ELASTIX_AFFINE_ITERATIONS),)
    pm["NumberOfSpatialSamples"] = (str(ELASTIX_SPATIAL_SAMPLES),)
    pm["NewSamplesEveryIteration"] = ("true",)
    pm["RandomSeed"] = (str(RANDOM_SEED),)
    # Masks are already widened past the placed atlas footprint; eroding them at
    # coarse pyramid levels would cut away exactly that informative edge.
    pm["ErodeMask"] = ("false",)
    pm["ErodeFixedMask"] = ("false",)
    pm["ErodeMovingMask"] = ("false",)
    pm["WriteResultImage"] = ("false",)
    parameter_object.AddParameterMap(pm)
    requested = {key: list(value) for key, value in pm.items()}
    extra_pairs: list[tuple[Any, Any]] = []
    if edges:
        assert inputs.fixed_edges is not None and inputs.moving_edges is not None
        extra_pairs = [(_elastix_image(inputs.fixed_edges, inputs),
                        _elastix_image(inputs.moving_edges, inputs))]
    _moving, transform = _elastix_register(inputs, parameter_object, extra_pairs)
    maps = _parameter_maps(transform)
    final = maps[-1]
    try:
        values = [float(v) for v in final["TransformParameters"]]
        centre = [float(v) for v in final["CenterOfRotationPoint"]]
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Elastix returned no affine parameters") from exc
    if len(values) != 6 or len(centre) != 2:
        raise RuntimeError(f"Unexpected Elastix affine parameters: {values}, {centre}")
    # Elastix's affine: T(x) = A (x - c) + c + t, A row-major.
    linear = np.array(values[:4], dtype=np.float64).reshape(2, 2)
    pivot = np.array(centre, dtype=np.float64)
    shift = pivot + np.array(values[4:], dtype=np.float64) - linear @ pivot
    matrix = np.array([[*linear[0], shift[0]], [*linear[1], shift[1]], [0.0, 0.0, 1.0]])
    if not np.isfinite(matrix).all() or abs(np.linalg.det(linear)) < 1e-6:
        raise RuntimeError("Elastix returned a non-finite or singular affine")
    return AffineResult(
        matrix_mm=matrix,
        native_parameters={"requested": requested, "transform_parameter_maps": maps,
                           "metric": ELASTIX_STAIN_METRIC, "edge_channel": edges,
                           "threads": FIT_THREADS, "random_seed": RANDOM_SEED},
        runtime_s=time.perf_counter() - start,
        engine_version=f"itk-elastix (itk {itk.__version__})",
    )
