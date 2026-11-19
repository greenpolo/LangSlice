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
import importlib
import importlib.metadata
import inspect
import io
import logging
import os
import sys
import threading
import time
import types
import warnings
from collections.abc import Callable, Generator, Iterator
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from itk import Image, ParameterObject

from langslice.core.fits.deformable.gpu_queue import (
    OOM_RETRIES,
    OOM_WAIT_S,
    gpu_turn,
    on_gpu,
    out_of_memory,
)
from langslice.core.fits.deformable.settings import (
    DETAIL,
    ELASTIX_STAIN_METRIC,
    TOTAL_SIGMA_FRACTION,
    FitSettings,
    Metric,
)
from langslice.core.hardware import (
    cpu_cores,
    drop_process_pool,
    global_threads,
    kept_process_pool,
)
from langslice.core.outside import untyped

logger = logging.getLogger(__name__)

MUTUAL_INFORMATION_BINS = 32
ELASTIX_SPATIAL_SAMPLES = 4096
FIT_THREADS = 8
RANDOM_SEED = 20261002
FIREANTS_LEARNING_RATE = 0.8
TRACED_LEARNING_RATE = 0.4
TRACED_TOTAL_SIGMA_FACTOR = 2.0
FIREANTS_SHRINK = 2
INVERSE_ITERATIONS = 200
INVERSE_RELAXATION = 0.5
FIREANTS_VERSION = importlib.metadata.version("fireants")
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
    fixed_labels: list[np.ndarray] = field(default_factory=list)
    moving_labels: list[np.ndarray] = field(default_factory=list)
    label_weights: list[float] = field(default_factory=list)
    fixed_edges: np.ndarray | None = None
    moving_edges: np.ndarray | None = None
    edge_weight: float = 0.0
    correlation_radius_px: int = 4
    brain_scale: float = 1.0


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
    left_out: np.ndarray | None = None
    gpu_wait_s: float = 0.0
    oom_retries: int = 0


def run_engine(inputs: EngineInputs, settings: FitSettings) -> EngineResult:
    """Fit one deformation with FireANTs' SyN. Raises when FireANTs is not
    installed or fails, and for any other engine (Elastix fits affines only:
    :func:`run_elastix_affine`)."""
    if settings.engine != "fireants":
        raise ValueError(
            f"No deformable fit with {settings.engine!r}: the deformable engine is SyN (FireANTs)"
        )
    return _run_fireants(inputs, settings, settings.metric)


_TORCH_LOCK = threading.Lock()


def _no_lock(*_args: object) -> None:
    """Take no lock: what the stand-in ``fcntl`` gives for ``flock``."""


@contextlib.contextmanager
def _windows_fcntl() -> Iterator[None]:
    """On Windows, a stand-in ``fcntl`` for the length of FireANTs' import:
    FireANTs (1.5.0) imports the Unix-only module at the top of
    ``fireants.io.image`` for a file lock only its multi-GPU training takes,
    which LangSlice never runs. Removed afterwards, so no other library takes
    the stand-in for a working lock."""
    if sys.platform != "win32" or "fcntl" in sys.modules:
        yield
        return
    stand_in = types.ModuleType("fcntl")
    vars(stand_in).update(LOCK_EX=2, LOCK_UN=8, flock=_no_lock)
    sys.modules["fcntl"] = stand_in
    try:
        yield
    finally:
        del sys.modules["fcntl"]


_IMPORTING = threading.Lock()


def import_fireants() -> Any:
    """Import torch and FireANTs' SyN (``RuntimeError`` with the install hint).

    Sets ``CUBLAS_WORKSPACE_CONFIG`` first when nothing set it, so the
    deterministic algorithms can run on the GPU.
    """
    os.environ.setdefault(*_CUBLAS_ENV)
    _IMPORTING.acquire()
    try:
        import torch

        for name in (
            "fireants",
            "fireants.registration.optimizers.adam",
            "fireants.interpolator",
            "fireants.registration.abstract",
            "fireants.io.image",
        ):
            logging.getLogger(name).setLevel(logging.ERROR)
        with contextlib.redirect_stdout(io.StringIO()), _windows_fcntl():
            importlib.import_module("fireants.registration.syn")
    except ImportError as exc:
        raise RuntimeError(
            "The deformable engine needs FireANTs and torch, LangSlice dependencies "
            "(reinstall LangSlice)"
        ) from exc
    finally:
        _IMPORTING.release()
    return torch


def fit_device() -> str:
    """``cuda`` when torch sees a GPU, else ``cpu``."""
    torch = import_fireants()
    return "cuda" if torch.cuda.is_available() else "cpu"


def engine_worker() -> None:
    """Process-pool initializer for CPU fits: torch on :data:`FIT_THREADS` threads."""
    torch = import_fireants()
    torch.set_num_threads(FIT_THREADS)


@dataclass
class _TorchSettings:
    """Count this process's running fits; keep torch's settings from before.

    :func:`_deterministic` puts them back after the last fit.
    """

    running: int = 0
    saved: tuple[bool, bool, bool, int] = (False, True, False, 1)


_TORCH_SETTINGS = _TorchSettings()


@contextlib.contextmanager
def _deterministic(device: str) -> Generator[None, None, None]:
    """Hold torch's deterministic settings while any fit of this process runs.

    Several fits run at once on threads: the first sets the deterministic
    algorithms, the seed and (CPU) thread count, the last puts the process's
    own back. No fit draws a random number, so the fits beside it never
    change its result.
    """
    import torch

    settings = _TORCH_SETTINGS
    with _TORCH_LOCK:
        if settings.running == 0:
            settings.saved = (
                torch.are_deterministic_algorithms_enabled(),
                torch.backends.cudnn.enabled,
                torch.backends.cudnn.benchmark,
                torch.get_num_threads(),
            )
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            if device == "cpu":
                torch.set_num_threads(FIT_THREADS)
            seed = cast("Callable[[int], object]", untyped("torch", "manual_seed"))
            seed(RANDOM_SEED)
        settings.running += 1
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have a deterministic")
            yield
    finally:
        with _TORCH_LOCK:
            settings.running -= 1
            if settings.running == 0:
                enabled, cudnn, benchmark, threads = settings.saved
                torch.use_deterministic_algorithms(enabled)
                torch.backends.cudnn.enabled = cudnn
                torch.backends.cudnn.benchmark = benchmark
                torch.set_num_threads(threads)


@contextlib.contextmanager
def _own_stream(device: str) -> Generator[None, None, None]:
    """Run a GPU fit on a CUDA stream of its own.

    Fits on several threads then overlap on the GPU; on the shared default
    stream they would queue behind each other.
    """
    if not on_gpu(device):
        yield
        return
    import torch

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        yield
    stream.synchronize()


def _channel_loss(
    torch: Any,
    terms: list[tuple[str, float]],
    kernel_px: int,
    *,
    balanced: bool = False,
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
                scale = torch.where(
                    length > 1e-12, 1.0 / length.clamp(min=1e-12), torch.zeros_like(length)
                )
            return (weights.to(p.device, p.dtype) * scale * values).sum()

    if balanced:
        return BalancedLoss()

    class ChannelLoss(torch.nn.Module):
        def forward(self, pred: Any, target: Any) -> Any:
            mask = (pred[:, -1:] * target[:, -1:]).detach()
            area = mask.sum() + 1e-8
            loss = pred.new_zeros(())
            for k, (metric, weight) in enumerate(terms):
                p, t = pred[:, k : k + 1], target[:, k : k + 1]
                if metric == "local_correlation":
                    value = correlation(p, t, mask)
                elif metric == "mean_squares":
                    value = ((p - t) ** 2 * mask).sum() / area
                else:
                    value = information(
                        torch.cat([p, mask], 1), torch.cat([t, torch.ones_like(mask)], 1)
                    )
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


def _sgd_options() -> dict[str, int]:
    """Return FireANTs' SGD options for one composition per step.

    FireANTs 1.5.0's SGD composes each step once and takes no
    ``compose_n``; a newer FireANTs takes ``compose_n`` and composes ten
    times by default.
    """
    if "compose_n" in inspect.signature(_global_step_sgd()).parameters:
        return {"compose_n": 1}
    return {}


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
        sampled = F.grid_sample(
            u, base + v, mode="bilinear", padding_mode="border", align_corners=True
        ).permute(0, 2, 3, 1)
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
    voxel_mm = float(np.mean(inputs.spacing_mm))
    traced = settings.section_image == "lines"
    learning_rate = TRACED_LEARNING_RATE if traced else FIREANTS_LEARNING_RATE
    scale = float(inputs.brain_scale)
    update_sigma_mm = settings.smoothness_um / 1000.0 * scale
    total_sigma_mm = (
        update_sigma_mm * TOTAL_SIGMA_FRACTION * (TRACED_TOTAL_SIGMA_FACTOR if traced else 1.0)
    )
    grad_sigma = update_sigma_mm / voxel_mm
    warp_sigma = total_sigma_mm / voxel_mm
    radius = max(1, int(inputs.correlation_radius_px))
    edges = inputs.fixed_edges is not None and inputs.moving_edges is not None
    terms: list[tuple[str, float]] = [(metric, 1.0)]
    fixed_channels = [inputs.fixed]
    moving_channels = [inputs.moving]
    if edges:
        assert inputs.fixed_edges is not None and inputs.moving_edges is not None
        terms.append((metric, float(inputs.edge_weight)))
        fixed_channels.append(inputs.fixed_edges)
        moving_channels.append(inputs.moving_edges)
    for f, m, w in zip(
        inputs.fixed_labels, inputs.moving_labels, inputs.label_weights, strict=True
    ):
        terms.append(("mean_squares", float(w)))
        fixed_channels.append(f)
        moving_channels.append(m)

    def image(channels: list[np.ndarray], mask: np.ndarray) -> Any:
        stack = np.stack([*channels, mask.astype(np.float32)], axis=-1).astype(np.float64)
        itk = sitk.GetImageFromArray(stack, isVector=True)
        itk.SetSpacing([float(v) for v in inputs.spacing_mm])
        itk.SetOrigin([float(v) for v in inputs.origin_mm])
        return BatchedImages([Image(itk, device=device, dtype=torch.float32)])

    levels = len(level.iterations)
    scales = [FIREANTS_SHRINK**k for k in range(levels - 1, -1, -1)]
    height, width = inputs.fixed.shape
    waited_s, retries = 0.0, 0
    for attempt in range(OOM_RETRIES + 1):
        try:
            with (
                gpu_turn(device) as turn,
                _deterministic(device),
                _own_stream(device),
            ):
                waited_s += turn.waited_s
                fixed = image(fixed_channels, inputs.fixed_mask)
                moving = image(moving_channels, inputs.moving_mask)
                registration: Any = SyNRegistration(
                    scales=scales,
                    iterations=list(level.iterations),
                    fixed_images=fixed,
                    moving_images=moving,
                    loss_type="masked_custom",
                    custom_loss=_channel_loss(torch, terms, 2 * radius + 1, balanced=traced),
                    optimizer="SGD",
                    optimizer_lr=learning_rate,
                    optimizer_params=_sgd_options(),
                    smooth_grad_sigma=grad_sigma,
                    smooth_warp_sigma=warp_sigma,
                    progress_bar=False,
                    dtype=torch.float32,
                )
                for half in (registration.fwd_warp, registration.rev_warp):
                    half.optimizer.__class__ = _global_step_sgd()
                registration.optimize()
                with torch.no_grad():
                    forward_warp = registration.fwd_warp.get_warp().detach()
                    reverse_warp = registration.rev_warp.get_warp().detach()
                    forward = compose_warp(forward_warp, _invert(torch, reverse_warp))
                    inverse = compose_warp(reverse_warp, _invert(torch, forward_warp))
                to_mm = np.array(
                    [
                        (width - 1) / 2.0 * inputs.spacing_mm[0],
                        (height - 1) / 2.0 * inputs.spacing_mm[1],
                    ],
                    dtype=np.float64,
                )
                forward_mm = forward[0].detach().cpu().double().numpy() * to_mm
                inverse_mm = inverse[0].detach().cpu().double().numpy() * to_mm
                del registration, fixed, moving
                if device == "cuda":
                    torch.cuda.empty_cache()
            break
        except Exception as exc:
            if device != "cuda" or attempt == OOM_RETRIES or not out_of_memory(exc):
                raise
            torch.cuda.empty_cache()
            retries += 1
            time.sleep(OOM_WAIT_S * (attempt + 1))
    else:
        raise RuntimeError("the fit ran out of GPU memory on every attempt")
    options: dict[str, Any] = {
        "metric": metric,
        "loss_terms": [list(t) for t in terms],
        "scales": scales,
        "iterations": list(level.iterations),
        "optimizer": "SGD, ANTs' step rule",
        "optimizer_lr": learning_rate,
        "smooth_grad_sigma_px": grad_sigma,
        "smooth_warp_sigma_px": warp_sigma,
        "update_sigma_mm": update_sigma_mm,
        "total_sigma_mm": total_sigma_mm,
        "brain_scale": scale,
        "balanced_channels": traced,
        "edge_channel": edges,
        "edge_weight": float(inputs.edge_weight) if edges else 0.0,
        "label_channels": len(inputs.fixed_labels),
        "label_weights": [float(w) for w in inputs.label_weights],
        "device": device,
        "random_seed": RANDOM_SEED,
        "deterministic": True,
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
        gpu_wait_s=waited_s,
        oom_retries=retries,
    )


ELASTIX_METRIC = "AdvancedMattesMutualInformation"


def _elastix_image(
    array: np.ndarray,
    inputs: EngineInputs,
    pixel: type[np.generic] = np.float32,
) -> Image:
    """An ITK image on the working grid (x = columns, y = rows, millimetres)."""
    import itk

    result = itk.image_from_array(np.ascontiguousarray(array, dtype=pixel))
    result.SetSpacing([float(v) for v in inputs.spacing_mm])
    result.SetOrigin([float(v) for v in inputs.origin_mm])
    return result


def _elastix_register(
    inputs: EngineInputs,
    parameter_object: ParameterObject,
    extra_pairs: list[tuple[Image, Image]],
) -> tuple[Image, ParameterObject]:
    """Run one Elastix registration of the intensity pair plus *extra_pairs*.

    Every pair shares the fixed and moving masks. Returns the moving image
    and the transform parameter object.
    """
    import itk

    fixed = _elastix_image(inputs.fixed, inputs)
    moving = _elastix_image(inputs.moving, inputs)
    fixed_mask = _elastix_image(inputs.fixed_mask.astype(np.uint8), inputs, np.uint8)
    moving_mask = _elastix_image(inputs.moving_mask.astype(np.uint8), inputs, np.uint8)
    with global_threads(itk.MultiThreaderBase, FIT_THREADS):
        method = itk.ElastixRegistrationMethod.New(fixed, moving)
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
    return moving, method.GetTransformParameterObject()


def _parameter_maps(transform: ParameterObject) -> list[dict[str, list[str]]]:
    return [
        {
            key: list(transform.GetParameterMap(i)[key])
            for key in transform.GetParameterMap(i).keys()
        }
        for i in range(transform.GetNumberOfParameterMaps())
    ]


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


ELASTIX_PROCESSES = True
ELASTIX_POOL = "elastix"


def elastix_in_workers() -> bool:
    """Say whether Elastix fits run in worker processes.

    Returns
    -------
    bool
        True when :data:`ELASTIX_PROCESSES` is on and the cores hold more
        than one worker of :data:`FIT_THREADS` threads.
    """
    return ELASTIX_PROCESSES and cpu_cores() // FIT_THREADS > 1


def fit_elastix_affine(inputs: EngineInputs) -> AffineResult:
    """Run :func:`run_elastix_affine` in a kept Elastix worker process.

    Several sections fit at once from threads, each in a worker that loaded
    ITK once; without workers (one would do, or :data:`ELASTIX_PROCESSES` is
    off) the fit runs here, and so does a fit whose worker died.

    Parameters
    ----------
    inputs
        The prepared fit's engine inputs.

    Returns
    -------
    AffineResult
        :func:`run_elastix_affine`'s result.
    """
    pool = None
    if elastix_in_workers():
        pool = kept_process_pool(ELASTIX_POOL, threads_each=FIT_THREADS)
    if pool is None:
        return run_elastix_affine(inputs)
    try:
        return pool.submit(run_elastix_affine, inputs).result()
    except BrokenProcessPool as exc:
        logger.warning("an Elastix worker failed (%s): fitting here", exc)
        drop_process_pool(ELASTIX_POOL, pool)
        return run_elastix_affine(inputs)


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
    parameter_object = itk.ParameterObject.New()
    pm = parameter_object.GetDefaultParameterMap("affine")
    pm["Transform"] = ("AffineTransform",)
    pm["AutomaticTransformInitialization"] = ("false",)
    pm["AutomaticScalesEstimation"] = ("true",)
    pm["Metric"] = (ELASTIX_METRIC,) * metrics
    if edges:
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
    pm["ErodeMask"] = ("false",)
    pm["ErodeFixedMask"] = ("false",)
    pm["ErodeMovingMask"] = ("false",)
    pm["WriteResultImage"] = ("false",)
    parameter_object.AddParameterMap(pm)
    requested = {key: list(value) for key, value in pm.items()}
    extra_pairs: list[tuple[Image, Image]] = []
    if edges:
        assert inputs.fixed_edges is not None and inputs.moving_edges is not None
        extra_pairs = [
            (
                _elastix_image(inputs.fixed_edges, inputs),
                _elastix_image(inputs.moving_edges, inputs),
            )
        ]
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
    linear = np.array(values[:4], dtype=np.float64).reshape(2, 2)
    pivot = np.array(centre, dtype=np.float64)
    shift = pivot + np.array(values[4:], dtype=np.float64) - linear @ pivot
    matrix = np.array([[*linear[0], shift[0]], [*linear[1], shift[1]], [0.0, 0.0, 1.0]])
    if not np.isfinite(matrix).all() or abs(np.linalg.det(linear)) < 1e-6:
        raise RuntimeError("Elastix returned a non-finite or singular affine")
    return AffineResult(
        matrix_mm=matrix,
        native_parameters={
            "requested": requested,
            "transform_parameter_maps": maps,
            "metric": ELASTIX_STAIN_METRIC,
            "edge_channel": edges,
            "threads": FIT_THREADS,
            "random_seed": RANDOM_SEED,
        },
        runtime_s=time.perf_counter() - start,
        engine_version=f"itk-elastix (itk {itk.__version__})",
    )
