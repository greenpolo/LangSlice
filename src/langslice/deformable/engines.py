"""ANTs (SyN) and Elastix (B-spline) adapters behind one call.

Both engines see the same thing: a fixed image (the section), a moving image
(the placed atlas, already on the section's working grid), a fixed and a
moving mask, and the physical spacing and origin of that grid. Both return a
displacement field in millimetres on the working grid that points from a
section point to the matching point of the placed atlas image — the
convention of brainglobe-registration and of both libraries' own resampling
fields: ``atlas_point = section_point + field(section_point)``.

No deformation solver lives here; the libraries do the fitting. Pure arrays
in, pure arrays out, so the adapters run unchanged in a worker process.

Determinism: identical inputs give identical fields. Neither engine samples
randomly with these settings except Elastix's random coordinate sampler, which
gets a fixed seed; what remains is the thread count, because a multithreaded
metric sums partial results in an order that depends on how the image is
split (ANTs fields differ by ~2 um between 1 and 8 threads, Elastix by
~1e-9 mm). Every fit therefore runs on :data:`FIT_THREADS` threads. Elastix
takes the number per call. ANTs (its own bundled ITK) reads the count from
the environment once per process, at its first use, so :func:`import_ants`
sets it around that first use and restores the environment afterwards (the
rest of the process keeps its own default); pool workers set it before
anything loads (:func:`ants_worker`). Only if something outside LangSlice
loaded ANTs first is the count unknown, and the record says so.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from langslice.deformable.settings import (
    ANTS_STIFFNESS,
    DETAIL,
    ELASTIX_MEAN_SQUARES_BENDING_SCALE,
    ELASTIX_STIFFNESS,
    FitSettings,
    Metric,
)

#: Iterations of the fixed-point inversion used when an engine gives no inverse.
INVERSE_ITERATIONS = 30
#: Fixed step for ANTs SyN's gradient descent (ANTs' own default).
ANTS_GRADIENT_STEP = 0.2
#: Histogram bins for mutual information, both engines.
MUTUAL_INFORMATION_BINS = 32
#: Random samples per Elastix iteration.
ELASTIX_SPATIAL_SAMPLES = 4096
#: Threads every engine call uses, whatever the machine or pool size: the
#: thread count changes a multithreaded fit's last bits (module docstring).
FIT_THREADS = 8
#: ITK's environment variable for its default thread count (read once).
ITK_THREADS_ENV = "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
#: antsRegistration's --random-seed and Elastix's RandomSeed.
RANDOM_SEED = 20261002


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
    #: their metric weights. ANTs only.
    fixed_labels: list[np.ndarray] = field(default_factory=list)
    moving_labels: list[np.ndarray] = field(default_factory=list)
    label_weights: list[float] = field(default_factory=list)
    #: Edge channel (stain fits): smoothed gradient magnitudes of both images
    #: and its metric weight; None without one.
    fixed_edges: np.ndarray | None = None
    moving_edges: np.ndarray | None = None
    edge_weight: float = 0.0
    #: ANTs local correlation window radius in working-grid pixels.
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


def run_engine(inputs: EngineInputs, settings: FitSettings) -> EngineResult:
    """Fit one candidate. Raises when the engine is not installed or fails."""
    if settings.engine == "ants":
        return _run_ants(inputs, settings, settings.metric)
    return _run_elastix(inputs, settings, settings.metric)


#: ANTs' thread count in this process: FIT_THREADS once :func:`import_ants`
#: or :func:`ants_worker` fixed it, None if ANTs was loaded by someone else.
_ANTS_THREADS: int | None = None
_ANTS_LOCK = threading.Lock()


def ants_worker() -> None:
    """Process-pool initializer: fix ITK's thread count before anything loads."""
    os.environ[ITK_THREADS_ENV] = str(FIT_THREADS)


def import_ants(purpose: str = "The ANTs engine") -> Any:
    """Import antspyx with its thread count fixed at :data:`FIT_THREADS`.

    The first import sets ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS``, runs one
    tiny filter so ANTs' ITK reads it, and puts the variable back.
    """
    global _ANTS_THREADS
    with _ANTS_LOCK:
        first = "ants" not in sys.modules
        held = os.environ.get(ITK_THREADS_ENV)
        if first:
            os.environ[ITK_THREADS_ENV] = str(FIT_THREADS)
        try:
            import ants

            if first:
                ants.smooth_image(ants.from_numpy(np.zeros((4, 4), np.float32)), 1.0)
                _ANTS_THREADS = FIT_THREADS
            elif _ANTS_THREADS is None and held == str(FIT_THREADS):
                _ANTS_THREADS = FIT_THREADS  # a pool worker (ants_worker)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                f"{purpose} needs antspyx: install LangSlice's 'registration' extra "
                "(pip install 'langslice[registration]')"
            ) from exc
        finally:
            if first:
                if held is None:
                    os.environ.pop(ITK_THREADS_ENV, None)
                else:
                    os.environ[ITK_THREADS_ENV] = held
    return ants


def _run_ants(inputs: EngineInputs, settings: FitSettings, metric: Metric) -> EngineResult:
    ants = import_ants()
    start = time.perf_counter()
    level = DETAIL[settings.detail]
    stiffness = ANTS_STIFFNESS[settings.stiffness]
    voxel_mm = float(np.mean(inputs.spacing_mm))
    # ANTs takes Gaussian *variances* in working-grid voxels.
    flow_variance = (stiffness.update_sigma_mm / voxel_mm) ** 2
    total_variance = (stiffness.total_sigma_mm / voxel_mm) ** 2

    def image(array: np.ndarray) -> Any:
        # ANTs arrays are x-first; transposing keeps x = columns, y = rows.
        return ants.from_numpy(
            np.ascontiguousarray(np.asarray(array, dtype=np.float32).T),
            origin=tuple(float(v) for v in inputs.origin_mm),
            spacing=tuple(float(v) for v in inputs.spacing_mm),
        )

    radius = max(1, int(inputs.correlation_radius_px))
    if metric == "local_correlation":
        # CC's parameter is the window radius in voxels of each pyramid level.
        syn_metric, sampling = "CC", radius
    elif metric == "mutual_information":
        syn_metric, sampling = "mattes", MUTUAL_INFORMATION_BINS
    elif metric == "mean_squares":
        syn_metric, sampling = "meansquares", 0
    else:
        raise ValueError(f"ANTs has no {metric!r} metric here")
    extras: list[list[Any]] = []
    if inputs.fixed_edges is not None and inputs.moving_edges is not None:
        # The edge channel uses the intensity pair's metric.
        extras.append([syn_metric, image(inputs.fixed_edges), image(inputs.moving_edges),
                       float(inputs.edge_weight), sampling])
    # Each label pair is a mean-squares metric on region indicators, exactly
    # the construction ants.label_image_registration builds internally; it is
    # assembled here so the SyN regularization (stiffness) stays ours —
    # label_image_registration hard-codes SyN[0.2,3,0].
    extras += [
        ["MeanSquares", image(f), image(m), float(w), 0]
        for f, m, w in zip(inputs.fixed_labels, inputs.moving_labels, inputs.label_weights,
                           strict=True)
    ]
    options: dict[str, Any] = {
        "type_of_transform": "SyNOnly", "initial_transform": "Identity",
        "syn_metric": syn_metric, "syn_sampling": sampling,
        "reg_iterations": tuple(level.ants_iterations), "grad_step": ANTS_GRADIENT_STEP,
        "flow_sigma": flow_variance, "total_sigma": total_variance,
        "mask_all_stages": True,
    }
    # antspyx passes --random-seed from this module-level value. Dense
    # sampling draws no random numbers; the seed pins whatever else might.
    held_seed = ants.config._random_seed
    ants.config._random_seed = RANDOM_SEED
    try:
        with tempfile.TemporaryDirectory(prefix="langslice-syn-") as directory:
            result = ants.registration(
                image(inputs.fixed), image(inputs.moving),
                outprefix=os.path.join(directory, "syn_"),
                mask=image(inputs.fixed_mask.astype(np.float32)),
                moving_mask=image(inputs.moving_mask.astype(np.float32)),
                multivariate_extras=extras or None, verbose=False, **options,
            )
            forward_files = [p for p in result["fwdtransforms"] if p.endswith(".nii.gz")]
            inverse_files = [p for p in result["invtransforms"] if p.endswith(".nii.gz")]
            affines = [p for p in result["fwdtransforms"] if p.endswith(".mat")]
            if len(forward_files) != 1 or len(inverse_files) != 1:
                raise RuntimeError(f"Unexpected SyN transform chain: {result['fwdtransforms']}")
            for path in affines:
                parameters = np.asarray(ants.read_transform(path).parameters, dtype=float)
                if not np.allclose(parameters, [1, 0, 0, 1, 0, 0], atol=1e-6):
                    raise RuntimeError("SyNOnly produced a non-identity affine stage")
            # The forward warp lives on the fixed (section) grid and is what
            # ANTs resamples the moving image through: section point -> atlas
            # point.
            forward = ants.image_read(forward_files[0]).numpy().transpose(1, 0, 2)
            inverse = ants.image_read(inverse_files[0]).numpy().transpose(1, 0, 2)
    finally:
        ants.config._random_seed = held_seed
    options["metric"] = metric
    if metric == "local_correlation":
        options["correlation_radius_mm"] = radius * voxel_mm
    options["edge_channel"] = inputs.fixed_edges is not None
    options["edge_weight"] = float(inputs.edge_weight) if inputs.fixed_edges is not None else 0.0
    options["random_seed"] = RANDOM_SEED
    options["threads"] = _ANTS_THREADS
    options["flow_sigma_mm"] = stiffness.update_sigma_mm
    options["total_sigma_mm"] = stiffness.total_sigma_mm
    options["reg_iterations"] = list(level.ants_iterations)
    options["label_channels"] = len(inputs.fixed_labels)
    options["label_weights"] = [float(w) for w in inputs.label_weights]
    return EngineResult(
        field_mm=np.ascontiguousarray(forward, dtype=np.float32),
        inverse_field_mm=np.ascontiguousarray(inverse, dtype=np.float32),
        inverse_source="engine",
        native_parameters={"antsRegistration": options,
                           "shrink_factors": [2 ** k for k in
                                              range(len(level.ants_iterations) - 1, -1, -1)]},
        runtime_s=time.perf_counter() - start,
        engine_version=f"antspyx {ants.__version__}",
        notes=[] if _ANTS_THREADS is not None else [
            "ANTs was loaded before LangSlice could fix its thread count; this field "
            "may differ in its last digits from the same fit run elsewhere"],
    )


#: Elastix's name for each metric it fits with.
ELASTIX_METRICS: dict[str, str] = {
    "mutual_information": "AdvancedMattesMutualInformation",
    "normalized_correlation": "AdvancedNormalizedCorrelation",
    "mean_squares": "AdvancedMeanSquares",
}


def _elastix_parameters(
    settings: FitSettings, metric: Metric, spacing_mm: tuple[float, float], *,
    edge_weight: float | None = None,
) -> tuple[Any, dict[str, list[str]]]:
    import itk

    level = DETAIL[settings.detail]
    stiffness = ELASTIX_STIFFNESS[settings.stiffness]
    bending = stiffness.bending_weight
    if metric == "mean_squares":
        bending *= ELASTIX_MEAN_SQUARES_BENDING_SCALE
    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]
    pm = parameter_object.GetDefaultParameterMap("bspline")
    if metric not in ELASTIX_METRICS:
        raise ValueError(f"Elastix has no {metric!r} metric")
    data_metric = ELASTIX_METRICS[metric]
    pm["Registration"] = ("MultiMetricMultiResolutionRegistration",)
    # Metric k reads image pair k; the edge pair (when present) is pair 1, and
    # the bending penalty, which reads no image, comes last.
    data = [data_metric] if edge_weight is None else [data_metric, data_metric]
    weights = [1.0] if edge_weight is None else [1.0, float(edge_weight)]
    pm["Metric"] = (*data, "TransformBendingEnergyPenalty")
    for k, weight in enumerate([*weights, float(bending)]):
        pm[f"Metric{k}Weight"] = (repr(weight),)
    pm["NumberOfHistogramBins"] = (str(MUTUAL_INFORMATION_BINS),)
    if edge_weight is not None:
        # Elastix's multi-image rules: pyramids, interpolators and samplers
        # (set below) number one or one per metric, the penalty included, and
        # at least one per image.
        for key in ("FixedImagePyramid", "MovingImagePyramid", "Interpolator"):
            pm[key] = tuple(pm[key]) * 3
    pm["FinalGridSpacingInPhysicalUnits"] = (repr(float(stiffness.grid_spacing_mm)),)
    pm["NumberOfResolutions"] = (str(level.elastix_resolutions),)
    # Control points double in spacing at each coarser level.
    pm["GridSpacingSchedule"] = tuple(
        str(2 ** k) for k in range(level.elastix_resolutions - 1, -1, -1)
    )
    pm["MaximumNumberOfIterations"] = (str(level.elastix_iterations),)
    pm["ImageSampler"] = ("RandomCoordinate",) * (1 if edge_weight is None else 3)
    pm["NumberOfSpatialSamples"] = (str(ELASTIX_SPATIAL_SAMPLES),)
    pm["NewSamplesEveryIteration"] = ("true",)
    pm["RandomSeed"] = (str(RANDOM_SEED),)
    # Masks are already widened past the tissue outline; eroding them at
    # coarse pyramid levels would cut away exactly that informative edge.
    pm["ErodeMask"] = ("false",)
    pm["ErodeFixedMask"] = ("false",)
    pm["ErodeMovingMask"] = ("false",)
    pm["FinalBSplineInterpolationOrder"] = ("1",)
    pm["WriteResultImage"] = ("false",)
    parameter_object.AddParameterMap(pm)
    return parameter_object, {key: list(value) for key, value in pm.items()}


def _run_elastix(inputs: EngineInputs, settings: FitSettings, metric: Metric) -> EngineResult:
    import itk

    if inputs.fixed_labels:
        raise ValueError("Label-map channels are ANTs-only")

    start = time.perf_counter()

    def image(array: np.ndarray, pixel: type = np.float32) -> Any:
        result = itk.image_from_array(np.ascontiguousarray(array, dtype=pixel))
        result.SetSpacing([float(v) for v in inputs.spacing_mm])
        result.SetOrigin([float(v) for v in inputs.origin_mm])
        return result

    fixed = image(inputs.fixed)
    moving = image(inputs.moving)
    edges = inputs.fixed_edges is not None and inputs.moving_edges is not None
    parameter_object, requested = _elastix_parameters(
        settings, metric, inputs.spacing_mm,
        edge_weight=inputs.edge_weight if edges else None)
    fixed_mask = image(inputs.fixed_mask.astype(np.uint8), np.uint8)
    moving_mask = image(inputs.moving_mask.astype(np.uint8), np.uint8)
    pairs = [(fixed, moving)]
    if edges:
        assert inputs.fixed_edges is not None and inputs.moving_edges is not None
        # The bending penalty (metric 2) reads no image, but with several
        # pairs Elastix wants one per metric: the intensity pair again.
        pairs += [(image(inputs.fixed_edges), image(inputs.moving_edges)), (fixed, moving)]
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
        for extra_fixed, extra_moving in pairs[1:]:
            method.AddFixedImage(extra_fixed)
            method.AddMovingImage(extra_moving)
            method.AddFixedMask(fixed_mask)
            method.AddMovingMask(moving_mask)
        method.SetParameterObject(parameter_object)
        method.SetLogToConsole(False)
        method.UpdateLargestPossibleRegion()
    finally:
        threader.SetGlobalDefaultNumberOfThreads(held_threads)
    transform = method.GetTransformParameterObject()
    # Transformix writes deformationField.nii to its output directory, the
    # process's working directory by default (the user's folder, shared by
    # every pool worker); point it at a private scratch directory instead.
    with tempfile.TemporaryDirectory(prefix="langslice-transformix-") as scratch:
        deformation = itk.transformix_deformation_field(  # type: ignore[attr-defined]
            moving, transform, output_directory=scratch)
        forward = np.array(itk.array_from_image(deformation), dtype=np.float32)
    maps = [
        {key: list(transform.GetParameterMap(i)[key])
         for key in transform.GetParameterMap(i).keys()}
        for i in range(transform.GetNumberOfParameterMaps())
    ]
    inverse, residual = invert_field(forward, inputs.spacing_mm)
    notes = []
    if settings.section_image == "stain" and settings.stain_metric == "local_correlation":
        notes.append(f"Elastix has no local correlation metric; this stain fit used "
                     f"{ELASTIX_METRICS[metric]} instead")
    return EngineResult(
        field_mm=forward,
        inverse_field_mm=inverse,
        inverse_source="numerical_fixed_point",
        native_parameters={"requested": requested, "transform_parameter_maps": maps,
                           "metric": metric, "edge_channel": edges,
                           "threads": FIT_THREADS},
        runtime_s=time.perf_counter() - start,
        engine_version=f"itk-elastix (itk {itk.__version__})",
        notes=[
            *notes,
            "Elastix B-splines have no closed-form inverse; the inverse field was "
            "approximated by fixed-point iteration on the working grid "
            f"(max residual {residual:.4g} mm over the grid)",
        ],
    )


def invert_field(
    field_mm: np.ndarray, spacing_mm: tuple[float, float], iterations: int = INVERSE_ITERATIONS,
) -> tuple[np.ndarray, float]:
    """Approximate the inverse of a displacement field by fixed-point iteration.

    Solves ``v(q) = -u(q + v(q))`` on the same grid. Returns the inverse and
    the largest remaining ``|u(q + v(q)) + v(q)|`` in millimetres; where the
    forward map folds there is no inverse and that residual stays large.
    """
    height, width = field_mm.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    sx, sy = spacing_mm
    inverse = -field_mm.astype(np.float64)
    residual = np.zeros((height, width))
    for _ in range(iterations):
        coordinates = [yy + inverse[..., 1] / sy, xx + inverse[..., 0] / sx]
        sampled = np.stack([
            ndi.map_coordinates(field_mm[..., k], coordinates, order=1, mode="nearest")
            for k in range(2)
        ], axis=-1)
        residual = np.linalg.norm(sampled + inverse, axis=-1)
        inverse = -sampled
    return inverse.astype(np.float32), float(residual.max())


def ants_preprocess(
    image: np.ndarray, mask: np.ndarray, steps: tuple[str, ...],
    spacing_mm: tuple[float, float],
) -> np.ndarray:
    """N4 bias-field correction and/or ANTs denoising of a section image.

    Runs in the listed order on the working grid, restricted to *mask*.
    Intensities are shifted positive first (N4 works on their logarithm).
    """
    if not steps:
        return image
    ants = import_ants()
    array = np.asarray(image, dtype=np.float32)
    array = array - float(array.min()) + 1.0
    current = ants.from_numpy(np.ascontiguousarray(array.T), spacing=tuple(spacing_mm))
    ants_mask = ants.from_numpy(
        np.ascontiguousarray(mask.astype(np.float32).T), spacing=tuple(spacing_mm),
    )
    for step in steps:
        if step == "n4":
            current = ants.n4_bias_field_correction(current, mask=ants_mask)
        elif step == "denoise":
            current = ants.denoise_image(current, mask=ants_mask)
        else:
            raise ValueError(f"Unknown preprocessing step: {step!r}")
    return np.ascontiguousarray(current.numpy().T, dtype=np.float32)
