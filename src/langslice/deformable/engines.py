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
"""

from __future__ import annotations

import os
import tempfile
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
    metric_for,
)

#: Iterations of the fixed-point inversion used when an engine gives no inverse.
INVERSE_ITERATIONS = 30
#: Fixed step for ANTs SyN's gradient descent (ANTs' own default).
ANTS_GRADIENT_STEP = 0.2
#: Histogram bins for mutual information, both engines.
MUTUAL_INFORMATION_BINS = 32
#: Random samples per Elastix iteration.
ELASTIX_SPATIAL_SAMPLES = 4096


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
    metric = metric_for(settings.section_image, settings.atlas_image)
    if settings.engine == "ants":
        return _run_ants(inputs, settings, metric)
    return _run_elastix(inputs, settings, metric)


def _import_ants() -> Any:
    try:
        import ants
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The ANTs engine needs antspyx: install LangSlice's 'registration' extra "
            "(pip install 'langslice[registration]')"
        ) from exc
    return ants


def _run_ants(inputs: EngineInputs, settings: FitSettings, metric: Metric) -> EngineResult:
    ants = _import_ants()
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

    syn_metric = "mattes" if metric == "mutual_information" else "meansquares"
    # Each label pair is a mean-squares metric on region indicators, exactly
    # the construction ants.label_image_registration builds internally; it is
    # assembled here so the SyN regularization (stiffness) stays ours —
    # label_image_registration hard-codes SyN[0.2,3,0].
    extras = [
        ["MeanSquares", image(f), image(m), float(w), 0]
        for f, m, w in zip(inputs.fixed_labels, inputs.moving_labels, inputs.label_weights,
                           strict=True)
    ]
    options = {
        "type_of_transform": "SyNOnly", "initial_transform": "Identity",
        "syn_metric": syn_metric, "syn_sampling": MUTUAL_INFORMATION_BINS,
        "reg_iterations": tuple(level.ants_iterations), "grad_step": ANTS_GRADIENT_STEP,
        "flow_sigma": flow_variance, "total_sigma": total_variance,
        "mask_all_stages": True,
    }
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
        # The forward warp lives on the fixed (section) grid and is what ANTs
        # resamples the moving image through: section point -> atlas point.
        forward = ants.image_read(forward_files[0]).numpy().transpose(1, 0, 2)
        inverse = ants.image_read(inverse_files[0]).numpy().transpose(1, 0, 2)
    options["flow_sigma_mm"] = stiffness.update_sigma_mm
    options["total_sigma_mm"] = stiffness.total_sigma_mm
    options["reg_iterations"] = list(level.ants_iterations)
    options["label_channels"] = len(extras)
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
    )


def _elastix_parameters(
    settings: FitSettings, metric: Metric, spacing_mm: tuple[float, float],
) -> tuple[Any, dict[str, list[str]]]:
    import itk

    level = DETAIL[settings.detail]
    stiffness = ELASTIX_STIFFNESS[settings.stiffness]
    bending = stiffness.bending_weight
    if metric == "mean_squares":
        bending *= ELASTIX_MEAN_SQUARES_BENDING_SCALE
    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]
    pm = parameter_object.GetDefaultParameterMap("bspline")
    data_metric = ("AdvancedMattesMutualInformation" if metric == "mutual_information"
                   else "AdvancedMeanSquares")
    pm["Registration"] = ("MultiMetricMultiResolutionRegistration",)
    pm["Metric"] = (data_metric, "TransformBendingEnergyPenalty")
    pm["Metric0Weight"] = ("1.0",)
    pm["Metric1Weight"] = (repr(float(bending)),)
    pm["NumberOfHistogramBins"] = (str(MUTUAL_INFORMATION_BINS),)
    pm["FinalGridSpacingInPhysicalUnits"] = (repr(float(stiffness.grid_spacing_mm)),)
    pm["NumberOfResolutions"] = (str(level.elastix_resolutions),)
    # Control points double in spacing at each coarser level.
    pm["GridSpacingSchedule"] = tuple(
        str(2 ** k) for k in range(level.elastix_resolutions - 1, -1, -1)
    )
    pm["MaximumNumberOfIterations"] = (str(level.elastix_iterations),)
    pm["ImageSampler"] = ("RandomCoordinate",)
    pm["NumberOfSpatialSamples"] = (str(ELASTIX_SPATIAL_SAMPLES),)
    pm["NewSamplesEveryIteration"] = ("true",)
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
    parameter_object, requested = _elastix_parameters(settings, metric, inputs.spacing_mm)
    method = itk.ElastixRegistrationMethod.New(fixed, moving)  # type: ignore[attr-defined]
    method.SetFixedMask(image(inputs.fixed_mask.astype(np.uint8), np.uint8))
    method.SetMovingMask(image(inputs.moving_mask.astype(np.uint8), np.uint8))
    method.SetParameterObject(parameter_object)
    method.SetLogToConsole(False)
    method.UpdateLargestPossibleRegion()
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
    return EngineResult(
        field_mm=forward,
        inverse_field_mm=inverse,
        inverse_source="numerical_fixed_point",
        native_parameters={"requested": requested, "transform_parameter_maps": maps},
        runtime_s=time.perf_counter() - start,
        engine_version=f"itk-elastix (itk {itk.__version__})",
        notes=[
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
    ants = _import_ants()
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
