"""Executable contract checks for landmark-only Elastix and persisted geometry."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from langslice.core.landmark_elastix import (
    _affine,
    _bounds,
    _check_landmarks,
    _extent,
    _points,
    load_elastix,
)


def _corners(bounds):
    lo, hi = bounds
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                     [lo[0], hi[1]], [hi[0], hi[1]]])


def fit_landmark_elastix(source_mm, target_mm, extent_mm, affine_mm, domain_bounds_mm=None):
    """A landmark spline payload as checkpoints carry one: a bounded,
    regularized residual fitted to landmark pairs (bending weight scales
    with length cubed, so coordinate units do not change the tradeoff)."""
    import itk

    source, target = _points(source_mm), _points(target_mm)
    _check_landmarks(source, target)
    extent, affine = _extent(extent_mm), _affine(affine_mm)
    aligned = source @ affine[:, :2].T + affine[:, 2]
    original_corners = _corners(np.array([np.zeros(2), extent]))
    aligned_corners = original_corners @ affine[:, :2].T + affine[:, 2]
    coordinates = np.vstack((original_corners, aligned_corners, source, target, aligned))
    if domain_bounds_mm is not None:
        coordinates = np.vstack((coordinates, _bounds(domain_bounds_mm)))
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    scale = float(np.max(upper - lower))
    lower, upper = lower - scale * 0.15, upper + scale * 0.15
    bounds = np.array([lower, upper])
    image_spacing = scale / 96
    shape = np.ceil((upper - lower) / image_spacing).astype(int) + 1
    blank = itk.image_from_array(np.zeros((int(shape[1]), int(shape[0])), dtype=np.float32))
    blank.SetOrigin(tuple(lower))
    blank.SetSpacing((image_spacing, image_spacing))
    # ITK loads these wrapped extension classes dynamically.
    runtime = itk
    parameter_object = runtime.ParameterObject.New()
    parameters = parameter_object.GetDefaultParameterMap("bspline", 1, scale / 4)
    parameters.update({
        "Registration": ["MultiMetricMultiResolutionRegistration"],
        "Metric": ["TransformBendingEnergyPenalty", "CorrespondingPointsEuclideanDistanceMetric"],
        "Metric0Weight": [str(0.05 * scale**3)],
        "Metric1Weight": ["1"], "UseRelativeWeights": ["false"],
        "Optimizer": ["AdaptiveStochasticGradientDescent"],
        "MaximumNumberOfIterations": ["160"],
        "ImageSampler": ["Full"], "NewSamplesEveryIteration": ["false"],
        "WriteResultImage": ["false"], "UseDirectionCosines": ["true"],
        "PassiveEdgeWidth": ["1"],
    })
    if np.allclose(aligned, target, rtol=0, atol=1e-12):
        # Avoid estimating optimizer gains from an exactly zero residual.
        parameters["MaximumNumberOfIterations"] = ["0"]
    parameter_object.AddParameterMap(parameters)
    with TemporaryDirectory(prefix="langslice-landmark-elastix-") as directory:
        root = Path(directory)
        for name, points in (("fixed", target), ("moving", aligned)):
            rows = "\n".join(" ".join(format(float(v), ".17g") for v in row) for row in points)
            (root / f"{name}.txt").write_text(f"point\n{len(points)}\n{rows}\n")
        registration = runtime.ElastixRegistrationMethod.New(blank, blank)
        registration.SetParameterObject(parameter_object)
        registration.SetFixedPointSetFileName(str(root / "fixed.txt"))
        registration.SetMovingPointSetFileName(str(root / "moving.txt"))
        registration.SetOutputDirectory(directory)
        registration.SetLogToConsole(False)
        registration.SetNumberOfThreads(1)
        try:
            registration.UpdateLargestPossibleRegion()
        except RuntimeError as exc:
            raise ValueError("Elastix could not fit the anatomical landmarks") from exc
        result = registration.GetTransformParameterObject().GetParameterMap(0)
        saved = {str(key): [str(value) for value in values] for key, values in result.items()}
    saved["InitialTransformParameterFileName"] = ["NoInitialTransform"]
    payload = {
        "backend": "elastix", "source": (source / extent).tolist(),
        "target": (target / extent).tolist(), "extent_mm": extent.tolist(),
        "affine_mm": affine.tolist(), "domain_mm": bounds.tolist(),
        "parameter_maps": [saved],
    }
    transform = load_elastix(payload)
    residuals = np.linalg.norm(transform.forward(source) - target, axis=1)
    before = np.linalg.norm(aligned - target, axis=1)
    payload["diagnostics"] = {
        **transform.validate_domain(extent),
        "affine_landmark_rms_mm": float(np.sqrt(np.mean(before**2))),
        "landmark_rms_mm": float(np.sqrt(np.mean(residuals**2))),
        "landmark_max_error_mm": float(np.max(residuals)),
    }
    return payload


@pytest.fixture(scope="module")
def fitted_payload():
    source = np.array([[1, 1], [7, 1], [1, 6], [7, 6], [4, 3.5]], dtype=float)
    affine = np.array([[1.1, 0.15, 0.4], [-0.05, 0.9, -0.2]])
    target = source @ affine[:, :2].T + affine[:, 2]
    target[-1] += [0.35, 0.1]
    return fit_landmark_elastix(source, target, [8, 7], affine)


def test_fit_keeps_affine_and_improves_landmarks(fitted_payload):
    transform = load_elastix(fitted_payload)
    diagnostics = fitted_payload["diagnostics"]
    assert diagnostics["landmark_rms_mm"] < diagnostics["affine_landmark_rms_mm"] * 0.7
    assert diagnostics["minimum_residual_jacobian"] > 0
    assert 0.5 <= diagnostics["minimum_residual_stretch"]
    assert diagnostics["maximum_residual_stretch"] <= 2
    source = transform.source
    np.testing.assert_allclose(transform.inverse(transform.forward(source)), source, atol=2e-6)
    # The residual is identity outside its domain, but the affine still applies.
    remote = np.array([[-100.0, -100.0], [100.0, 100.0]])
    expected = remote @ transform.affine[:, :2].T + transform.affine[:, 2]
    np.testing.assert_allclose(transform.forward(remote), expected, atol=1e-10)
    assert not transform.is_identity()


def test_serialized_evaluator_matches_native_transformix(fitted_payload, tmp_path):
    import itk

    # Compare the saved spline directly against native Elastix evaluation at
    # irregular locations, valid-domain edges and outside the coefficient grid.
    transform = load_elastix(json.loads(json.dumps(fitted_payload)))
    rng = np.random.default_rng(401)
    probes = np.vstack((rng.uniform(-5, 15, size=(80, 2)), transform.target,
                        [[-50, -50], [50, 50]]))
    points = tmp_path / "points.txt"
    rows = "\n".join(" ".join(format(float(v), ".17g") for v in row) for row in probes)
    points.write_text(f"point\n{len(probes)}\n{rows}\n")
    parameters = itk.ParameterObject.New()
    parameters.AddParameterMap(fitted_payload["parameter_maps"][0])
    image = itk.image_from_array(np.zeros((8, 8), dtype=np.float32))
    native = itk.TransformixFilter.New(image)
    native.SetTransformParameterObject(parameters)
    native.SetFixedPointSetFileName(str(points))
    native.SetOutputDirectory(str(tmp_path))
    native.UpdateLargestPossibleRegion()
    lines = (tmp_path / "outputpoints.txt").read_text().splitlines()
    residual_native = np.array([
        [float(v) for v in re.search(r"OutputPoint = \[ ([^]]+) \]", line).group(1).split()]
        for line in lines
    ])
    expected = ((residual_native - transform.affine[:, 2])
                @ np.linalg.inv(transform.affine[:, :2]).T)
    np.testing.assert_allclose(transform.inverse(probes), expected, atol=1e-6)


def test_forward_jacobian_matches_finite_difference(fitted_payload):
    transform = load_elastix(fitted_payload)
    points = np.array([[2.3, 1.7], [4.1, 3.2], [6.2, 5.4]])
    step = 1e-4
    measured = np.stack([
        (transform.forward(points + np.eye(2)[axis] * step, tolerance_mm=1e-10)
         - transform.forward(points - np.eye(2)[axis] * step, tolerance_mm=1e-10)) / (2 * step)
        for axis in range(2)
    ], axis=-1)
    np.testing.assert_allclose(transform.jacobian(points), measured, atol=1e-5)


def test_affine_only_and_identity_are_not_refit():
    points = np.array([[1., 1.], [4., 1.], [1., 4.], [4., 4.]])
    payload = fit_landmark_elastix(points, points, [5, 5], np.eye(2, 3))
    assert load_elastix(payload).is_identity()
    affine = np.array([[1.1, 0.2, -0.3], [0.1, 0.9, 0.4]])
    payload = fit_landmark_elastix(points, points @ affine[:, :2].T + affine[:, 2],
                                  [5, 5], affine)
    transform = load_elastix(payload)
    assert not transform.is_identity()
    np.testing.assert_allclose(transform.forward(points), transform.target, atol=1e-10)


@pytest.mark.parametrize("change", [
    {"affine_mm": [[1, 0, 0], [0, 0, 0]]},
    {"extent_mm": [8, -1]},
    {"domain_mm": [[0, 0], [0, 8]]},
    {"source": [[0, 0], [0, 0], [1, 1], [1, 0]]},
    {"parameter_maps": []},
])
def test_rejects_malformed_payload(fitted_payload, change):
    payload = copy.deepcopy(fitted_payload)
    payload.update(change)
    with pytest.raises(ValueError):
        load_elastix(payload)


def test_rejects_external_reference_and_bad_coefficients(fitted_payload):
    payload = copy.deepcopy(fitted_payload)
    pm = payload["parameter_maps"][0]
    pm["InitialTransformParameterFileName"] = ["/tmp/another-transform.txt"]
    with pytest.raises(ValueError, match="external"):
        load_elastix(payload)
    pm["InitialTransformParameterFileName"] = ["NoInitialTransform"]
    pm["TransformParameters"][0] = "nan"
    with pytest.raises(ValueError, match="lattice"):
        load_elastix(payload)


def test_rejects_folded_residual(fitted_payload):
    payload = copy.deepcopy(fitted_payload)
    pm = payload["parameter_maps"][0]
    nx, ny = map(int, pm["GridSize"])
    # A large alternating displacement is a smooth spline but folds the domain.
    coefficient = np.zeros((2, ny, nx))
    coefficient[0, :, ::2] = 20
    pm["TransformParameters"] = [str(v) for v in coefficient.ravel()]
    with pytest.raises(ValueError, match="folds|compresses"):
        load_elastix(payload)
