"""Geometry and rejection checks for the shared BigWarp-compatible TPS."""
import numpy as np
import pytest

from langslice.landmark_warp import ThinPlateSpline, fit_spline, warp_section

POINTS = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [.5, .5]], dtype=float)


def spec(target=None):
    return {"source": POINTS.tolist(),
            "target": (POINTS if target is None else target).tolist(), "extent_mm": [8., 5.]}


def test_identity_and_resampling():
    transform = fit_spline(spec())
    points = np.array([[.1, 3.2], [5, 2], [8, 5]])
    np.testing.assert_allclose(transform.forward(points), points, atol=1e-12)
    np.testing.assert_allclose(transform.inverse(points), points, atol=1e-12)
    section = np.arange(16 * 10 * 3, dtype=np.uint8).reshape(10, 16, 3)
    warped = warp_section(section, spec(), (20, 14), (2, 2), 500)
    np.testing.assert_array_equal(warped[2:12, 2:18], section)
    assert np.count_nonzero(warped[:2]) == 0


def test_affine_shear_translation_and_physical_anisotropy():
    matrix = np.array([[1.1, .18], [-.08, .9]])
    shift = np.array([.3, -.2])
    source = POINTS * [8., 5.]
    target = (source @ matrix.T + shift) / [8., 5.]
    transform = fit_spline(spec(target))
    probes = np.random.default_rng(4).uniform([0, 0], [8, 5], (100, 2))
    expected = probes @ matrix.T + shift
    np.testing.assert_allclose(transform.forward(probes), expected, atol=1e-11)
    np.testing.assert_allclose(
        transform.jacobian(probes), np.broadcast_to(matrix, (100, 2, 2)), atol=1e-11)
    np.testing.assert_allclose(transform.inverse(expected), probes, atol=1e-8)


def test_nonlinear_landmarks_inverse_and_analytic_jacobian():
    target = POINTS.copy()
    target[-1] += [.05, -.04]
    transform = fit_spline(spec(target))
    np.testing.assert_allclose(transform.forward(POINTS * [8, 5]), target * [8, 5], atol=2e-6)
    probes = np.random.default_rng(7).uniform([0, 0], [8, 5], (120, 2))
    np.testing.assert_allclose(transform.inverse(transform.forward(probes)), probes, atol=2e-6)
    numerical = np.stack([(transform.forward(probes + np.eye(2)[i] * 1e-5, tolerance_mm=1e-11)
                           - transform.forward(probes - np.eye(2)[i] * 1e-5, tolerance_mm=1e-11))
                          / 2e-5
                          for i in range(2)], axis=-1)
    np.testing.assert_allclose(transform.jacobian(probes), numerical, atol=2e-6)
    # A separately reversed TPS is only an approximation to the true inverse.
    reverse = ThinPlateSpline(target * [8, 5], POINTS * [8, 5])
    assert np.max(np.abs(reverse.forward(transform.forward(probes)) - probes)) > 1e-3


@pytest.mark.parametrize("bad", [np.zeros((4, 2)), [[0, 0], [1, 0], [2, 0], [3, 0]],
                                      [[0, 0], [1, 0], [0, 1], [float('nan'), 1]]])
def test_degenerate_landmarks_rejected(bad):
    with pytest.raises(ValueError):
        ThinPlateSpline(bad, bad)


def test_fold_and_reflection_rejected():
    target = POINTS.copy()
    target[-1] = [1.5, .5]
    with pytest.raises(ValueError, match="fold|singular|converge"):
        fit_spline(spec(target))
    with pytest.raises(ValueError, match="fold"):
        fit_spline(spec(POINTS * [-1, 1]))


def test_forward_inverse_solver_failure_is_explicit():
    target = POINTS.copy()
    target[-1] += [.08, -.04]
    transform = fit_spline(spec(target))
    with pytest.raises(ValueError, match="converge"):
        transform.forward([[4, 2]], max_iterations=1, tolerance_mm=1e-12)


@pytest.mark.parametrize("field,value", [
    ("extent_mm", [0, 5]), ("extent_mm", [8, float('inf')]),
    ("source", POINTS[:3].tolist()),
    ("target", [[0, 0], [1, 0], [0, 1], [1, 1], [1, 1]]),
])
def test_invalid_persisted_geometry_rejected(field, value):
    candidate = spec()
    candidate[field] = value
    with pytest.raises(ValueError):
        fit_spline(candidate)


def test_large_inverse_crosses_chunk_boundaries_without_geometry_change():
    target = POINTS.copy()
    target[-1] += [.05, -.04]
    transform = fit_spline(spec(target))
    points = np.random.default_rng(2).uniform([0, 0], [8, 5], (8200, 2))
    recovered = transform.inverse(transform.forward(points))
    np.testing.assert_allclose(recovered, points, atol=2e-6)


def test_resampling_uses_original_extent_despite_thumbnail_rounding():
    # Height rounded to 11 when the ideal thumbnail height would have been 10.
    # A normalized 2-pixel shift must still be exactly two pixels on both axes.
    section = np.arange(16 * 11 * 3, dtype=np.uint8).reshape(11, 16, 3)
    shifted = spec(POINTS + [2 / 16, 2 / 11])
    warped = warp_section(section, shifted, (22, 17), (2, 2), 500)
    np.testing.assert_array_equal(warped[4:15, 4:20], section)
