"""Geometry boundary for native BigWarp landmarks, independent of Java."""
import numpy as np
import pytest

from langslice.integrations.abba_spline import spline_world_landmarks


def test_spline_world_landmarks_preserve_axis_direction_and_orientation():
    source = np.array([[0, 0], [1, 0], [0, 1], [.5, .5]])
    target = source + [.1, -.2]
    src, tgt = spline_world_landmarks(
        {"source": source, "target": target, "extent_mm": [15, 20]},
        size=(800, 600), pixel_size_um=25., rotation_deg=90,
    )
    np.testing.assert_allclose(src, [[-7.5, -10], [7.5, -10], [-7.5, 10], [0, 0]])
    np.testing.assert_allclose(tgt-src, np.tile([1.5, -4], (4, 1)))


@pytest.mark.parametrize("points", [
    [[0, 0], [0, 0], [1, 1]], [[0, 0], [.5, .5], [1, 1]],
    [[0, 0], [1, 0], [float("nan"), 1]],
])
def test_spline_rejects_invalid_geometry(points):
    with pytest.raises(ValueError):
        spline_world_landmarks(
            {"source": points, "target": points, "extent_mm": [20, 15]},
            size=(800, 600), pixel_size_um=25.,
        )


def test_elastix_export_samples_full_pullback_and_centers_world_frame(monkeypatch):
    from langslice import landmark_warp

    class Exact:
        def inverse(self, points):
            p = np.asarray(points)
            return p @ np.array([[1.1, .1], [.05, .9]]) + [.3, -.2]

    exact = Exact()
    monkeypatch.setattr(landmark_warp, "fit_spline", lambda _: exact)
    payload = {"backend": "elastix", "extent_mm": [20, 15],
               "domain_mm": [[-2, -1], [22, 16]]}
    diagnostics = {}
    src, tgt = spline_world_landmarks(
        payload, size=(800, 600), pixel_size_um=25., diagnostics=diagnostics,
    )
    assert len(src) == 81  # Export samples are independent of agent's 64-pair limit.
    np.testing.assert_allclose(src + [10, 7.5], exact.inverse(tgt + [10, 7.5]))
    np.testing.assert_allclose(tgt.min(axis=0), [-12, -8.5])
    np.testing.assert_allclose(tgt.max(axis=0), [12, 8.5])
    assert diagnostics["max_error_mm"] < 1e-10
    assert diagnostics["min_sampled_jacobian"] > 0


def test_elastix_export_verifies_nonlinear_offgrid_accuracy(monkeypatch):
    from langslice import landmark_warp
    from langslice.integrations.abba_spline import _sample_elastix_landmarks

    class Exact:
        def inverse(self, points):
            p = np.asarray(points).copy()
            p[:, 0] += .1*np.sin(3*p[:, 1])
            return p

    exact = Exact()
    monkeypatch.setattr(landmark_warp, "fit_spline", lambda _: exact)
    diagnostics = {}
    src, tgt = _sample_elastix_landmarks(
        {"domain_mm": [[0, 0], [4, 4]]}, diagnostics=diagnostics,
    )
    fit = landmark_warp._ThinPlateKernel(tgt, src, max_points=1089)
    probes = np.random.default_rng(7).uniform(0, 4, (100, 2))
    assert np.max(np.linalg.norm(fit.forward(probes)-exact.inverse(probes), axis=1)) < .005
    assert diagnostics["validation_points"] > len(src)
    assert diagnostics["grid_size"] > 9  # Coarse sampling cannot meet the tolerance.


def test_elastix_export_refuses_folded_map_before_native_replacement(monkeypatch):
    from langslice import landmark_warp
    from langslice.integrations.abba_spline import _sample_elastix_landmarks

    class Folded:
        def inverse(self, points):
            return np.asarray(points) * [-1, 1]

    monkeypatch.setattr(landmark_warp, "fit_spline", lambda _: Folded())
    with pytest.raises(ValueError, match="folds or collapses"):
        _sample_elastix_landmarks({"domain_mm": [[0, 0], [4, 4]]})


def test_elastix_export_requires_saved_domain():
    from langslice.integrations.abba_spline import _sample_elastix_landmarks

    with pytest.raises(ValueError, match="domain_mm"):
        _sample_elastix_landmarks({})


def test_unknown_backend_cannot_silently_export_as_legacy_tps():
    points = [[0, 0], [1, 0], [0, 1], [.5, .5]]
    with pytest.raises(ValueError, match="Unknown landmark warp backend"):
        spline_world_landmarks(
            {"backend": "future", "source": points, "target": points,
             "extent_mm": [20, 15]}, size=(800, 600), pixel_size_um=25.,
        )
