"""Checks for registration solver math."""

from __future__ import annotations

import math

from langslice_harness.registration.solver import (
    fit_affine_from_correspondences,
    fit_tps_from_correspondences,
)
from langslice_harness.registration.types import RegistrationCorrespondence


def _sample_correspondences() -> list[RegistrationCorrespondence]:
    return [
        RegistrationCorrespondence(
            slice_xy=(20.0, 20.0), atlas_xy=(10.0, 10.0), label="a", confidence="high"
        ),
        RegistrationCorrespondence(
            slice_xy=(60.0, 22.0), atlas_xy=(50.0, 12.0), label="b", confidence="high"
        ),
        RegistrationCorrespondence(
            slice_xy=(98.0, 25.0), atlas_xy=(88.0, 15.0), label="c", confidence="high"
        ),
        RegistrationCorrespondence(
            slice_xy=(24.0, 70.0), atlas_xy=(14.0, 60.0), label="d", confidence="high"
        ),
        RegistrationCorrespondence(
            slice_xy=(62.0, 72.0), atlas_xy=(52.0, 62.0), label="e", confidence="high"
        ),
        RegistrationCorrespondence(
            slice_xy=(95.0, 78.0), atlas_xy=(85.0, 68.0), label="f", confidence="high"
        ),
    ]


def test_registration_solver_math() -> None:
    correspondences = _sample_correspondences()

    affine, residuals = fit_affine_from_correspondences(
        correspondences,
        source_size=(100, 80),
        output_size=(120, 100),
        backend="test_affine",
        reasoning="test",
    )
    assert affine.backend == "test_affine"
    assert affine.output_size == (120, 100)
    assert residuals.max() < 1e-6
    assert math.isclose(affine.translation_px[0], 10.0, abs_tol=1e-6)
    assert math.isclose(affine.translation_px[1], 10.0, abs_tol=1e-6)

    nonlinear = fit_tps_from_correspondences(
        correspondences,
        output_size=(120, 100),
        smoothing=0.5,
        reasoning="test",
    )
    assert nonlinear.backend == "tps"
    assert nonlinear.qc_metrics["max_landmark_residual_px"] < 1e-4
    assert nonlinear.qc_metrics["negative_jacobian_fraction"] == 0.0
