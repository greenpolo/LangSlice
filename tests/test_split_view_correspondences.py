"""Checks for split-view correspondence mapping helpers."""

from __future__ import annotations

import numpy as np

from langslice.gui.main_window import (
    build_split_view_correspondence_points,
    should_apply_affine_to_slice,
)
from langslice.registration import (
    AffineResult,
    NonlinearResult,
    RegistrationCorrespondence,
    RegistrationResult,
    build_annotation_session_from_correspondences,
)


def test_split_view_correspondence_mapping() -> None:
    corr_a = RegistrationCorrespondence(
        slice_xy=(10.0, 20.0), atlas_xy=(40.0, 50.0), label="A", confidence="high"
    )
    corr_b = RegistrationCorrespondence(
        slice_xy=(30.0, 25.0), atlas_xy=(60.0, 55.0), label="B", confidence="high"
    )

    affine = AffineResult(
        matrix=np.array(
            [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        source_size=(100, 80),
        output_size=(100, 80),
        backend="test",
        reasoning="test affine",
    )

    nonlinear = NonlinearResult(
        atlas_points=np.array([[40.0, 50.0], [60.0, 55.0]], dtype=np.float64),
        slice_points=np.array([[10.0, 20.0], [30.0, 25.0]], dtype=np.float64),
        smoothing=1.0,
        backend="tps",
        reasoning="test",
        output_size=(100, 80),
    )

    registration_result = RegistrationResult(
        correspondences=[corr_a, corr_b],
        accepted_correspondences=[corr_a, corr_b],
        affine_result=affine,
        nonlinear_result=nonlinear,
        annotation_session=build_annotation_session_from_correspondences([corr_a, corr_b]),
    )

    slice_points, atlas_points = build_split_view_correspondence_points(registration_result, affine)
    assert slice_points == [(10.0, 20.0, "A"), (30.0, 25.0, "B")]
    assert atlas_points == [(40.0, 50.0, "A"), (60.0, 55.0, "B")]

    empty_slice, empty_atlas = build_split_view_correspondence_points(None, affine)
    assert empty_slice == []
    assert empty_atlas == []


def test_affine_display_direction_guard() -> None:
    affine = AffineResult(
        matrix=np.array(
            [[8.5, 0.0, 120.0], [0.0, 8.5, 80.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        source_size=(456, 320),
        output_size=(4003, 2997),
        backend="test",
        reasoning="atlas_to_slice",
    )
    assert not should_apply_affine_to_slice(affine, (4003, 2997))
    assert should_apply_affine_to_slice(affine, (456, 320))

    equal_size_atlas_to_slice = AffineResult(
        matrix=np.array(
            [[1.0, 0.0, 4.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        source_size=(512, 512),
        output_size=(512, 512),
        backend="test",
        reasoning="direction guard",
        provenance={"transform_direction": "atlas_to_slice"},
    )
    assert not should_apply_affine_to_slice(equal_size_atlas_to_slice, (512, 512))
