"""Checks for registration runtime orchestration."""

import numpy as np
from PIL import Image

import langslice.registration.core as registration_core
from langslice.registration import (
    AffineResult,
    NonlinearResult,
    RegistrationCorrespondence,
    RegistrationResult,
    estimate_registration_runtime,
)


def fake_runtime(**kwargs: object) -> RegistrationResult:
    image_obj = kwargs["image"]
    assert isinstance(image_obj, Image.Image)
    assert kwargs["target_landmark_count"] == 12
    assert kwargs["workflow"] == "multimodal_tool_loop"
    assert kwargs.get("on_correspondences") is None
    assert kwargs.get("on_trace") is None
    assert kwargs.get("debug_dir") is None
    affine = AffineResult(
        matrix=np.array(
            [[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        source_size=(image_obj.width, image_obj.height),
        output_size=(140, 100),
        backend="landmark_affine",
        reasoning="runtime affine",
    )
    nonlinear = NonlinearResult(
        atlas_points=np.array(
            [[0.0, 0.0], [10.0, 5.0], [20.0, 8.0], [15.0, 18.0], [4.0, 16.0], [11.0, 10.0]]
        ),
        slice_points=np.array(
            [[1.0, 2.0], [12.0, 7.0], [24.0, 12.0], [18.0, 21.0], [6.0, 17.0], [14.0, 12.0]]
        ),
        smoothing=1.0,
        backend="tps",
        reasoning="runtime nonlinear",
        output_size=(140, 100),
    )
    corr = RegistrationCorrespondence(
        slice_xy=(1.0, 2.0), atlas_xy=(0.0, 0.0), label="test", confidence="high"
    )
    return RegistrationResult(
        correspondences=[corr],
        accepted_correspondences=[corr],
        affine_result=affine,
        nonlinear_result=nonlinear,
    )


def test_registration_runtime_orchestration(monkeypatch) -> None:
    image = Image.new("L", (120, 90), color=0)
    monkeypatch.setattr(registration_core, "estimate_registration", fake_runtime)
    full_result = estimate_registration_runtime(
        image=image,
        atlas_name="allen_mouse_25um",
        position_mm=1.2,
    )
    assert full_result.affine_result.backend == "landmark_affine"
    assert full_result.nonlinear_result.backend == "tps"
