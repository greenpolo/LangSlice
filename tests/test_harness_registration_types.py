from __future__ import annotations

import numpy as np
from PIL import Image

from langslice_harness.harness.registration import (
    GeneratedSegmentation,
    RegistrationCandidate,
    candidate_to_registration_result,
)
from langslice_harness.registration.types import (
    RegistrationAnnotationSession,
    annotation_session_to_dict,
)


def test_generated_segmentation_and_candidate_default_metadata() -> None:
    image = Image.new("RGB", (8, 6), color="white")
    session = RegistrationAnnotationSession(workflow="dense")

    segmentation = GeneratedSegmentation(
        image=image,
        provider="google",
        model="gemini-2.0-flash",
        route="image_gen",
    )
    candidate = RegistrationCandidate(
        candidate_id="cand-1",
        generated_segmentation=segmentation.image,
        warped_atlas=image,
        warped_border_overlay=image,
        markers=[[1.0, 2.0], [3.0, 4.0]],
        annotation_session=session,
    )

    assert segmentation.metadata == {}
    assert candidate.metadata == {}


def test_candidate_to_registration_result_uses_dense_placeholders_and_metadata() -> None:
    image = Image.new("RGB", (20, 12), color="white")
    session = RegistrationAnnotationSession(
        workflow="dense",
        metadata={"existing": "keep", "candidate_id": "old", "n_markers": 99},
    )
    candidate = RegistrationCandidate(
        candidate_id="cand-2",
        generated_segmentation=image,
        warped_atlas=image,
        warped_border_overlay=image,
        markers=[[10.0, 11.0], [12.0, 13.0]],
        annotation_session=session,
        metadata={
            "source": "unit-test",
            "generation": {
                "provider": "google",
                "model": "gemini-2.0-flash",
                "route": "image_gen",
                "revised_prompt": None,
                "details": {"pass": 1},
            },
        },
    )

    result = candidate_to_registration_result(candidate, image_size=(20, 12), debug_dir=None)
    serialized_before = annotation_session_to_dict(result.annotation_session)
    assert serialized_before is not None

    assert result.correspondences == []
    assert result.accepted_correspondences == []
    assert np.array_equal(result.affine_result.matrix, np.eye(3, dtype=np.float64))
    assert result.affine_result.backend == "image_gen_registration_dense"
    assert result.affine_result.output_size == (20, 12)
    assert result.nonlinear_result.atlas_points.shape == (0, 2)
    assert result.nonlinear_result.slice_points.shape == (0, 2)
    assert result.nonlinear_result.backend == "elastix_bspline_visualign"
    assert result.nonlinear_result.output_size == (20, 12)
    assert serialized_before["metadata"]["existing"] == "keep"
    assert serialized_before["metadata"]["visualign_markers"] == [[10.0, 11.0], [12.0, 13.0]]
    assert serialized_before["metadata"]["n_markers"] == 2
    assert serialized_before["metadata"]["candidate_id"] == "cand-2"
    assert serialized_before["metadata"]["candidate_metadata"]["source"] == "unit-test"
    assert serialized_before["metadata"]["candidate_metadata"]["generation"]["provider"] == "google"
    assert serialized_before["metadata"]["candidate_metadata"]["generation"]["details"] == {
        "pass": 1
    }
    assert serialized_before["metadata"]["n_markers"] != 99

    candidate.markers[0][0] = 99.0
    candidate.metadata["generation"]["provider"] = "changed"
    candidate.metadata["generation"]["details"]["pass"] = 2

    serialized_after = annotation_session_to_dict(result.annotation_session)
    assert serialized_after == serialized_before
