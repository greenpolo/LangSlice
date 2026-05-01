"""Checks for the registration runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import langslice_harness.cli as cli
import langslice_harness.registration.core as registration_core
import langslice_harness.registration.runtime as runtime
from langslice_harness.harness.registration.types import RegistrationCandidate
from langslice_harness.registration import (
    AffineResult,
    NonlinearResult,
    RegistrationResult,
    estimate_registration_runtime,
)
from langslice_harness.registration import (
    RegistrationCorrespondence as CoreRegistrationCorrespondence,
)
from langslice_harness.registration.types import (
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
)


def test_registration_runtime_direct_image_gen_registration_uses_dense_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    correspondences_called: list[list[RegistrationCorrespondence]] = []
    annotation_sessions: list[RegistrationAnnotationSession] = []
    progress_messages: list[str] = []
    trace_events: list[dict[str, object]] = []
    generator_calls: dict[str, object] = {}

    def fake_generate_registration_candidate(
        image,
        *,
        atlas_name,
        position_mm,
        provider="google",
        image_model=None,
        debug_dir=None,
        on_progress=None,
        on_trace=None,
        openai_image_route="images",
        review_model=None,
        **kwargs,
    ):
        _ = image, atlas_name, position_mm, kwargs
        generator_calls.update(
            provider=provider,
            image_model=image_model,
            debug_dir=debug_dir,
            openai_image_route=openai_image_route,
            review_model=review_model,
        )
        if on_progress is not None:
            on_progress("candidate progress")
        if on_trace is not None:
            on_trace({"stage": "registration", "title": "candidate trace"})
        return RegistrationCandidate(
            candidate_id="dense-candidate",
            generated_segmentation=Image.new("RGB", (16, 16), (220, 30, 30)),
            warped_atlas=Image.new("RGB", (16, 16), (30, 220, 30)),
            warped_border_overlay=Image.new("RGB", (16, 16), (30, 30, 220)),
            markers=[[10.0, 11.0], [12.0, 13.0]],
            annotation_session=RegistrationAnnotationSession(
                workflow="image_gen_registration",
                target_count=0,
                metadata={"source": "fake-candidate"},
            ),
            metadata={"candidate": "fake"},
        )

    monkeypatch.setattr(
        runtime,
        "generate_registration_candidate",
        fake_generate_registration_candidate,
    )
    monkeypatch.setenv("LANGSLICE_VLM_DEBUG_DIR", str(tmp_path))

    result = runtime.estimate_registration(
        Image.new("RGB", (120, 100), (255, 255, 255)),
        atlas_name="allen_mouse_25um",
        position_mm=1.0,
        registration_mode="direct",
        provider="fallback-provider",
        image_provider="direct-provider",
        image_model="direct-image-model",
        openai_image_route="responses",
        review_model=object(),
        on_correspondences=correspondences_called.append,
        on_annotation_session=annotation_sessions.append,
        on_progress=progress_messages.append,
        on_trace=trace_events.append,
        debug_dir=str(tmp_path),
    )

    assert generator_calls["provider"] == "direct-provider"
    assert generator_calls["image_model"] == "direct-image-model"
    assert generator_calls["openai_image_route"] == "responses"
    assert generator_calls["review_model"] is None
    assert generator_calls["debug_dir"] == str(tmp_path)
    assert correspondences_called == [[]]
    assert len(annotation_sessions) == 1
    assert annotation_sessions[0].metadata["visualign_markers"] == [
        [10.0, 11.0],
        [12.0, 13.0],
    ]
    assert result.correspondences == []
    assert result.accepted_correspondences == []
    assert result.annotation_session is not None
    assert result.annotation_session.metadata["visualign_markers"] == [
        [10.0, 11.0],
        [12.0, 13.0],
    ]
    assert result.debug_dir == str(tmp_path / "registration")
    assert (tmp_path / "registration" / "registration.json").exists()
    assert progress_messages
    assert trace_events


def test_registration_runtime_direct_image_gen_registration_emits_runtime_trace_without_debug(
    monkeypatch,
) -> None:
    trace_events: list[dict[str, object]] = []

    def fake_generate_registration_candidate(
        image,
        *,
        atlas_name,
        position_mm,
        provider="google",
        image_model=None,
        debug_dir=None,
        on_progress=None,
        on_trace=None,
        openai_image_route="images",
        review_model=None,
        **kwargs,
    ):
        _ = image, atlas_name, position_mm, provider, image_model, debug_dir, kwargs
        _ = openai_image_route, review_model
        if on_trace is not None:
            on_trace({"stage": "registration", "title": "candidate trace"})
        return RegistrationCandidate(
            candidate_id="dense-candidate",
            generated_segmentation=Image.new("RGB", (16, 16), (220, 30, 30)),
            warped_atlas=Image.new("RGB", (16, 16), (30, 220, 30)),
            warped_border_overlay=Image.new("RGB", (16, 16), (30, 30, 220)),
            markers=[[10.0, 11.0]],
            annotation_session=RegistrationAnnotationSession(
                workflow="image_gen_registration",
                target_count=0,
                metadata={"source": "fake-candidate"},
            ),
            metadata={"candidate": "fake"},
        )

    monkeypatch.setattr(
        runtime,
        "generate_registration_candidate",
        fake_generate_registration_candidate,
    )
    monkeypatch.delenv("LANGSLICE_VLM_DEBUG_DIR", raising=False)

    result = runtime.estimate_registration(
        Image.new("RGB", (120, 100), (255, 255, 255)),
        atlas_name="allen_mouse_25um",
        position_mm=1.0,
        registration_mode="direct",
        on_trace=trace_events.append,
    )

    assert result.debug_dir is None
    assert any(event.get("title") == "Registration solve completed" for event in trace_events)


def test_registration_runtime_agentic_image_gen_registration_uses_review_session(
    monkeypatch,
) -> None:
    review_calls: list[dict[str, object]] = []
    correspondences_calls: list[list[RegistrationCorrespondence]] = []
    annotation_sessions: list[RegistrationAnnotationSession] = []
    progress_messages: list[str] = []
    trace_events: list[dict[str, object]] = []

    def fake_run_registration_review_session(**kwargs):
        review_calls.append(dict(kwargs))
        return RegistrationCandidate(
            candidate_id="agentic-candidate",
            generated_segmentation=Image.new("RGB", (16, 16), (210, 40, 40)),
            warped_atlas=Image.new("RGB", (16, 16), (40, 210, 40)),
            warped_border_overlay=Image.new("RGB", (16, 16), (40, 40, 210)),
            markers=[[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            annotation_session=RegistrationAnnotationSession(
                workflow="image_gen_registration",
                target_count=0,
                metadata={
                    "visualign_markers": [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
                    "n_markers": 3,
                },
            ),
            metadata={"source": "agentic-review"},
        )

    monkeypatch.setattr(
        runtime,
        "run_registration_review_session",
        fake_run_registration_review_session,
    )

    result = runtime.estimate_registration(
        Image.new("RGB", (128, 96), (255, 255, 255)),
        atlas_name="allen_mouse_25um",
        position_mm=1.0,
        registration_mode="agentic",
        provider="fallback-provider",
        image_provider="agentic-provider",
        image_model="agentic-image-model",
        openai_image_route="responses",
        review_model="agentic-review-model",
        max_candidates=2,
        on_correspondences=correspondences_calls.append,
        on_annotation_session=annotation_sessions.append,
        on_progress=progress_messages.append,
        on_trace=trace_events.append,
    )

    assert len(review_calls) == 1
    assert review_calls[0]["image_provider"] == "agentic-provider"
    assert review_calls[0]["image_model"] == "agentic-image-model"
    assert review_calls[0]["openai_image_route"] == "responses"
    assert review_calls[0]["model"] == "agentic-review-model"
    assert review_calls[0]["pipeline_review_model"] == "agentic-review-model"
    assert review_calls[0]["max_candidates"] == 2
    assert review_calls[0]["position_mm"] == 1.0
    assert review_calls[0]["on_progress"] is not None
    assert review_calls[0]["on_trace"] is not None
    assert correspondences_calls == [[]]
    assert len(annotation_sessions) == 1
    assert annotation_sessions[0].metadata["visualign_markers"] == [
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ]
    assert result.correspondences == []
    assert result.accepted_correspondences == []
    assert result.annotation_session is not None
    assert result.annotation_session.metadata["n_markers"] == 3
    assert result.annotation_session.metadata["candidate_metadata"] == {
        "source": "agentic-review",
    }
    assert progress_messages
    assert trace_events


def _fake_core_registration_runtime(**kwargs: object) -> RegistrationResult:
    image_obj = kwargs["image"]
    assert isinstance(image_obj, Image.Image)
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
            [
                [0.0, 0.0],
                [10.0, 5.0],
                [20.0, 8.0],
                [15.0, 18.0],
                [4.0, 16.0],
                [11.0, 10.0],
            ]
        ),
        slice_points=np.array(
            [
                [1.0, 2.0],
                [12.0, 7.0],
                [24.0, 12.0],
                [18.0, 21.0],
                [6.0, 17.0],
                [14.0, 12.0],
            ]
        ),
        smoothing=1.0,
        backend="tps",
        reasoning="runtime nonlinear",
        output_size=(140, 100),
    )
    corr = CoreRegistrationCorrespondence(
        slice_xy=(1.0, 2.0), atlas_xy=(0.0, 0.0), label="test", confidence="high"
    )
    return RegistrationResult(
        correspondences=[corr],
        accepted_correspondences=[corr],
        affine_result=affine,
        nonlinear_result=nonlinear,
    )


def test_estimate_registration_runtime_orchestration(monkeypatch) -> None:
    image = Image.new("L", (120, 90), color=0)
    monkeypatch.setattr(
        registration_core,
        "estimate_registration",
        _fake_core_registration_runtime,
    )
    full_result = estimate_registration_runtime(
        image=image,
        atlas_name="allen_mouse_25um",
        position_mm=1.2,
    )
    assert full_result.affine_result.backend == "landmark_affine"
    assert full_result.nonlinear_result.backend == "tps"


def test_resolve_register_models_defaults_review_to_effective_model() -> None:
    image_model, review_model = cli._resolve_register_models(
        default_image_model="gpt-image-2",
        default_review_model="gpt-4.1",
        image_model=None,
        review_model=None,
    )
    assert image_model == "gpt-image-2"
    assert review_model == "gpt-4.1"


def test_resolve_register_models_keeps_explicit_review_model() -> None:
    image_model, review_model = cli._resolve_register_models(
        default_image_model="gpt-image-1.5",
        default_review_model="gpt-4.1",
        image_model="gpt-image-2",
        review_model="gpt-4.1-mini",
    )
    assert image_model == "gpt-image-2"
    assert review_model == "gpt-4.1-mini"
