"""Checks for the registration runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import langslice.core.nonlinear.runtime as runtime
from langslice.core.nonlinear.types import RegistrationAnnotationSession, RegistrationCandidate
from langslice.doors.cli.register import resolve_register_models


@pytest.fixture(autouse=True)
def offline_atlas(monkeypatch):
    monkeypatch.setattr(runtime, "load_atlas", lambda _: object())
    monkeypatch.setattr(
        runtime, "get_composite_slice", lambda *args, **kwargs: Image.new("RGB", (16, 16))
    )


def test_registration_runtime_direct_image_gen_registration_uses_dense_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
        provider="fallback-provider",
        image_provider="direct-provider",
        image_model="direct-image-model",
        openai_image_route="responses",
        review_model=object(),
        on_progress=progress_messages.append,
        on_trace=trace_events.append,
        debug_dir=str(tmp_path),
    )

    assert generator_calls["provider"] == "direct-provider"
    assert generator_calls["image_model"] == "direct-image-model"
    assert generator_calls["openai_image_route"] == "responses"
    assert generator_calls["review_model"] is None
    assert generator_calls["debug_dir"] == str(tmp_path)
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
        on_trace=trace_events.append,
    )

    assert result.debug_dir is None
    assert any(event.get("title") == "Registration solve completed" for event in trace_events)
def test_resolve_register_models_defaults_review_to_effective_model() -> None:
    image_model, review_model = resolve_register_models(
        default_image_model="gpt-image-2",
        default_review_model="gpt-4.1",
        image_model=None,
        review_model=None,
    )
    assert image_model == "gpt-image-2"
    assert review_model == "gpt-4.1"


def test_resolve_register_models_keeps_explicit_review_model() -> None:
    image_model, review_model = resolve_register_models(
        default_image_model="gpt-image-1.5",
        default_review_model="gpt-4.1",
        image_model="gpt-image-2",
        review_model="gpt-4.1-mini",
    )
    assert image_model == "gpt-image-2"
    assert review_model == "gpt-4.1-mini"
