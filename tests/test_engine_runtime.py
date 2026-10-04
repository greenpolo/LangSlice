from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from PIL import Image

from langslice.doors.api import runtime
from langslice.doors.api.models import RegisterRequest


def _stub_image_prep(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("PIL.Image.open", lambda _p: Image.new("RGB", (32, 24), "white"))
    monkeypatch.setattr("langslice.core.image_prep.normalize_image", lambda img: img)
    monkeypatch.setattr(
        "langslice.core.image_prep.prepare_image_for_vlm",
        lambda img, **_kwargs: SimpleNamespace(image=img, effective_pixel_size_um=None),
    )


def _stub_vlm_mutators(monkeypatch) -> None:  # noqa: ANN001
    import langslice.providers.vlm_config as vlm_config

    def set_temp(value: float) -> None:
        vlm_config.TEMPERATURE = value

    def set_thinking(value: str) -> None:
        vlm_config.THINKING_LEVEL = value

    monkeypatch.setattr("langslice.providers.vlm_config.set_temperature", set_temp)
    monkeypatch.setattr("langslice.providers.vlm_config.set_thinking_level", set_thinking)


def test_run_register_surfaces_artifact_paths_from_metadata(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice.providers.vlm_config.MODEL_NAME", "fake-model")

    fake_affine = SimpleNamespace(
        rotation_deg=1.5,
        translation_px=(2.0, 3.0),
        scale=(1.0, 1.1),
        shear=0.05,
    )
    fake_result = SimpleNamespace(
        accepted_correspondences=[object()],
        affine_result=fake_affine,
        debug_dir="out/debug",
        annotation_session=SimpleNamespace(metadata={}),
    )

    monkeypatch.setattr(
        "langslice.core.nonlinear.runtime.estimate_registration",
        lambda **_kwargs: fake_result,
    )
    monkeypatch.setattr(
        "langslice.core.nonlinear.types.annotation_session_to_dict",
        lambda _session: {
            "metadata": {
                "warped_atlas_path": "/tmp/warped_atlas.png",
                "candidate_metadata": {
                    "raw_correction_path": "/tmp/raw_correction.png",
                    "rough_border_overlay_path": "/tmp/rough.png",
                    "corrected_border_overlay_path": "/tmp/corrected.png",
                    "warped_border_overlay_path": "/tmp/warped_border.png",
                    "generated_segmentation_path": "/tmp/generated_seg.png",
                    "generated_border_overlay_path": "/tmp/generated_border.png",
                    "slice_warped_to_atlas_path": "/tmp/inverse_slice.png",
                    "slice_atlas_border_overlay_path": "/tmp/inverse_border.png",
                    "inverse_warp_status": "ok",
                },
            }
        },
    )

    request = RegisterRequest(
        image_path="slice.png",
        atlas="allen_mouse_25um",
        position_mm=1.2,
    )
    result = runtime.run_register(request)
    assert result.warped_atlas_path == "/tmp/warped_atlas.png"
    assert result.warped_border_overlay_path == "/tmp/warped_border.png"
    assert result.generated_segmentation_path == "/tmp/generated_seg.png"
    assert result.generated_border_overlay_path == "/tmp/generated_border.png"
    assert result.slice_warped_to_atlas_path == "/tmp/inverse_slice.png"
    assert result.slice_atlas_border_overlay_path == "/tmp/inverse_border.png"
    assert result.inverse_warp_status == "ok"
    assert result.raw_correction_path == "/tmp/raw_correction.png"
    assert result.rough_border_overlay_path == "/tmp/rough.png"
    assert result.corrected_border_overlay_path == "/tmp/corrected.png"


def test_run_register_restores_runtime_globals_after_exception(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice.providers.vlm_config.MODEL_NAME", "fake-model")
    monkeypatch.setattr("langslice.providers.vlm_config.TEMPERATURE", 0.7)
    monkeypatch.setattr("langslice.providers.vlm_config.THINKING_LEVEL", "HIGH")
    _stub_vlm_mutators(monkeypatch)
    monkeypatch.setenv("LANGSLICE_ENDPOINT", "http://prior-endpoint")
    monkeypatch.setenv("LANGSLICE_VLM_DEBUG_DIR", "prior-debug")
    monkeypatch.setattr(
        "langslice.core.nonlinear.runtime.estimate_registration",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("register failed")),
    )
    monkeypatch.setattr(
        "langslice.core.nonlinear.types.annotation_session_to_dict",
        lambda _session: {"metadata": {}},
    )
    request = RegisterRequest(
        image_path="slice.png",
        atlas="allen_mouse_25um",
        position_mm=1.2,
        endpoint="http://new-endpoint",
        output_dir="new-debug",
        temperature=0.1,
        thinking="LOW",
    )
    with pytest.raises(RuntimeError, match="register failed"):
        runtime.run_register(request)
    assert os.environ["LANGSLICE_ENDPOINT"] == "http://prior-endpoint"
    assert os.environ["LANGSLICE_VLM_DEBUG_DIR"] == "prior-debug"
    import langslice.providers.vlm_config as vlm_config

    assert vlm_config.TEMPERATURE == 0.7
    assert vlm_config.THINKING_LEVEL == "HIGH"


def test_run_register_restores_runtime_globals_after_success(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice.providers.vlm_config.MODEL_NAME", "fake-model")
    monkeypatch.setattr("langslice.providers.vlm_config.TEMPERATURE", 0.7)
    monkeypatch.setattr("langslice.providers.vlm_config.THINKING_LEVEL", "HIGH")
    _stub_vlm_mutators(monkeypatch)
    monkeypatch.setenv("LANGSLICE_ENDPOINT", "http://prior-endpoint")
    monkeypatch.setenv("LANGSLICE_VLM_DEBUG_DIR", "prior-debug")

    fake_affine = SimpleNamespace(
        rotation_deg=0.0,
        translation_px=(0.0, 0.0),
        scale=(1.0, 1.0),
        shear=0.0,
    )
    fake_result = SimpleNamespace(
        accepted_correspondences=[],
        affine_result=fake_affine,
        debug_dir=None,
        annotation_session=SimpleNamespace(metadata={}),
    )
    monkeypatch.setattr(
        "langslice.core.nonlinear.runtime.estimate_registration",
        lambda **_kwargs: fake_result,
    )
    monkeypatch.setattr(
        "langslice.core.nonlinear.types.annotation_session_to_dict",
        lambda _session: {"metadata": {}},
    )

    request = RegisterRequest(
        image_path="slice.png",
        atlas="allen_mouse_25um",
        position_mm=1.2,
        endpoint="http://new-endpoint",
        output_dir="new-debug",
        temperature=0.1,
        thinking="LOW",
    )
    runtime.run_register(request)
    assert os.environ["LANGSLICE_ENDPOINT"] == "http://prior-endpoint"
    assert os.environ["LANGSLICE_VLM_DEBUG_DIR"] == "prior-debug"
    import langslice.providers.vlm_config as vlm_config

    assert vlm_config.TEMPERATURE == 0.7
    assert vlm_config.THINKING_LEVEL == "HIGH"

