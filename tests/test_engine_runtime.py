from __future__ import annotations

import builtins
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from langslice_harness.api import runtime
from langslice_harness.api.models import EstimateRequest, RegisterRequest


def _stub_image_prep(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("PIL.Image.open", lambda _p: Image.new("RGB", (32, 24), "white"))
    monkeypatch.setattr("langslice_harness.image_prep.normalize_image", lambda img: img)
    monkeypatch.setattr(
        "langslice_harness.image_prep.prepare_image_for_vlm",
        lambda img, **_kwargs: SimpleNamespace(image=img),
    )


def _stub_vlm_mutators(monkeypatch) -> None:  # noqa: ANN001
    import langslice_harness.vlm_config as vlm_config

    def set_temp(value: float) -> None:
        vlm_config.TEMPERATURE = value

    def set_thinking(value: str) -> None:
        vlm_config.THINKING_LEVEL = value

    monkeypatch.setattr("langslice_harness.vlm_config.set_temperature", set_temp)
    monkeypatch.setattr("langslice_harness.vlm_config.set_thinking_level", set_thinking)


def test_run_register_surfaces_artifact_paths_from_metadata(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice_harness.vlm_config.MODEL_NAME", "fake-model")

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
        "langslice_harness.registration.core.estimate_registration_runtime",
        lambda **_kwargs: fake_result,
    )
    monkeypatch.setattr(
        "langslice_harness.registration.types.annotation_session_to_dict",
        lambda _session: {
            "metadata": {
                "warped_atlas_path": "/tmp/warped_atlas.png",
                "candidate_metadata": {
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


def test_run_estimate_tool_use_forwards_on_progress_to_estimator(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice_harness.vlm_config.MODEL_NAME", "fake-model")
    monkeypatch.setattr("langslice_harness.vlm_config.is_image_generation_model", lambda _m: False)

    def fake_estimate_position(**kwargs):  # noqa: ANN003
        kwargs["on_progress"]("tool-use progress")
        return SimpleNamespace(position_mm=1.0, reasoning="ok", debug_dir=None)

    monkeypatch.setattr("langslice_harness.estimation.estimate_position", fake_estimate_position)

    events = []
    request = EstimateRequest(
        image_path="slice.png",
        atlas="allen_mouse_25um",
        workflow="tool_use",
    )
    runtime.run_estimate(request, emit=events.append)
    progress_messages = [event.message for event in events if event.kind == "progress"]
    assert any("tool-use progress" == msg for msg in progress_messages)


def test_run_estimate_restores_runtime_globals_after_success(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice_harness.vlm_config.MODEL_NAME", "fake-model")
    monkeypatch.setattr("langslice_harness.vlm_config.TEMPERATURE", 0.7)
    monkeypatch.setattr("langslice_harness.vlm_config.THINKING_LEVEL", "HIGH")
    monkeypatch.setattr("langslice_harness.vlm_config.is_image_generation_model", lambda _m: False)
    _stub_vlm_mutators(monkeypatch)
    monkeypatch.setenv("LANGSLICE_ENDPOINT", "http://prior-endpoint")
    monkeypatch.setenv("LANGSLICE_VLM_DEBUG_DIR", "prior-debug")
    monkeypatch.setattr(
        "langslice_harness.estimation.estimate_position",
        lambda **_kwargs: SimpleNamespace(position_mm=1.0, reasoning="ok", debug_dir=None),
    )

    request = EstimateRequest(
        image_path="slice.png",
        atlas="allen_mouse_25um",
        workflow="tool_use",
        endpoint="http://new-endpoint",
        output_dir="new-debug",
        temperature=0.1,
        thinking="LOW",
    )
    runtime.run_estimate(request)
    assert os.environ["LANGSLICE_ENDPOINT"] == "http://prior-endpoint"
    assert os.environ["LANGSLICE_VLM_DEBUG_DIR"] == "prior-debug"
    import langslice_harness.vlm_config as vlm_config

    assert vlm_config.TEMPERATURE == 0.7
    assert vlm_config.THINKING_LEVEL == "HIGH"


def test_run_estimate_restores_runtime_globals_after_exception(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice_harness.vlm_config.MODEL_NAME", "fake-model")
    monkeypatch.setattr("langslice_harness.vlm_config.TEMPERATURE", 0.7)
    monkeypatch.setattr("langslice_harness.vlm_config.THINKING_LEVEL", "HIGH")
    monkeypatch.setattr("langslice_harness.vlm_config.is_image_generation_model", lambda _m: False)
    _stub_vlm_mutators(monkeypatch)
    monkeypatch.setenv("LANGSLICE_ENDPOINT", "http://prior-endpoint")
    monkeypatch.setenv("LANGSLICE_VLM_DEBUG_DIR", "prior-debug")

    def boom(**_kwargs):  # noqa: ANN003
        raise RuntimeError("estimate failed")

    monkeypatch.setattr("langslice_harness.estimation.estimate_position", boom)
    request = EstimateRequest(
        image_path="slice.png",
        atlas="allen_mouse_25um",
        workflow="tool_use",
        endpoint="http://new-endpoint",
        output_dir="new-debug",
        temperature=0.1,
        thinking="LOW",
    )
    with pytest.raises(RuntimeError, match="estimate failed"):
        runtime.run_estimate(request)
    assert os.environ["LANGSLICE_ENDPOINT"] == "http://prior-endpoint"
    assert os.environ["LANGSLICE_VLM_DEBUG_DIR"] == "prior-debug"
    import langslice_harness.vlm_config as vlm_config

    assert vlm_config.TEMPERATURE == 0.7
    assert vlm_config.THINKING_LEVEL == "HIGH"


def test_run_register_restores_runtime_globals_after_exception(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice_harness.vlm_config.MODEL_NAME", "fake-model")
    monkeypatch.setattr("langslice_harness.vlm_config.TEMPERATURE", 0.7)
    monkeypatch.setattr("langslice_harness.vlm_config.THINKING_LEVEL", "HIGH")
    _stub_vlm_mutators(monkeypatch)
    monkeypatch.setenv("LANGSLICE_ENDPOINT", "http://prior-endpoint")
    monkeypatch.setenv("LANGSLICE_VLM_DEBUG_DIR", "prior-debug")
    monkeypatch.setattr(
        "langslice_harness.registration.core.estimate_registration_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("register failed")),
    )
    monkeypatch.setattr(
        "langslice_harness.registration.types.annotation_session_to_dict",
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
    import langslice_harness.vlm_config as vlm_config

    assert vlm_config.TEMPERATURE == 0.7
    assert vlm_config.THINKING_LEVEL == "HIGH"


def test_run_register_restores_runtime_globals_after_success(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice_harness.vlm_config.MODEL_NAME", "fake-model")
    monkeypatch.setattr("langslice_harness.vlm_config.TEMPERATURE", 0.7)
    monkeypatch.setattr("langslice_harness.vlm_config.THINKING_LEVEL", "HIGH")
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
        "langslice_harness.registration.core.estimate_registration_runtime",
        lambda **_kwargs: fake_result,
    )
    monkeypatch.setattr(
        "langslice_harness.registration.types.annotation_session_to_dict",
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
    import langslice_harness.vlm_config as vlm_config

    assert vlm_config.TEMPERATURE == 0.7
    assert vlm_config.THINKING_LEVEL == "HIGH"


def test_run_estimate_tool_use_does_not_import_image_gen_module(monkeypatch) -> None:  # noqa: ANN001
    _stub_image_prep(monkeypatch)
    monkeypatch.setattr("langslice_harness.vlm_config.MODEL_NAME", "fake-model")
    monkeypatch.setattr("langslice_harness.vlm_config.is_image_generation_model", lambda _m: False)
    monkeypatch.setattr(
        "langslice_harness.estimation.estimate_position",
        lambda **_kwargs: SimpleNamespace(position_mm=1.0, reasoning="ok", debug_dir=None),
    )

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
        if name == "langslice_harness.harness.estimation.image_gen":
            raise AssertionError("image_gen import should not happen in tool_use workflow")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    request = EstimateRequest(
        image_path="slice.png",
        atlas="allen_mouse_25um",
        workflow="tool_use",
    )
    result = runtime.run_estimate(request)
    assert result.position_mm == 1.0
