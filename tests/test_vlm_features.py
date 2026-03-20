"""Checks for Gemini feature-flag integrations."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import cast

import numpy as np
from PIL import Image

import langslice.registration.agents as registration_agents
import langslice.vlm.batch_eval as batch_eval
import langslice.vlm.config as vlm_config
import langslice.vlm.estimator as estimator


def test_feature_flags_for_ai_studio(monkeypatch) -> None:
    monkeypatch.setenv("LANGSLICE_GENAI_BACKEND", "ai_studio")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LANGSLICE_GENAI_COUNT_TOKENS", "1")
    monkeypatch.setenv("LANGSLICE_GENAI_AP_USE_FILE_API", "1")
    monkeypatch.setenv("LANGSLICE_GENAI_AP_USE_CONTEXT_CACHE", "1")
    monkeypatch.setenv("LANGSLICE_GENAI_AP_USE_INTERACTIONS", "1")

    flags = vlm_config.feature_flags()
    assert flags["count_tokens_enabled"] is True
    assert flags["ap_use_file_api"] is True
    assert flags["ap_use_context_cache"] is True
    assert flags["ap_use_interactions"] is True
    assert flags["supports_file_api"] is True
    assert flags["supports_interactions_api"] is True
    assert flags["supports_batch_api"] is False


def test_set_temperature_updates_runtime_value() -> None:
    original = vlm_config.TEMPERATURE
    try:
        vlm_config.set_temperature(0.35)
        assert vlm_config.TEMPERATURE == 0.35
        vlm_config.set_temperature(3.0)
        assert vlm_config.TEMPERATURE == 2.0
        vlm_config.set_temperature(-1.0)
        assert vlm_config.TEMPERATURE == 0.0
    finally:
        vlm_config.set_temperature(original)


def test_set_code_execution_updates_runtime_value() -> None:
    original = vlm_config.CODE_EXECUTION_ENABLED
    try:
        vlm_config.set_code_execution_enabled(False)
        assert vlm_config.CODE_EXECUTION_ENABLED is False
        vlm_config.set_code_execution_enabled(True)
        assert vlm_config.CODE_EXECUTION_ENABLED is True
    finally:
        vlm_config.set_code_execution_enabled(original)


def test_supports_code_execution_only_for_supported_models() -> None:
    assert vlm_config.supports_code_execution("gemini-3-flash-preview") is True
    assert vlm_config.supports_code_execution("gemini-3.1-pro-preview") is False
    assert vlm_config.supports_code_execution(None) is False


def test_set_thinking_level_updates_runtime_value() -> None:
    original = vlm_config.THINKING_LEVEL
    try:
        vlm_config.set_thinking_level("low")
        assert vlm_config.THINKING_LEVEL == "LOW"
        vlm_config.set_thinking_level("HIGH")
        assert vlm_config.THINKING_LEVEL == "HIGH"
    finally:
        vlm_config.set_thinking_level(original)


def test_ap_progress_heartbeat_reports_wait_and_completion() -> None:
    messages: list[str] = []

    result = estimator._run_with_progress_heartbeat(
        lambda: (time.sleep(0.03), "ok")[1],
        request_label="AP test request",
        on_progress=messages.append,
        heartbeat_interval_s=0.01,
    )

    assert result == "ok"
    assert messages[0] == "AP test request: request started"
    assert any("still waiting for Gemini" in message for message in messages)
    assert messages[-1].startswith("AP test request: response received in ")


def test_registration_workflow_options_gate_image_models() -> None:
    assert "gemini-3-pro-image-preview" in vlm_config.AVAILABLE_MODELS
    assert "gemini-3.1-flash-image-preview" in vlm_config.AVAILABLE_MODELS

    image_options = vlm_config.get_registration_workflow_options("gemini-3-pro-image-preview")
    assert image_options == [("Image Gen (2-Shot)", "image_gen_two_shot")]

    text_options = vlm_config.get_registration_workflow_options("gemini-3-flash-preview")
    assert text_options == [
        ("Single Pass", "single_pass"),
        ("Tool Loop", "multimodal_tool_loop"),
    ]


def test_history_metrics_counts_file_parts() -> None:
    content = estimator.types.Content(
        role="user",
        parts=[
            estimator.types.Part(text="hello"),
            estimator.types.Part.from_uri(file_uri="gs://bucket/image.jpg", mime_type="image/jpeg"),
        ],
    )

    metrics = estimator._history_metrics([content])
    assert metrics["content_count"] == 1
    assert metrics["text_parts"] == 1
    assert metrics["image_parts"] == 1
    assert metrics["image_bytes"] == 0


def test_build_ap_batch_requests() -> None:
    requests = batch_eval.build_ap_batch_requests(
        [
            batch_eval.APBatchCase(
                key="case-1",
                image_uri="gs://bucket/trace-1.jpg",
                prompt="Estimate AP position for this slice.",
                metadata={"split": "eval"},
            )
        ],
        model="gemini-3-flash-preview",
        system_instruction="You are a neuroanatomy assistant.",
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.model == "gemini-3-flash-preview"
    assert request.metadata == {"key": "case-1", "split": "eval"}
    assert request.config is not None
    assert request.contents is not None
    contents = cast(list[estimator.types.Content], request.contents)
    content = contents[0]
    assert content.role == "user"
    assert content.parts is not None
    parts = content.parts
    assert parts[0].text == "Estimate AP position for this slice."
    file_data = getattr(parts[1], "file_data", None)
    assert getattr(file_data, "file_uri", None) == "gs://bucket/trace-1.jpg"


def test_create_ap_batch_job_uses_vertex_batch_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeBatches:
        def create(self, *, model: str, src: object, config: object | None = None) -> object:
            captured["model"] = model
            captured["src"] = src
            captured["config"] = config
            return {"name": "jobs/demo"}

    fake_client = SimpleNamespace(batches=FakeBatches())
    monkeypatch.setattr(batch_eval.vlm_config, "supports_batch_api", lambda: True)
    monkeypatch.setattr(batch_eval.vlm_config, "create_batch_client", lambda: fake_client)

    job = batch_eval.create_ap_batch_job(
        [
            batch_eval.APBatchCase(
                key="case-1",
                image_uri="gs://bucket/trace-1.jpg",
                prompt="Estimate AP position for this slice.",
            )
        ],
        model="gemini-3-flash-preview",
        dest_gcs_uri="gs://bucket/out/",
        display_name="langslice-ap-eval",
    )

    assert job == {"name": "jobs/demo"}
    assert captured["model"] == "gemini-3-flash-preview"
    config = captured["config"]
    assert getattr(config, "dest", None) == "gs://bucket/out/"
    assert getattr(config, "display_name", None) == "langslice-ap-eval"


def test_registration_count_tokens_runs_before_generate(monkeypatch) -> None:
    progress_messages: list[str] = []
    model_calls: list[str] = []

    class FakeModels:
        def count_tokens(
            self, *, model: str, contents: object, config: object | None = None
        ) -> object:
            _ = model, contents, config
            model_calls.append("count_tokens")
            return SimpleNamespace(total_tokens=123, total_billable_characters=456)

        def generate_content(self, *, model: str, contents: object, config: object) -> object:
            _ = model, contents, config
            model_calls.append("generate_content")
            return SimpleNamespace(
                parsed={
                    "correspondences": [
                        {
                            "atlas_point_2d": [int(80 + idx * 70), int(50 + idx * 75)],
                            "slice_point_2d": [int(100 + idx * 65), int(60 + idx * 70)],
                            "label": f"pt-{idx}",
                            "status": "found",
                        }
                        for idx in range(8)
                    ]
                }
            )

    fake_module = SimpleNamespace(
        MODEL_NAME="gemini-3-flash-preview",
        THINKING_LEVEL="HIGH",
        TEMPERATURE=0.5,
        count_tokens_enabled=lambda: True,
        get_client=lambda: SimpleNamespace(models=FakeModels()),
    )

    monkeypatch.setattr(
        registration_agents.importlib,
        "import_module",
        lambda name: fake_module if name == "langslice.vlm.config" else None,
    )
    atlas = SimpleNamespace(
        atlas_name="allen_mouse_25um",
        orientation="asr",
        reference=np.ones((1, 100, 120), dtype=np.uint8),
        annotation=np.ones((1, 100, 120), dtype=np.int32),
        resolution=(25.0, 25.0, 25.0),
        metadata={},
        structures={1: {"acronym": "CTX", "name": "Cortex"}},
    )
    monkeypatch.setattr(registration_agents, "load_atlas", lambda atlas_name: atlas)
    monkeypatch.setattr(
        registration_agents,
        "get_atlas_info",
        lambda atlas: {"shape": (1, 2, 3), "resolution_um": (25.0, 25.0, 25.0)},
    )
    monkeypatch.setattr(
        registration_agents,
        "get_slice_region_metadata",
        lambda atlas, position_mm: [
            {
                "acronym": "CTX",
                "name": "Cortex",
                "centroid_normalized": (500, 500),
                "area_fraction": 0.5,
            }
        ],
    )
    monkeypatch.setattr(
        registration_agents,
        "get_composite_slice",
        lambda atlas, position_mm: Image.new("RGB", (120, 100), "white"),
    )

    result = registration_agents.estimate_registration_correspondences(
        Image.new("RGB", (140, 120), "black"),
        atlas_name="allen_mouse_25um",
        position_mm=1.5,
        on_progress=progress_messages.append,
    )

    assert len(result) == 8
    assert model_calls == ["count_tokens", "generate_content"]
    assert any("Registration token preflight" in message for message in progress_messages)
