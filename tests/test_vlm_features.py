"""Checks for Gemini feature-flag integrations."""

from __future__ import annotations

import langslice_harness.vlm_config as vlm_config


def test_feature_flags_for_ai_studio(monkeypatch) -> None:
    monkeypatch.setenv("LANGSLICE_GENAI_BACKEND", "ai_studio")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LANGSLICE_GENAI_COUNT_TOKENS", "1")
    monkeypatch.setenv("LANGSLICE_GENAI_AP_USE_FILE_API", "1")
    monkeypatch.setenv("LANGSLICE_GENAI_AP_USE_CONTEXT_CACHE", "1")

    flags = vlm_config.feature_flags()
    assert flags["count_tokens_enabled"] is True
    assert flags["ap_use_file_api"] is True
    assert flags["ap_use_context_cache"] is True
    assert flags["supports_file_api"] is True
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


def test_image_generation_model_detection() -> None:
    assert vlm_config.is_image_generation_model("gemini-3-pro-image-preview") is True
    assert vlm_config.is_image_generation_model("gemini-3-flash-preview") is False


def test_set_thinking_level_updates_runtime_value() -> None:
    original = vlm_config.THINKING_LEVEL
    try:
        vlm_config.set_thinking_level("low")
        assert vlm_config.THINKING_LEVEL == "LOW"
        vlm_config.set_thinking_level("HIGH")
        assert vlm_config.THINKING_LEVEL == "HIGH"
    finally:
        vlm_config.set_thinking_level(original)


def test_is_gemma_model_detects_gemma_4():
    from langslice_harness.vlm_config import is_gemma_model

    assert is_gemma_model("gemma-4-31b-it") is True
    assert is_gemma_model("gemma-4-26b-a4b-it") is True
    assert is_gemma_model("models/gemma-4-26b-a4b-it") is True
    assert is_gemma_model("gemini-3-flash-preview") is False
    assert is_gemma_model(None) is False


def test_build_thinking_config_gemma_maps_to_high_or_none():
    from langslice_harness.vlm_config import build_thinking_config

    cfg = build_thinking_config("gemma-4-31b-it", "HIGH")
    assert cfg is not None
    cfg = build_thinking_config("gemma-4-31b-it", "MEDIUM")
    assert cfg is not None
    cfg = build_thinking_config("gemma-4-31b-it", "LOW")
    assert cfg is None
    cfg = build_thinking_config("gemma-4-31b-it", "MINIMAL")
    assert cfg is None


def test_build_thinking_config_gemini_passes_through():
    from langslice_harness.vlm_config import build_thinking_config

    cfg = build_thinking_config("gemini-3-flash-preview", "LOW")
    assert cfg is not None
