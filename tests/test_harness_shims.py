"""Wiring tests for synchronous estimation compat shims.

These tests confirm the shims convert legacy units/kwargs into the ADK runner
call shape.

The runner is monkeypatched; this is pure wiring. End-to-end paths live in
``tests/test_harness_runner.py``.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from langslice.linear._types import PositionResult


def _fake_run_single_slice_session_factory(captured: dict[str, Any]):
    async def _fake(**kwargs: Any) -> PositionResult:
        captured.update(kwargs)
        return PositionResult(position_mm=6.0, reasoning="fake shim test")

    return _fake


# --- estimate_position shim ---


def test_estimate_position_forwards_supported_legacy_kwargs(monkeypatch):
    """Legacy single-slice kwargs should reach the ADK runner when supported."""
    from langslice.linear import estimate_position

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice.linear.runner.run_single_slice_session",
        _fake_run_single_slice_session_factory(captured),
    )

    image = Image.new("L", (16, 16), color=128)
    result = estimate_position(
        image=image,
        atlas_name="allen_mouse_25um",
        model_name="gemini-3-flash-preview",
        max_iterations=12,
        media_resolution="medium",
        thinking="LOW",
        temperature=0.7,
        apply_clahe=False,
        debug_dir="/tmp/does-not-exist",
        send_individually=True,
        show_borders=False,
        some_future_kwarg="ignored",
    )

    assert isinstance(result, PositionResult)
    assert captured["image"] is image
    assert captured["atlas_name"] == "allen_mouse_25um"
    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["max_iterations"] == 12
    assert captured["media_resolution"] == "medium"
    assert captured["thinking_level"] == "LOW"
    assert captured["temperature"] == 0.7
    assert captured["apply_clahe"] is False

    for key in (
        "model_name",
        "thinking",
        "debug_dir",
        "send_individually",
        "show_borders",
        "some_future_kwarg",
    ):
        assert key not in captured, f"{key} unexpectedly forwarded to runner"


def test_estimate_position_uses_vlm_config_defaults(monkeypatch):
    """CLI-set Gemini config should be honored even without explicit kwargs."""
    from langslice.linear import estimate_position
    from langslice.providers import vlm_config

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice.linear.runner.run_single_slice_session",
        _fake_run_single_slice_session_factory(captured),
    )
    original_model = vlm_config.MODEL_NAME
    original_thinking = vlm_config.THINKING_LEVEL
    original_temperature = vlm_config.TEMPERATURE

    try:
        vlm_config.set_model_name("gemini-configured")
        vlm_config.set_thinking_level("HIGH")
        vlm_config.set_temperature(0.4)

        estimate_position(Image.new("L", (16, 16), color=128), "allen_mouse_25um")
    finally:
        vlm_config.set_model_name(original_model)
        vlm_config.set_thinking_level(original_thinking)
        vlm_config.set_temperature(original_temperature)

    assert captured["model"] == "gemini-configured"
    assert captured["thinking_level"] == "HIGH"
    assert captured["temperature"] == 0.4
