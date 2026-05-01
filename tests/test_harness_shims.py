"""Wiring tests for synchronous estimation compat shims.

These tests confirm the shims:
- Exist at both the harness and legacy import paths.
- Convert legacy units/kwargs into the ADK runner call shape.

The runner is monkeypatched; this is pure wiring. End-to-end paths live in
``tests/test_harness_runner.py``.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from langslice_harness.harness.estimation._types import MultiSliceResult, PositionResult


def _fake_run_single_slice_session_factory(captured: dict[str, Any]):
    async def _fake(**kwargs: Any) -> PositionResult:
        captured.update(kwargs)
        return PositionResult(position_mm=6.0, reasoning="fake shim test")

    return _fake


def _fake_run_group_session_factory(captured: dict[str, Any]):
    async def _fake(**kwargs: Any) -> MultiSliceResult:
        captured.update(kwargs)
        n = len(kwargs["images"])
        return MultiSliceResult(
            positions=[PositionResult(position_mm=float(i), reasoning="fake") for i in range(n)],
            group_reasoning="fake shim test",
        )

    return _fake


# --- estimate_position shim ---


def test_estimate_position_exposed_on_legacy_shim():
    """The legacy ``langslice_harness.estimation`` re-exports the harness shim."""
    from langslice_harness.estimation import estimate_position as legacy
    from langslice_harness.harness.estimation import estimate_position as harness

    assert legacy is harness


def test_estimate_position_forwards_supported_legacy_kwargs(monkeypatch):
    """Legacy single-slice kwargs should reach the ADK runner when supported."""
    from langslice_harness.harness.estimation import estimate_position

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice_harness.harness.estimation.runner.run_single_slice_session",
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
    from langslice_harness import vlm_config
    from langslice_harness.harness.estimation import estimate_position

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice_harness.harness.estimation.runner.run_single_slice_session",
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


# --- estimate_group shim ---


def test_estimate_group_exposed_on_legacy_shim():
    """The legacy ``langslice_harness.estimation`` re-exports the harness shim."""
    from langslice_harness.estimation import estimate_group as legacy
    from langslice_harness.harness.estimation import estimate_group as harness

    # Same underlying function object — re-export, not a wrapper.
    assert legacy is harness


def test_estimate_group_converts_interval_um_to_mm(monkeypatch):
    """Legacy micron interval → runner millimetre interval."""
    from langslice_harness.harness.estimation import estimate_group

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice_harness.harness.estimation.runner.run_group_session",
        _fake_run_group_session_factory(captured),
    )

    images = [Image.new("L", (16, 16), color=128) for _ in range(3)]
    result = estimate_group(
        images=images,
        atlas_name="allen_mouse_25um",
        interval_um=200,
        thickness_um=50,
    )
    assert isinstance(result, MultiSliceResult)
    assert captured["interval_mm"] == 0.200
    assert captured["atlas_name"] == "allen_mouse_25um"
    assert captured["thickness_um"] == 50
    assert captured["plane"] == "coronal"
    # model_name=None should NOT be forwarded — runner owns its own default.
    assert "model" not in captured


def test_estimate_group_forwards_model_name_as_model(monkeypatch):
    """Legacy ``model_name`` kwarg → runner ``model`` kwarg."""
    from langslice_harness.harness.estimation import estimate_group

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice_harness.harness.estimation.runner.run_group_session",
        _fake_run_group_session_factory(captured),
    )

    images = [Image.new("L", (16, 16), color=128) for _ in range(2)]
    estimate_group(
        images=images,
        atlas_name="allen_mouse_25um",
        interval_um=150,
        thickness_um=50,
        model_name="gemini-3-flash-preview",
        max_iterations=12,
    )
    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["max_iterations"] == 12
    assert captured["interval_mm"] == 0.150


def test_estimate_group_forwards_supported_legacy_kwargs(monkeypatch):
    """Legacy kwargs that the ADK runner understands are preserved."""
    from langslice_harness.harness.estimation import estimate_group

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice_harness.harness.estimation.runner.run_group_session",
        _fake_run_group_session_factory(captured),
    )

    images = [Image.new("L", (16, 16), color=128) for _ in range(4)]

    def dummy_progress(msg: str) -> None:
        pass

    result = estimate_group(
        images=images,
        atlas_name="allen_mouse_25um",
        interval_um=250,
        thickness_um=50,
        model_name=None,
        max_iterations=25,
        send_individually=True,
        on_progress=dummy_progress,
        media_resolution="medium",
        show_borders=False,
        debug_dir="/tmp/does-not-exist",
        some_future_kwarg="whatever",
    )
    assert isinstance(result, MultiSliceResult)
    assert captured["media_resolution"] == "medium"

    for key in (
        "send_individually",
        "on_progress",
        "show_borders",
        "debug_dir",
        "some_future_kwarg",
        "model_name",
    ):
        assert key not in captured, f"{key} unexpectedly forwarded to runner"


def test_estimate_group_positional_interval_um(monkeypatch):
    """Legacy positional call shape still works: images, atlas_name, interval_um."""
    from langslice_harness.harness.estimation import estimate_group

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice_harness.harness.estimation.runner.run_group_session",
        _fake_run_group_session_factory(captured),
    )

    images = [Image.new("L", (16, 16), color=128) for _ in range(2)]
    estimate_group(images, "allen_mouse_25um", 200)
    assert captured["interval_mm"] == 0.200
    assert captured["atlas_name"] == "allen_mouse_25um"

