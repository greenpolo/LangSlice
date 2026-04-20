"""Wiring tests for the synchronous ``estimate_group`` compat shim.

These tests confirm the shim:
  1. Exists at both the harness and legacy import paths.
  2. Translates ``interval_um`` to ``interval_mm``.
  3. Forwards ``model_name`` as ``model``, falling through to the runner default
     when ``model_name`` is ``None``.
  4. Accepts (and silently drops) the legacy kwargs that current callers pass
     — ``send_individually``, ``on_progress``, ``media_resolution``,
     ``show_borders``, ``debug_dir``, and any unknown extras.

The runner is monkeypatched; this is pure wiring.  The end-to-end paths live in
``tests/test_harness_runner.py``.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from langslice.harness.estimation._types import MultiSliceResult, PositionResult


def _fake_run_group_session_factory(captured: dict[str, Any]):
    async def _fake(**kwargs: Any) -> MultiSliceResult:
        captured.update(kwargs)
        n = len(kwargs["images"])
        return MultiSliceResult(
            positions=[
                PositionResult(position_mm=float(i), reasoning="fake")
                for i in range(n)
            ],
            group_reasoning="fake shim test",
        )

    return _fake


def test_estimate_group_exposed_on_harness():
    """The sync shim is importable from the harness public API."""
    from langslice.harness.estimation import estimate_group

    assert callable(estimate_group)


def test_estimate_group_exposed_on_legacy_shim():
    """The legacy ``langslice.estimation`` re-exports the harness shim."""
    from langslice.estimation import estimate_group as legacy
    from langslice.harness.estimation import estimate_group as harness

    # Same underlying function object — re-export, not a wrapper.
    assert legacy is harness


def test_estimate_group_converts_interval_um_to_mm(monkeypatch):
    """Legacy micron interval → runner millimetre interval."""
    from langslice.harness.estimation import estimate_group

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice.harness.estimation.runner.run_group_session",
        _fake_run_group_session_factory(captured),
    )
    # The shim imports run_group_session inside the function body, so the
    # monkeypatch hits it the first time the shim is called.

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
    from langslice.harness.estimation import estimate_group

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice.harness.estimation.runner.run_group_session",
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


def test_estimate_group_swallows_legacy_kwargs(monkeypatch):
    """Legacy kwargs that the ADK runner doesn't know about must not explode."""
    from langslice.harness.estimation import estimate_group

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice.harness.estimation.runner.run_group_session",
        _fake_run_group_session_factory(captured),
    )

    images = [Image.new("L", (16, 16), color=128) for _ in range(4)]

    def dummy_progress(msg: str) -> None:
        pass

    # All of these are passed by eval_group.py and/or cli.py today.
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
        # And something completely unknown — must fall into **_ignored silently.
        some_future_kwarg="whatever",
    )
    assert isinstance(result, MultiSliceResult)
    # None of the ignored kwargs leaked to the runner.
    for key in (
        "send_individually",
        "on_progress",
        "media_resolution",
        "show_borders",
        "debug_dir",
        "some_future_kwarg",
        "model_name",
    ):
        assert key not in captured, f"{key} unexpectedly forwarded to runner"


def test_estimate_group_positional_interval_um(monkeypatch):
    """Legacy positional call shape still works: images, atlas_name, interval_um."""
    from langslice.harness.estimation import estimate_group

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "langslice.harness.estimation.runner.run_group_session",
        _fake_run_group_session_factory(captured),
    )

    images = [Image.new("L", (16, 16), color=128) for _ in range(2)]
    estimate_group(images, "allen_mouse_25um", 200)
    assert captured["interval_mm"] == 0.200
    assert captured["atlas_name"] == "allen_mouse_25um"
