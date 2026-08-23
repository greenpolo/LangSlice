"""Tests for async agent wrappers with mocked tool-use estimator."""

import asyncio
from unittest.mock import patch

from PIL import Image

from langslice.linear import APResult
from langslice.linear.whole_brain.estimation_agents import (
    run_anchor_estimation,
    run_slice_estimation,
)

_FAKE_IMAGE = Image.new("RGB", (64, 64), (128, 128, 128))
_ESTIMATION_AGENTS = "langslice.linear.whole_brain.estimation_agents"


def test_run_anchor_estimation_default_model_none():
    """Without an explicit model, the estimator receives None (its own default)."""
    anchor_result = APResult(position_mm=3.45, reasoning="anchor", debug_dir=None)

    captured_kwargs_list: list[dict] = []

    def fake_estimate_position(image, atlas_name, **kwargs):
        captured_kwargs_list.append(dict(kwargs))
        return anchor_result

    with (
        patch(
            f"{_ESTIMATION_AGENTS}.estimate_position",
            fake_estimate_position,
        ),
        patch(f"{_ESTIMATION_AGENTS}._load_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_anchor_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )

    assert result.position_mm == 3.45
    assert len(captured_kwargs_list) == 1
    assert captured_kwargs_list[0]["model_name"] is None


def test_run_anchor_estimation_honors_explicit_model():
    """An explicit model is forwarded verbatim."""
    anchor_result = APResult(position_mm=2.0, reasoning="anchor", debug_dir=None)

    captured_kwargs_list: list[dict] = []

    def fake_estimate_position(image, atlas_name, **kwargs):
        captured_kwargs_list.append(dict(kwargs))
        return anchor_result

    with (
        patch(
            f"{_ESTIMATION_AGENTS}.estimate_position",
            fake_estimate_position,
        ),
        patch(f"{_ESTIMATION_AGENTS}._load_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_anchor_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
                model_name="custom-model",
            )
        )

    assert result.position_mm == 2.0
    assert captured_kwargs_list[0]["model_name"] == "custom-model"


def test_run_anchor_estimation_midpoint_fallback():
    """Estimator failure falls back to the atlas midpoint."""

    def fake_estimate_position_fail(image, atlas_name, **kwargs):
        raise RuntimeError("API quota exhausted")

    with (
        patch(
            f"{_ESTIMATION_AGENTS}.estimate_position",
            fake_estimate_position_fail,
        ),
        patch(f"{_ESTIMATION_AGENTS}._load_slice", return_value=_FAKE_IMAGE),
        patch(f"{_ESTIMATION_AGENTS}.load_atlas"),
        patch(
            f"{_ESTIMATION_AGENTS}.get_position_range_mm",
            return_value=(0.0, 13.175),
        ),
    ):
        result = asyncio.run(
            run_anchor_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )

    # Midpoint of (0.0 + 13.175) / 2 = 6.5875
    assert abs(result.position_mm - 6.5875) < 0.01
    assert "fell back" in result.reasoning.lower()


def test_run_slice_estimation_forwards_model_name():
    """Non-anchor estimation forwards model_name to the tool-use estimator."""
    slice_result = APResult(position_mm=5.5, reasoning="slice", debug_dir=None)

    captured_kwargs_list: list[dict] = []

    def fake_estimate_position(image, atlas_name, **kwargs):
        captured_kwargs_list.append(dict(kwargs))
        return slice_result

    with (
        patch(
            f"{_ESTIMATION_AGENTS}.estimate_position",
            fake_estimate_position,
        ),
        patch(f"{_ESTIMATION_AGENTS}._load_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_slice_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
                model_name="flash-class-model",
            )
        )

    assert result.position_mm == 5.5
    assert len(captured_kwargs_list) == 1
    assert captured_kwargs_list[0]["model_name"] == "flash-class-model"


def test_run_slice_estimation_default_model_name_none():
    """Without an explicit model_name the estimator receives None (its own default)."""
    slice_result = APResult(position_mm=7.25, reasoning="slice", debug_dir=None)

    captured_kwargs_list: list[dict] = []

    def fake_estimate_position(image, atlas_name, **kwargs):
        captured_kwargs_list.append(dict(kwargs))
        return slice_result

    with (
        patch(
            f"{_ESTIMATION_AGENTS}.estimate_position",
            fake_estimate_position,
        ),
        patch(f"{_ESTIMATION_AGENTS}._load_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_slice_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )

    assert result.position_mm == 7.25
    assert captured_kwargs_list[0]["model_name"] is None
