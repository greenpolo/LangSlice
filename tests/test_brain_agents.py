"""Single-slice escalation worker, with the tool-use estimator mocked."""

import asyncio
from unittest.mock import patch

import pytest
from PIL import Image

from langslice.linear import APResult
from langslice.linear.whole_brain.estimation_agents import run_slice_estimation

_FAKE_IMAGE = Image.new("RGB", (64, 64), (128, 128, 128))
_ESTIMATION_AGENTS = "langslice.linear.whole_brain.estimation_agents"


def test_run_slice_estimation_forwards_model_name():
    slice_result = APResult(position_mm=5.5, reasoning="slice", debug_dir=None)
    captured_kwargs_list: list[dict] = []

    def fake_estimate_position(image, atlas_name, **kwargs):
        captured_kwargs_list.append(dict(kwargs))
        return slice_result

    with (
        patch(f"{_ESTIMATION_AGENTS}.estimate_position", fake_estimate_position),
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
        patch(f"{_ESTIMATION_AGENTS}.estimate_position", fake_estimate_position),
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


def test_run_slice_estimation_propagates_failures():
    """No midpoint fallback: the positioning step decides what a failure means."""

    def fake_estimate_position_fail(image, atlas_name, **kwargs):
        raise RuntimeError("API quota exhausted")

    with (
        patch(f"{_ESTIMATION_AGENTS}.estimate_position", fake_estimate_position_fail),
        patch(f"{_ESTIMATION_AGENTS}._load_slice", return_value=_FAKE_IMAGE),
        pytest.raises(RuntimeError, match="API quota exhausted"),
    ):
        asyncio.run(
            run_slice_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )
