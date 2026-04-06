"""Tests for async agent wrappers with mocked estimators."""

import asyncio
from unittest.mock import patch

from PIL import Image

from langslice.ai.estimator import APResult
from langslice.brain.agents import run_anchor_estimation, run_slice_estimation

_FAKE_IMAGE = Image.new("RGB", (64, 64), (128, 128, 128))


def test_run_anchor_estimation():
    """Anchor estimation runs coarse tool-use then nano-banana fine pass."""
    coarse_result = APResult(position_mm=3.45, reasoning="coarse", debug_dir=None)
    fine_result = APResult(position_mm=3.42, reasoning="fine", debug_dir=None)

    call_log = []
    captured_kwargs: dict = {}

    def fake_estimate(image, atlas_name, **kwargs):
        call_log.append("coarse")
        return coarse_result

    def fake_image_gen(image, atlas_name, **kwargs):
        call_log.append("fine")
        captured_kwargs.update(kwargs)
        return fine_result

    with (
        patch("langslice.brain.agents.estimate_position", fake_estimate),
        patch("langslice.brain.agents.estimate_position_image_gen", fake_image_gen),
        patch("langslice.brain.agents._prepare_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_anchor_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )

    assert result.position_mm == 3.42
    assert call_log == ["coarse", "fine"]
    assert captured_kwargs["center_mm"] == 3.45


def test_run_slice_estimation():
    """Non-anchor estimation uses center_mm and bounds."""
    result_obj = APResult(position_mm=5.5, reasoning="centered", debug_dir=None)

    captured_kwargs: dict = {}

    def fake_image_gen(image, atlas_name, **kwargs):
        captured_kwargs.update(kwargs)
        return result_obj

    with (
        patch("langslice.brain.agents.estimate_position_image_gen", fake_image_gen),
        patch("langslice.brain.agents._prepare_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_slice_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
                center_mm=5.0,
                window_half_mm=2.0,
            )
        )

    assert result.position_mm == 5.5
    assert captured_kwargs["center_mm"] == 5.0
    assert captured_kwargs["bounds"] == (3.0, 7.0)
