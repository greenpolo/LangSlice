"""Tests for async agent wrappers with mocked estimators."""

import asyncio
from unittest.mock import patch

from PIL import Image

from langslice.ai.estimator import APResult
from langslice.brain.agents import run_anchor_estimation, run_slice_estimation

_FAKE_IMAGE = Image.new("RGB", (64, 64), (128, 128, 128))


def test_run_anchor_estimation():
    """Anchor estimation runs 3-pass nano-banana (no coarse stage)."""
    result_obj = APResult(position_mm=3.42, reasoning="nano-banana", debug_dir=None)

    def fake_image_gen(image, atlas_name, **kwargs):
        # Should NOT receive center_mm (full atlas range)
        assert "center_mm" not in kwargs or kwargs["center_mm"] is None
        return result_obj

    with (
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
