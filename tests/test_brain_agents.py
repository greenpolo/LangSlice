"""Tests for async agent wrappers with mocked estimators."""

import asyncio
from unittest.mock import patch

from PIL import Image

from langslice.ai.estimator import APResult
from langslice.brain.agents import run_anchor_estimation, run_refinement

_FAKE_IMAGE = Image.new("RGB", (64, 64), (128, 128, 128))


def test_run_anchor_estimation():
    """Anchor estimation calls estimate_position then nano-banana."""
    coarse_result = APResult(position_mm=3.45, reasoning="coarse", debug_dir=None)
    fine_result = APResult(position_mm=3.42, reasoning="fine", debug_dir=None)

    call_log = []

    def fake_estimate(image, atlas_name, **kwargs):
        call_log.append("coarse")
        return coarse_result

    def fake_image_gen(image, atlas_name, **kwargs):
        call_log.append("fine")
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


def test_run_refinement():
    """Refinement calls nano-banana with window-constrained positions."""
    fine_result = APResult(position_mm=2.55, reasoning="refined", debug_dir=None)

    def fake_image_gen(image, atlas_name, **kwargs):
        return fine_result

    with (
        patch("langslice.brain.agents.estimate_position_image_gen", fake_image_gen),
        patch("langslice.brain.agents._prepare_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_refinement(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
                window_lo=2.3,
                window_hi=2.7,
                window_center=2.5,
                n_images=8,
            )
        )

    assert result is not None
    assert result.position_mm == 2.55


def test_run_refinement_returns_none_on_skip():
    """When n_images=0, returns None without calling any API."""
    result = asyncio.run(
        run_refinement(
            image_path="/fake/slice.tif",
            atlas_name="allen_mouse_25um",
            window_lo=2.48,
            window_hi=2.52,
            window_center=2.5,
            n_images=0,
        )
    )
    assert result is None
