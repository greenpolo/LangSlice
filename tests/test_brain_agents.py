"""Tests for async agent wrappers with mocked estimators."""

import asyncio
from unittest.mock import patch

from PIL import Image

from langslice.estimation import APResult
from langslice.whole_brain.estimation_agents import run_anchor_estimation, run_slice_estimation

_FAKE_IMAGE = Image.new("RGB", (64, 64), (128, 128, 128))


def test_run_anchor_estimation():
    """Anchor estimation runs image-gen 2-pass coarse then fine pass."""
    coarse_result = APResult(position_mm=3.45, reasoning="coarse", debug_dir=None)
    fine_result = APResult(position_mm=3.42, reasoning="fine", debug_dir=None)

    call_log: list[str] = []
    captured_kwargs_list: list[dict] = []

    def fake_image_gen(image, atlas_name, **kwargs):
        captured_kwargs_list.append(dict(kwargs))
        if kwargs.get("max_passes") == 2:
            call_log.append("coarse")
            return coarse_result
        call_log.append("fine")
        return fine_result

    with (
        patch(
            "langslice.whole_brain.estimation_agents.estimate_position_image_gen",
            fake_image_gen,
        ),
        patch("langslice.whole_brain.estimation_agents._prepare_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_anchor_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )

    assert result.position_mm == 3.42
    assert call_log == ["coarse", "fine"]
    # Stage A (coarse): image-gen 2-pass, no center_mm, no bounds
    assert captured_kwargs_list[0]["max_passes"] == 2
    assert "center_mm" not in captured_kwargs_list[0] or captured_kwargs_list[0].get("center_mm") is None
    # Stage B (fine): centered on coarse, ±0.5mm bounds
    assert captured_kwargs_list[1]["center_mm"] == 3.45
    assert captured_kwargs_list[1]["bounds"] == (2.95, 3.95)


def test_run_anchor_estimation_midpoint_fallback():
    """Stage A failure falls back to atlas midpoint, Stage B still runs."""
    fine_result = APResult(position_mm=6.5, reasoning="fine", debug_dir=None)

    call_log: list[str] = []

    def fake_image_gen_fail_first(image, atlas_name, **kwargs):
        if kwargs.get("max_passes") == 2:
            call_log.append("coarse_fail")
            raise RuntimeError("API quota exhausted")
        call_log.append("fine")
        return fine_result

    with (
        patch(
            "langslice.whole_brain.estimation_agents.estimate_position_image_gen",
            fake_image_gen_fail_first,
        ),
        patch("langslice.whole_brain.estimation_agents._prepare_slice", return_value=_FAKE_IMAGE),
        patch("langslice.whole_brain.estimation_agents.load_atlas") as mock_atlas,
        patch("langslice.whole_brain.estimation_agents.get_position_range_mm", return_value=(0.0, 13.175)),
    ):
        result = asyncio.run(
            run_anchor_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )

    assert result.position_mm == 6.5
    assert call_log == ["coarse_fail", "fine"]


def test_run_anchor_estimation_both_stages_fail():
    """Both stages fail returns midpoint APResult."""

    def fake_image_gen_always_fail(image, atlas_name, **kwargs):
        raise RuntimeError("API quota exhausted")

    with (
        patch(
            "langslice.whole_brain.estimation_agents.estimate_position_image_gen",
            fake_image_gen_always_fail,
        ),
        patch("langslice.whole_brain.estimation_agents._prepare_slice", return_value=_FAKE_IMAGE),
        patch("langslice.whole_brain.estimation_agents.load_atlas"),
        patch("langslice.whole_brain.estimation_agents.get_position_range_mm", return_value=(0.0, 13.175)),
    ):
        result = asyncio.run(
            run_anchor_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
            )
        )

    # Midpoint of (0.0 + 13.175) / 2 = 6.5875
    assert abs(result.position_mm - 6.5875) < 0.01


def test_run_slice_estimation():
    """Non-anchor estimation uses center_mm, bounds, and fine_resolution_mm=0.10."""
    result_obj = APResult(position_mm=5.5, reasoning="centered", debug_dir=None)

    captured_kwargs: dict = {}

    def fake_image_gen(image, atlas_name, **kwargs):
        captured_kwargs.update(kwargs)
        return result_obj

    with (
        patch(
            "langslice.whole_brain.estimation_agents.estimate_position_image_gen",
            fake_image_gen,
        ),
        patch("langslice.whole_brain.estimation_agents._prepare_slice", return_value=_FAKE_IMAGE),
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
    assert captured_kwargs["fine_resolution_mm"] == 0.10
