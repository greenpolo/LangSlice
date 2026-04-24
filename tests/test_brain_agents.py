"""Tests for async agent wrappers with mocked estimators."""

import asyncio
from unittest.mock import patch

from PIL import Image

from langslice_harness.estimation import APResult
from langslice_harness.whole_brain.estimation_agents import (
    run_anchor_estimation,
    run_slice_estimation,
)

_FAKE_IMAGE = Image.new("RGB", (64, 64), (128, 128, 128))
_ESTIMATION_AGENTS = "langslice_harness.whole_brain.estimation_agents"


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
            f"{_ESTIMATION_AGENTS}.estimate_position_image_gen",
            fake_image_gen,
        ),
        patch(f"{_ESTIMATION_AGENTS}._prepare_slice", return_value=_FAKE_IMAGE),
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
    assert captured_kwargs_list[0].get("center_mm") is None
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
            f"{_ESTIMATION_AGENTS}.estimate_position_image_gen",
            fake_image_gen_fail_first,
        ),
        patch(f"{_ESTIMATION_AGENTS}._prepare_slice", return_value=_FAKE_IMAGE),
        patch(f"{_ESTIMATION_AGENTS}.load_atlas"),
        patch(f"{_ESTIMATION_AGENTS}.get_coronal_long_edge", return_value=528),
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

    assert result.position_mm == 6.5
    assert call_log == ["coarse_fail", "fine"]


def test_run_anchor_estimation_both_stages_fail():
    """Both stages fail returns midpoint APResult."""

    def fake_image_gen_always_fail(image, atlas_name, **kwargs):
        raise RuntimeError("API quota exhausted")

    with (
        patch(
            f"{_ESTIMATION_AGENTS}.estimate_position_image_gen",
            fake_image_gen_always_fail,
        ),
        patch(f"{_ESTIMATION_AGENTS}._prepare_slice", return_value=_FAKE_IMAGE),
        patch(f"{_ESTIMATION_AGENTS}.load_atlas"),
        patch(f"{_ESTIMATION_AGENTS}.get_coronal_long_edge", return_value=528),
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


def test_run_slice_estimation():
    """Non-anchor estimation uses center_mm, bounds, confirmation pass."""
    main_result = APResult(position_mm=5.5, reasoning="main", debug_dir=None)
    confirm_result = APResult(position_mm=5.48, reasoning="confirmed", debug_dir=None)

    call_log: list[str] = []
    captured_kwargs_list: list[dict] = []

    def fake_image_gen(image, atlas_name, **kwargs):
        captured_kwargs_list.append(dict(kwargs))
        if kwargs.get("max_passes") == 1:
            call_log.append("confirm")
            return confirm_result
        call_log.append("main")
        return main_result

    with (
        patch(
            f"{_ESTIMATION_AGENTS}.estimate_position_image_gen",
            fake_image_gen,
        ),
        patch(f"{_ESTIMATION_AGENTS}._prepare_slice", return_value=_FAKE_IMAGE),
    ):
        result = asyncio.run(
            run_slice_estimation(
                image_path="/fake/slice.tif",
                atlas_name="allen_mouse_25um",
                center_mm=5.0,
                window_half_mm=2.0,
            )
        )

    # Final result comes from confirmation pass
    assert result.position_mm == 5.48
    assert call_log == ["main", "confirm"]
    # Main pass: center_mm=5.0, bounds=(3.0, 7.0), fine_resolution_mm=0.10
    assert captured_kwargs_list[0]["center_mm"] == 5.0
    assert captured_kwargs_list[0]["bounds"] == (3.0, 7.0)
    assert captured_kwargs_list[0]["fine_resolution_mm"] == 0.10
    # Confirmation pass: center_mm=5.5 (main result), ±0.25mm bounds, 9 slices, max_passes=1
    assert captured_kwargs_list[1]["center_mm"] == 5.5
    assert captured_kwargs_list[1]["bounds"] == (5.25, 5.75)
    assert captured_kwargs_list[1]["slices_per_pass"] == 9
    assert captured_kwargs_list[1]["max_passes"] == 1


def test_run_slice_estimation_confirm_fallback():
    """If the confirmation pass fails, the main result is returned."""
    main_result = APResult(position_mm=5.5, reasoning="main", debug_dir=None)

    call_count = 0

    def fake_image_gen(image, atlas_name, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("max_passes") == 1:
            raise RuntimeError("API error")
        return main_result

    with (
        patch(
            f"{_ESTIMATION_AGENTS}.estimate_position_image_gen",
            fake_image_gen,
        ),
        patch(f"{_ESTIMATION_AGENTS}._prepare_slice", return_value=_FAKE_IMAGE),
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
    assert call_count == 2
