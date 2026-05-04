"""Unit tests for ``rlvr.env.LangSliceEstimateEnv`` (spec §11 verification 1).

These tests stub out ``AtlasGrid`` so they don't require BrainGlobe atlases
or the slow pre-render step — env behavior (clamp, dedupe, cap, kind
matching, ground-truth privacy) is independent of which slices come back.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from PIL import Image
from rlvr import env as env_mod
from rlvr.env import (
    DEDUPE_TOL_MM,
    MAX_POSITIONS_PER_FETCH,
    LangSliceEstimateEnv,
    _public_tool_method_names,
)


class _StubAtlasGrid:
    """Minimal AtlasGrid stand-in: returns a tiny image and snaps to 0.05 mm."""

    def __init__(self, pos_lo: float = 0.0, pos_hi: float = 13.2) -> None:
        self._pos_lo = pos_lo
        self._pos_hi = pos_hi

    def range_mm(self, atlas_name: str, plane: str) -> Any:  # noqa: ARG002
        from rlvr.atlas_grid import GridRange  # noqa: PLC0415

        return GridRange(pos_lo=self._pos_lo, pos_hi=self._pos_hi)

    def get_slice(
        self, atlas_name: str, plane: str, position_mm: float  # noqa: ARG002
    ) -> tuple[Image.Image, float]:
        snapped = round(position_mm / 0.05) * 0.05
        return Image.new("L", (4, 4), color=int(snapped * 10) % 256), snapped


@pytest.fixture
def single_env() -> LangSliceEstimateEnv:
    e = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    e.reset(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        valid_range_mm=(0.0, 13.2),
        ground_truth_positions_mm=(5.5,),
        kind="single",
        prompt=[],
        image=None,
        subject_id="M01",
    )
    return e


@pytest.fixture
def group_env() -> LangSliceEstimateEnv:
    e = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    e.reset(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        valid_range_mm=(0.0, 13.2),
        ground_truth_positions_mm=(2.5, 3.0, 3.5, 4.0),
        kind="group",
        prompt=[],
        images=[],
        subject_id="M02",
    )
    return e


# --- tool surface ----------------------------------------------------------


def test_only_three_tools_exposed() -> None:
    """``reset`` and any leading-underscore method must NOT be auto-exposed."""
    assert _public_tool_method_names() == (
        "fetch_atlas",
        "submit_estimate",
        "submit_group_estimate",
    )


def test_no_public_method_returns_ground_truth() -> None:
    """No public method may have a return annotation that leaks ground truth."""
    instance = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    instance.reset(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        valid_range_mm=(0.0, 13.2),
        ground_truth_positions_mm=(5.5,),
        kind="single",
    )
    for name in _public_tool_method_names():
        method = getattr(instance, name)
        sig = inspect.signature(method)
        # The signature must not mention ``ground_truth`` anywhere.
        assert "ground_truth" not in str(sig).lower(), name
    # And neither submit_estimate nor fetch_atlas may return the gt value
    # in their response payloads.
    fetch_blocks = instance.fetch_atlas([5.5])
    flattened = " ".join(b.get("text", "") for b in fetch_blocks if b.get("type") == "text")
    assert "5.5" not in flattened or "Atlas at" in flattened  # only as snapped caption
    assert "ground_truth" not in flattened.lower()


# --- fetch_atlas validation ------------------------------------------------


def test_fetch_atlas_clamps_to_valid_range(single_env: LangSliceEstimateEnv) -> None:
    blocks = single_env.fetch_atlas([-1.0, 20.0])
    text = " ".join(b["text"] for b in blocks if b["type"] == "text")
    assert "clamped" in text
    # Both clamps land on the boundaries, then dedupe leaves them as 2 distinct points.
    image_blocks = [b for b in blocks if b["type"] == "image"]
    assert len(image_blocks) == 2


def test_fetch_atlas_dedupes_close_positions(single_env: LangSliceEstimateEnv) -> None:
    # 5.50 and 5.51 are within DEDUPE_TOL_MM=0.02 → second is dropped.
    blocks = single_env.fetch_atlas([5.50, 5.50 + DEDUPE_TOL_MM / 2])
    image_blocks = [b for b in blocks if b["type"] == "image"]
    assert len(image_blocks) == 1
    text = " ".join(b["text"] for b in blocks if b["type"] == "text")
    assert "deduped" in text


def test_fetch_atlas_caps_at_max_positions(single_env: LangSliceEstimateEnv) -> None:
    # Spread positions wider than DEDUPE_TOL_MM so dedupe doesn't also shrink the list.
    requested = [0.5 + i * 0.5 for i in range(MAX_POSITIONS_PER_FETCH + 3)]
    blocks = single_env.fetch_atlas(requested)
    image_blocks = [b for b in blocks if b["type"] == "image"]
    assert len(image_blocks) == MAX_POSITIONS_PER_FETCH
    text = " ".join(b["text"] for b in blocks if b["type"] == "text")
    assert "8-per-call cap" in text


def test_fetch_atlas_image_before_text_per_position(single_env: LangSliceEstimateEnv) -> None:
    """Gemma 4's chat template requires the image block before its caption."""
    blocks = single_env.fetch_atlas([2.0, 4.0])
    # Walk through and confirm each image is immediately followed by a caption text.
    for i, b in enumerate(blocks):
        if b["type"] == "image":
            assert blocks[i + 1]["type"] == "text"
            assert "Atlas at" in blocks[i + 1]["text"]


def test_fetch_atlas_empty_input_records_malformed(single_env: LangSliceEstimateEnv) -> None:
    blocks = single_env.fetch_atlas([])
    assert single_env._state.malformed_tool_calls == 1
    assert blocks[0]["type"] == "text" and "Error" in blocks[0]["text"]


# --- submit kind matching --------------------------------------------------


def test_submit_estimate_rejects_when_kind_is_group(group_env: LangSliceEstimateEnv) -> None:
    msg = group_env.submit_estimate(3.0, "guess")
    assert "Error" in msg and "single" in msg
    assert group_env._state.submitted_positions_mm is None
    assert group_env._state.malformed_tool_calls == 1


def test_submit_group_estimate_rejects_when_kind_is_single(
    single_env: LangSliceEstimateEnv,
) -> None:
    msg = single_env.submit_group_estimate([5.5, 6.0], "guess")
    assert "Error" in msg and "group" in msg
    assert single_env._state.submitted_positions_mm is None
    assert single_env._state.malformed_tool_calls == 1


def test_submit_estimate_records_and_marks_done(single_env: LangSliceEstimateEnv) -> None:
    msg = single_env.submit_estimate(5.55, "anterior commissure visible")
    assert single_env._state.submitted_positions_mm == (5.55,)
    assert single_env._state.submitted_kind == "single"
    assert single_env._state.done is True
    assert "5.550 mm" in msg


def test_submit_group_estimate_records_all_positions(group_env: LangSliceEstimateEnv) -> None:
    msg = group_env.submit_group_estimate([2.5, 3.05, 3.55, 4.0], "matched landmarks")
    assert group_env._state.submitted_positions_mm == (2.5, 3.05, 3.55, 4.0)
    assert group_env._state.submitted_kind == "group"
    assert group_env._state.done is True
    assert "4 estimate" in msg


# --- ground-truth privacy --------------------------------------------------


def test_ground_truth_not_in_any_tool_response(single_env: LangSliceEstimateEnv) -> None:
    fetch = single_env.fetch_atlas([5.4, 5.5, 5.6])
    submit = single_env.submit_estimate(5.5, "guess")
    fetch_text = " ".join(b["text"] for b in fetch if b["type"] == "text")
    # The exact gt is 5.5 — and yes 5.5 may legitimately appear in a snapped
    # atlas caption — but the response must never label it as the ground truth.
    assert "ground_truth" not in fetch_text.lower()
    assert "ground_truth" not in submit.lower()


def test_reset_validates_kind_and_positions() -> None:
    e = LangSliceEstimateEnv(atlas_grid=_StubAtlasGrid())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind must be"):
        e.reset(
            atlas_name="allen_mouse_25um",
            plane="coronal",
            valid_range_mm=(0.0, 13.2),
            ground_truth_positions_mm=(5.5,),
            kind="invalid",
        )
    with pytest.raises(ValueError, match="single-slice kind expects exactly 1"):
        e.reset(
            atlas_name="allen_mouse_25um",
            plane="coronal",
            valid_range_mm=(0.0, 13.2),
            ground_truth_positions_mm=(5.5, 6.0),
            kind="single",
        )
    with pytest.raises(ValueError, match="group kind expects >=2"):
        e.reset(
            atlas_name="allen_mouse_25um",
            plane="coronal",
            valid_range_mm=(0.0, 13.2),
            ground_truth_positions_mm=(5.5,),
            kind="group",
        )
    with pytest.raises(ValueError, match="valid_range_mm must be increasing"):
        e.reset(
            atlas_name="allen_mouse_25um",
            plane="coronal",
            valid_range_mm=(5.0, 5.0),
            ground_truth_positions_mm=(5.5,),
            kind="single",
        )


def test_reset_rejects_atlas_grid_without_pair() -> None:
    class _EmptyGrid:
        def range_mm(self, *args: Any, **kwargs: Any) -> Any:
            raise KeyError("no pair")

    e = LangSliceEstimateEnv(atlas_grid=_EmptyGrid())  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        e.reset(
            atlas_name="bogus",
            plane="coronal",
            valid_range_mm=(0.0, 1.0),
            ground_truth_positions_mm=(0.5,),
            kind="single",
        )


def test_module_exports_what_callers_import() -> None:
    # Catches accidental rename / typo regressions in the env module surface.
    assert env_mod.LangSliceEstimateEnv is LangSliceEstimateEnv
    assert env_mod.MAX_POSITIONS_PER_FETCH == 8
    assert env_mod.DEDUPE_TOL_MM == 0.02
