import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types
from PIL import Image

from langslice_harness.harness.estimation.adk_plugins import MULTIMODAL_PARTS_RESULT_KEY
from langslice_harness.harness.estimation.session import atlas_key, build_initial_state
from langslice_harness.harness.estimation.tools import (
    _clamp_and_dedupe_positions,
    _image_to_jpeg_bytes,
    _is_broad_sweep,
    _is_narrow_sweep,
    _parse_atlas_key,
    fetch_atlas,
    submit_estimate,
    submit_group_estimate,
)


def _image_part_bytes(part: types.Part) -> bytes:
    assert part.inline_data is not None and part.inline_data.data is not None
    return part.inline_data.data


def _is_image_part(part: object) -> bool:
    inline = getattr(part, "inline_data", None)
    return inline is not None and getattr(inline, "data", None) is not None


def _is_text_part(part: object) -> bool:
    text = getattr(part, "text", None)
    return isinstance(text, str) and text != ""


def _text_of(part: types.Part) -> str:
    """Narrow ``Part.text`` to ``str`` for assertions — basedpyright treats it as ``str | None``."""
    assert part.text is not None
    return part.text


def _multimodal_parts(result: object) -> list[types.Part]:
    assert isinstance(result, dict)
    parts = result[MULTIMODAL_PARTS_RESULT_KEY]
    assert isinstance(parts, list)
    assert all(isinstance(part, types.Part) for part in parts)
    return parts


def test_parse_atlas_key():
    assert _parse_atlas_key("atlas:3.20") == 3.20
    assert _parse_atlas_key("atlas:10.00") == 10.0
    with pytest.raises(ValueError):
        _parse_atlas_key("target")
    with pytest.raises(ValueError):
        _parse_atlas_key("atlas:not-a-number")


def test_is_broad_sweep_threshold():
    assert _is_broad_sweep([1.0, 4.0, 7.0]) is True
    assert _is_broad_sweep([1.0, 2.0]) is False  # too few positions


def test_is_narrow_sweep_threshold():
    assert _is_narrow_sweep([4.0, 4.3, 4.6]) is True  # span 0.6mm <= 1.0
    assert _is_narrow_sweep([4.0, 5.0, 6.5]) is False  # span 2.5mm > 1.0
    assert _is_narrow_sweep([4.0, 4.5]) is False  # too few


def test_clamp_and_dedupe_positions():
    out = _clamp_and_dedupe_positions(
        [1.0, 1.005, 2.0, -1.0, 99.0], pos_lo=0.0, pos_hi=10.0, dedupe_tol=0.02
    )
    # 1.0 kept, 1.005 coalesced, 2.0 kept, -1.0 clamped to 0.0, 99.0 clamped to 10.0
    assert 0.0 in out
    assert 10.0 in out
    assert out.count(1.0) == 1  # 1.005 dedupe'd into 1.0
    assert 2.0 in out


def test_image_to_jpeg_bytes_roundtrip():
    img = Image.new("RGB", (64, 64), (128, 64, 32))
    blob = _image_to_jpeg_bytes(img)
    assert isinstance(blob, bytes)
    assert len(blob) > 100  # has content
    assert blob[:2] == b"\xff\xd8"  # JPEG magic


def _fake_tool_context(state: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state
    ctx.save_artifact = AsyncMock(return_value=1)
    return ctx


def test_fetch_atlas_returns_labeled_parts_and_updates_state():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    result = asyncio.run(fetch_atlas(positions_mm=[2.0, 5.0, 8.0], tool_context=ctx))
    assert isinstance(result, dict)
    assert result["status"] == "ok"
    assert result["positions_mm"] == [2.0, 5.0, 8.0]
    assert "3 atlas sections" in str(result["description"])
    parts = _multimodal_parts(result)
    # Multimodal payload is header text + 3x (label_text, image_part).
    assert len(parts) == 1 + 2 * 3
    assert _is_text_part(parts[0])
    assert "3 atlas sections" in _text_of(parts[0])
    for i in range(3):
        assert _is_text_part(parts[1 + 2 * i])
        assert _is_image_part(parts[2 + 2 * i])
    joined = " ".join(_text_of(p) for p in parts if _is_text_part(p))
    for pos in (2.0, 5.0, 8.0):
        assert f"{pos:.2f} mm" in joined
    assert state["saw_broad_sweep"] is True
    assert state["images_fetched"] == 3
    assert ctx.save_artifact.call_count == 3


def test_fetch_atlas_rejects_empty_positions():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    result = asyncio.run(fetch_atlas(positions_mm=[], tool_context=ctx))
    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert result["error"] == "BAD_ARGS"


def test_submit_estimate_sets_state_and_escalates():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    ctx.actions = MagicMock()
    out = submit_estimate(position_mm=5.0, reasoning="hippocampus visible", tool_context=ctx)
    assert out["status"] == "ok"
    assert state["result"] == {"position_mm": 5.0, "reasoning": "hippocampus visible"}
    assert ctx.actions.escalate is True


def test_submit_group_estimate_sets_state_and_escalates():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=3,
        interval_mm=0.200, thickness_um=50, max_iterations=25,
    )
    ctx = _fake_tool_context(state)
    ctx.actions = MagicMock()
    out = submit_group_estimate(
        positions_mm=[5.0, 5.2, 5.4], reasoning="ok", tool_context=ctx,
    )
    assert out["status"] == "ok"
    assert state["result"]["positions_mm"] == [5.0, 5.2, 5.4]
    assert ctx.actions.escalate is True


def test_fetch_atlas_save_artifact_uses_canonical_keys():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    asyncio.run(fetch_atlas(positions_mm=[3.5, 7.25], tool_context=ctx))
    saved_keys = {call.kwargs["filename"] for call in ctx.save_artifact.call_args_list}
    assert saved_keys == {atlas_key(3.5), atlas_key(7.25)}
