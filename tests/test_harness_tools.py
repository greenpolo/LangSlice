import asyncio
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from langslice.harness.estimation.session import build_initial_state
from langslice.harness.estimation.tools import (
    _clamp_and_dedupe_positions,
    _image_to_jpeg_bytes,
    _image_to_part,
    _is_broad_sweep,
    _is_narrow_sweep,
    _parse_atlas_key,
    _short_hash,
    fetch_atlas,
    submit_estimate,
    submit_group_estimate,
    zoom,
)


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


def test_fetch_atlas_returns_ok_and_updates_state():
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    ctx = _fake_tool_context(state)
    result = asyncio.run(fetch_atlas(positions_mm=[2.0, 5.0, 8.0], tool_context=ctx))
    assert result["status"] == "ok"
    assert len(result["images"]) == 3
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


# --- Phase 5 Task 5.1: zoom tool --------------------------------------------


class _ArtifactStore:
    """In-memory stand-in for the ADK artifact service used in zoom tests."""

    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    async def load(self, *, filename: str, version: int | None = None):
        del version
        return self._store.get(filename)

    async def save(self, *, filename: str, artifact, custom_metadata=None) -> int:
        del custom_metadata
        self._store[filename] = artifact
        return 1


def _zoom_ctx(state: dict, store: _ArtifactStore) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state
    ctx.load_artifact = AsyncMock(side_effect=store.load)
    ctx.save_artifact = AsyncMock(side_effect=store.save)
    return ctx


def _seed_target(store: _ArtifactStore, img: Image.Image) -> None:
    store._store["target"] = _image_to_part(img)


def _state_for_zoom() -> dict:
    return build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )


def test_zoom_happy_path_resizes_and_saves_artifact():
    store = _ArtifactStore()
    _seed_target(store, Image.new("RGB", (400, 300), (200, 100, 50)))
    state = _state_for_zoom()
    ctx = _zoom_ctx(state, store)

    result = asyncio.run(
        zoom(source="target", bbox=[100, 200, 900, 800], tool_context=ctx)
    )

    assert result["status"] == "ok"
    # bbox covers (px_x1=80, px_y1=30, px_x2=320, px_y2=270) → 240x240
    # atlas long edge is 456 (allen_mouse_25um coronal), so 240→456.
    assert result["key"].startswith("zoom:target:")
    assert len(result["images"]) == 1
    part = result["images"][0]
    assert part.inline_data is not None
    decoded = Image.open(io.BytesIO(part.inline_data.data))
    w, h = decoded.size
    assert max(w, h) == 456
    # Saved under the same key returned.
    assert result["key"] in store._store
    # Hash is reproducible from the bbox.
    assert result["key"].endswith(_short_hash([100, 200, 900, 800]))


def test_zoom_bad_bbox_out_of_range():
    store = _ArtifactStore()
    _seed_target(store, Image.new("RGB", (100, 100)))
    ctx = _zoom_ctx(_state_for_zoom(), store)
    result = asyncio.run(
        zoom(source="target", bbox=[-1, 0, 1000, 1000], tool_context=ctx)
    )
    assert result == {"status": "error", "error": "BAD_BBOX"}


def test_zoom_bad_bbox_wrong_length():
    store = _ArtifactStore()
    _seed_target(store, Image.new("RGB", (100, 100)))
    ctx = _zoom_ctx(_state_for_zoom(), store)
    result = asyncio.run(
        zoom(source="target", bbox=[0, 0, 1000, 1000, 1000], tool_context=ctx)
    )
    assert result == {"status": "error", "error": "BAD_BBOX"}


def test_zoom_bad_bbox_zero_area():
    store = _ArtifactStore()
    _seed_target(store, Image.new("RGB", (100, 100)))
    ctx = _zoom_ctx(_state_for_zoom(), store)
    result = asyncio.run(
        zoom(source="target", bbox=[500, 500, 500, 500], tool_context=ctx)
    )
    assert result == {"status": "error", "error": "BAD_BBOX"}


def test_zoom_bad_bbox_upper_bound_exceeded():
    store = _ArtifactStore()
    _seed_target(store, Image.new("RGB", (100, 100)))
    ctx = _zoom_ctx(_state_for_zoom(), store)
    result = asyncio.run(
        zoom(source="target", bbox=[1001, 0, 1000, 1000], tool_context=ctx)
    )
    assert result == {"status": "error", "error": "BAD_BBOX"}


def test_zoom_bad_source_missing_artifact():
    store = _ArtifactStore()
    # No seeded artifact.
    ctx = _zoom_ctx(_state_for_zoom(), store)
    result = asyncio.run(
        zoom(source="nonexistent", bbox=[100, 100, 900, 900], tool_context=ctx)
    )
    assert result == {"status": "error", "error": "BAD_SOURCE"}


def test_zoom_tiny_bbox_yields_single_pixel_crop():
    # Under floor/ceil semantics, any non-empty normalized bbox maps to a
    # non-empty pixel bbox — even on a 2x2 image. The tiny crop then upscales
    # to the atlas long edge so the model can still see it.
    store = _ArtifactStore()
    _seed_target(store, Image.new("RGB", (2, 2), (255, 0, 0)))
    ctx = _zoom_ctx(_state_for_zoom(), store)
    result = asyncio.run(
        zoom(source="target", bbox=[0, 0, 1, 1], tool_context=ctx)
    )
    assert result["status"] == "ok"
    part = result["images"][0]
    decoded = Image.open(io.BytesIO(part.inline_data.data))
    assert max(decoded.size) == 456  # upscaled to atlas long edge


def test_zoom_of_zoom_works():
    store = _ArtifactStore()
    _seed_target(store, Image.new("RGB", (400, 300), (123, 45, 67)))
    state = _state_for_zoom()
    ctx = _zoom_ctx(state, store)

    first = asyncio.run(
        zoom(source="target", bbox=[100, 100, 900, 900], tool_context=ctx)
    )
    assert first["status"] == "ok"
    first_key = first["key"]

    second = asyncio.run(
        zoom(source=first_key, bbox=[250, 250, 750, 750], tool_context=ctx)
    )
    assert second["status"] == "ok"
    assert second["key"].startswith(f"zoom:{first_key}:")
    assert second["key"] in store._store
