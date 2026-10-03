"""Separate positioning references reuse cached bytes, not composed canvases."""

from langslice.adk import TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY
from langslice.linear.atlas_fetch import atlas_part
from langslice.linear.render import picture_edge, reference_slice_part
from tests.test_linear_toolbox import _box, _tool, _ToolContext


def _images(parts):
    return [part.inline_data.data for part in parts if part.inline_data is not None]


def test_separate_comparison_reuses_cached_bytes_and_maps_each_pair(tmp_path):
    state, ctx, box = _box(tmp_path)
    position = 3.0
    result = _tool(box, "view_placement")([
        {"id": "s0.png", "positions_mm": [position, position]},
        {"id": "s1.png", "positions_mm": [position]},
    ], view={"mode": "side_by_side"})
    images = _images(result[TOOL_MEDIA_PARTS_KEY])
    assert len(images) == 5
    edge = picture_edge(ctx)
    assert images[0] == _images([reference_slice_part(ctx, state.by_id("s0.png"),
                                                      long_edge=edge)])[0]
    assert images[3] == _images([reference_slice_part(ctx, state.by_id("s1.png"),
                                                      long_edge=edge)])[0]
    assert images[1] == images[2] == images[4] == _images(
        [atlas_part(ctx, state, position, long_edge=edge)])[0]
    assert [row["image_indexes"] for row in result["compared"]] == [
        {"section": 0, "atlas": 1}, {"section": 0, "atlas": 2},
        {"section": 3, "atlas": 4},
    ]


def test_reference_reuse_survives_reorder_but_not_orientation_or_preprocess(tmp_path):
    state, ctx, box = _box(tmp_path)
    original = _images([reference_slice_part(ctx, state.by_id("s0.png"))])[0]
    record = state.by_id("s0.png")
    _tool(box, "reorder_slices")([r.id for r in reversed(state.in_order())])
    assert reference_slice_part(ctx, record).inline_data.data == original
    record.rotation_deg = 90
    turned = reference_slice_part(ctx, record).inline_data.data
    assert turned != original
    before = len(ctx.reference_parts)
    record.flip = True
    reference_slice_part(ctx, record)
    assert len(ctx.reference_parts) == before + 1
    before = len(ctx.reference_parts)
    ctx.spec.preprocess = "auto"
    reference_slice_part(ctx, record)
    assert len(ctx.reference_parts) == before + 1


def test_atlas_reuse_keys_on_exact_position_and_cutting_angles(tmp_path, monkeypatch):
    state, ctx, _box_value = _box(tmp_path)
    first = atlas_part(ctx, state, 3.0).inline_data.data
    assert atlas_part(ctx, state, 3.0).inline_data.data == first
    # Fake atlas does not support oblique interpolation; isolate cache identity.
    from langslice.linear import atlas_fetch
    original = atlas_fetch.atlas_section
    image = original(ctx, state, 3.0, frame=True)
    calls = []

    def render(_ctx, _state, position, **_kwargs):
        calls.append(position)
        return image

    monkeypatch.setattr(atlas_fetch, "atlas_section", render)
    state.cutting_angles_deg["pitch"] = 2.0
    assert atlas_part(ctx, state, 3.0).inline_data.data != first
    atlas_part(ctx, state, 3.0001)
    atlas_part(ctx, state, 3.0001)
    assert calls == [3.0, 3.0001]


def test_separate_full_views_suppress_only_after_delivery_and_reject_zoom(tmp_path):
    _, _, box = _box(tmp_path)
    compare = _tool(box, "view_placement")
    args = [{"id": "s0.png", "positions_mm": [3.0]}]
    assert compare(args, view={"mode": "side_by_side", "zoom": [0, 0, .5, .5]})["error"] == (
        "ZOOM_UNSUPPORTED"
    )
    assert not box.compared
    result = compare(args, view={"mode": "side_by_side"}, tool_context=_ToolContext("pair"))
    assert result[TOOL_MEDIA_DELIVERY_ID_KEY] == "pair"
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 2
    write = _tool(box, "set_positions")
    assert len(write([{"id": "s0.png", "position_mm": 3.0}])[TOOL_MEDIA_PARTS_KEY]) == 1
    box.mark_placement_views_delivered({"pair"})
    assert write([{"id": "s0.png", "position_mm": 3.0}])[TOOL_MEDIA_PARTS_KEY] == []


def test_separate_comparison_caps_pairs(tmp_path):
    state, _, box = _box(tmp_path)
    result = _tool(box, "view_placement")([
        {"id": record.id, "positions_mm": [3.0]} for record in state.in_order()
    ], view={"mode": "side_by_side"})
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 8
    assert result["dropped_pairs"] == 1
