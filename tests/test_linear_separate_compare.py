"""Separate positioning references reuse seed bytes, not composed canvases."""

from langslice.adk import TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY
from langslice.linear.atlas_fetch import atlas_part, atlas_strip_parts
from langslice.linear.render import reference_slice_part, stack_image_parts
from tests.test_linear_toolbox import _box, _tool, _ToolContext


def _images(parts):
    return [part.inline_data.data for part in parts if part.inline_data is not None]


def test_separate_comparison_reuses_seed_bytes_and_maps_each_pair(tmp_path):
    state, ctx, box = _box(tmp_path)
    seed = _images(stack_image_parts(state, ctx))
    atlas_seed = atlas_strip_parts(ctx, state)
    position = float(atlas_seed[1].text.split()[1])
    result = _tool(box, "compare_placement")([
        {"id": "s0.png", "positions_mm": [position, position]},
        {"id": "s1.png", "positions_mm": [position]},
    ], mode="side_by_side")
    images = _images(result[TOOL_MEDIA_PARTS_KEY])
    assert len(images) == 5
    assert images[0] == seed[0] and images[3] == seed[1]
    assert images[1] == images[2] == images[4] == _images(atlas_seed)[0]
    assert [row["image_indexes"] for row in result["compared"]] == [
        {"section": 0, "atlas": 1}, {"section": 0, "atlas": 2},
        {"section": 3, "atlas": 4},
    ]


def test_reference_reuse_survives_reorder_but_not_orientation_or_preprocess(tmp_path):
    state, ctx, box = _box(tmp_path)
    original = _images(stack_image_parts(state, ctx))[0]
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
    compare = _tool(box, "compare_placement")
    args = [{"id": "s0.png", "positions_mm": [3.0]}]
    assert compare(args, mode="side_by_side", zoom=[0, 0, .5, .5])["error"] == (
        "ZOOM_UNSUPPORTED"
    )
    assert not box.compared
    result = compare(args, mode="side_by_side", tool_context=_ToolContext("pair"))
    assert result[TOOL_MEDIA_DELIVERY_ID_KEY] == "pair"
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 2
    write = _tool(box, "set_positions")
    assert len(write([{"id": "s0.png", "position_mm": 3.0}])[TOOL_MEDIA_PARTS_KEY]) == 1
    box.mark_placement_views_delivered({"pair"})
    assert write([{"id": "s0.png", "position_mm": 3.0}])[TOOL_MEDIA_PARTS_KEY] == []


def test_completion_evidence_marks_only_atlas_eligible_and_caps_pairs(tmp_path):
    state, _, box = _box(tmp_path, image_retention="completion")
    result = _tool(box, "compare_placement")([
        {"id": record.id, "positions_mm": [3.0]} for record in state.in_order()
    ], mode="side_by_side")
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 8
    assert result["dropped_pairs"] == 1
    evidence = box.visual_context.media[result[TOOL_MEDIA_DELIVERY_ID_KEY]]
    assert [item.eligible for item in evidence] == [False, True] * 4
    assert [item.slice_id for item in evidence] == [
        f"s{i}.png" for i in range(4) for _ in range(2)
    ]
