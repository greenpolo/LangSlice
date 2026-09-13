"""Completion retirement preserves the actual trajectory and exact image slots."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from google.adk.flows.llm_flows.functions import _build_function_response_content
from google.genai import types
from test_linear_toolbox import _box, _tool

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.linear.spec import TransformSpec
from langslice.providers.openai_oauth import content_to_input_items

IDENTITY = dict(rotation_deg=0.0, scale_x=1.0, scale_y=1.0,
                translate_x_mm=0.0, translate_y_mm=0.0)


def test_retirement_through_native_adk_media_lifting(tmp_path):
    """Exercise actual ADK extraction, not just hand-built response objects."""
    _, _, box = _box(tmp_path, n=1, placed=True, tasks=["transform"],
                     image_retention="completion")

    class Tool:
        name = "adjust_transforms"
        response_scheduling = None

    old = _build_function_response_content(Tool(), batch(box, ["s0.png"]), "old")
    new = _build_function_response_content(
        Tool(), batch(box, ["s0.png"], scale_x=1.02), "new",
    )
    history = [old, new]
    original = copy.deepcopy(history)
    box.visual_context(history)
    assert _tool(box, "accept_views")(["s0.png"], "transform")["status"] == "ok"
    filtered = box.visual_context(history)
    assert old.parts[0].function_response.parts
    assert filtered[0].parts[0].function_response.parts is None
    assert len(filtered[1].parts[0].function_response.parts) == 1
    assert history == original
    assert wire_without_images(history) == wire_without_images(filtered)


def test_position_completion_keeps_shared_evidence_for_open_transforms(tmp_path):
    _, _, box = _box(tmp_path, n=1, placed=True, tasks=["position", "transform"],
                     image_retention="completion")
    shared = response("fetch_atlas", _tool(box, "fetch_atlas")([2.0]))
    comparison = response("compare_placement", _tool(box, "compare_placement")(
        [{"id": "s0.png"}],
    ))
    history = [shared, comparison]
    box.visual_context(history)
    assert _tool(box, "accept_views")(["s0.png"], "position")["status"] == "ok"
    assert box.visual_context(history)[0].parts[0].function_response.parts
    overlay = response("adjust_transforms", batch(box, ["s0.png"]))
    history.append(overlay)
    box.visual_context(history)
    assert _tool(box, "accept_views")(["s0.png"], "transform")["status"] == "ok"
    assert box.visual_context(history)[0].parts[0].function_response.parts is None


def response(name, result):
    media = [types.FunctionResponsePart(inline_data=types.FunctionResponseBlob(
        mime_type=part.inline_data.mime_type, data=part.inline_data.data,
    )) for part in result.get(TOOL_MEDIA_PARTS_KEY, []) if part.inline_data is not None]
    return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        name=name, id=result["media_delivery_id"],
        response={k: v for k, v in result.items() if k != TOOL_MEDIA_PARTS_KEY}, parts=media,
    ))])


def wire_without_images(contents):
    items = [item for content in contents for item in content_to_input_items(content)]
    for item in items:
        if isinstance(item.get("output"), list):
            item["output"] = [part for part in item["output"] if part["type"] != "input_image"]
    return items


def batch(box, ids, **overrides):
    return _tool(box, "adjust_transforms")(
        [{"id": name, **IDENTITY, **overrides} for name in ids]
    )


def test_selective_batch_retirement_preserves_seed_text_reasoning_and_other_slice(tmp_path):
    _, _, box = _box(tmp_path, n=2, placed=True, tasks=["transform"],
                     image_retention="completion")
    visual = box.visual_context
    first = response("adjust_transforms", batch(box, ["s0.png", "s1.png"]))
    blob = first.parts[0].function_response.parts[0].inline_data
    seed = types.Content(role="user", parts=[
        types.Part.from_text(text="Whole stack seed"),
        types.Part.from_bytes(data=blob.data, mime_type=blob.mime_type),
    ])
    thought = types.Content(role="model", parts=[types.Part(
        text="Cross-slice observation", thought=True,
        thought_signature=b'{"type":"reasoning","encrypted_content":"preserved"}',
    )])
    history = [seed, first, thought]
    original = copy.deepcopy(history)
    visual(history)
    latest = response("adjust_transforms", batch(box, ["s0.png"], translate_x_mm=0.1))
    history.append(latest)
    assert _tool(box, "accept_views")(["s0.png"], "transform")["error"] == "FINAL_VIEWS_NOT_SEEN"
    visual(history)
    assert _tool(box, "accept_views")(["s0.png"], "transform")["retired_images"] == 1
    filtered = visual(history)
    assert len(filtered[1].parts[0].function_response.parts) == 1
    assert (
        filtered[1].parts[0].function_response.parts[0]
        == first.parts[0].function_response.parts[1]
    )
    assert filtered[0] is seed and filtered[2] is thought
    assert history[:3] == original
    assert wire_without_images(filtered) == wire_without_images(history)
    assert visual(filtered) == filtered


def test_final_submit_retires_shared_and_intermediate_but_keeps_one_overlay_each(tmp_path):
    state, _, box = _box(tmp_path, n=2, placed=True, tasks=["transform"],
                         image_retention="completion")
    visual = box.visual_context
    shared = response("fetch_atlas", _tool(box, "fetch_atlas")([2.0]))
    old = response("adjust_transforms", batch(box, ["s0.png", "s1.png"]))
    latest = response("adjust_transforms", batch(box, ["s0.png", "s1.png"], scale_x=1.01))
    history = [shared, old, latest]
    assert _tool(box, "submit")("done", [], [])["error"] == "FINAL_VIEWS_NOT_SEEN"
    visual(history)
    assert _tool(box, "submit")("done", [], [])["status"] == "ok"
    filtered = visual(history)
    assert state.submitted
    assert [len(c.parts[0].function_response.parts or []) for c in filtered] == [0, 0, 2]
    assert wire_without_images(filtered) == wire_without_images(history)


@pytest.mark.parametrize("change", ["position", "flip", "angle", "transform"])
def test_reopened_geometry_requires_new_delivered_view(tmp_path, change):
    state, _, box = _box(tmp_path, n=1, placed=True, tasks=["transform"],
                         transform=TransformSpec(angles=True), image_retention="completion")
    old = response("adjust_transforms", batch(box, ["s0.png"]))
    box.visual_context([old])
    assert _tool(box, "accept_views")(["s0.png"], "transform")["status"] == "ok"
    record = state.slices[0]
    if change == "position":
        record.position_mm += 0.1
    elif change == "flip":
        record.flip = True
    elif change == "angle":
        _tool(box, "set_cutting_angles")(3.0, 0.0)
    else:
        batch(box, ["s0.png"], scale_x=1.02)
    assert _tool(box, "accept_views")(["s0.png"], "transform")["error"] == "FINAL_VIEWS_NOT_SEEN"
    new = response("adjust_transforms", batch(box, ["s0.png"]))
    box.visual_context([old, new])
    _tool(box, "accept_views")(["s0.png"], "transform")
    assert box.visual_context([old, new])[0].parts[0].function_response.parts is None


def test_position_acceptance_and_seen_write_require_delivered_full_atlas(tmp_path):
    _, _, box = _box(tmp_path, n=1, tasks=["position"], image_retention="completion")
    compare = _tool(box, "compare_placement")
    set_pos = _tool(box, "set_positions")
    zoom = response("compare_placement", compare(
        [{"id": "s0.png", "positions_mm": [2.0]}], zoom=[0.2, 0.2, 0.8, 0.8],
    ))
    box.visual_context([zoom])
    write = set_pos([{"id": "s0.png", "position_mm": 2.0}])
    assert not write["images_suppressed_seen"]
    full = response("compare_placement", compare([{"id": "s0.png", "positions_mm": [2.0]}]))
    box.visual_context([zoom, full])
    assert set_pos([{"id": "s0.png", "position_mm": 2.0}])["images_suppressed_seen"] == ["s0.png"]
    assert _tool(box, "accept_views")(["s0.png"], "position")["status"] == "ok"
    assert box.visual_context([zoom, full])[0].parts[0].function_response.parts is None


def test_ab_acceptance_keeps_current_not_reference_and_rejects_zoom(tmp_path):
    _, _, box = _box(tmp_path, n=1, placed=True, tasks=["transform"],
                     image_retention="completion")
    adjust = _tool(box, "adjust_transform")
    zoom = response("adjust_transform", adjust("s0.png", **IDENTITY, zoom=[0.2, 0.2, 0.8, 0.8]))
    box.visual_context([zoom])
    assert _tool(box, "accept_views")(["s0.png"], "transform")["error"] == "FINAL_VIEWS_NOT_SEEN"
    ab = response("adjust_transform", adjust("s0.png", **IDENTITY, mode="ab"))
    box.visual_context([zoom, ab])
    _tool(box, "accept_views")(["s0.png"], "transform")
    filtered = box.visual_context([zoom, ab])
    assert len(filtered[1].parts[0].function_response.parts) == 1
    assert filtered[1].parts[0].function_response.parts[0] == ab.parts[0].function_response.parts[0]
    assert wire_without_images(filtered) == wire_without_images([zoom, ab])


def test_separate_candidates_keep_corrected_section_dependency(tmp_path):
    state, _, box = _box(tmp_path, n=1, tasks=["position"], image_retention="completion")
    state.slices[0].flip = True
    compare = response("compare_placement", _tool(box, "compare_placement")(
        [{"id": "s0.png", "positions_mm": [2.0, 2.1]}], mode="side_by_side",
    ))
    visual = box.visual_context
    visual([compare])
    _tool(box, "set_positions")([{"id": "s0.png", "position_mm": 2.1}])
    assert _tool(box, "accept_views")(["s0.png"], "position")["status"] == "ok"
    filtered = visual([compare])
    original = compare.parts[0].function_response.parts
    assert filtered[0].parts[0].function_response.parts == [original[0], original[2]]
    assert wire_without_images(filtered) == wire_without_images([compare])
    assert visual.candidates(["s0.png"], "position")[1] == []


def test_accept_never_retires_unseen_local_or_shared_siblings(tmp_path):
    _, _, box = _box(tmp_path, n=1, placed=True, tasks=["transform"],
                     image_retention="completion")
    visual = box.visual_context
    inspected = response("adjust_transforms", batch(box, ["s0.png"]))
    visual([inspected])
    sibling = response("adjust_transforms", batch(box, ["s0.png"]))
    shared = response("fetch_atlas", _tool(box, "fetch_atlas")([2.0]))
    assert _tool(box, "accept_views")(["s0.png"], "transform")["retired_images"] == 0
    history = [inspected, sibling, shared]
    assert all(c.parts[0].function_response.parts for c in visual(history))
    # Once actually delivered, a later explicit acceptance can retire them.
    assert _tool(box, "accept_views")(["s0.png"], "transform")["retired_images"] == 2
    assert [len(c.parts[0].function_response.parts or []) for c in visual(history)] == [0, 1, 0]


def test_interleaved_acceptance_shared_boundary_and_undo(tmp_path):
    _, _, box = _box(tmp_path, n=2, placed=True, tasks=["transform"],
                     image_retention="completion")
    visual = box.visual_context
    shared = response("fetch_atlas", _tool(box, "fetch_atlas")([2.0]))
    refs = response("view_slices", _tool(box, "view_slices")(["s0.png", "s1.png"]))
    first = response("adjust_transforms", batch(box, ["s0.png", "s1.png"]))
    history = [shared, refs, first]
    visual(history)
    _tool(box, "accept_views")(["s0.png"], "transform")
    filtered = visual(history)
    assert len(filtered[0].parts[0].function_response.parts) == 1
    assert len(filtered[1].parts[0].function_response.parts) == 1
    # Reopen A while B is still active. Undo restores the accepted geometry
    # without resurrecting any retired historical image.
    changed = response("adjust_transforms", batch(box, ["s0.png"], scale_x=1.02))
    history.append(changed)
    visual(history)
    assert visual.candidates(["s0.png"], "transform")[1] == []
    _tool(box, "undo")()
    assert visual.accepted[("transform", "s0.png")] == visual.candidates(
        ["s0.png"], "transform",
    )[0]["s0.png"]
    _tool(box, "accept_views")(["s1.png"], "transform")
    filtered = visual(history)
    assert filtered[0].parts[0].function_response.parts is None
    assert filtered[1].parts[0].function_response.parts is None
    assert len(filtered[2].parts[0].function_response.parts) == 2
    # A's reopened evidence is not owned by B's acceptance.
    assert filtered[3].parts[0].function_response.parts
    assert wire_without_images(filtered) == wire_without_images(history)


def test_final_submit_does_not_discard_unread_sibling(tmp_path):
    _, _, box = _box(tmp_path, n=1, placed=True, tasks=["transform"],
                     image_retention="completion")
    visual = box.visual_context
    inspected = response("adjust_transforms", batch(box, ["s0.png"]))
    visual([inspected])
    unread = response("fetch_atlas", _tool(box, "fetch_atlas")([2.0]))
    assert _tool(box, "submit")("done", [], [])["status"] == "ok"
    filtered = visual([inspected, unread])
    assert all(content.parts[0].function_response.parts for content in filtered)


@pytest.mark.parametrize("sibling", ["adjust", "angles", "undo"])
def test_submit_closes_completion_state_before_later_siblings(tmp_path, sibling):
    state, ctx, box = _box(
        tmp_path, n=1, placed=True, tasks=["transform"],
        transform=TransformSpec(angles=True), image_retention="completion",
    )
    inspected = response("adjust_transforms", batch(box, ["s0.png"]))
    box.visual_context([inspected])
    assert _tool(box, "submit")("done", [], [])["status"] == "ok"
    checkpoint = Path(ctx.checkpoint_path).read_bytes()
    submitted = copy.deepcopy(state.to_dict())
    accepted = copy.deepcopy(box.visual_context.accepted)
    undo_stack = copy.deepcopy(box.undo_stack)
    if sibling == "adjust":
        result = batch(box, ["s0.png"], scale_x=1.02)
    elif sibling == "angles":
        result = _tool(box, "set_cutting_angles")(3.0, 0.0)
    else:
        result = _tool(box, "undo")()
    assert result == {"status": "refused", "error": "ALREADY_SUBMITTED"}
    assert state.to_dict() == submitted
    assert Path(ctx.checkpoint_path).read_bytes() == checkpoint
    assert box.visual_context.accepted == accepted
    assert box.undo_stack == undo_stack


def test_legacy_post_submit_behavior_is_unchanged(tmp_path):
    state, _, box = _box(tmp_path, n=1, placed=True, tasks=["transform"])
    batch(box, ["s0.png"])
    assert _tool(box, "submit")("done", [], [])["status"] == "ok"
    assert _tool(box, "undo")()["status"] == "ok"
    assert not state.submitted
