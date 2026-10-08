"""The wrappers every tool runs inside (``doors/tools/toolbox.py``): the
serialization lock and its host events, the strict argument check, the
picture saving, the background notices, the opening-read gate and the
host display targets."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from PIL import Image

from langslice.core import layers
from langslice.core.state import SliceState, StackState
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import (
    _saves_views,
    _serialized,
    _tool_target_ids,
    build_tools,
)
from langslice.job.background import Landed
from langslice.job.job import HOST_TRANSFORM_KIND
from langslice.job.views import PICTURE_FILE
from tests.linear_tool_helpers import box as _box
from tests.linear_tool_helpers import stack as _stack
from tests.linear_tool_helpers import tool_named as _tool

# --- _serialized ---------------------------------------------------------------------


def test_tools_keep_their_identity_and_run_one_at_a_time():
    """ADK runs a turn's several calls concurrently; every tool shares one
    state, so the box serializes them without changing what ADK sees."""
    lock = threading.Lock()
    inside = {"now": 0, "peak": 0}

    def slow(x: int) -> int:
        """Doc kept."""
        inside["now"] += 1
        inside["peak"] = max(inside["peak"], inside["now"])
        time.sleep(0.02)
        inside["now"] -= 1
        return x

    wrapped = _serialized(slow, lock)
    assert wrapped.__name__ == "slow" and wrapped.__doc__ == "Doc kept."
    threads = [threading.Thread(target=wrapped, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert inside["peak"] == 1


def test_tool_end_names_the_pictures_the_call_saved(tmp_path: Path):
    events: list[dict] = []
    state, ctx, spec = _stack(tmp_path, n=3, placed=True)
    box = build_tools(state, ctx, spec, on_event=events.append)
    result = _tool(box, "look")("overlay", sections=["s1.png"])
    start, end = events
    assert (start["kind"], end["kind"]) == ("tool_start", "tool_end")
    assert start["name"] == end["name"] == "look"
    assert start["args"]["mode"] == "overlay" and start["target_ids"] == ["s1.png"]
    assert TOOL_MEDIA_PARTS_KEY not in end["response"]
    assert end["response"]["pictures"] == result["pictures"]
    (path,) = end["views"]
    assert path.endswith(PICTURE_FILE)
    box.job.views.flush()
    assert Path(path).is_file()


# --- what ADK reads -----------------------------------------------------------------------


def test_every_tool_declares_a_plain_schema_with_closed_objects(tmp_path: Path):
    from google.adk.tools import FunctionTool

    from langslice.core.spec import NonlinearSpec
    from langslice.providers.openai_oauth import _json_schema_dict

    _, _, box = _box(tmp_path, tasks=["position", "transform", "nonlinear"],
                     nonlinear=NonlinearSpec(provider="none"))
    for tool in box.tools:
        schema = _json_schema_dict(FunctionTool(tool)._get_declaration())
        assert "$ref" not in str(schema), tool.__name__
        assert "landmark" not in str(schema).lower(), tool.__name__
    schemas = {tool.__name__: _json_schema_dict(FunctionTool(tool)._get_declaration())
               for tool in box.tools}
    placed = schemas["position_sections"]["properties"]
    assert placed["sections"]["items"]["additionalProperties"] is False
    assert set(placed["sections"]["items"]["properties"]) == {"id", "position_mm"}
    assert placed["cutting_angles"]["additionalProperties"] is False
    turned = schemas["interactive_transform"]["properties"]["sections"]["items"]
    assert set(turned["properties"]) == {
        "id", "flip", "rotate_quarter", "rotation_deg", "scale_x", "scale_y", "shear",
        "translate_x_mm", "translate_y_mm"}
    left = schemas["submit"]["properties"]["left_linear"]["items"]
    assert set(left["properties"]) == {"id", "reason"}


# --- _strict -----------------------------------------------------------------------------


def test_unknown_and_misplaced_arguments_are_refused_before_the_tool_runs(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    before = state.to_dict()
    look = _tool(box, "look")
    top = look("overlay", zoom=[0, 0, 1, 1])
    assert top["error"] == "UNKNOWN_ARGUMENTS" and top["tool"] == "look"
    assert top["problems"] == [{"argument": "(top level)", "unknown": ["zoom"],
                                "accepted": ["mode", "sections", "positions_mm", "channels",
                                             "atlas_layers", "atlas_opacity", "warp"]}]
    assert top["message"].endswith("Nothing was done.")

    write = _tool(box, "position_sections")
    misplaced = write(position_mm=3.0)
    assert misplaced["problems"][0]["notes"] == [
        "`position_mm` belongs inside each `sections` entry."]
    entry = write([{"id": "s0.png", "position": 4.0}])
    assert entry["problems"][0] == {"argument": "sections[0]", "unknown": ["position"],
                                    "accepted": ["id", "position_mm"]}
    angles = write(cutting_angles={"pitch": 1.0, "yaw_deg": 0.0})
    assert angles["problems"][0]["argument"] == "cutting_angles"
    assert angles["problems"][0]["accepted"] == ["pitch_deg", "yaw_deg"]
    turned = _tool(box, "interactive_transform")([{"id": "s0.png", "rotate_deg": 90}])
    assert turned["problems"][0]["argument"] == "sections[0]"
    assert turned["problems"][0]["unknown"] == ["rotate_deg"]
    assert state.to_dict() == before and not box.job.undo_stack


def test_the_adk_plugin_refuses_what_adk_would_drop(tmp_path: Path):
    from google.adk.tools import FunctionTool

    from langslice.agent.plugins import StrictArgumentsPlugin

    _, _, box = _box(tmp_path, placed=True)
    look = _tool(box, "look")
    plugin = StrictArgumentsPlugin()
    answer = asyncio.run(plugin.before_tool_callback(
        tool=FunctionTool(look), tool_args={"mode": "section", "zoom": [0, 0, 1, 1]},
        tool_context=None))
    assert answer is not None and answer["error"] == "UNKNOWN_ARGUMENTS"
    assert asyncio.run(plugin.before_tool_callback(
        tool=FunctionTool(look), tool_args={"mode": "section"}, tool_context=None)) is None


# --- _saves_views ----------------------------------------------------------------------------


def test_plain_pictures_are_saved_and_numbered_in_order(tmp_path: Path):
    state, ctx, spec = _stack(tmp_path, n=2, placed=True)
    job = build_tools(state, ctx, spec).job
    red, blue = Image.new("RGB", (20, 10), "red"), Image.new("RGB", (20, 10), "blue")

    def drawn(sections: list[str]) -> dict:
        layers.note(blue, sections=("s1.png",), mode="section", caption="the blue one")
        return {"status": "ok", TOOL_MEDIA_PARTS_KEY: [red, "a line of text", blue]}

    result = _saves_views(drawn, job, ctx)(["s0.png"])
    assert result["pictures"] == [{"id": 1}, {"id": 2, "caption": "the blue one"}]
    records = job.views.records()
    assert [(r.seq, r.tool) for r in records] == [(1, "drawn"), (2, "drawn")]
    assert records[1].sections == ("s1.png",)


def test_pictures_an_operation_saved_are_not_saved_again(tmp_path: Path):
    state, ctx, spec = _stack(tmp_path, n=2, placed=True)
    job = build_tools(state, ctx, spec).job
    image = Image.new("RGB", (20, 10), "green")
    listed = [{"id": 7, "caption": "saved by its operation"}]

    def looked() -> dict:
        return {"status": "ok", "pictures": list(listed), TOOL_MEDIA_PARTS_KEY: [image]}

    result = _saves_views(looked, job, ctx)()
    assert result["pictures"] == listed
    assert job.views.records() == []


# --- _announces_work ------------------------------------------------------------------------


def test_background_notices_open_the_next_reply_with_their_pictures_last(tmp_path: Path):
    state, _, box = _box(tmp_path, n=3, placed=True)
    work_picture = Image.new("RGB", (30, 20), "purple")

    def land() -> Landed:
        return Landed(status="done", text="it landed.", pictures=[work_picture])

    work = box.job.background.start("demo", ["s0.png"], land, wait_for_images=False)
    box.job.background.wait_all()
    result = _tool(box, "look")("section", sections=["s2.png"])
    assert list(result)[0] == "background"
    assert result["background"] == [f"{work.id} demo of s0.png finished: it landed. "
                                    f"Pictures #{work.pictures[0]}."]
    own, theirs = result["pictures"]
    assert "caption" in own and theirs == {"id": work.pictures[0], "work": work.id}
    assert result[TOOL_MEDIA_PARTS_KEY][1] is work_picture
    # Each notice is handed out once.
    again = _tool(box, "status")()
    assert "background" not in again and "pictures" not in again


# --- the opening-read gate --------------------------------------------------------------------


def test_the_opening_gate_refuses_writes_until_every_page_is_read(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    box.require_opening(2)
    note = _tool(box, "note")
    refused = note("first")
    assert refused["status"] == "refused" and refused["error"] == "OPENING_NOT_READ"
    assert refused["pages"] == [1, 2]
    assert "show_stack(page=1), show_stack(page=2)" in refused["detail"]
    assert _tool(box, "status")()["status"] == "ok"  # reads pass
    box.opening_read(1)
    assert note("first")["pages"] == [2]
    box.opening_read(2)
    assert note("first")["status"] == "ok"
    box.require_opening(2)  # armed once; a host that read the pages is not asked again
    assert note("second")["status"] == "ok"
    assert state.notes[-2:] == ["first", "second"]


# --- _tool_target_ids -------------------------------------------------------------------------


def test_host_display_targets_follow_each_tools_scope():
    state = StackState(slices=[
        SliceState("a", 0, 0, position_mm=2),
        SliceState("b", 1, 1, position_mm=3,
                   transform={"kind": HOST_TRANSFORM_KIND, "params": [1, 0, 0, 0, 1, 0]}),
        SliceState("c", 2, 2),
    ])
    every = ["a", "b", "c"]
    for name in ("status", "undo", "redo", "submit"):
        assert _tool_target_ids(state, name, {}) == every
    assert _tool_target_ids(state, "look", {"mode": "positioning", "sections": []}) == every
    assert _tool_target_ids(state, "look", {"mode": "atlas", "sections": []}) == []
    assert _tool_target_ids(state, "look", {"mode": "overlay", "sections": ["2"]}) == []
    assert _tool_target_ids(state, "look", {"mode": "overlay", "sections": ["c"]}) == ["c"]
    assert _tool_target_ids(state, "position_sections",
                            {"sections": [], "cutting_angles": {"pitch_deg": 1}}) == every
    assert _tool_target_ids(state, "position_sections", {
        "sections": [{"id": "c", "position_mm": 1.0}, {"id": "a", "position_mm": 4.0}],
    }) == ["c", "a"]
    assert _tool_target_ids(state, "set_preprocessed_channel_properties",
                            {"sections": []}) == every
    # elastix_affine with no sections fits the placed sections the host did not align.
    assert _tool_target_ids(state, "elastix_affine", {"sections": []}) == ["a"]
    assert _tool_target_ids(state, "interactive_transform", {
        "sections": [{"id": "b"}, {"id": "a"}, {"id": "1"}, {"id": "a"}],
    }) == ["b", "a"]
    assert _tool_target_ids(state, "mark_damage", {"section": "b", "regions": []}) == ["b"]
    assert _tool_target_ids(state, "export_maps", {"slices": ["c", "nope"]}) == ["c"]
