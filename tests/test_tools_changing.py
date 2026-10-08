"""The change tools and the bookkeeping, through the tool door:
position_sections, interactive_transform, mark_damage, note/undo/redo,
submit, and the look-before-commit gates (``PositionSpec.gated``)."""

from __future__ import annotations

import inspect
from pathlib import Path

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec, PositionSpec, TransformSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.toolbox import build_tools
from langslice.job.checkpoint import load_checkpoint
from langslice.job.job import Job
from langslice.ops.registry import RETIRED
from tests.deformable_synthetic import SyntheticAtlas
from tests.linear_tool_helpers import SLAB, ToolContext, single_transform
from tests.linear_tool_helpers import box as _box
from tests.linear_tool_helpers import stack as _stack
from tests.linear_tool_helpers import submit as _submit
from tests.linear_tool_helpers import tool_named as _tool


def _positions(state) -> dict[str, float | None]:
    return {record.id: record.position_mm for record in state.slices}


# --- which tools ------------------------------------------------------------------


def test_a_position_only_run_has_no_transform_or_fit_tools(tmp_path: Path):
    _, _, box = _box(tmp_path, tasks=["position"])
    names = set(box.names)
    assert {"status", "look", "position_sections", "mark_damage", "submit"} <= names
    assert not names & {"interactive_transform", "elastix_affine", "ants_syn",
                        "trace_borders"}
    assert not names & set(RETIRED)


# --- position_sections -------------------------------------------------------------


def test_a_write_checkpoints_the_whole_state(tmp_path: Path):
    state, ctx, box = _box(tmp_path)
    assert load_checkpoint(ctx.checkpoint_path) is None
    result = _tool(box, "position_sections")([{"id": "s0.png", "position_mm": 3.0}])
    assert result["status"] == "ok"
    assert result["written"] == [{"id": "s0.png", "position_mm": 3.0}]
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved is not None and saved.by_id("s0.png").position_mm == 3.0
    assert state.by_id("s0.png").position_mm == 3.0


def test_a_batch_write_undoes_and_redoes_as_one_step(tmp_path: Path):
    state, _, box = _box(tmp_path)
    _tool(box, "position_sections")([
        {"id": "s0.png", "position_mm": 1.0},
        {"id": "s1.png", "position_mm": 2.0},
        {"id": "s2.png", "position_mm": 3.0},
    ], view=False)
    assert [s.position_mm for s in state.in_order()][:3] == [1.0, 2.0, 3.0]
    assert len(box.job.undo_stack) == 1

    assert _tool(box, "undo")()["status"] == "ok"
    assert all(s.position_mm is None for s in state.slices)
    assert _tool(box, "redo")()["status"] == "ok"
    assert [s.position_mm for s in state.in_order()][:3] == [1.0, 2.0, 3.0]
    assert _tool(box, "undo")()["status"] == "ok"
    assert _tool(box, "undo")()["error"] == "NOTHING_TO_UNDO"
    assert _tool(box, "redo")()["status"] == "ok"
    assert _tool(box, "redo")()["error"] == "NOTHING_TO_REDO"


def test_unknown_ids_are_named_and_nothing_is_written(tmp_path: Path):
    state, ctx, box = _box(tmp_path, placed=True)
    before = state.to_dict()
    result = _tool(box, "position_sections")([{"id": "nope.png", "position_mm": 1.0}],
                                             view=False)
    assert result["unknown_ids"] == ["nope.png"]
    assert result["error"] == "NOTHING_WRITTEN"
    assert state.to_dict() == before
    assert box.job.undo_stack == []
    assert load_checkpoint(ctx.checkpoint_path) is None


def test_position_sections_clamps_to_the_atlas_range(tmp_path: Path):
    state, _, box = _box(tmp_path)
    result = _tool(box, "position_sections")([{"id": "s0.png", "position_mm": 500.0}],
                                             view=False)
    assert result["clamped"] == [{"id": "s0.png", "requested_mm": 500.0,
                                  "clamped_to_mm": 19.0}]
    assert result["atlas_range_mm"] == [0.0, 19.0]
    assert state.by_id("s0.png").position_mm == 19.0


def test_the_stack_is_renumbered_by_position_after_every_write(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    before = [s.id for s in state.in_order()]
    result = _tool(box, "position_sections")([{"id": "s0.png", "position_mm": 9.0}],
                                             view=False)
    order = ["s1.png", "s2.png", "s3.png", "s4.png", "s0.png"]
    assert result["order"] == order
    assert result["reordered"] == order  # every section changed its place
    assert [s.id for s in state.in_order()] == order
    assert [s.index_corrected for s in state.in_order()] == list(range(5))
    assert result["changed"][0]["id"] == "s0.png" and "index" not in result["changed"][0]
    # Moving one section within its neighbours renumbers only the pair.
    swapped = _tool(box, "position_sections")([{"id": "s2.png", "position_mm": 2.4}],
                                              view=False)
    assert swapped["order"][:2] == ["s2.png", "s1.png"]
    assert sorted(swapped["reordered"]) == ["s1.png", "s2.png"]
    # The order is part of the write: undo puts it back.
    _tool(box, "undo")()
    _tool(box, "undo")()
    assert [s.id for s in state.in_order()] == before


def test_a_write_returns_only_the_rows_it_touched(tmp_path: Path):
    _, _, box = _box(tmp_path, placed=True)
    result = _tool(box, "position_sections")(
        [{"id": "s0.png", "position_mm": 1.0}, {"id": "s3.png", "position_mm": 6.0}],
        view=False)
    assert sorted(row["id"] for row in result["changed"]) == ["s0.png", "s3.png"]
    assert result["n_sections"] == 5
    assert result["cutting_angles_deg"] == {"pitch": 0.0, "yaw": 0.0}
    assert result["interval_breaks"] == []
    assert "rows" not in result  # the whole table is `status`, one call away
    assert len(_tool(box, "status")()["rows"]) == 5
    assert len(_tool(box, "undo")()["rows"]) == 5
    assert len(_tool(box, "redo")()["rows"]) == 5


def test_an_exact_rewrite_of_a_starting_position_makes_it_the_agents(tmp_path: Path):
    _stack(tmp_path, tasks=["position"])  # the images
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none",
                   tasks=["position"])
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: SLAB)
    job = Job.open(spec, ctx)
    box = build_tools(job.state, ctx, spec, job=job)
    rows = {row["id"]: row for row in _tool(box, "status")()["rows"]}
    assert all(row["position_source"] == "default" for row in rows.values())
    start = rows["s1.png"]["position_mm"]

    result = _tool(box, "position_sections")([{"id": "s1.png", "position_mm": start}],
                                             view=False)
    assert result["written"] == [{"id": "s1.png", "position_mm": start}]
    assert "position_source" not in result["changed"][0]
    assert job.state.by_id("s1.png").position_source == ""
    assert job.state.by_id("s1.png").position_mm == start
    refusal = _submit(box)
    assert refusal["error"] == "MISSING_POSITIONS"
    assert "s1.png" not in refusal["missing_ids"] and len(refusal["missing_ids"]) == 4
    assert refusal["at_default"] == refusal["missing_ids"]
    # Undo gives the starting mark back.
    _tool(box, "undo")()
    assert job.state.by_id("s1.png").position_source == "default"


def test_cutting_angles_are_one_write_for_the_stack(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True, tasks=["position"])
    write = _tool(box, "position_sections")
    result = write(cutting_angles={"pitch_deg": 2.0, "yaw_deg": -1.0}, view=False)
    assert result["status"] == "ok" and result["written"] == []
    assert result["cutting_angles_deg"] == {"pitch": 2.0, "yaw": -1.0}
    assert state.cutting_angles_deg == {"pitch": 2.0, "yaw": -1.0}
    assert len(box.job.undo_stack) == 1
    assert write(cutting_angles={"pitch_deg": "steep", "yaw_deg": 0})["error"] == "BAD_ARGS"
    assert write()["error"] == "BAD_ARGS"  # neither sections nor angles
    _tool(box, "undo")()
    assert state.cutting_angles_deg == {"pitch": 0.0, "yaw": 0.0}


def test_the_picture_is_a_positioning_picture_of_the_written_sections(tmp_path: Path):
    _, _, box = _box(tmp_path, placed=True)
    write = _tool(box, "position_sections")
    result = write([{"id": "s0.png", "position_mm": 3.2}])
    (entry,) = result["pictures"]
    assert set(entry) == {"id", "caption"}
    assert entry["caption"].startswith("positioning:") and "s0.png 3.20 mm" in entry["caption"]
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 1
    records = box.job.views.records()
    assert [(r.seq, r.tool) for r in records] == [(entry["id"], "position_sections")]

    quiet = write([{"id": "s1.png", "position_mm": 3.4}], view=False)
    assert quiet["status"] == "ok"
    assert "pictures" not in quiet and TOOL_MEDIA_PARTS_KEY not in quiet
    assert len(box.job.views.records()) == 1


def test_force_view_drops_the_view_argument_and_always_draws(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True, force_view=True)
    for name in ("position_sections", "interactive_transform", "elastix_affine"):
        assert "view" not in inspect.signature(_tool(box, name)).parameters, name
    refused = _tool(box, "position_sections")([{"id": "s0.png", "position_mm": 3.0}],
                                              view=False)
    assert refused["error"] == "UNKNOWN_ARGUMENTS"
    assert state.by_id("s0.png").position_mm == 2.0
    drawn = _tool(box, "position_sections")([{"id": "s0.png", "position_mm": 3.0}])
    assert len(drawn["pictures"]) == 1 and len(drawn[TOOL_MEDIA_PARTS_KEY]) == 1
    turned = _tool(box, "interactive_transform")([{"id": "s1.png", "rotation_deg": 2.0}])
    assert len(turned["pictures"]) == 1


# --- the look-before-commit gates -----------------------------------------------------


def _gated(tmp_path: Path, **kwargs):
    return _box(tmp_path, placed=True, tasks=["position"], position=PositionSpec(gated=True),
                **kwargs)


def test_gated_writes_need_an_overlay_or_positioning_look_first(tmp_path: Path):
    state, _, box = _gated(tmp_path)
    write = _tool(box, "position_sections")
    look = _tool(box, "look")
    refused = write([{"id": "s0.png", "position_mm": 3.0}])
    assert refused["status"] == "refused" and refused["error"] == "NOT_COMPARED"
    assert refused["rejected"][0]["id"] == "s0.png"
    assert "look at it in mode overlay or positioning" in refused["rejected"][0]["reason"]
    # A look at the section alone is not a comparison with the atlas, nor is
    # a refused look.
    look("section", sections=["s0.png"])
    assert look("overlay", sections=["s0.png"], warp="bent")["error"] == "BAD_WARP"
    assert write([{"id": "s0.png", "position_mm": 3.0}])["error"] == "NOT_COMPARED"
    assert state.by_id("s0.png").position_mm == 2.0

    look("overlay", sections=["s0.png"])
    assert write([{"id": "s0.png", "position_mm": 3.0}], view=False)["status"] == "ok"
    assert state.by_id("s0.png").position_mm == 3.0
    # The write resets the record: the section needs a new look.
    assert write([{"id": "s0.png", "position_mm": 3.2}])["error"] == "NOT_COMPARED"
    look("positioning", sections=["s0.png"], positions_mm=[3.2])
    assert write([{"id": "s0.png", "position_mm": 3.2}], view=False)["status"] == "ok"


def test_a_gated_call_writes_the_compared_and_names_the_rest(tmp_path: Path):
    state, _, box = _gated(tmp_path)
    _tool(box, "look")("overlay", sections=["s0.png"])
    result = _tool(box, "position_sections")(
        [{"id": "s0.png", "position_mm": 3.1}, {"id": "s1.png", "position_mm": 3.3}],
        view=False)
    assert result["status"] == "ok"
    assert result["written"] == [{"id": "s0.png", "position_mm": 3.1}]
    assert [row["id"] for row in result["rejected"]] == ["s1.png"]
    assert state.by_id("s1.png").position_mm == 2.5
    # Cutting angles alone pass the gate.
    angled = _tool(box, "position_sections")(cutting_angles={"pitch_deg": 1, "yaw_deg": 0},
                                             view=False)
    assert angled["status"] == "ok" and state.cutting_angles_deg["pitch"] == 1.0


def test_gated_submit_waits_for_a_positioning_review_of_every_section(tmp_path: Path):
    state, _, box = _gated(tmp_path)
    look = _tool(box, "look")
    result = _submit(box)
    assert result["status"] == "refused" and result["error"] == "NOT_REVIEWED"
    look("positioning", sections=["s0.png", "s1.png"])
    assert _submit(box)["error"] == "NOT_REVIEWED"  # not every section
    look("positioning")
    look("overlay", sections=["s1.png"])
    _tool(box, "position_sections")([{"id": "s1.png", "position_mm": 2.6}], view=False)
    assert _submit(box)["error"] == "NOT_REVIEWED"  # the write undid the review
    look("positioning")
    assert _submit(box)["status"] == "ok" and state.submitted is True


def test_undo_of_a_position_is_a_write_to_it_for_the_gates(tmp_path: Path):
    _, _, box = _gated(tmp_path)
    look = _tool(box, "look")
    write = _tool(box, "position_sections")
    look("overlay", sections=["s0.png"])
    write([{"id": "s0.png", "position_mm": 3.0}], view=False)
    look("overlay", sections=["s0.png"])
    look("positioning")
    assert box.compared.get("s0.png") and box.reviewed
    _tool(box, "undo")()  # s0 moves back
    assert "s0.png" not in box.compared and box.reviewed is False
    assert write([{"id": "s0.png", "position_mm": 3.0}])["error"] == "NOT_COMPARED"


# --- interactive_transform --------------------------------------------------------------


def test_one_section_is_written_shown_and_undone(tmp_path: Path):
    state, ctx, box = _box(tmp_path, placed=True)
    transform = single_transform(_tool(box, "interactive_transform"))
    state.by_id("s1.png").position_mm = None
    # A section with no position has nothing to align against.
    assert transform("s1.png", 2.0)["error"] == "NO_POSITION"

    row = transform("s0.png", 5.0, 1.1, 1.0, 0.25, -0.1)
    assert row["status"] == "ok" and row["id"] == "s0.png" and row["written"] is True
    assert row["transform"]["rotation_deg"] == 5.0 and row["transform"]["scale_x"] == 1.1
    assert row["orientation"] == {"flip": False, "rotate_quarter": 0, "changed": False}
    # The write and the look are one call: the result comes back as a picture.
    (picture,) = row["pictures"]
    assert "s0.png overlay" in picture["caption"]
    assert "interactive transform" in picture["caption"]
    assert len(row[TOOL_MEDIA_PARTS_KEY]) == 1
    assert row["changed"][0]["transform"] == "interactive"
    stored = state.by_id("s0.png").transform
    assert stored["kind"] == "interactive" and len(stored["params"]) == 6
    assert stored["physical"]["rotation_deg"] == 5.0
    # Nothing states a pixel size here, and the record says so.
    assert stored["calibration"]["source"] == "estimated"
    assert load_checkpoint(ctx.checkpoint_path).by_id("s0.png").transform is not None

    # The same numbers again draw, and write nothing.
    again = transform("s0.png", 5.0, 1.1, 1.0, 0.25, -0.1)
    assert again["written"] is False and len(again["pictures"]) == 1
    assert "changed" not in again
    assert len(box.job.undo_stack) == 1
    assert _tool(box, "undo")()["status"] == "ok"
    assert state.by_id("s0.png").transform is None


def test_values_left_out_keep_the_sections_current_ones(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    transform = single_transform(_tool(box, "interactive_transform"))
    transform("s0.png", 5.0, translate_x_mm=0.2)
    row = transform("s0.png", scale_x=1.2)
    assert row["transform"]["rotation_deg"] == 5.0
    assert row["transform"]["translate_x_mm"] == 0.2
    assert row["transform"]["scale_x"] == 1.2
    assert state.by_id("s0.png").transform["physical"]["rotation_deg"] == 5.0


def test_a_batch_is_one_undo_step(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    result = _tool(box, "interactive_transform")([
        {"id": "s0.png", "rotation_deg": 2.0, "translate_x_mm": 0.1},
        {"id": "s1.png", "rotation_deg": -3.0, "scale_x": 1.1, "scale_y": 0.9},
    ])
    assert result["status"] == "ok"
    assert [row["id"] for row in result["results"]] == ["s0.png", "s1.png"]
    assert len(result["pictures"]) == 2 and len(result[TOOL_MEDIA_PARTS_KEY]) == 2
    assert len(box.job.undo_stack) == 1
    assert state.by_id("s1.png").transform["physical"]["rotation_deg"] == -3.0
    _tool(box, "undo")()
    assert state.by_id("s0.png").transform is None
    assert state.by_id("s1.png").transform is None


def test_an_orientation_change_keeps_the_values_and_says_so(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    transform = single_transform(_tool(box, "interactive_transform"))
    transform("s0.png", 7.0)
    row = transform("s0.png", flip=True)
    assert row["orientation"] == {"flip": True, "rotate_quarter": 0, "changed": True}
    assert "The orientation changed." in row["message"]
    assert row["transform"]["rotation_deg"] == 7.0
    record = state.by_id("s0.png")
    assert record.flip is True and record.transform["physical"]["rotation_deg"] == 7.0
    # Without a transform, the orientation alone is set.
    quarter = transform("s1.png", rotate_quarter=90)
    assert quarter["written"] is True and "transform" not in quarter
    assert state.by_id("s1.png").rotation_deg == 90
    assert state.by_id("s1.png").transform is None
    assert transform("s2.png", rotate_quarter=45)["error"] == "BAD_ROTATION"


def test_flip_is_refused_when_the_spec_switches_it_off(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True, transform=TransformSpec(flip=False))
    result = _tool(box, "interactive_transform")(
        [{"id": "s0.png", "flip": True}, {"id": "s1.png", "rotate_quarter": 90}])
    assert result["results"][0] == {"status": "error", "error": "FLIP_DISABLED",
                                    "id": "s0.png"}
    assert result["results"][1]["status"] == "ok"
    assert state.by_id("s0.png").flip is False
    assert state.by_id("s1.png").rotation_deg == 90


def test_interactive_transform_refuses_bad_calls_without_writing(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    tool = _tool(box, "interactive_transform")
    duplicate = tool([{"id": "s0.png", "rotation_deg": 1}, {"id": "s0", "rotation_deg": 2}])
    assert duplicate["error"] == "DUPLICATE_SLICE_IDS"
    assert duplicate["duplicate_ids"] == ["s0.png"]
    many = tool([{"id": f"s{i}.png"} for i in range(5)])
    assert many == {"status": "error", "error": "TOO_MANY_SECTIONS", "max_sections": 4}
    assert tool([])["error"] == "BAD_ARGS"
    nothing = tool([{"id": "nope.png", "rotation_deg": 1}])
    assert nothing["status"] == "error" and nothing["error"] == "NOTHING_WRITTEN"
    assert nothing["results"][0]["error"] == "UNKNOWN_SLICE_IDS"
    assert "pictures" not in nothing
    assert all(record.transform is None for record in state.slices)
    assert box.job.undo_stack == []


def test_a_picture_that_fails_leaves_the_write_standing(tmp_path: Path, monkeypatch):
    import langslice.ops.look as ops_look

    state, ctx, box = _box(tmp_path, placed=True)

    def broken(*_args, **_kwargs):
        raise RuntimeError("render broke")

    monkeypatch.setattr(ops_look.looks, "look", broken)
    result = _tool(box, "interactive_transform")([{"id": "s0.png", "rotation_deg": 2.0},
                                                  {"id": "s1.png", "rotation_deg": 3.0}])
    assert result["status"] == "ok"
    assert [row["status"] for row in result["results"]] == ["ok", "ok"]
    assert [row["id"] for row in result["render_failed"]] == ["s0.png", "s1.png"]
    assert result["render_failed"][0]["message"] == "render broke"
    assert result["pictures"] == []
    assert state.by_id("s0.png").transform is not None
    assert load_checkpoint(ctx.checkpoint_path).by_id("s1.png").transform is not None
    assert len(box.job.undo_stack) == 1


def test_view_false_returns_no_picture(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    result = _tool(box, "interactive_transform")([{"id": "s0.png", "rotation_deg": 3.0}],
                                                 view=False)
    assert result["status"] == "ok" and result["results"][0]["written"] is True
    assert "pictures" not in result and TOOL_MEDIA_PARTS_KEY not in result
    assert state.by_id("s0.png").transform is not None
    assert box.job.views.records() == []


# --- mark_damage ------------------------------------------------------------------------


def _regions_box(tmp_path: Path, **kwargs):
    state, ctx, spec = _stack(tmp_path, n=3, atlas=SyntheticAtlas(), **kwargs)
    for record in state.slices:
        record.position_mm = 0.1
    return state, ctx, build_tools(state, ctx, spec)


def test_mark_damage_marks_regions_shows_them_and_undoes(tmp_path: Path):
    state, ctx, box = _regions_box(tmp_path)
    mark = _tool(box, "mark_damage")
    result = mark("s0.png", ["TH"], "torn")
    assert result["status"] == "ok" and result["written"] is True
    assert (result["id"], result["regions"], result["note"], result["damaged"]) == (
        "s0.png", ["TH"], "torn", True)
    assert result["changed"][0]["damaged_regions"] == ["TH"]
    # Its picture is saved by the door, with a number.
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 1 and len(result["pictures"]) == 1
    assert box.job.views.records()[0].tool == "mark_damage"
    record = state.by_id("s0.png")
    assert record.damaged and record.damaged_regions == ["TH"]
    assert load_checkpoint(ctx.checkpoint_path).by_id("s0.png").damaged_regions == ["TH"]

    again = mark("s0.png", ["TH"], "torn")
    assert again["written"] is False and len(box.job.undo_stack) == 1
    cleared = mark("s0.png", [], "")
    assert cleared["damaged"] is False and cleared["note"] == ""
    assert TOOL_MEDIA_PARTS_KEY not in cleared  # nothing marked, nothing to draw
    _tool(box, "undo")()
    assert state.by_id("s0.png").damaged_regions == ["TH"]
    _tool(box, "undo")()
    assert state.by_id("s0.png").damaged is False


def test_mark_damage_refuses_unknown_regions_and_sections(tmp_path: Path):
    state, _, box = _regions_box(tmp_path)
    mark = _tool(box, "mark_damage")
    assert mark("s0.png", ["XYZ"])["error"] == "UNKNOWN_REGIONS"
    assert mark("nope.png", ["TH"])["error"] == "UNKNOWN_SLICE_IDS"
    assert not any(record.damaged for record in state.slices)
    assert box.job.undo_stack == []


# --- note, undo, redo --------------------------------------------------------------------


def test_a_note_is_one_undoable_line(tmp_path: Path):
    state, _, box = _box(tmp_path)
    assert _tool(box, "undo")()["error"] == "NOTHING_TO_UNDO"
    result = _tool(box, "note")("sections 3-5 are faint")
    assert result["status"] == "ok" and result["notes"][-1] == "sections 3-5 are faint"
    assert state.notes[-1] == "sections 3-5 are faint"
    assert _tool(box, "undo")()["status"] == "ok"
    assert "sections 3-5 are faint" not in state.notes
    assert _tool(box, "redo")()["redo_depth"] == 0


# --- submit ------------------------------------------------------------------------------


def test_submit_refuses_an_unplaced_stack(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"])
    result = _submit(box)
    assert result["error"] == "MISSING_POSITIONS"
    assert len(result["missing_ids"]) == 5
    assert state.submitted is False


def test_submit_ends_the_run_and_escalates(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    context = ToolContext()
    result = _tool(box, "submit")("Placed.", ["faint"], [], tool_context=context)
    assert result["status"] == "ok" and len(result["rows"]) == 5
    assert context.actions.escalate is True
    assert state.submitted is True
    assert box.submission["summary"] == "Placed."


def test_submit_takes_positions_that_run_against_the_order(tmp_path: Path):
    """Order follows position: a dip in an increasing stack is not refused."""
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    state.by_id("s2.png").position_mm = 0.5
    assert _submit(box)["status"] == "ok"
    assert state.submitted is True


def test_submit_accepts_a_stack_that_runs_backwards_and_emits_atlas_order(tmp_path: Path):
    """Direction is the code's job: a posterior-first stack is reversed at submit."""
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    ordered = state.in_order()
    n = len(ordered)
    for index, record in enumerate(ordered):
        record.position_mm = 10.0 - index
    ids_before = [r.id for r in ordered]
    for record in ordered[3:]:  # a real gap between old index 2 and 3
        record.position_mm = float(record.position_mm) - 4.0
    result = _submit(box, interval_breaks=[ids_before[3]])
    assert result["status"] == "ok"
    after = state.in_order()
    assert [r.id for r in after] == ids_before[::-1]
    positions = [float(r.position_mm) for r in after]
    assert positions == sorted(positions)
    assert state.interval_breaks == [n - 3]  # the same physical gap
    assert box.submission["interval_breaks"] == [after[n - 3].id]
    assert any("reversed" in note for note in state.notes)


def test_submit_refuses_a_break_the_positions_do_not_show(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    result = _submit(box, interval_breaks=["s3.png"])
    assert result["error"] == "INTERVAL_BREAKS_UNSUPPORTED"
    assert result["failures"][0]["written_interval_mm"] == 0.5
    assert result["failures"][0]["id"] == "s3.png"
    numbered = _submit(box, interval_breaks=[3])
    assert numbered["error"] == "UNKNOWN_SLICE_IDS" and "filename" in numbered["message"]
    assert state.submitted is False


def test_submit_accepts_a_break_the_positions_do_show(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    for record in state.in_order()[3:]:
        record.position_mm = float(record.position_mm) + 4.0
    assert _submit(box, interval_breaks=["s3"])["status"] == "ok"
    assert state.interval_breaks == [3]
    assert box.submission["interval_breaks"] == ["s3.png"]


def test_strict_interval_refuses_uneven_spacing_and_any_break(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"], placed=True,
                         position=PositionSpec(strict_interval=True, interval_um=500))
    assert _submit(box)["status"] == "ok"  # 0.5 mm apart, exactly the interval
    state.submitted = False
    state.by_id("s4.png").position_mm = 5.0
    refusal = _submit(box)
    assert refusal["error"] == "STRICT_INTERVAL"
    assert refusal["failures"][0]["between"] == ["s3.png", "s4.png"]
    state.by_id("s4.png").position_mm = 4.0
    strict = _submit(box, interval_breaks=["s2.png"])
    assert strict["error"] == "STRICT_INTERVAL" and strict["reported_breaks"] == ["s2.png"]


def test_a_refused_submit_writes_nothing(tmp_path: Path):
    state, ctx, box = _box(tmp_path, tasks=["position"], placed=True)
    state.by_id("s2.png").position_mm = None
    assert _submit(box)["error"] == "MISSING_POSITIONS"
    state.by_id("s2.png").position_mm = 3.0
    assert _submit(box, interval_breaks=["s3.png"])["error"] == "INTERVAL_BREAKS_UNSUPPORTED"
    assert state.submitted is False
    assert box.job.undo_stack == []
    assert load_checkpoint(ctx.checkpoint_path) is None


def test_the_position_gates_do_not_apply_when_position_is_off(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["reorder"])
    assert _submit(box)["status"] == "ok"
    assert state.submitted is True


def test_submit_names_the_sections_with_no_transform(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["transform"], placed=True)
    refusal = _submit(box)
    assert refusal["error"] == "MISSING_TRANSFORMS"
    assert len(refusal["missing_ids"]) == 5
    assert state.submitted is False
    for record in state.slices:
        record.transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    assert _submit(box)["status"] == "ok"


def test_left_linear_belongs_to_runs_with_nonlinear(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    # Without Nonlinear the argument is not offered at all.
    assert "left_linear" not in inspect.signature(_tool(box, "submit")).parameters
    refused = _tool(box, "submit")("done", [], [], left_linear=[
        {"id": "s0.png", "reason": "no deformation needed"}])
    assert refused["error"] == "UNKNOWN_ARGUMENTS"
    assert state.submitted is False
