"""The toolbox: gating, undo/redo, the reorder rule, and the submit gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.core.spec import JobSpec, PositionSpec, TransformSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.checkpoint import load_checkpoint
from langslice.job.job import ingest
from tests.fakes import EllipseAtlas, SlabAtlas, ellipse_section
from tests.linear_tool_helpers import single_adjust

_ATLAS = SlabAtlas()


class _Actions:
    escalate = False


class _ToolContext:
    def __init__(self, function_call_id: str | None = None) -> None:
        self.actions = _Actions()
        self.function_call_id = function_call_id


def _stack(folder: Path, n: int = 5, *, placed: bool = False, **spec_kwargs):
    for index in range(n):
        Image.fromarray(
            np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)
        ).save(folder / f"s{index}.png")
    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none", **spec_kwargs
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    state = ingest(spec, ctx)
    if placed:
        for index, record in enumerate(state.in_order()):
            record.position_mm = 2.0 + 0.5 * index
    return state, ctx, spec


def _box(folder: Path, **kwargs):
    state, ctx, spec = _stack(folder, **kwargs)
    return state, ctx, build_tools(state, ctx, spec)


def _tool(box, name: str) -> Any:
    return next(tool for tool in box.tools if tool.__name__ == name)


# --- gating --------------------------------------------------------------


def test_a_position_only_spec_has_no_reorder_or_transform_tools(tmp_path: Path):
    _, _, box = _box(tmp_path, tasks=["position"])
    names = set(box.names)
    assert {"status", "view_slices", "view_atlas", "set_positions", "submit"} <= names
    assert not names & {"reorder_slices", "move_slice", "orient_slices"}
    assert not names & {"fit_affine", "adjust_transforms", "view_landmarks",
        "edit_landmarks", "warp_landmarks"}


def test_optional_tools_follow_their_flags(tmp_path: Path):
    _, _, box = _box(
        tmp_path,
        position=PositionSpec(deepslice=True, bayesian=True),
        transform=TransformSpec(angles=True),
    )
    names = set(box.names)
    assert {"search_position", "set_cutting_angles"} <= names

    _, _, plain = _box(tmp_path)
    assert not set(plain.names) & {"search_position", "set_cutting_angles"}
    # The interactive transform rides in the main trajectory, always on with
    # the task.
    assert not {"adjust_transform", "unmark_damaged", "validate", "move_slice"} & set(plain.names)
    assert "adjust_transforms" in plain.names
    assert not set(plain.names) & {"view_landmarks", "edit_landmarks", "warp_landmarks"}
    from google.adk.tools import FunctionTool

    schemas = [FunctionTool(tool)._get_declaration().model_dump() for tool in plain.tools]
    assert len(schemas) == 15
    assert "landmark" not in str(schemas).lower()

    _, _, refinement = _box(
        tmp_path, tasks=["transform"],
        transform=TransformSpec(interactive=True, automatic=False),
    )
    assert len(refinement.tools) == 11
    assert "orient_slices" in refinement.names
    # The read-only view of the complete registration exists without positioning.
    assert "view_placement" in refinement.names and "set_positions" not in refinement.names


def test_orientation_is_a_transform_tool_not_a_positioning_one(tmp_path: Path):
    # 2026-09-29: a mirror is the sign of the in-plane affine, so flip and
    # rotation belong to the transform task.
    _, _, positioning = _box(tmp_path, tasks=["reorder", "position"])
    assert "reorder_slices" in positioning.names
    assert "orient_slices" not in positioning.names
    _, _, linear = _box(tmp_path, tasks=["transform"])
    assert "orient_slices" in linear.names
    assert "reorder_slices" not in linear.names


def test_flip_is_refused_when_the_spec_switches_it_off(tmp_path: Path):
    state, _, box = _box(tmp_path, transform=TransformSpec(flip=False))
    result = _tool(box, "orient_slices")(
        [{"id": "s0.png", "flip": True}, {"id": "s1.png", "rotate_deg": 90}]
    )
    assert result["rejected"] == [{"id": "s0.png", "error": "FLIP_DISABLED"}]
    assert state.by_id("s0.png").flip is False
    assert state.by_id("s1.png").rotation_deg == 90


# --- writes, checkpoints, undo/redo --------------------------------------


def test_a_write_checkpoints_the_whole_state(tmp_path: Path):
    state, ctx, box = _box(tmp_path)
    assert load_checkpoint(ctx.checkpoint_path) is None
    _tool(box, "set_positions")([{"id": "s0.png", "position_mm": 3.0}])
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved is not None
    assert saved.by_id("s0.png").position_mm == 3.0
    assert state.by_id("s0.png").position_mm == 3.0


def test_a_batch_write_undoes_and_redoes_as_one_step(tmp_path: Path):
    state, _, box = _box(tmp_path)
    _tool(box, "set_positions")(
        [
            {"id": "s0.png", "position_mm": 1.0},
            {"id": "s1.png", "position_mm": 2.0},
            {"id": "s2.png", "position_mm": 3.0},
        ]
    )
    assert [s.position_mm for s in state.in_order()][:3] == [1.0, 2.0, 3.0]

    assert _tool(box, "undo")()["status"] == "ok"
    assert all(s.position_mm is None for s in state.slices)

    assert _tool(box, "redo")()["status"] == "ok"
    assert [s.position_mm for s in state.in_order()][:3] == [1.0, 2.0, 3.0]

    assert _tool(box, "undo")()["status"] == "ok"
    assert _tool(box, "undo")()["error"] == "NOTHING_TO_UNDO"


def test_a_failed_write_leaves_no_undo_step(tmp_path: Path):
    _, _, box = _box(tmp_path)
    result = _tool(box, "set_positions")([{"id": "nope.png", "position_mm": 1.0}])
    assert result["error"] == "NOTHING_WRITTEN"
    assert box.job.undo_stack == []


def test_set_positions_clamps_to_the_atlas_range(tmp_path: Path):
    state, _, box = _box(tmp_path)
    result = _tool(box, "set_positions")([{"id": "s0.png", "position_mm": 500.0}])
    assert result["clamped"][0]["clamped_to_mm"] == 19.0
    assert state.by_id("s0.png").position_mm == 19.0


# --- what a write answers with -------------------------------------------


def test_writes_return_the_picture_of_what_they_did(tmp_path: Path):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    _, _, box = _box(tmp_path)
    placed = _tool(box, "set_positions")(
        [{"id": "s0.png", "position_mm": 1.0}, {"id": "s3.png", "position_mm": 6.0}]
    )
    # A pair per written section: the section, then the atlas at its position.
    assert len(placed[TOOL_MEDIA_PARTS_KEY]) == 2  # one stitched picture per section
    assert "s0.png, s3.png" in placed["description"]
    assert "unpictured" not in placed

    turned = _tool(box, "orient_slices")([{"id": "s1.png", "flip": True}])
    assert len(turned[TOOL_MEDIA_PARTS_KEY]) == 1
    assert "s1.png" in turned["description"]


def test_set_positions_suppresses_only_a_placement_seen_in_an_earlier_round(
    tmp_path: Path,
):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    state, _, box = _box(tmp_path, tasks=["position"])
    compare = _tool(box, "view_placement")
    write = _tool(box, "set_positions")

    compared = compare(
        [{"id": "s0.png", "positions_mm": [3.0]}],
        tool_context=_ToolContext("adk-compare-1"),
    )
    assert len(compared[TOOL_MEDIA_PARTS_KEY]) == 1

    # Both calls emitted by one model response are planned before either
    # result is visible, so a same-round write still returns its own picture.
    same_round = write(
        [{"id": "s0.png", "position_mm": 3.0}],
        tool_context=_ToolContext("write-1"),
    )
    assert len(same_round[TOOL_MEDIA_PARTS_KEY]) == 1

    # The opaque ids ride in response JSON even where ADK strips generated
    # FunctionResponse ids from replayed history.
    box.mark_placement_views_delivered({"adk-compare-1", "write-1"})
    later = write(
        [{"id": "s0.png", "position_mm": 3.0}],
        tool_context=_ToolContext("write-2"),
    )
    assert later[TOOL_MEDIA_PARTS_KEY] == []
    assert later["images_suppressed_seen"] == ["s0.png"]
    assert state.by_id("s0.png").position_mm == 3.0


def test_seen_placement_identity_includes_orientation_and_cutting_angles(
    tmp_path: Path,
):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    state, _, box = _box(tmp_path, tasks=["position"])
    compare = _tool(box, "view_placement")
    write = _tool(box, "set_positions")
    compare(
        [{"id": "s0.png", "positions_mm": [3.0]}],
        tool_context=_ToolContext("compare-1"),
    )
    box.mark_placement_views_delivered({"compare-1"})

    state.by_id("s0.png").flip = True
    reoriented = write(
        [{"id": "s0.png", "position_mm": 3.0}],
        tool_context=_ToolContext("write-1"),
    )
    assert len(reoriented[TOOL_MEDIA_PARTS_KEY]) == 1
    box.mark_placement_views_delivered({"write-1"})

    state.cutting_angles_deg = {"pitch": 2.0, "yaw": 0.0}
    reangled = write(
        [{"id": "s0.png", "position_mm": 3.0}],
        tool_context=_ToolContext("write-2"),
    )
    assert len(reangled[TOOL_MEDIA_PARTS_KEY]) == 1


def test_seen_placement_uses_the_actual_position_not_rounded_reply_text(
    tmp_path: Path,
):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    _state, _, box = _box(tmp_path, tasks=["position"])
    compare = _tool(box, "view_placement")
    write = _tool(box, "set_positions")
    compare(
        [{"id": "s0.png", "positions_mm": [3.0004]}],
        tool_context=_ToolContext("compare-1"),
    )
    box.mark_placement_views_delivered({"compare-1"})

    different = write(
        [{"id": "s0.png", "position_mm": 3.00049}],
        tool_context=_ToolContext("write-1"),
    )
    assert len(different[TOOL_MEDIA_PARTS_KEY]) == 1


def test_only_a_full_atlas_bearing_compare_suppresses_the_write_image(
    tmp_path: Path,
):
    from langslice.doors.tools import TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY

    _state, _, box = _box(tmp_path, tasks=["position"])
    compare = _tool(box, "view_placement")
    write = _tool(box, "set_positions")

    section_only = compare(
        [{"id": "s0.png", "positions_mm": [3.0]}],
        view={"mode": "section"},
        tool_context=_ToolContext("section-only"),
    )
    assert TOOL_MEDIA_DELIVERY_ID_KEY not in section_only
    after_section = write(
        [{"id": "s0.png", "position_mm": 3.0}],
        tool_context=_ToolContext("write-0"),
    )
    assert len(after_section[TOOL_MEDIA_PARTS_KEY]) == 1

    zoomed = compare(
        [{"id": "s1.png", "positions_mm": [3.5]}],
        view={"mode": "overlay", "zoom": [0.0, 0.0, 0.5, 0.5]},
        tool_context=_ToolContext("zoomed"),
    )
    assert TOOL_MEDIA_DELIVERY_ID_KEY not in zoomed
    after_zoom = write(
        [{"id": "s1.png", "position_mm": 3.5}],
        tool_context=_ToolContext("write-1"),
    )
    assert len(after_zoom[TOOL_MEDIA_PARTS_KEY]) == 1

    full = compare(
        [{"id": "s2.png", "positions_mm": [4.0]}],
        view={"mode": "template"},
        tool_context=_ToolContext("full"),
    )
    assert full[TOOL_MEDIA_DELIVERY_ID_KEY] == "full"
    box.mark_placement_views_delivered({"full"})
    after_full = write(
        [{"id": "s2.png", "position_mm": 4.0}],
        tool_context=_ToolContext("write-2"),
    )
    assert after_full[TOOL_MEDIA_PARTS_KEY] == []


def test_historical_delivery_token_cannot_promote_a_new_pending_call(
    tmp_path: Path,
):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    _state, _, box = _box(tmp_path, tasks=["position"])
    compare = _tool(box, "view_placement")
    compare(
        [{"id": "s0.png", "positions_mm": [3.0]}],
        tool_context=_ToolContext("new-call"),
    )

    # A replayed old response may still carry media, but its stable token is
    # different and cannot make the new, possibly-pruned response look seen.
    box.mark_placement_views_delivered({"old-call"})
    result = _tool(box, "set_positions")(
        [{"id": "s0.png", "position_mm": 3.0}],
        tool_context=_ToolContext("write-1"),
    )
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 1


def test_failed_compare_is_not_counted_as_seen_or_compared(
    tmp_path: Path, monkeypatch,
):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    _state, _, box = _box(tmp_path, tasks=["position"])

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("render broke")

    monkeypatch.setattr("langslice.core.placement.physical_views", fail_render)
    result = _tool(box, "view_placement")(
        [{"id": "s0.png", "positions_mm": [3.0]}],
        tool_context=_ToolContext("compare-1"),
    )
    assert result["error"] == "RENDER_FAILED"
    assert "s0.png" not in box.compared
    assert box.pending_placement_views == {}

    written = _tool(box, "set_positions")(
        [{"id": "s0.png", "position_mm": 3.0}],
        tool_context=_ToolContext("write-1"),
    )
    assert len(written[TOOL_MEDIA_PARTS_KEY]) == 1


def test_view_stack_orders_by_position_and_plots_it(tmp_path: Path):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    state, ctx, box = _box(tmp_path, placed=True)
    state.by_id("s0.png").position_mm = 9.0  # placed after the others
    result = _tool(box, "view_stack")()
    assert [row["id"] for row in result["rows"]][-1] == "s0.png"
    parts = result[TOOL_MEDIA_PARTS_KEY]
    # Two images however big the stack: the contact sheet and the plot.
    assert sum(1 for item in parts if not isinstance(item, str)) == 2
    # The sheet's labels carry position and spacing.
    from langslice.core.sheets import stack_pictures

    labels = [label for label, _ in stack_pictures(state, ctx, by_position=True)]
    assert labels[-1].startswith("0: s0.png  9.00 mm")
    assert "to next)" in labels[0]


def test_a_write_returns_only_the_rows_it_touched(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    result = _tool(box, "set_positions")(
        [{"id": "s0.png", "position_mm": 1.0}, {"id": "s3.png", "position_mm": 6.0}]
    )
    assert [row["id"] for row in result["changed"]] == ["s0.png", "s3.png"]
    assert result["n_sections"] == 5
    assert "rows" not in result  # the whole table is `status`, one call away

    marked = _tool(box, "mark_damaged")([{"id": "s2.png", "note": "torn"}])
    assert [row["id"] for row in marked["changed"]] == ["s2.png"]
    assert marked["changed"][0]["damaged"] is True

    # status, undo and redo still answer with the whole stack.
    assert len(_tool(box, "status")()["rows"]) == 5
    assert len(_tool(box, "undo")()["rows"]) == 5
    assert len(_tool(box, "redo")()["rows"]) == 5
    assert state.by_id("s2.png").damaged is True


def test_confidence_is_gone_from_the_package():
    """Nash: "What use is there for confidence?" — none downstream."""
    import langslice.agent
    import langslice.core
    import langslice.doors.tools
    import langslice.job

    # The former linear package's modules, in their layer packages.
    packages = [Path(module.__file__).parent for module in (
        langslice.core, langslice.job, langslice.doors.tools, langslice.agent)]
    hits = [
        path.name
        for package in packages
        for path in sorted(package.glob("*.py"))
        if "confidence" in path.read_text(encoding="utf-8")
    ]
    assert hits == []


# --- the reorder rule ----------------------------------------------------


def test_reorder_keeps_positions_and_transforms(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    for record in state.slices:
        record.transform = {"kind": "silhouette", "params": [], "iou": 0.8}
    before = {s.id: s.position_mm for s in state.slices}
    order = ["s0.png", "s2.png", "s1.png", "s3.png", "s4.png"]
    result = _tool(box, "reorder_slices")(order)

    assert [s.id for s in state.in_order()] == order
    assert sorted(result["moved"]) == ["s1.png", "s2.png"]  # index changed
    assert "cleared_positions" not in result
    # Nothing but the corrected index moves; the submit gate is what holds
    # order and position together.
    assert {s.id: s.position_mm for s in state.slices} == before
    assert all(s.transform is not None for s in state.slices)


def test_reorder_moves_one_section_and_keeps_every_position(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    before_positions = {s.id: s.position_mm for s in state.slices}
    result = _tool(box, "reorder_slices")(["s3.png"])
    assert [s.id for s in state.in_order()] == [
        "s3.png", "s0.png", "s1.png", "s2.png", "s4.png",
    ]
    # Every section it passed is renumbered; nothing loses its position.
    assert sorted(result["moved"]) == ["s0.png", "s1.png", "s2.png", "s3.png"]
    assert {s.id: s.position_mm for s in state.slices} == before_positions

    result = _tool(box, "reorder_slices")(["s3.png"], after="s4.png")
    assert result["status"] == "ok"
    assert [s.id for s in state.in_order()] == [
        "s0.png", "s1.png", "s2.png", "s4.png", "s3.png",
    ]
    assert {s.id: s.position_mm for s in state.slices} == before_positions


@pytest.mark.parametrize(
    ("after", "expected"),
    [
        ("start", ["s4.png", "s1.png", "s0.png", "s2.png", "s3.png"]),
        ("s2.png", ["s0.png", "s2.png", "s4.png", "s1.png", "s3.png"]),
        ("s3.png", ["s0.png", "s2.png", "s3.png", "s4.png", "s1.png"]),
    ],
)
def test_reorder_moves_a_block_and_checkpoints_one_undo_step(
    tmp_path: Path, after: str, expected: list[str],
):
    state, ctx, box = _box(tmp_path, placed=True)
    before = state.to_dict()
    result = _tool(box, "reorder_slices")(["s4.png", "s1.png"], after=after)
    assert result["status"] == "ok"
    assert [s.id for s in state.in_order()] == expected
    assert [s.index_corrected for s in state.in_order()] == list(range(5))
    assert len(box.job.undo_stack) == 1
    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved is not None
    assert saved.to_dict() == state.to_dict()
    assert _tool(box, "undo")()["status"] == "ok"
    assert state.to_dict() == before
    assert not box.job.undo_stack


@pytest.mark.parametrize(
    ("new_order", "after"),
    [
        ([], "start"),
        (["s1.png", "s1.png"], "start"),
        (["s1.png", "missing.png"], "start"),
        (["0", "2", "1", "3", "4"], "start"),
        (["s1.png"], "missing.png"),
        (["s1.png"], "0"),
        (["s1.png", "s3.png"], "s3.png"),
    ],
)
def test_reorder_rejects_invalid_input_without_writing_or_changing_history(
    tmp_path: Path, new_order: list[str], after: str,
):
    state, ctx, box = _box(tmp_path, placed=True)
    _tool(box, "note")("keep this undo step")
    _tool(box, "note")("keep this redo step")
    _tool(box, "undo")()
    before = state.to_dict()
    checkpoint_before = Path(ctx.checkpoint_path).read_bytes()
    undo_before = list(box.job.undo_stack)
    redo_before = list(box.job.redo_stack)

    result = _tool(box, "reorder_slices")(new_order, after=after)

    assert result["status"] == "error"
    assert result["error"]
    assert state.to_dict() == before
    assert Path(ctx.checkpoint_path).read_bytes() == checkpoint_before
    assert box.job.undo_stack == undo_before
    assert box.job.redo_stack == redo_before


# --- the submit gates ----------------------------------------------------


def _submit(box, **kwargs):
    args = {"summary": "done", "notes": [], "interval_breaks": []}
    args.update(kwargs)
    return _tool(box, "submit")(**args, tool_context=_ToolContext())


def test_submit_refuses_an_unplaced_stack(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"])
    result = _submit(box)
    assert result["error"] == "MISSING_POSITIONS"
    assert len(result["missing_ids"]) == 5
    assert state.submitted is False


def test_submit_refuses_positions_that_run_against_the_order(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    state.by_id("s2.png").position_mm = 0.5  # a dip in an increasing stack
    result = _submit(box)
    assert result["error"] == "ORDER_POSITION_MISMATCH"
    assert result["pairs"] == [
        {
            "before": "s1.png",
            "after": "s2.png",
            "before_position_mm": 2.5,
            "after_position_mm": 0.5,
        }
    ]
    assert state.submitted is False


def test_submit_accepts_a_stack_that_runs_backwards_and_emits_atlas_order(tmp_path: Path):
    """Direction is the code's job: a posterior-first stack is reversed at submit."""
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    ordered = state.in_order()
    n = len(ordered)
    for index, record in enumerate(ordered):
        record.position_mm = 10.0 - index
    ids_before = [r.id for r in ordered]
    # a real gap between old index 2 and 3 (positions 8.0 -> 3.0)
    for record in ordered[3:]:
        record.position_mm = float(record.position_mm) - 4.0
    result = _submit(box, interval_breaks=[3])
    assert result["status"] == "ok"
    assert state.submitted is True
    after = state.in_order()
    assert [r.id for r in after] == ids_before[::-1]
    positions = [float(r.position_mm) for r in after]
    assert positions == sorted(positions)
    # the break still names the section after the same physical gap
    assert state.interval_breaks == [n - 3]
    assert any("reversed" in note for note in state.notes)


def test_submit_refuses_a_break_the_positions_do_not_show(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    result = _submit(box, interval_breaks=[3])
    assert result["error"] == "INTERVAL_BREAKS_UNSUPPORTED"
    assert result["failures"][0]["written_interval_mm"] == 0.5
    assert state.submitted is False


def test_submit_accepts_a_break_the_positions_do_show(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["position"], placed=True)
    for record in state.in_order()[3:]:
        record.position_mm = float(record.position_mm) + 4.0
    assert _submit(box, interval_breaks=[3])["status"] == "ok"
    assert state.interval_breaks == [3]


def test_strict_interval_refuses_uneven_spacing_and_any_break(tmp_path: Path):
    state, ctx, spec = _stack(
        tmp_path,
        tasks=["position"],
        placed=True,
        position=PositionSpec(strict_interval=True, interval_um=500),
    )
    box = build_tools(state, ctx, spec)
    assert _submit(box)["status"] == "ok"  # 0.5 mm apart, exactly the interval

    state.submitted = False
    state.by_id("s4.png").position_mm = 5.0
    refusal = _submit(box)
    assert refusal["error"] == "STRICT_INTERVAL"
    assert refusal["failures"][0]["between"] == ["s3.png", "s4.png"]

    state.by_id("s4.png").position_mm = 4.0
    assert _submit(box, interval_breaks=[2])["error"] == "STRICT_INTERVAL"


def test_refused_submit_runs_the_gates_without_writing(tmp_path: Path):
    state, ctx, box = _box(tmp_path, tasks=["position"], placed=True)

    state.by_id("s2.png").position_mm = None
    assert _submit(box)["error"] == "MISSING_POSITIONS"

    state.by_id("s2.png").position_mm = 0.5  # placed, but against the order
    assert _submit(box)["error"] == "ORDER_POSITION_MISMATCH"

    state.by_id("s2.png").position_mm = 3.0
    assert _submit(box, interval_breaks=[3])["error"] == "INTERVAL_BREAKS_UNSUPPORTED"

    # Writes nothing: no checkpoint, no undo step, no submission.
    assert state.submitted is False
    assert box.job.undo_stack == []
    assert load_checkpoint(ctx.checkpoint_path) is None


def test_gates_do_not_apply_when_position_is_off(tmp_path: Path):
    state, _, box = _box(tmp_path, tasks=["reorder"])
    assert _submit(box)["status"] == "ok"
    assert state.submitted is True


# --- fits ----------------------------------------------------------------


def test_fit_affine_records_a_transform_and_refuses_damaged_sections(tmp_path: Path):
    atlas = EllipseAtlas()
    for index in range(2):
        ellipse_section().save(tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    for record in state.slices:
        record.position_mm = 10.0
    state.by_id("s1.png").damaged = True
    box = build_tools(state, ctx, spec)

    result = _tool(box, "fit_affine")([], "silhouette")
    assert result["status"] == "ok"
    assert [row["id"] for row in result["results"]] == ["s0.png"]  # damaged is skipped
    assert result["results"][0]["iou"] > 0.5
    # One representation: a fit reports the same five knobs adjust_transforms takes.
    assert set(result["results"][0]["physical"]) == {
        "rotation_deg", "scale_x", "scale_y", "shear",
        "translate_x_mm", "translate_y_mm", "pivot",
    }
    assert "decomposition" not in result["results"][0]
    assert state.by_id("s0.png").transform["kind"] == "silhouette"
    assert len(state.by_id("s0.png").transform["params"]) == 6
    assert state.by_id("s0.png").transform["mirrored"] == (
        result["results"][0]["mirrored"]
    )
    assert state.by_id("s0.png").transform["physical"]["rotation_deg"] is not None
    assert state.by_id("s1.png").transform is None
    assert "changed" not in result  # fit results already identify every written transform

    named = _tool(box, "fit_affine")(["s1.png"], "silhouette")
    assert named["results"][0]["error"] == "DAMAGED"

    # The default method (elastix) refuses damaged sections the same way.
    assert _tool(box, "fit_affine")(["s1.png"])["results"][0]["error"] == "DAMAGED"
    assert _tool(box, "fit_affine")([], "spline")["error"] == "BAD_ARGS"


def test_one_entry_adjustment_writes_shows_and_undoes(tmp_path: Path):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    atlas = EllipseAtlas()
    for index in range(2):
        ellipse_section().save(tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", preprocess="none")
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    state.by_id("s0.png").position_mm = 10.0
    state.by_id("s0.png").damaged = True  # the hand path is for exactly these
    box = build_tools(state, ctx, spec)

    adjust = single_adjust(_tool(box, "adjust_transforms"))
    # A section with no position has nothing to align against.
    assert adjust("s1.png", 0.0, 1.0, 1.0, 0.0, 0.0)["error"] == "NO_POSITION"

    result = adjust(
        "s0.png", 5.0, 1.1, 1.0, 0.25, -0.1, note="lined up the intact border"
    )
    assert result["status"] == "ok"
    # The write and the look are one call: the result comes back as a picture.
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 1
    assert result["written"] is True
    assert result["id"] == "s0.png"
    assert "changed" not in result  # the physical result is the canonical reply
    transform = state.by_id("s0.png").transform
    assert transform["kind"] == "interactive"
    assert len(transform["params"]) == 6
    assert transform["physical"]["rotation_deg"] == 5.0
    assert transform["note"] == "lined up the intact border"
    # Nothing states a pixel size here, and the payload says so rather than
    # pretending.
    assert transform["calibration"]["source"] == "estimated"
    assert load_checkpoint(ctx.checkpoint_path).by_id("s0.png").transform is not None

    # The same numbers again are a look, not a write: one undo clears the lot.
    again = adjust("s0.png", 5.0, 1.1, 1.0, 0.25, -0.1, "side_by_side")
    assert len(again[TOOL_MEDIA_PARTS_KEY]) == 2 and again["written"] is False
    assert _tool(box, "undo")()["status"] == "ok"
    assert state.by_id("s0.png").transform is None


def test_adjust_transforms_batches_independent_sections_as_one_undo_step(
    tmp_path: Path,
):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    state, _, box = _box(tmp_path, placed=True)
    adjust_many = _tool(box, "adjust_transforms")
    result = adjust_many(
        [
            {
                "id": "s0.png",
                "rotation_deg": 2.0,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "translate_x_mm": 0.1,
                "translate_y_mm": 0.0,
            },
            {
                "id": "s1.png",
                "rotation_deg": -3.0,
                "scale_x": 1.1,
                "scale_y": 0.9,
                "translate_x_mm": 0.0,
                "translate_y_mm": -0.1,
            },
        ],
        view={"mode": "outlines"},
    )
    assert result["status"] == "ok"
    assert [row["id"] for row in result["results"]] == ["s0.png", "s1.png"]
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 2
    assert len(box.job.undo_stack) == 1
    assert state.by_id("s0.png").transform["physical"]["rotation_deg"] == 2.0
    assert state.by_id("s1.png").transform["physical"]["rotation_deg"] == -3.0

    assert _tool(box, "undo")()["status"] == "ok"
    assert state.by_id("s0.png").transform is None
    assert state.by_id("s1.png").transform is None


def test_adjust_transforms_refuses_two_planned_edits_to_the_same_section(
    tmp_path: Path,
):
    state, _, box = _box(tmp_path, placed=True)
    base = {
        "rotation_deg": 0.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x_mm": 0.0,
        "translate_y_mm": 0.0,
    }
    result = _tool(box, "adjust_transforms")(
        [{"id": "s0.png", **base}, {"id": "0", **base}]
    )
    assert result["error"] == "DUPLICATE_SLICE_IDS"
    assert state.by_id("s0.png").transform is None
    assert box.job.undo_stack == []


def test_adjust_transforms_checkpoints_successes_when_another_render_fails(
    tmp_path: Path, monkeypatch,
):
    import langslice.core.placement as placement_module

    state, ctx, box = _box(tmp_path, placed=True)
    real_views = placement_module.physical_views
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second render broke")
        return real_views(*args, **kwargs)

    monkeypatch.setattr(placement_module, "physical_views", fail_second)
    base = {
        "rotation_deg": 0.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x_mm": 0.0,
        "translate_y_mm": 0.0,
    }
    result = _tool(box, "adjust_transforms")(
        [{"id": "s0.png", **base}, {"id": "s1.png", **base}]
    )

    assert [row["status"] for row in result["results"]] == ["ok", "error"]
    assert state.by_id("s0.png").transform is not None
    assert state.by_id("s1.png").transform is None
    assert load_checkpoint(ctx.checkpoint_path).by_id("s0.png").transform is not None
    assert len(box.job.undo_stack) == 1
    _tool(box, "undo")()
    assert state.by_id("s0.png").transform is None


def test_one_entry_adjustment_render_failure_does_not_mutate_state(
    tmp_path: Path, monkeypatch,
):
    state, ctx, box = _box(tmp_path, placed=True)

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("render broke")

    monkeypatch.setattr("langslice.core.placement.physical_views", fail_render)
    result = single_adjust(_tool(box, "adjust_transforms"))(
        "s0.png", 2.0, 1.0, 1.0, 0.0, 0.0
    )
    assert result["error"] == "RENDER_FAILED"
    assert state.by_id("s0.png").transform is None
    assert load_checkpoint(ctx.checkpoint_path) is None
    assert box.job.undo_stack == []


def test_submit_names_the_sections_with_no_transform(tmp_path: Path):
    state, ctx, box = _box(tmp_path, tasks=["transform"], placed=True)

    refusal = _submit(box)
    assert refusal["error"] == "MISSING_TRANSFORMS"
    assert len(refusal["missing_ids"]) == 5
    assert _submit(box)["error"] == "MISSING_TRANSFORMS"
    assert state.submitted is False

    for record in state.slices:
        record.transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    assert _submit(box)["status"] == "ok"
    assert state.submitted is True


def test_orient_change_clears_a_stale_transform(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    for record in state.slices:
        record.transform = {"kind": "silhouette", "params": [], "iou": 0.8}
    result = _tool(box, "orient_slices")(
        [{"id": "s0.png", "flip": True}, {"id": "s1.png", "flip": False}]
    )
    assert result["cleared_transforms"] == ["s0.png"]
    assert state.by_id("s0.png").transform is None
    assert state.by_id("s1.png").transform is not None  # unchanged orientation


def test_view_atlas_images_are_section_sized(tmp_path: Path):
    from langslice.core.atlas_fetch import atlas_section
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    state, ctx, box = _box(tmp_path)
    result = _tool(box, "view_atlas")([1.0])
    assert result["status"] == "ok"
    image = result[TOOL_MEDIA_PARTS_KEY][0]
    native = atlas_section(ctx, state, 1.0, frame=True)
    # The later-picture size at most, and never upsampled past the plane's voxels.
    shrink = min(1.0, 512 / max(native.size))
    assert abs(image.width - native.width * shrink) <= 1


def test_tools_keep_their_identity_and_run_one_at_a_time():
    """ADK runs a turn's several calls concurrently; every tool shares one
    state, so the box serializes them without changing what ADK sees."""
    import threading
    import time

    from langslice.doors.tools.toolbox import _serialized

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
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert inside["peak"] == 1


# --- the look-before-you-write gates ---------------------------------------


def test_gated_set_positions_refuses_an_uncompared_section(tmp_path: Path):
    from langslice.core.spec import PositionSpec

    state, _, box = _box(tmp_path, tasks=["position"], position=PositionSpec(gated=True))
    result = _tool(box, "set_positions")([{"id": "s0.png", "position_mm": 3.0}])
    assert result["status"] == "error" and result["error"] == "NOTHING_WRITTEN"
    assert "view_placement first" in result["rejected"][0]["reason"]
    assert state.by_id("s0.png").position_mm is None
    # one compare is enough (Astra confirms at one hypothesised position);
    # the write then resets the record
    _tool(box, "view_placement")([{"id": "s0.png", "positions_mm": [3.0]}])
    result = _tool(box, "set_positions")([{"id": "s0.png", "position_mm": 3.0}])
    assert result["status"] == "ok" and state.by_id("s0.png").position_mm == 3.0
    result = _tool(box, "set_positions")([{"id": "s0.png", "position_mm": 3.2}])
    assert result["status"] == "error"


def test_gated_submit_waits_for_a_review_after_the_last_write(tmp_path: Path):
    from langslice.core.spec import PositionSpec

    state, _, box = _box(
        tmp_path, placed=True, tasks=["position"], position=PositionSpec(gated=True)
    )
    result = _submit(box)
    assert result["status"] == "refused" and result["error"] == "NOT_REVIEWED"
    _tool(box, "view_stack")()
    _tool(box, "view_placement")([{"id": "s1.png", "positions_mm": [2.6]}])
    _tool(box, "set_positions")([{"id": "s1.png", "position_mm": 2.6}])
    assert _submit(box)["error"] == "NOT_REVIEWED"  # the write undid the review
    _tool(box, "view_stack")()
    assert _submit(box)["status"] == "ok" and state.submitted is True


def test_the_playbook_puts_astras_method_in_the_job_statement(tmp_path: Path):
    from langslice.agent.prompt import build_job_statement
    from langslice.core.spec import PositionSpec

    kwargs = dict(tool_names=["set_positions", "view_placement", "view_stack", "submit"],
                  species="mouse", pos_lo=0.0, pos_hi=10.0, axis_ends=("anterior", "posterior"))
    state, _, spec = _stack(tmp_path, tasks=["position"], position=PositionSpec(playbook=True))
    text = build_job_statement(spec, state, **kwargs)
    assert "form a complete hypothesis" in text and "four sections per call" in text
    assert "one at a time" not in text
    state, _, spec = _stack(tmp_path, tasks=["position"], position=PositionSpec())
    plain = build_job_statement(spec, state, **kwargs)
    assert "complete hypothesis" not in plain and "Method:" in plain
    method = plain.split("Method:", 1)[1]
    assert "batch" not in method.lower()
    assert "candidate atlas positions before writing" in method
    assert "nominal interval stand in for a look" in method
    assert "After writing, review the whole stack" in method
    assert "side of any gap before reporting an interval break" in method
    assert "Submit when the work is complete" in method


def test_damage_flags_can_be_set_and_cleared_together_with_undo(tmp_path: Path):
    state, ctx, box = _box(tmp_path)
    state.by_id("s0.png").damaged = True
    state.by_id("s0.png").damage_note = "previous damage"
    before = state.to_dict()
    result = _tool(box, "mark_damaged")([
        {"id": "s0.png", "damaged": False},
        {"id": "s1.png", "damaged": True, "note": "missing hemisphere"},
    ])
    assert result["status"] == "ok"
    assert state.by_id("s0.png").damaged is False
    assert state.by_id("s0.png").damage_note == ""
    assert state.by_id("s1.png").damaged is True
    assert state.by_id("s1.png").damage_note == "missing hemisphere"
    assert load_checkpoint(ctx.checkpoint_path).to_dict() == state.to_dict()
    assert len(box.job.undo_stack) == 1
    _tool(box, "undo")()
    assert state.to_dict() == before


def test_adjust_transforms_one_view_draws_every_entry(tmp_path: Path):
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY

    state, _, box = _box(tmp_path, placed=True)
    base = {"rotation_deg": 2, "scale_x": 1, "scale_y": 1,
            "translate_x_mm": 0, "translate_y_mm": 0}
    entries = [{"id": f"s{i}.png", **base} for i in range(3)]
    # Picture options belong to the call, not to an entry: refused, nothing written.
    misplaced = _tool(box, "adjust_transforms")([{**entries[0], "mode": "ab"}])
    assert misplaced["error"] == "UNKNOWN_ARGUMENTS"
    assert misplaced["problems"][0]["argument"] == "entries[0]"
    assert "`mode` belongs inside `view`." in misplaced["message"]
    assert all(record.transform is None for record in state.slices)
    result = _tool(box, "adjust_transforms")(entries, view={"mode": "ab"})
    assert result["status"] == "ok"
    assert [row["image_indexes"] for row in result["results"]] == [[0, 1], [2, 3], [4, 5]]
    assert result["view"]["mode"] == "ab"
    assert len(result[TOOL_MEDIA_PARTS_KEY]) == 6
    assert len(box.job.undo_stack) == 1
    _tool(box, "undo")()
    assert all(record.transform is None for record in state.slices)
