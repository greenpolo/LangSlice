"""The toolbox: gating, undo/redo, the reorder rule, and the submit gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from langslice.linear.checkpoint import load_checkpoint
from langslice.linear.engine import build_context, ingest
from langslice.linear.spec import JobSpec, PositionSpec, ReorderSpec, TransformSpec
from langslice.linear.toolbox import build_tools
from tests.fakes import EllipseAtlas, SlabAtlas, ellipse_section

_ATLAS = SlabAtlas()


class _Actions:
    escalate = False


class _ToolContext:
    def __init__(self) -> None:
        self.actions = _Actions()


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
    assert {"status", "view_slices", "fetch_atlas", "set_positions", "submit"} <= names
    assert not names & {"reorder_slices", "move_slice", "orient_slices"}
    assert not names & {"fit_affine", "align_slice", "copy_transform"}


def test_optional_tools_follow_their_flags(tmp_path: Path):
    _, _, box = _box(
        tmp_path,
        position=PositionSpec(deepslice=True, bayesian=True),
        transform=TransformSpec(angles=True, subagents=False),
    )
    names = set(box.names)
    assert {"run_deepslice", "fit_position", "set_cutting_angles"} <= names
    assert "align_slice" not in names  # subagents off

    _, _, plain = _box(tmp_path)
    assert not set(plain.names) & {"run_deepslice", "fit_position", "set_cutting_angles"}
    assert "align_slice" in plain.names


def test_flip_is_refused_when_the_spec_switches_it_off(tmp_path: Path):
    state, _, box = _box(tmp_path, reorder=ReorderSpec(flip=False))
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
    assert box.undo_stack == []


def test_set_positions_clamps_to_the_atlas_range(tmp_path: Path):
    state, _, box = _box(tmp_path)
    result = _tool(box, "set_positions")([{"id": "s0.png", "position_mm": 500.0}])
    assert result["clamped"][0]["clamped_to_mm"] == 19.0
    assert state.by_id("s0.png").position_mm == 19.0


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


def test_reorder_refuses_anything_that_is_not_a_permutation(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    result = _tool(box, "reorder_slices")(["s0.png", "s1.png"])
    assert result["error"] == "NOT_A_PERMUTATION"
    assert state.by_id("s1.png").position_mm is not None


def test_move_slice_moves_one_section_and_keeps_every_position(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    result = _tool(box, "move_slice")("s3.png", "start")
    assert [s.id for s in state.in_order()][0] == "s3.png"
    # Every section it passed is renumbered; nothing loses its position.
    assert sorted(result["moved"]) == ["s0.png", "s1.png", "s2.png", "s3.png"]
    assert state.by_id("s3.png").position_mm == 3.5
    assert state.by_id("s0.png").position_mm == 2.0
    assert state.by_id("s4.png").position_mm == 4.0

    box.undo_stack.clear()
    result = _tool(box, "move_slice")("s3.png", "s4.png")
    assert [s.id for s in state.in_order()][-1] == "s3.png"


# --- distribute_spacing --------------------------------------------------


def test_distribute_spacing_previews_without_writing(tmp_path: Path):
    state, _, box = _box(tmp_path)
    result = _tool(box, "distribute_spacing")(
        [{"id": "s0.png", "position_mm": 1.0}, {"id": "s4.png", "position_mm": 5.0}],
        [],
        False,
    )
    assert [row["position_mm"] for row in result["suggestions"]] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["implied_interval_mm"] == 1.0
    assert result["applied"] is False
    assert all(s.position_mm is None for s in state.slices)


def test_distribute_spacing_keeps_the_sections_named_in_keep(tmp_path: Path):
    state, _, box = _box(tmp_path)
    state.by_id("s2.png").position_mm = 4.0
    result = _tool(box, "distribute_spacing")(
        [{"id": "s0.png", "position_mm": 1.0}], ["s2.png"], True
    )
    positions = [s.position_mm for s in state.in_order()]
    assert result["applied"] is True
    assert positions[0] == 1.0
    assert positions[2] == 4.0  # kept, and used as an anchor
    assert positions[1] == 2.5
    assert positions[3] == 5.5 and positions[4] == 7.0  # implied interval of 1.5


def test_distribute_spacing_needs_two_anchors(tmp_path: Path):
    _, _, box = _box(tmp_path)
    result = _tool(box, "distribute_spacing")(
        [{"id": "s0.png", "position_mm": 1.0}], [], False
    )
    assert result["error"] == "NEED_TWO_ANCHORS"


# --- the submit gates ----------------------------------------------------


def _submit(box, **kwargs):
    args = {"summary": "done", "notes": [], "interval_breaks": []}
    args.update(kwargs)
    return _tool(box, "submit")(**args, tool_context=_ToolContext())


def test_submit_refuses_an_unplaced_stack(tmp_path: Path):
    state, _, box = _box(tmp_path)
    result = _submit(box)
    assert result["error"] == "MISSING_POSITIONS"
    assert len(result["missing_ids"]) == 5
    assert state.submitted is False


def test_submit_refuses_positions_that_run_against_the_order(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
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


def test_submit_accepts_a_stack_that_runs_backwards(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 10.0 - index
    assert _submit(box)["status"] == "ok"
    assert state.submitted is True


def test_submit_refuses_a_break_the_positions_do_not_show(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    result = _submit(box, interval_breaks=[3])
    assert result["error"] == "INTERVAL_BREAKS_UNSUPPORTED"
    assert result["failures"][0]["written_interval_mm"] == 0.5
    assert state.submitted is False


def test_submit_accepts_a_break_the_positions_do_show(tmp_path: Path):
    state, _, box = _box(tmp_path, placed=True)
    for record in state.in_order()[3:]:
        record.position_mm = float(record.position_mm) + 4.0
    assert _submit(box, interval_breaks=[3])["status"] == "ok"
    assert state.interval_breaks == [3]


def test_strict_interval_refuses_uneven_spacing_and_any_break(tmp_path: Path):
    state, ctx, spec = _stack(
        tmp_path, placed=True, position=PositionSpec(strict_interval=True, interval_um=500)
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


def test_validate_runs_the_gates_without_writing(tmp_path: Path):
    state, ctx, box = _box(tmp_path, placed=True)
    validate = _tool(box, "validate")

    state.by_id("s2.png").position_mm = None
    assert validate([])["error"] == "MISSING_POSITIONS"

    state.by_id("s2.png").position_mm = 0.5  # placed, but against the order
    assert validate([])["error"] == "ORDER_POSITION_MISMATCH"

    state.by_id("s2.png").position_mm = 3.0
    assert validate([]) == {"status": "ok", "would_submit": True}
    assert validate([3])["error"] == "INTERVAL_BREAKS_UNSUPPORTED"

    # Writes nothing: no checkpoint, no undo step, no submission.
    assert state.submitted is False
    assert box.undo_stack == []
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

    result = _tool(box, "fit_affine")([], "silhouette", True)
    assert result["status"] == "ok"
    assert [row["id"] for row in result["results"]] == ["s0.png"]  # damaged is skipped
    assert result["results"][0]["iou"] > 0.5
    assert result["results"][0]["decomposition"]["mirrored"] in (True, False)
    assert state.by_id("s0.png").transform["kind"] == "silhouette"
    assert len(state.by_id("s0.png").transform["params"]) == 6
    assert state.by_id("s0.png").transform["mirrored"] == (
        result["results"][0]["decomposition"]["mirrored"]
    )
    assert state.by_id("s1.png").transform is None

    named = _tool(box, "fit_affine")(["s1.png"], "silhouette", True)
    assert named["results"][0]["error"] == "DAMAGED"

    assert _tool(box, "fit_affine")([], "elastix", True)["error"] == "UNAVAILABLE"


def test_copy_transform_copies_onto_named_sections(tmp_path: Path):
    state, _, box = _box(tmp_path)
    state.by_id("s0.png").transform = {"kind": "silhouette", "params": [1, 0, 0, 0, 1, 0]}
    result = _tool(box, "copy_transform")("s0.png", ["s1.png", "s2.png"])
    assert result["copied_to"] == ["s1.png", "s2.png"]
    assert state.by_id("s1.png").transform["copied_from"] == "s0.png"


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


def test_fetch_atlas_images_are_section_sized(tmp_path: Path):
    import io

    from PIL import Image

    from langslice.adk import TOOL_MEDIA_PARTS_KEY
    from langslice.linear.atlas_fetch import ATLAS_LONG_EDGE

    _, _, box = _box(tmp_path)
    result = _tool(box, "fetch_atlas")([1.0])
    assert result["status"] == "ok"
    image = Image.open(io.BytesIO(result[TOOL_MEDIA_PARTS_KEY][0].inline_data.data))
    assert max(image.size) == ATLAS_LONG_EDGE
