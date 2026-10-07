"""The ABBA fitting switches control the tools offered to the linear agent."""

import inspect
from pathlib import Path

import pytest

from langslice.core.spec import TransformSpec
from langslice.job.job import submit_errors
from tests.linear_tool_helpers import box as _box
from tests.linear_tool_helpers import single_transform
from tests.linear_tool_helpers import stack as _stack
from tests.linear_tool_helpers import submit as _submit
from tests.linear_tool_helpers import tool_named as _tool


@pytest.mark.parametrize(
    "interactive,automatic", [(True, False), (False, True), (True, True), (False, False)],
)
def test_transform_controls_independently_gate_tools(
    tmp_path: Path, interactive: bool, automatic: bool,
):
    _, _, box = _box(
        tmp_path, tasks=["transform"],
        transform=TransformSpec(interactive=interactive, automatic=automatic),
    )
    names = set(box.names)
    assert ("interactive_transform" in names) is interactive
    assert ("elastix_affine" in names) is automatic
    assert "position_sections" not in names  # positions and angles are supplied
    assert {"look", "status", "submit"} <= names


def test_disabled_transform_task_overrides_enabled_fitting_switches(tmp_path: Path):
    _, _, box = _box(
        tmp_path, tasks=["position"],
        transform=TransformSpec(interactive=True, automatic=True, angles=True),
    )
    names = set(box.names)
    assert not names & {"interactive_transform", "elastix_affine"}
    # The cutting angles come with Positioning.
    assert "position_sections" in names


def test_angles_alone_give_position_sections_without_sections(tmp_path: Path):
    """``transform.angles`` with Positioning off: position_sections sets the
    cutting angles, and its ``sections`` argument is not offered."""
    state, _, box = _box(tmp_path, tasks=["transform"], placed=True,
                         transform=TransformSpec(angles=True))
    write = _tool(box, "position_sections")
    assert list(inspect.signature(write).parameters) == ["cutting_angles", "view"]
    result = write(cutting_angles={"pitch_deg": 2.0, "yaw_deg": 1.0}, view=False)
    assert result["status"] == "ok"
    assert state.cutting_angles_deg == {"pitch": 2.0, "yaw": 1.0}
    refused = write(sections=[{"id": "s0.png", "position_mm": 9.0}])
    assert refused["error"] == "UNKNOWN_ARGUMENTS"
    assert state.by_id("s0.png").position_mm == 2.0


def test_a_manual_transform_meets_the_transform_gate_and_undo_restores_it(tmp_path):
    state, ctx, box = _box(tmp_path, tasks=["transform"], placed=True)
    for record in state.slices[1:]:
        record.transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    assert _submit(box)["error"] == "MISSING_TRANSFORMS"
    row = single_transform(_tool(box, "interactive_transform"))("s0.png", 5, 1.1, 1, .25, 0)
    assert row["status"] == "ok"
    assert submit_errors(state, ctx.spec, []) is None
    _tool(box, "undo")()
    assert _submit(box)["error"] == "MISSING_TRANSFORMS"
    _tool(box, "redo")()
    assert _submit(box)["status"] == "ok"


@pytest.mark.parametrize("transform", [
    {"kind": "silhouette", "params": [1, 0, .1, 0, 1, 0]},
    {"kind": "elastix", "params": [1, 0, .1, 0, 1, 0],
     "regions": {"include": ["CTX"], "exclude": []}},
    {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]},
    {"kind": "imported", "params": [1, 0, .1, 0, 1, 0]},
])
def test_a_damaged_section_takes_any_transform(tmp_path, transform):
    """A section with marked regions is fitted like any other: its regions
    are left out of the fits, and any transform counts at submit."""
    state, _, spec = _stack(tmp_path, tasks=["transform"], placed=True)
    for record in state.slices:
        record.transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    state.slices[0].damaged_regions = ["CTX"]
    state.slices[0].transform = transform
    assert state.slices[0].damaged
    assert submit_errors(state, spec, []) is None
