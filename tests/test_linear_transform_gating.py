"""The ABBA fitting switches control the tools offered to the linear agent."""

from pathlib import Path

import pytest

from langslice.linear.spec import TransformSpec
from tests.test_linear_toolbox import _box


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
    manual = {"adjust_transform", "adjust_transforms", "view_landmarks",
        "edit_landmarks", "warp_landmarks"}
    assert names & manual == (manual if interactive else set())
    assert ("fit_affine" in names) is automatic
    assert "submit" in names


def test_disabled_transform_task_overrides_enabled_fitting_switches(tmp_path: Path):
    _, _, box = _box(
        tmp_path, tasks=["position"],
        transform=TransformSpec(interactive=True, automatic=True, angles=True, elastix=True),
    )
    assert not set(box.names) & {
        "adjust_transform", "adjust_transforms", "view_landmarks",
        "edit_landmarks", "warp_landmarks", "fit_affine", "set_cutting_angles",
    }


@pytest.mark.parametrize("transform,reason", [
    (None, "missing_transform"),
    ({"kind": "silhouette", "params": [1, 0, .1, 0, 1, 0]}, "not_interactive"),
    ({"kind": "interactive", "params": [1, 0, 0, 0, 1, 0],
      "note": "Reviewed manual transform"}, "identity_transform"),
    ({"kind": "interactive", "params": [1, 1e-16, 0, -1e-16, 1, 0]},
     "identity_transform"),
    ({"kind": "interactive", "params": [1, 0, float("nan"), 0, 1, 0]},
     "invalid_transform"),
])
def test_damaged_submission_requires_applied_manual_transform(tmp_path, transform, reason):
    from tests.test_linear_toolbox import _submit, _tool

    state, _, box = _box(tmp_path, tasks=["transform"], placed=True)
    for record in state.slices:
        record.transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    damaged = state.by_id("s0.png")
    assert damaged is not None
    damaged.damaged = True
    damaged.transform = transform
    validation = _tool(box, "validate")([])
    assert validation["error"] == "DAMAGED_REQUIRES_MANUAL_TRANSFORM"
    assert validation["failures"] == [{"id": damaged.id, "reason": reason}]
    assert "adjust_transform" in validation["message"]
    assert _submit(box) == validation
    assert not state.submitted


def test_real_manual_adjustment_resolves_damage_gate_and_undo_restores_it(tmp_path):
    from tests.test_linear_toolbox import _submit, _tool

    state, _, box = _box(tmp_path, tasks=["transform"], placed=True)
    for record in state.slices:
        record.transform = {"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]}
    damaged = state.by_id("s0.png")
    assert damaged is not None
    damaged.damaged = True
    result = _tool(box, "adjust_transform")("s0.png", 5, 1.1, 1, .25, 0)
    assert result["status"] == "ok"
    assert _tool(box, "validate")([])["would_submit"]
    _tool(box, "undo")()
    assert _tool(box, "validate")([])["error"] == "DAMAGED_REQUIRES_MANUAL_TRANSFORM"
    _tool(box, "redo")()
    assert _submit(box)["status"] == "ok"


def test_damage_gate_reports_disabled_manual_tools_and_respects_task_switch(tmp_path):
    from langslice.linear.toolbox import submit_errors
    from tests.test_linear_toolbox import _stack

    state, _, spec = _stack(
        tmp_path, tasks=["transform"], placed=True,
        transform=TransformSpec(interactive=False, automatic=True),
    )
    damaged = state.by_id("s0.png")
    assert damaged is not None
    damaged.damaged = True
    refusal = submit_errors(state, spec, [])
    assert refusal is not None
    assert refusal["interactive_enabled"] is False
    assert "host must enable" in refusal["message"]
    spec.tasks = ["position"]
    assert submit_errors(state, spec, []) is None
