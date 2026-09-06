"""Spec and state: round-trips, addressing, and the in-place restore."""

from __future__ import annotations

import pytest

from langslice.linear.spec import JobSpec, PositionSpec, ReorderSpec, TransformSpec
from langslice.linear.state import SliceState, StackState


def _stack(n: int = 3) -> StackState:
    return StackState(
        atlas="fake",
        slices=[
            SliceState(id=f"s{i}.png", index_original=i, index_corrected=i)
            for i in range(n)
        ],
    )


def test_spec_round_trips_through_dict():
    spec = JobSpec(
        image_folder="/tmp/x",
        tasks=["position"],
        reorder=ReorderSpec(flip=False, hemisphere_cue="notch on the left"),
        position=PositionSpec(interval_um=300, strict_interval=True, bayesian=True),
        transform=TransformSpec(angles=True, subagents=False),
        facts=["the block was cut back to front"],
        inputs={"positions": {"s0.png": 1.0}},
    )
    again = JobSpec.from_dict(spec.to_dict())
    assert again == spec
    assert again.interval_mm == 0.3
    assert again.thickness_mm == 0.05
    assert again.has("position") and not again.has("reorder")


def test_spec_rejects_an_unknown_plane_and_task():
    with pytest.raises(ValueError, match="plane"):
        JobSpec(image_folder="/tmp/x", plane="axial")
    with pytest.raises(ValueError, match="task"):
        JobSpec(image_folder="/tmp/x", tasks=["reorder", "colorize"])


def test_state_round_trips_and_resolves_by_name_or_index():
    state = _stack()
    state.slices[0].position_mm = 4.0
    state.slices[2].transform = {"kind": "silhouette", "params": [1, 0, 0, 0, 1, 0]}
    again = StackState.from_dict(state.to_dict())
    assert again == state
    assert again.resolve("s1.png") is again.slices[1]
    assert again.resolve(2) is again.slices[2]
    assert again.resolve("nope.png") is None


def test_in_order_follows_the_corrected_index():
    state = _stack()
    state.slices[0].index_corrected = 2
    state.slices[2].index_corrected = 0
    assert [s.id for s in state.in_order()] == ["s2.png", "s1.png", "s0.png"]


def test_restore_refills_the_same_object():
    state = _stack()
    snapshot = state.to_dict()
    state.slices[0].position_mm = 9.0
    state.notes.append("changed")
    state.restore(snapshot)
    assert state.slices[0].position_mm is None
    assert state.notes == []


def test_cutting_angles_report_obliqueness():
    state = _stack()
    assert not state.is_oblique
    state.cutting_angles_deg = {"pitch": 3.0, "yaw": -1.0}
    assert state.is_oblique
    assert (state.pitch_deg, state.yaw_deg) == (3.0, -1.0)
