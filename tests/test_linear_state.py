"""Spec and state: round-trips, addressing, and the in-place restore."""

from __future__ import annotations

import pytest

from langslice.core.spec import JobSpec, PositionSpec, TransformSpec
from langslice.core.state import SliceState, StackState, unknown_sections


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
        position=PositionSpec(interval_um=300, strict_interval=True),
        transform=TransformSpec(
            angles=True, flip=False, hemisphere_cue="notch on the left"
        ),
        facts=["the block was cut back to front"],
        inputs={"positions": {"s0.png": 1.0}},
    )
    again = JobSpec.from_dict(spec.to_dict())
    assert again == spec
    assert again.interval_mm == 0.3
    assert again.thickness_mm is None
    assert again.has("position") and not again.has("reorder")


def test_spec_rejects_an_unknown_plane_and_task():
    with pytest.raises(ValueError, match="plane"):
        JobSpec(image_folder="/tmp/x", plane="axial")
    with pytest.raises(ValueError, match="task"):
        JobSpec(image_folder="/tmp/x", tasks=["reorder", "colorize"])


def test_state_round_trips_and_resolves_by_filename_only():
    state = _stack()
    state.slices[0].position_mm = 4.0
    state.slices[2].transform = {"kind": "silhouette", "params": [1, 0, 0, 0, 1, 0]}
    again = StackState.from_dict(state.to_dict())
    assert again == state
    assert again.resolve("s1.png") is again.slices[1]
    assert again.resolve(" s2 ") is again.slices[2]  # the stem names one file
    assert again.resolve(2) is None and again.resolve("2") is None  # never a number
    assert again.resolve("nope.png") is None


def test_a_stem_two_files_share_names_neither_and_the_refusal_says_why():
    state = StackState(slices=[SliceState("a.png", 0, 0), SliceState("a.tif", 1, 1),
                               SliceState("3.png", 2, 2)])
    assert state.resolve("a") is None
    assert state.resolve("a.tif") is state.slices[1]
    assert state.resolve("3") is state.slices[2]  # a file named by a number is a name
    facts = unknown_sections(state, ["a", "1"])
    assert facts["unknown"] == ["a", "1"]
    assert facts["filenames"] == ["3.png", "a.png", "a.tif"]
    assert "named by filename" in facts["message"]
    assert "no numbers" in facts["message"] and "whole filename" in facts["message"]


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


@pytest.mark.parametrize("protocol", [{}, {"interval_um": 200}, {"thickness_um": 50},
                                      {"interval_um": 200, "thickness_um": 50}])
def test_optional_protocol_round_trips_without_inventing_values(protocol):
    spec = JobSpec.from_dict({"image_folder": ".", "position": protocol})
    saved = spec.to_dict()
    assert JobSpec.from_dict(saved) == spec
    for name in ("interval_um", "thickness_um"):
        assert saved["position"][name] == protocol.get(name)
    state = StackState(interval_mm=spec.interval_mm, thickness_mm=spec.thickness_mm)
    assert StackState.from_dict(state.to_dict()) == state


def test_strict_spacing_requires_a_supplied_interval():
    with pytest.raises(ValueError, match="requires a supplied interval"):
        PositionSpec(strict_interval=True)
    assert PositionSpec(interval_um=200, strict_interval=True).thickness_um is None


@pytest.mark.parametrize("value", [0, -50, True, "50"])
def test_protocol_values_are_positive_micrometres_or_unknown(value):
    with pytest.raises(ValueError, match="positive integer"):
        JobSpec.from_dict({"image_folder": ".", "position": {"thickness_um": value}})
