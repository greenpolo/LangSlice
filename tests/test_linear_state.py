"""Spec and state: round-trips, addressing, and the in-place restore."""

from __future__ import annotations

import pytest

from langslice.core.spec import JobSpec, PositionSpec, TransformSpec
from langslice.core.stack.codec import affine_from_json, mask_from_json
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack, unknown_sections
from langslice.job.state_file import stack_from_json, stack_to_json


def _stack(n: int = 3) -> Stack:
    return Stack(
        atlas="fake",
        slices=[
            Section(id=f"s{i}.png", index_original=i, index_corrected=i)
            for i in range(n)
        ],
    )


def test_spec_round_trips_through_dict():
    spec = JobSpec(
        image_folder="/tmp/x",
        tasks=["position"],
        position=PositionSpec(interval_um=300, strict_interval=True),
        transform=TransformSpec(angles=True, hemisphere_cue="notch on the left"),
        disabled=["interactive_transform:flip", "zoom"],
        facts=["the block was cut back to front"],
        inputs={"positions": {"s0.png": 1.0}},
    )
    again = JobSpec.from_dict(spec.to_dict())
    assert again == spec
    assert again.interval_mm == 0.3
    assert again.thickness_mm is None
    assert again.has("position")


def test_spec_rejects_an_unknown_plane_and_task():
    with pytest.raises(ValueError, match="plane"):
        JobSpec(image_folder="/tmp/x", plane="axial")
    with pytest.raises(ValueError, match="task"):
        JobSpec(image_folder="/tmp/x", tasks=["reorder", "colorize"])


def test_state_round_trips_and_resolves_by_filename_only():
    state = _stack()
    state.slices[0].position_mm = 4.0
    state.slices[2].transform = affine_from_json(
        {"kind": "silhouette", "params": [1, 0, 0, 0, 1, 0]})
    again = stack_from_json(stack_to_json(state))
    assert again == state
    assert again.resolve("s1.png") is again.slices[1]
    assert again.resolve(" s2 ") is again.slices[2]  # the stem names one file
    assert again.resolve(2) is None and again.resolve("2") is None  # never a number
    assert again.resolve("nope.png") is None


def test_masks_are_left_out_of_the_serialized_state_while_empty():
    state = _stack()
    assert all("masks" not in row for row in stack_to_json(state)["slices"])
    assert stack_from_json(stack_to_json(state)) == state
    entry = {"kind": "em", "section_image": "preprocessed", "atlas_image": "template",
             "params": {"match_sd": 0.1}, "inputs": "abc",
             "file": "sections/s1/masks/hole.npz", "left_out": 0.05}
    state.slices[1].masks["hole"] = mask_from_json(entry)
    rows = stack_to_json(state)["slices"]
    assert "masks" not in rows[0] and rows[1]["masks"] == {"hole": entry}
    assert stack_from_json(stack_to_json(state)) == state
    # A state written before masks existed still loads.
    assert stack_from_json(stack_to_json(_stack())).slices[1].masks == {}


def test_a_stem_two_files_share_names_neither_and_the_refusal_says_why():
    state = Stack(slices=[Section("a.png", 0, 0), Section("a.tif", 1, 1),
                               Section("3.png", 2, 2)])
    assert state.resolve("a") is None
    assert state.resolve("a.tif") is state.slices[1]
    assert state.resolve("3") is state.slices[2]  # a file named by a number is a name
    facts = unknown_sections(state, ["a", "1"])
    assert facts["unknown"] == ["a", "1"]
    assert facts["filenames"] == ["3.png", "a.png", "a.tif"]


def test_in_order_follows_the_corrected_index():
    state = _stack()
    state.slices[0].index_corrected = 2
    state.slices[2].index_corrected = 0
    assert [s.id for s in state.in_order()] == ["s2.png", "s1.png", "s0.png"]


def test_restore_refills_the_same_object():
    state = _stack()
    snapshot = stack_to_json(state)
    state.slices[0].position_mm = 9.0
    state.notes.append("changed")
    state.replace_with(stack_from_json(snapshot))
    assert state.slices[0].position_mm is None
    assert state.notes == []


@pytest.mark.parametrize("protocol", [{}, {"interval_um": 200, "thickness_um": 50}])
def test_optional_protocol_round_trips_without_inventing_values(protocol):
    spec = JobSpec.from_dict({"image_folder": ".", "position": protocol})
    saved = spec.to_dict()
    assert JobSpec.from_dict(saved) == spec
    for name in ("interval_um", "thickness_um"):
        assert saved["position"][name] == protocol.get(name)
    state = Stack(interval_mm=spec.interval_mm, thickness_mm=spec.thickness_mm)
    assert stack_from_json(stack_to_json(state)) == state


def test_strict_spacing_requires_a_supplied_interval():
    with pytest.raises(ValueError, match="requires a supplied interval"):
        PositionSpec(strict_interval=True)
    assert PositionSpec(interval_um=200, strict_interval=True).thickness_um is None


@pytest.mark.parametrize("value", [0, -50, True])
def test_protocol_values_are_positive_micrometres_or_unknown(value):
    with pytest.raises(ValueError, match="positive integer"):
        JobSpec.from_dict({"image_folder": ".", "position": {"thickness_um": value}})
