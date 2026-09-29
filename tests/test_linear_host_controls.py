"""Host dialog controls on the job spec: notes, parallel cap, damage, locked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.linear.engine import apply_host_inputs, build_context, ingest
from langslice.linear.prompt import build_job_statement
from langslice.linear.spec import JobSpec, TransformSpec
from langslice.linear.toolbox import HOST_TRANSFORM_KIND, build_tools, submit_errors
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()


def _stack(folder: Path, n: int = 5, *, placed: bool = True, **spec_kwargs: Any):
    for index in range(n):
        Image.fromarray(
            np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)
        ).save(folder / f"s{index}.png")
    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none", **spec_kwargs
    )
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    state = ingest(spec, ctx)
    apply_host_inputs(state, spec)
    if placed:
        for index, record in enumerate(state.in_order()):
            record.position_mm = 2.0 + 0.5 * index
    box = build_tools(state, ctx, spec)
    return state, spec, {tool.__name__: tool for tool in box.tools}


def _statement(state, spec, tools) -> str:
    return build_job_statement(
        spec, state, tool_names=list(tools), species="mouse",
        pos_lo=0.0, pos_hi=19.0, axis_ends=("anterior", "posterior"),
    )


def _entry(slice_id: str) -> dict[str, Any]:
    return {"id": slice_id, "rotation_deg": 1.0, "scale_x": 1.0, "scale_y": 1.0,
            "translate_x_mm": 0.1, "translate_y_mm": 0.0}


# --- the spec --------------------------------------------------------------


def test_new_fields_default_and_round_trip_through_from_dict(tmp_path: Path):
    spec = JobSpec(image_folder=str(tmp_path))
    assert spec.image_resolution == "low"
    assert spec.agent_damage is True
    assert spec.transform.max_parallel == 4
    assert spec.position.notes == spec.transform.notes == spec.nonlinear.notes == ""

    rebuilt = JobSpec.from_dict({
        "image_folder": str(tmp_path), "image_resolution": "high", "agent_damage": False,
        "position": {"notes": "p"}, "transform": {"notes": "t", "max_parallel": 2},
        "nonlinear": {"notes": "n"},
    })
    assert rebuilt.image_resolution == "high" and rebuilt.agent_damage is False
    assert (rebuilt.position.notes, rebuilt.transform.notes, rebuilt.nonlinear.notes) == (
        "p", "t", "n")
    assert rebuilt.transform.max_parallel == 2
    assert JobSpec.from_dict(rebuilt.to_dict()) == rebuilt


@pytest.mark.parametrize("data, message", [
    ({"image_resolution": "ultra"}, "image_resolution"),
    ({"agent_damage": "yes"}, "agent_damage"),
    ({"transform": {"max_parallel": 0}}, "max_parallel"),
    ({"transform": {"max_parallel": 5}}, "max_parallel"),
    ({"transform": {"max_parallel": True}}, "max_parallel"),
    ({"transform": {"max_parallel": "2"}}, "max_parallel"),
])
def test_bad_values_are_refused(tmp_path: Path, data: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        JobSpec.from_dict({"image_folder": str(tmp_path), **data})


# --- per-task notes --------------------------------------------------------


def test_each_note_sits_with_its_task_not_in_the_facts(tmp_path: Path):
    state, spec, tools = _stack(
        tmp_path, facts=["global fact"], tasks=["reorder", "position", "transform"],
    )
    spec.position.notes = "Sections 3-5 are from a second brain.\n\n  second line "
    spec.transform.notes = "Tissue shrank about 10%."
    spec.nonlinear.notes = "never shown: nonlinear is off"
    text = _statement(state, spec, tools)

    job_end = text.index("Your job:")
    facts_start = text.index("Run facts:")
    block = text[job_end:facts_start]
    assert "Positioning notes from the user:\n- Sections 3-5 are from a second brain.\n" \
        "- second line" in block
    assert "In-plane alignment notes from the user:\n- Tissue shrank about 10%." in block
    assert block.index("Positioning") < block.index("In-plane alignment")
    assert "never shown" not in text
    facts = text[facts_start:text.index("Tools:")]
    assert "- global fact" in facts  # global facts still work
    assert "second brain" not in facts and "shrank" not in facts


def test_an_empty_note_or_an_off_task_adds_nothing(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, tasks=["position"])
    spec.transform.notes = "transform is off"
    text = _statement(state, spec, tools)
    assert "notes from the user" not in text


# --- max parallel ----------------------------------------------------------


def test_the_cap_is_a_fact_only_below_four(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, transform=TransformSpec(max_parallel=2))
    assert "Each transform tool call takes at most 2 sections." in _statement(state, spec, tools)
    spec.transform.max_parallel = 1
    assert "at most 1 section." in _statement(state, spec, tools)
    spec.transform.max_parallel = 4
    assert "transform tool call takes at most" not in _statement(state, spec, tools)


def test_over_cap_transform_calls_are_refused_naming_the_cap(tmp_path: Path):
    state, _spec, tools = _stack(tmp_path, transform=TransformSpec(max_parallel=2))
    refused = tools["adjust_transforms"]([_entry("s0.png"), _entry("s1.png"), _entry("s2.png")])
    assert refused == {"status": "error", "error": "TOO_MANY_SECTIONS",
                       "max_sections": 2, "requested": 3}
    assert all(record.transform is None for record in state.slices)

    refused = tools["fit_affine"](["s0.png", "s1.png", "s2.png"], "silhouette")
    assert refused["error"] == "TOO_MANY_SECTIONS" and refused["max_sections"] == 2
    # An empty list means every eligible section, which is five here.
    assert tools["fit_affine"]([], "silhouette")["requested"] == 5

    allowed = tools["adjust_transforms"]([_entry("s0.png"), _entry("s1.png")])
    assert allowed["status"] == "ok"


def test_at_four_fit_affine_keeps_taking_any_number(tmp_path: Path):
    _state, _spec, tools = _stack(tmp_path)
    result = tools["fit_affine"](["s0.png", "s1.png", "s2.png", "s3.png", "s4.png"], "silhouette")
    assert result.get("error") != "TOO_MANY_SECTIONS"
    assert len(result["results"]) == 5


# --- damage ----------------------------------------------------------------


def test_agent_damage_off_builds_no_mark_damaged(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, agent_damage=False)
    assert "mark_damaged" not in tools
    assert "`mark_damaged`" not in _statement(state, spec, tools)
    _state, _spec, default = _stack(tmp_path)
    assert "mark_damaged" in default


def test_the_agent_can_never_clear_host_damage(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, inputs={"damaged": {"s1.png": "torn"}})
    result = tools["mark_damaged"]([
        {"id": "s1.png", "damaged": False}, {"id": "s2.png", "note": "fold"},
    ])
    assert result["rejected"] == [{"id": "s1.png", "error": "DAMAGE_SET_BY_USER"}]
    assert state.by_id("s1.png").damaged and state.by_id("s1.png").damage_note == "torn"
    assert state.by_id("s2.png").damaged
    # Agent-set flags stay clearable.
    cleared = tools["mark_damaged"]([{"id": "s2.png", "damaged": False}])
    assert cleared["unmarked"] == ["s2.png"] and "rejected" not in cleared
    assert "cannot be cleared: s1.png" in _statement(state, spec, tools)


# --- locked ----------------------------------------------------------------


def test_locked_sections_carry_the_host_identity(tmp_path: Path):
    state, _spec, _tools = _stack(tmp_path, inputs={"locked": ["s1.png"]})
    transform = state.by_id("s1.png").transform
    assert transform["kind"] == HOST_TRANSFORM_KIND
    assert transform["params"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert state.by_id("s0.png").transform is None
    with pytest.raises(ValueError, match="unknown section"):
        _stack(tmp_path, inputs={"locked": ["nope.png"]})


def test_locked_geometry_is_refused_per_section_and_positions_still_move(tmp_path: Path):
    state, _spec, tools = _stack(tmp_path, inputs={"locked": ["s1.png"]})
    held = dict(state.by_id("s1.png").transform)

    oriented = tools["orient_slices"]([
        {"id": "s1.png", "flip": True}, {"id": "s2.png", "rotate_deg": 90},
    ])
    assert {"id": "s1.png", "error": "LOCKED"} in oriented["rejected"]
    assert state.by_id("s1.png").flip is False
    assert state.by_id("s2.png").rotation_deg == 90

    adjusted = tools["adjust_transforms"]([_entry("s1.png"), _entry("s0.png")])
    assert adjusted["results"][0] == {"status": "error", "error": "LOCKED", "id": "s1.png"}
    assert adjusted["results"][1]["status"] == "ok"

    fitted = tools["fit_affine"](["s1.png"], "silhouette")
    assert fitted["results"] == [{"id": "s1.png", "status": "error", "error": "LOCKED"}]
    everything = tools["fit_affine"]([], "silhouette")
    assert "s1.png" not in [row.get("id") for row in everything["results"]]
    assert state.by_id("s1.png").transform == held

    moved = tools["set_positions"]([{"id": "s1.png", "position_mm": 2.6}])
    assert moved["written"] == [{"id": "s1.png", "position_mm": 2.6}]


def test_locked_transforms_count_at_submit_even_when_damaged(tmp_path: Path):
    state, spec, _tools = _stack(
        tmp_path, n=2, tasks=["position", "transform"],
        inputs={"locked": ["s0.png", "s1.png"], "damaged": {"s0.png": ""}},
    )
    assert submit_errors(state, spec, []) is None

    # Without the lock the same damaged identity transform is refused.
    unlocked = JobSpec.from_dict({**spec.to_dict(), "inputs": {"damaged": {"s0.png": ""}}})
    refusal = submit_errors(state, unlocked, [])
    assert refusal is not None and refusal["error"] == "DAMAGED_REQUIRES_MANUAL_TRANSFORM"


def test_the_job_statement_names_locked_sections_and_why(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, inputs={"locked": ["s3.png", "s1.png"]})
    text = _statement(state, spec, tools)
    assert "already done by the user" in text
    assert "s1.png, s3.png" in text
    assert "Their positions can still be changed." in text


def test_the_worker_never_emits_geometry_for_locked_sections():
    from langslice.api.abba_worker import _host_updates

    def row(transform, flip=False, position=4.0):
        return {"id": "a.tif", "position_mm": position, "index_corrected": 0,
                "flip": flip, "rotation_deg": 0, "transform": transform}

    before = {"slices": [row(None)]}
    after = {"slices": [row({"params": [1, 0, .1, 0, 1, 0]}, flip=True, position=4.5)]}
    args = (["reorder", "position", "transform"], {"a.tif": (80, 40)}, 25.0)
    assert set(_host_updates(after, before, *args)[0]) == {
        "id", "position_mm", "flip", "rotation_deg", "affine_mm"}
    locked = _host_updates(after, before, *args, frozenset({"a.tif"}))
    assert locked == [{"id": "a.tif", "position_mm": 4.5}]
