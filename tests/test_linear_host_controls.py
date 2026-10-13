"""Host dialog controls on the job spec: notes, parallel cap, damage, frozen
properties, forced pictures and the picture size."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.agent.prompt import build_job_statement
from langslice.core import deformation
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import HOST_TRANSFORM_KIND, apply_host_inputs, ingest, submit_errors
from langslice.ops import transforms as ops_transforms
from langslice.ops.transforms import AffineFit
from tests.deformable_synthetic import SyntheticAtlas
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()
needs_ants = pytest.mark.skipif(not deformation.ants_available(), reason="needs antspyx")


def _stack(folder: Path, n: int = 5, *, placed: bool = True, atlas: Any = None,
           image_model: Any = None, **spec_kwargs: Any):
    for index in range(n):
        Image.fromarray(
            np.full((30, 40, 3), 40 + 10 * index, dtype=np.uint8)
        ).save(folder / f"s{index}.png")
    spec = JobSpec(
        image_folder=str(folder), model="fake-model", preprocess="none", **spec_kwargs
    )
    the_atlas = atlas or _ATLAS
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: the_atlas)
    state = ingest(spec, ctx)
    apply_host_inputs(state, spec)
    if placed:
        low, high = ctx.position_range
        for index, record in enumerate(state.in_order()):
            record.position_mm = min(high, max(low, 2.0 + 0.5 * index))
    box = build_tools(state, ctx, spec, image_model=image_model)
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
    assert spec.disabled == [] and spec.force_view is False
    assert spec.transform.max_parallel == 4
    assert spec.position.notes == spec.transform.notes == spec.nonlinear.notes == ""

    rebuilt = JobSpec.from_dict({
        "image_folder": str(tmp_path), "image_resolution": "high",
        "disabled": ["mark_damage"], "force_view": True,
        "position": {"notes": "p"}, "transform": {"notes": "t", "max_parallel": 2},
        "nonlinear": {"notes": "n"},
    })
    assert rebuilt.image_resolution == "high" and rebuilt.disabled == ["mark_damage"]
    assert rebuilt.force_view is True
    assert (rebuilt.position.notes, rebuilt.transform.notes, rebuilt.nonlinear.notes) == (
        "p", "t", "n")
    assert rebuilt.transform.max_parallel == 2
    assert JobSpec.from_dict(rebuilt.to_dict()) == rebuilt


@pytest.mark.parametrize("data, message", [
    ({"image_resolution": "ultra"}, "image_resolution"),
    ({"disabled": "mark_damage"}, "disabled"),
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
        tmp_path, facts=["global fact"], tasks=["position", "transform"],
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
    facts = text[facts_start:]
    assert "- global fact" in facts  # global facts still work
    assert "second brain" not in facts and "shrank" not in facts


def test_an_empty_note_or_an_off_task_adds_nothing(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, tasks=["position"])
    spec.transform.notes = "transform is off"
    text = _statement(state, spec, tools)
    assert "notes from the user" not in text


def test_the_playbook_puts_a_method_in_the_job_statement(tmp_path: Path):
    from langslice.core.spec import PositionSpec

    state, spec, tools = _stack(tmp_path, tasks=["position"],
                                position=PositionSpec(playbook=True))
    text = " ".join(_statement(state, spec, tools).split())
    assert "initial position hypotheses from the anatomy" in text
    assert "`grep_atlas_view`" in text
    assert "Write supported placements with `position_sections`" in text
    spec.position.playbook = False
    plain = " ".join(_statement(state, spec, tools).split())
    assert "initial position hypotheses from the anatomy" not in plain and "Method:" in plain
    method = plain.split("Method:", 1)[1]
    assert "batch" not in method.lower()
    assert "candidate atlas positions before writing" in method
    assert "nominal interval stand in for a look" in method
    assert "After writing, review the whole stack" in method
    assert "side of any gap before reporting an interval break" in method
    assert "Submit when the work is complete" in method


# --- max parallel ----------------------------------------------------------


def test_the_cap_is_a_fact_only_below_four(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, transform=TransformSpec(max_parallel=2))
    assert "Each transform tool call takes at most 2 sections." in _statement(state, spec, tools)
    spec.transform.max_parallel = 1
    assert "at most 1 section." in _statement(state, spec, tools)
    spec.transform.max_parallel = 4
    assert "transform tool call takes at most" not in _statement(state, spec, tools)


def _no_fit(monkeypatch) -> list[list[str]]:
    """Stand in for the Elastix fit: the cap is checked before it runs."""
    asked: list[list[str]] = []

    def fit(job, ctx, sections, regions, atlas_image):
        names = list(sections) or [record.id for record in ops_transforms.fit_targets(job)]
        asked.append(names)
        return AffineFit(rows=[{"id": name, "status": "error", "error": "FIT_FAILED"}
                               for name in names])

    monkeypatch.setattr(ops_transforms, "elastix_affine", fit)
    return asked


def test_over_cap_transform_calls_are_refused_naming_the_cap(tmp_path: Path, monkeypatch):
    asked = _no_fit(monkeypatch)
    state, _spec, tools = _stack(tmp_path, transform=TransformSpec(max_parallel=2))
    refused = tools["interactive_transform"](
        [_entry("s0.png"), _entry("s1.png"), _entry("s2.png")])
    assert refused == {"status": "error", "error": "TOO_MANY_SECTIONS",
                       "max_sections": 2, "requested": 3}
    assert all(record.transform is None for record in state.slices)

    refused = tools["elastix_affine"](["s0.png", "s1.png", "s2.png"])
    assert refused == {"status": "error", "error": "TOO_MANY_SECTIONS",
                       "max_sections": 2, "requested": 3}
    # An empty list means every placed section, which is five here.
    assert tools["elastix_affine"]([])["requested"] == 5
    assert asked == []

    allowed = tools["interactive_transform"]([_entry("s0.png"), _entry("s1.png")])
    assert allowed["status"] == "ok"
    tools["elastix_affine"](["s0.png", "s1.png"])
    assert asked == [["s0.png", "s1.png"]]


def test_at_four_elastix_affine_keeps_taking_any_number(tmp_path: Path, monkeypatch):
    asked = _no_fit(monkeypatch)
    _state, _spec, tools = _stack(tmp_path)
    names = ["s0.png", "s1.png", "s2.png", "s3.png", "s4.png"]
    result = tools["elastix_affine"](names)
    assert result.get("error") != "TOO_MANY_SECTIONS"
    assert len(result["results"]) == 5 and asked == [names]
    # interactive_transform keeps its own limit of four.
    assert tools["interactive_transform"]([_entry(name) for name in names]) == {
        "status": "error", "error": "TOO_MANY_SECTIONS", "max_sections": 4}


# --- damage ----------------------------------------------------------------


def test_disabling_mark_damage_builds_no_mark_damage(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, disabled=["mark_damage"])
    assert "mark_damage" not in tools
    assert "`mark_damage`" not in _statement(state, spec, tools)
    _state, _spec, default = _stack(tmp_path)
    assert "mark_damage" in default


def test_a_host_damage_note_is_a_note_not_damage(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, n=3, atlas=SyntheticAtlas(),
                                inputs={"damaged": {"s1.png": "torn"}})
    record = state.by_id("s1.png")
    assert record.damage_note == "torn" and record.damaged is False
    assert "The user noted damage on these sections" in _statement(state, spec, tools)
    marked = tools["mark_damage"]("s1.png", ["TH"], "fold")
    assert marked["damaged"] is True and marked["note"] == "torn; fold"
    # Clearing the regions leaves the user's note.
    cleared = tools["mark_damage"]("s1.png", [], "")
    assert cleared["damaged"] is False and cleared["note"] == "torn"
    assert state.by_id("s1.png").damage_note == "torn"


# --- frozen ----------------------------------------------------------------


ALIGNED = ["orientation", "affine"]


def test_a_frozen_affine_carries_the_host_identity(tmp_path: Path):
    state, _spec, _tools = _stack(tmp_path, frozen={"s1.png": ALIGNED})
    transform = state.by_id("s1.png").transform
    assert transform["kind"] == HOST_TRANSFORM_KIND
    assert transform["params"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert state.by_id("s0.png").transform is None
    with pytest.raises(ValueError, match="not here"):
        _stack(tmp_path, frozen={"nope.png": ALIGNED})


def test_frozen_geometry_is_refused_per_section_and_positions_still_move(tmp_path: Path):
    state, _spec, tools = _stack(tmp_path, frozen={"s1.png": ALIGNED})
    held = dict(state.by_id("s1.png").transform)

    turned = tools["interactive_transform"]([
        {"id": "s1.png", "flip": True}, {"id": "s2.png", "rotate_quarter": 90},
        _entry("s0.png"),
    ])
    assert turned["results"][0]["error"] == "FROZEN" and turned["results"][0]["id"] == "s1.png"
    assert turned["results"][0]["properties"] == ["orientation"]
    assert turned["results"][1]["status"] == "ok" and turned["results"][2]["status"] == "ok"
    assert state.by_id("s1.png").flip is False
    assert state.by_id("s2.png").rotation_deg == 90

    # Elastix leaves it out: by name it is refused, and it is no default target.
    fitted = tools["elastix_affine"](["s1.png"])
    assert fitted["status"] == "error"
    assert [(row["id"], row["error"], row["properties"]) for row in fitted["results"]] == [
        ("s1.png", "FROZEN", ["affine"])]
    assert state.by_id("s1.png").transform == held

    moved = tools["position_sections"]([{"id": "s1.png", "position_mm": 2.6}], view=False)
    assert moved["written"] == [{"id": "s1.png", "position_mm": 2.6}]


def test_the_default_fit_targets_leave_out_frozen_affines(tmp_path: Path, monkeypatch):
    asked = _no_fit(monkeypatch)
    _state, _spec, tools = _stack(tmp_path, frozen={"s1.png": ALIGNED})
    tools["elastix_affine"]([])
    assert asked == [["s0.png", "s2.png", "s3.png", "s4.png"]]


def test_frozen_transforms_count_at_submit(tmp_path: Path):
    state, spec, _tools = _stack(
        tmp_path, n=2, tasks=["position", "transform"],
        frozen={"s0.png": ALIGNED, "s1.png": ALIGNED}, inputs={"damaged": {"s0.png": ""}},
    )
    assert submit_errors(state, spec, []) is None
    unlocked_state, unlocked, _ = _stack(tmp_path, n=2, tasks=["position", "transform"])
    refusal = submit_errors(unlocked_state, unlocked, [])
    assert refusal is not None and refusal["error"] == "MISSING_TRANSFORMS"


def test_the_job_statement_names_frozen_sections_and_what_is_frozen(tmp_path: Path):
    state, spec, tools = _stack(tmp_path, frozen={"s3.png": ALIGNED, "s1.png": ALIGNED})
    text = _statement(state, spec, tools)
    assert ("- Frozen by the user: orientation (flip and quarter turn) and affine "
            "(in-plane transform) of s1.png, s3.png.") in text


def test_the_worker_never_emits_geometry_for_frozen_properties():
    from langslice.doors.api.abba_worker import _host_updates

    def row(transform, flip=False, position=4.0):
        return {"id": "a.tif", "position_mm": position, "index_corrected": 0,
                "flip": flip, "rotation_deg": 0, "transform": transform}

    before = {"slices": [row(None)]}
    after = {"slices": [row({"params": [1, 0, .1, 0, 1, 0]}, flip=True, position=4.5)]}
    args = ({"a.tif": (80, 40)}, 25.0)
    assert set(_host_updates(after, before, *args)[0]) == {
        "id", "position_mm", "flip", "rotation_deg", "affine_mm"}
    frozen = _host_updates(after, before, *args, lambda _name: frozenset(ALIGNED))
    assert frozen == [{"id": "a.tif", "position_mm": 4.5}]
    nothing = _host_updates(after, before, *args, lambda _name: frozenset({"position", *ALIGNED}))
    assert nothing == []


# --- a frozen deformation ---------------------------------------------------------


class _NeverCalled:
    """An image model whose call must not happen (the refusal comes first)."""

    def __init__(self) -> None:
        from langslice.providers.registry import resolve_image_model

        resolved = resolve_image_model("openai-oauth", None)
        self.model = dataclasses.replace(resolved, call=self.generate)

    def generate(self, request: Any) -> Any:
        raise AssertionError("the image model was called")


@needs_ants
def test_sections_with_a_frozen_deformation_are_refused(tmp_path: Path):
    code = "FROZEN"
    state, _spec, tools = _stack(
        tmp_path, n=2, tasks=["nonlinear"], image_model=_NeverCalled().model,
        nonlinear=NonlinearSpec(provider="openai-oauth"),
        frozen={"s0.png": ["deformation"]},
        inputs={
                "transforms": {"s0.png": {"kind": "imported", "params": [1, 0, 0, 0, 1, 0]},
                               "s1.png": {"kind": "imported", "params": [1, 0, 0, 0, 1, 0]}}},
    )
    fitted = tools["ants_syn"](["s0.png"])
    assert fitted["status"] == "error" and fitted["error"] == "NOTHING_FITTED"
    assert fitted["results"][0]["id"] == "s0.png"
    assert fitted["results"][0]["error"] == code
    traced = tools["trace_borders"]("s0.png")
    assert traced["status"] == "error" and traced["error"] == code
    assert traced["id"] == "s0.png"
    assert state.by_id("s0.png").deformation is None
    assert state.by_id("s0.png").image_correction is None


# --- forced pictures and the picture size --------------------------------------------


def test_force_view_takes_the_switch_away_from_every_change_tool(tmp_path: Path):
    _state, _spec, tools = _stack(tmp_path, force_view=True)
    for name in ("position_sections", "interactive_transform", "elastix_affine"):
        assert "view" not in inspect.signature(tools[name]).parameters, name
    _state, _spec, free = _stack(tmp_path)
    for name in ("position_sections", "interactive_transform", "elastix_affine"):
        assert "view" in inspect.signature(free[name]).parameters, name


def test_image_resolution_auto_gives_look_a_resolution(tmp_path: Path):
    _state, _spec, fixed = _stack(tmp_path, n=2)
    assert "resolution" not in inspect.signature(fixed["look"]).parameters
    assert fixed["look"]("section", resolution=300)["error"] == "UNKNOWN_ARGUMENTS"

    _state, _spec, auto = _stack(tmp_path, n=2, image_resolution="auto")
    assert "resolution" in inspect.signature(auto["look"]).parameters
    plain = auto["look"]("section", sections=["s0.png"])
    assert plain["status"] == "ok" and "resolution_note" not in plain
    clamped = auto["look"]("section", sections=["s0.png"], resolution=10**6)
    assert clamped["status"] == "ok"
    assert clamped["resolution_note"].startswith("resolution 1000000 is outside")
    assert auto["look"]("section", resolution="big")["error"] == "BAD_ARGS"


@pytest.mark.parametrize("interval,thickness", [(None, None), (0.2, None),
                                               (None, 0.05), (0.2, 0.05)])
def test_prompt_distinguishes_known_protocol_from_inferred_spacing(interval, thickness):
    from langslice.agent.prompt import cutting_protocol_fact
    from langslice.core.state import StackState

    text = cutting_protocol_fact(StackState(interval_mm=interval, thickness_mm=thickness))
    assert ("0.200 mm center-to-center" in text) == (interval is not None)
    assert ("0.050 mm" in text) == (thickness is not None)
    unknown = interval is None or thickness is None
    assert ("not supplied" in text) == unknown
    assert ("Infer positions and spacing from anatomy" in text) == unknown
    assert ("uncertainty in your notes" in text) == unknown
