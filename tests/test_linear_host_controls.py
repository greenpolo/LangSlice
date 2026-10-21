"""Host dialog controls on the job spec: validation, parallel cap, damage, frozen
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
from langslice.core import deformation
from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import HOST_TRANSFORM_KIND, apply_host_inputs, ingest, submit_errors
from langslice.ops import transforms as ops_transforms
from langslice.ops.transforms import AffineFit
from tests.deformable_synthetic import SyntheticAtlas
from tests.fakes import SlabAtlas

_ATLAS = SlabAtlas()
needs_engine = pytest.mark.skipif(not deformation.engine_available(), reason="needs FireANTs")


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


def _entry(slice_id: str) -> dict[str, Any]:
    return {"id": slice_id, "rotation_deg": 1.0, "scale_x": 1.0, "scale_y": 1.0,
            "translate_x_mm": 0.1, "translate_y_mm": 0.0}


# --- the spec --------------------------------------------------------------


def test_new_fields_round_trip_through_from_dict(tmp_path: Path):
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


# --- max parallel ----------------------------------------------------------


def _no_fit(monkeypatch) -> list[list[str]]:
    """Stand in for the Elastix fit: the cap is checked before it runs."""
    asked: list[list[str]] = []

    def fit(job, ctx, sections, regions, *_inputs):
        names = list(sections)
        asked.append(names)
        return AffineFit(rows=[{"id": name, "status": "error", "error": "FIT_FAILED"}
                               for name in names])

    monkeypatch.setattr(ops_transforms, "elastix_affine", fit)
    return asked


@pytest.mark.parametrize("cap", [2, 4])
def test_over_cap_transform_calls_are_refused_naming_the_cap(
        tmp_path: Path, monkeypatch, cap: int):
    asked = _no_fit(monkeypatch)
    state, _spec, tools = _stack(tmp_path, transform=TransformSpec(max_parallel=cap))
    names = [f"s{i}.png" for i in range(cap + 1)]
    for refused in (tools["interactive_transform"]([_entry(n) for n in names]),
                    tools["elastix_affine"](names)):
        assert refused["error"] == "TOO_MANY_SECTIONS" and refused["max_sections"] == cap
    if cap == 2:
        assert tools["elastix_affine"](names)["requested"] == 3
    assert all(record.transform is None for record in state.slices)
    assert tools["elastix_affine"]([])["error"] == "BAD_ARGS"
    assert asked == []

    allowed = tools["interactive_transform"]([_entry(n) for n in names[:cap]])
    assert allowed["status"] == "ok"
    assert len(tools["elastix_affine"](names[:cap])["results"]) == cap
    assert asked == [names[:cap]]


# --- damage ----------------------------------------------------------------


def test_a_host_damage_note_is_a_note_not_damage(tmp_path: Path):
    state, _spec, tools = _stack(tmp_path, n=3, atlas=SyntheticAtlas(),
                                inputs={"damaged": {"s1.png": "torn"}})
    record = state.by_id("s1.png")
    assert record.damage_note == "torn" and record.damaged is False
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


def test_frozen_transforms_count_at_submit(tmp_path: Path):
    state, spec, _tools = _stack(
        tmp_path, n=2, tasks=["position", "transform"],
        frozen={"s0.png": ALIGNED, "s1.png": ALIGNED}, inputs={"damaged": {"s0.png": ""}},
    )
    assert submit_errors(state, spec, []) is None
    unlocked_state, unlocked, _ = _stack(tmp_path, n=2, tasks=["position", "transform"])
    refusal = submit_errors(unlocked_state, unlocked, [])
    assert refusal is not None and refusal["error"] == "MISSING_TRANSFORMS"


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


@needs_engine
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
    fitted = tools["fireants_syn"](["s0.png"])
    assert fitted["status"] == "error" and fitted["error"] == "FROZEN"
    assert fitted["results"][0]["id"] == "s0.png"
    assert fitted["results"][0]["error"] == code
    traced = tools["trace_borders"]("s0.png")
    assert traced["status"] == "error" and traced["error"] == code
    assert traced["id"] == "s0.png"
    assert state.by_id("s0.png").deformation is None
    assert state.by_id("s0.png").image_correction is None


# --- forced pictures and the picture size --------------------------------------------


def test_tool_signatures_follow_the_spec(tmp_path: Path):
    """force_view removes the view switch; image_resolution=auto gives look a
    resolution, clamped when out of range."""
    _state, _spec, forced = _stack(tmp_path, n=2, force_view=True)
    _state, _spec, free = _stack(tmp_path, n=2)
    for name in ("position_sections", "interactive_transform", "elastix_affine"):
        assert "view" not in inspect.signature(forced[name]).parameters, name
        assert "view" in inspect.signature(free[name]).parameters, name

    assert "resolution" not in inspect.signature(free["look"]).parameters
    assert free["look"]("section", resolution=300)["error"] == "UNKNOWN_ARGUMENTS"
    _state, _spec, auto = _stack(tmp_path, n=2, image_resolution="auto")
    assert "resolution" in inspect.signature(auto["look"]).parameters
    plain = auto["look"]("section", sections=["s0.png"])
    assert plain["status"] == "ok" and "resolution_note" not in plain
    clamped = auto["look"]("section", sections=["s0.png"], resolution=10**6)
    assert clamped["status"] == "ok"
    assert clamped["resolution_note"].startswith("resolution 1000000 is outside")
    assert auto["look"]("section", resolution="big")["error"] == "BAD_ARGS"
