"""The run: ingest, host inputs, a fake-model session, results.

No live model calls: the session is driven by a scripted fake BaseLlm swapped
in through ``LLMRegistry.new_llm`` (tests/fakes.py).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from langslice.agent.engine import build_context, run
from langslice.agent.prompt import build_job_statement
from langslice.core.spec import JobSpec, PositionSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.job.checkpoint import load_checkpoint
from langslice.job.job import apply_host_inputs, ingest
from tests.fakes import SlabAtlas, install_fake_adk_model_stack

_ATLAS = SlabAtlas()

def _make_stack(folder: Path, n: int = 4) -> list[str]:
    """Tiny generated PNGs with non-lexicographic numbering."""
    names = [f"slice_{i}.png" for i in (1, 2, 3, 10, 11)][:n]
    for index, name in enumerate(names):
        Image.new("RGB", (40, 30), (10 * index, 60, 120)).save(folder / name)
    (folder / "notes.txt").write_text("not an image")
    return names


def _spec(folder: Path, **kwargs) -> JobSpec:
    kwargs.setdefault("model", "fake-model")
    kwargs.setdefault("preprocess", "none")
    return JobSpec(image_folder=str(folder), **kwargs)


def _ctx(spec: JobSpec, atlas=_ATLAS):
    return build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)


# --- ingest --------------------------------------------------------------


def test_ingest_discovers_the_folder_in_natural_order(tmp_path: Path):
    names = _make_stack(tmp_path, n=5)
    spec = _spec(tmp_path)
    state = ingest(spec, _ctx(spec))

    assert [s.id for s in state.slices] == names
    assert [s.index_corrected for s in state.slices] == [0, 1, 2, 3, 4]
    assert all(s.position_mm is None and not s.flip for s in state.slices)
    assert state.interval_mm == 0.2 and state.thickness_mm == 0.05
    assert state.spec["atlas"] == "allen_mouse_25um"
    assert "0.00-19.00 mm" in state.notes[0]


def test_ingest_rejects_an_empty_folder(tmp_path: Path):
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="No slice images"):
        ingest(spec, _ctx(spec))


def test_host_inputs_set_the_order_positions_and_angles(tmp_path: Path):
    names = _make_stack(tmp_path, n=3)
    spec = _spec(
        tmp_path,
        tasks=["transform"],
        inputs={
            "order": list(reversed(names)),
            "positions": {names[0]: 3.0, names[1]: 2.0},
            "angles": {"pitch": 2.5, "yaw": -1.0},
        },
    )
    state = ingest(spec, _ctx(spec))
    apply_host_inputs(state, spec)

    assert [s.id for s in state.in_order()] == list(reversed(names))
    assert state.by_id(names[0]).position_mm == 3.0
    assert state.cutting_angles_deg == {"pitch": 2.5, "yaw": -1.0}


def test_a_mirrored_host_transform_is_accepted_and_kept(tmp_path: Path):
    # A host's own alignment (ABBA) carries a flip inside its affine: a
    # negative determinant. The harness keeps it as supplied.
    from langslice.job.job import submit_errors

    names = _make_stack(tmp_path, n=2)
    mirrored = {"kind": "host", "params": [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0], "mirrored": True}
    plain = {"kind": "host", "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
    spec = _spec(
        tmp_path,
        tasks=["transform"],
        inputs={
            "positions": {names[0]: 2.0, names[1]: 2.5},
            "transforms": {names[0]: mirrored, names[1]: plain},
        },
    )
    state = ingest(spec, _ctx(spec))
    apply_host_inputs(state, spec)

    record = state.by_id(names[0])
    assert record.transform == mirrored and record.transform is not mirrored
    assert record.flip is False  # the mirror stays inside the transform
    assert submit_errors(state, spec, []) is None


def test_host_inputs_reject_an_unknown_filename(tmp_path: Path):
    _make_stack(tmp_path, n=2)
    spec = _spec(tmp_path, inputs={"positions": {"ghost.png": 1.0}})
    state = ingest(spec, _ctx(spec))
    with pytest.raises(ValueError, match="ghost.png"):
        apply_host_inputs(state, spec)


# --- the prompt ----------------------------------------------------------


def test_the_job_statement_lists_only_the_tools_that_exist(tmp_path: Path):
    _make_stack(tmp_path, n=3)
    spec = _spec(tmp_path, tasks=["position"], facts=["the block was cut back to front"])
    ctx = _ctx(spec)
    state = ingest(spec, ctx)
    box = build_tools(state, ctx, spec)
    pos_lo, pos_hi = ctx.position_range

    text = build_job_statement(
        spec,
        state,
        tool_names=box.names,
        species="mouse",
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        axis_ends=ctx.axis_ends,
    )
    assert "`set_positions`" in text and "`submit`" in text
    assert "`reorder_slices`" not in text and "`fit_affine`" not in text
    assert "the block was cut back to front" in text
    assert "0.00-19.00 mm" in text
    # the one direction fact, derived from the atlas orientation
    assert ctx.axis_ends == ("anterior", "posterior")
    assert "0.00 mm is the anterior edge" in text
    assert "positions increase toward posterior" in text
    # atlas-agnostic: no region names, no strategy
    for banned in ("cortex", "hippocampus", "strategy", "tip:", "you should"):
        assert banned not in text.lower()


def _statement(spec: JobSpec) -> str:
    ctx = _ctx(spec)
    state = ingest(spec, ctx)
    pos_lo, pos_hi = ctx.position_range
    return build_job_statement(
        spec, state, tool_names=build_tools(state, ctx, spec).names, species="mouse",
        pos_lo=pos_lo, pos_hi=pos_hi, axis_ends=ctx.axis_ends,
    )


def test_mirroring_is_part_of_linear_never_of_positioning(tmp_path: Path):
    from langslice.core.spec import TransformSpec

    _make_stack(tmp_path, n=3)
    cue = TransformSpec(hemisphere_cue="ink on the right")
    positioning = _statement(_spec(tmp_path, tasks=["reorder", "position"], transform=cue))
    for word in ("mirror", "flip", "hemisphere", "orient_slices"):
        assert word not in positioning.lower()

    linear = _statement(_spec(tmp_path, tasks=["transform"], transform=cue))
    assert "`orient_slices`" in linear
    assert "orientation of any section that is mirrored or turned" in linear
    assert "ink on the right" in linear

    no_flip = _statement(_spec(
        tmp_path, tasks=["transform"],
        transform=TransformSpec(flip=False, hemisphere_cue="ink on the right"),
    ))
    assert "Flipping sections is switched off" in no_flip
    assert "mirrored or turned" not in no_flip and "ink on the right" not in no_flip


# --- end to end ----------------------------------------------------------


def test_run_places_the_stack_and_writes_results(tmp_path: Path, monkeypatch):
    names = _make_stack(tmp_path, n=4)
    positions = {name: 2.0 + index for index, name in enumerate(names)}
    install_fake_adk_model_stack(monkeypatch, positions=positions)

    spec = _spec(tmp_path, tasks=["position"])
    state = asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS))

    assert state.submitted is True
    assert [s.position_mm for s in state.in_order()] == list(positions.values())

    results = json.loads((tmp_path / "langslice" / "exports" / "linear_results.json").read_text())
    assert [row["id"] for row in results["slices"]] == names
    assert results["submitted"] is True

    checkpoint = load_checkpoint(str(tmp_path / "langslice" / "state.json"))
    assert checkpoint is not None and checkpoint.submitted is True


def test_a_tools_plain_pictures_reach_the_model_as_message_images(
    tmp_path: Path, monkeypatch
):
    """The tools return plain pictures; the ADK door packages them, so the
    model's next request carries the write's picture as an image in the
    function response."""
    from tests import fakes

    names = _make_stack(tmp_path, n=2)
    install_fake_adk_model_stack(monkeypatch, positions={names[0]: 2.0, names[1]: 3.0})
    requests: list = []
    generate = fakes._StackLlm.generate_content_async

    def recording(self, llm_request, stream=False):
        requests.append(llm_request.model_copy(deep=True))
        return generate(self, llm_request, stream)

    monkeypatch.setattr(fakes._StackLlm, "generate_content_async", recording)
    state = asyncio.run(run(_spec(tmp_path, tasks=["position"]), emit=lambda _m: None,
                            atlas_loader=lambda _n: _ATLAS))
    assert state.submitted is True

    def images(request) -> list[bytes]:
        found = []
        for content in request.contents or []:
            for part in content.parts or []:
                response = part.function_response
                if response is not None and response.name == "set_positions":
                    found += [item.inline_data.data for item in response.parts or []
                              if item.inline_data is not None]
        return found

    pictures = images(requests[1])
    assert len(pictures) == 2 and all(data[:2] == b"\xff\xd8" for data in pictures)


def test_run_calls_on_write_with_the_initial_state_and_every_checkpoint(
    tmp_path: Path, monkeypatch
):
    names = _make_stack(tmp_path, n=3)
    positions = {name: 2.0 + index for index, name in enumerate(names)}
    install_fake_adk_model_stack(monkeypatch, positions=positions)
    seen: list[bool] = []  # snapshot of "has any position yet" per call

    def on_write(state):
        seen.append(any(s.position_mm is not None for s in state.slices))

    spec = _spec(tmp_path, tasks=["position"])
    state = asyncio.run(
        run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS, on_write=on_write)
    )

    assert state.submitted is True
    # the very first call is the freshly-ingested stack, before any write
    assert seen[0] is False
    # at least one later call saw the positions land
    assert seen[-1] is True


def test_run_resumes_from_the_checkpoint(tmp_path: Path, monkeypatch):
    names = _make_stack(tmp_path, n=3)
    positions = {name: 2.0 + index for index, name in enumerate(names)}
    install_fake_adk_model_stack(monkeypatch, positions=positions)
    spec = _spec(tmp_path, tasks=["position"])
    asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS))

    # A second run resumes: the fake never re-writes positions (the stack is
    # already placed) and still submits.
    install_fake_adk_model_stack(monkeypatch, positions=None)
    state = asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS))
    assert state.submitted is True
    assert [s.position_mm for s in state.in_order()] == list(positions.values())


def test_a_session_that_never_submits_keeps_its_writes(tmp_path: Path, monkeypatch):
    names = _make_stack(tmp_path, n=3)
    positions = {names[0]: 4.0}  # incomplete: submit is refused every turn
    install_fake_adk_model_stack(monkeypatch, positions=positions)

    spec = _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500))
    state = asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS))

    assert state.submitted is False
    assert state.by_id(names[0]).position_mm == 4.0
    assert any("no submission" in note for note in state.notes)
    assert (tmp_path / "langslice" / "exports" / "linear_results.json").exists()


def test_one_request_over_the_input_context_limit_stops_with_grace(
    tmp_path: Path, monkeypatch
):
    names = _make_stack(tmp_path, n=3)
    positions = {names[0]: 4.0}  # incomplete: it would loop to max_iterations
    install_fake_adk_model_stack(monkeypatch, positions=positions, input_tokens_per_call=1000)
    lines: list[str] = []

    spec = _spec(
        tmp_path,
        tasks=["position"],
        position=PositionSpec(interval_um=500),
        max_input_tokens=999,
    )
    state = asyncio.run(run(spec, emit=lines.append, atlas_loader=lambda _n: _ATLAS))

    assert state.submitted is False
    assert state.by_id(names[0]).position_mm == 4.0  # the write survived the stop
    budget = [line for line in lines if "exceeded the context limit" in line]
    assert len(budget) == 1 and "request input 1000" in budget[0]
    assert any(line.startswith("[tokens] call 1: request_input=1000") for line in lines)
    # the stop, then ONE grace call to submit (refused: the stack is incomplete)
    assert any("run total: 2 calls, cumulative_input=2000" in line for line in lines)


@pytest.mark.parametrize("limit", [None, 1000, 2500])
def test_repeated_inputs_are_not_cumulative_context(tmp_path: Path, monkeypatch, limit):
    names = _make_stack(tmp_path, n=3)
    install_fake_adk_model_stack(
        monkeypatch, positions=dict(zip(names, [4.0, 4.5, 5.0], strict=True)),
        input_tokens_per_call=1000,
    )
    lines: list[str] = []
    state = asyncio.run(run(
        _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500),
              max_input_tokens=limit),
        emit=lines.append, atlas_loader=lambda _n: _ATLAS,
    ))
    assert state.submitted
    assert not any("exceeded the context limit" in line for line in lines)
    assert any("cumulative_input=3000" in line and "peak_input=1000" in line for line in lines)


def test_input_context_safeguard_is_disabled_by_default(tmp_path):
    assert _spec(tmp_path).max_input_tokens is None


def test_token_tally_tracks_last_and_peak_including_cached_input():
    from types import SimpleNamespace

    from langslice.agent.session import TokenTally

    tally = TokenTally()
    for prompt, cached in [(2000, 1900), (1000, 900)]:
        tally.add(SimpleNamespace(prompt_token_count=prompt,
                                  cached_content_token_count=cached,
                                  candidates_token_count=10))
    assert tally.as_dict()["input"] == 3000
    assert tally.as_dict()["latest_input"] == 1000
    assert tally.as_dict()["peak_input"] == 2000


def test_the_quota_budget_is_measured_from_the_first_call(tmp_path: Path, monkeypatch):
    """The window was at 40% before the run; the run may spend 25 points of
    it, not reach 25%."""
    names = _make_stack(tmp_path, n=3)
    positions = {names[0]: 4.0}  # incomplete: it would loop to max_iterations
    install_fake_adk_model_stack(monkeypatch, positions=positions, quota_percent_per_call=10)
    lines: list[str] = []

    spec = _spec(
        tmp_path,
        tasks=["position"],
        position=PositionSpec(interval_um=500),
        max_quota_percent=25,
    )
    state = asyncio.run(run(spec, emit=lines.append, atlas_loader=lambda _n: _ATLAS))

    assert state.submitted is False
    stops = [line for line in lines if "the budget of 25%" in line]
    assert len(stops) == 1 and "used 30%" in stops[0]  # 50, 60, 70, 80: 0, 10, 20, 30
    assert any("window used by this run 10%" in line for line in lines)
    assert any("run total: 5 calls" in line for line in lines)  # the stop + one grace call


def test_the_seed_carries_section_strips_then_the_atlas_reference(tmp_path: Path):
    from langslice.agent.engine import build_context, build_seed_message
    from langslice.job.job import ingest

    _make_stack(tmp_path, n=3)
    spec = _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    state = ingest(spec, ctx)
    parts = build_seed_message(state, ctx).parts or []
    texts = [p.text for p in parts if p.text]
    # No positions: one section-only strip, then the atlas reference strips.
    assert texts[0].startswith("The 3 sections of the stack follow in 1 strip,")
    assert "Beneath each section" not in texts[0]
    assert texts[1] == "Strip 1 of 1: " + ", ".join(
        f"{r.index_corrected}: {r.id}" for r in state.in_order())
    reference = next(t for t in texts if t.startswith("Atlas reference strip"))
    n_atlas = int(reference.split("the atlas at ")[1].split()[0])
    assert 1 < n_atlas <= 48
    atlas_strips = [t for t in texts if t.startswith("Atlas strip ")]
    images = [p for p in parts if p.inline_data is not None]
    assert len(images) == 1 + len(atlas_strips) >= 2
    assert texts.index(reference) > 1
    assert texts[-1].startswith("Status table")


def test_the_job_statement_states_the_alignment_frame_when_transforms_are_on(tmp_path: Path):
    names = _make_stack(tmp_path, n=2)
    del names
    on = _spec(tmp_path, tasks=["transform"])
    off = _spec(tmp_path, tasks=["position"])
    from langslice.agent.engine import build_context
    from langslice.agent.prompt import build_job_statement
    from langslice.doors.tools.toolbox import build_tools
    from langslice.job.job import ingest

    for spec, expected in ((on, True), (off, False)):
        ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
        state = ingest(spec, ctx)
        box = build_tools(state, ctx, spec)
        text = build_job_statement(
            spec, state, tool_names=box.names, species="mouse", pos_lo=0.0, pos_hi=10.0,
            axis_ends=("anterior", "posterior"),
        )
        assert ("TRUE physical size" in text) is expected
        assert "landmark" not in text.lower()
        assert "regularized spline" not in text.lower()
