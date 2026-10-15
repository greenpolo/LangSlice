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
from langslice.job.job import apply_host_inputs, ingest
from tests import fakes
from tests.fakes import SlabAtlas, install_fake_adk_model_stack
from tests.linear_tool_helpers import load_checkpoint

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
    assert state.interval_mm is None and state.thickness_mm is None
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
    assert "`position_sections`" in text and "`look`" in text and "`submit`" in text
    # Positioning only: no transform, fit or retired tool is named.
    for absent in ("`interactive_transform`", "`elastix_affine`", "`ants_syn`",
                   "`set_positions`", "`reorder_slices`", "`fit_affine`"):
        assert absent not in text
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
    positioning = _statement(_spec(tmp_path, tasks=["position"], transform=cue))
    for word in ("mirror", "flip", "hemisphere", "interactive_transform"):
        assert word not in positioning.lower()

    linear = _statement(_spec(tmp_path, tasks=["transform"], transform=cue))
    assert "`interactive_transform`" in linear
    assert "orientation of any section that is mirrored or turned" in linear
    assert "ink on the right" in linear

    no_flip = _statement(_spec(
        tmp_path, tasks=["transform"],
        transform=TransformSpec(hemisphere_cue="ink on the right"),
        disabled=["interactive_transform:flip"],
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
                if response is not None and response.name == "position_sections":
                    found += [item.inline_data.data for item in response.parts or []
                              if item.inline_data is not None]
        return found

    # position_sections shows its positioning picture of the written sections.
    pictures = images(requests[1])
    assert len(pictures) == 1 and all(data[:2] == b"\xff\xd8" for data in pictures)


def test_a_run_with_disabled_tools_shows_the_model_neither_the_tools_nor_the_options(
    tmp_path: Path, monkeypatch
):
    """The system instruction and the tools of every request name nothing the
    job disables; position_sections draws one overlay per written section
    where the positioning look is off."""
    names = _make_stack(tmp_path, n=2)
    install_fake_adk_model_stack(monkeypatch, positions={names[0]: 2.0, names[1]: 3.0})
    requests: list = []
    generate = fakes._StackLlm.generate_content_async

    def recording(self, llm_request, stream=False):
        requests.append(llm_request.model_copy(deep=True))
        return generate(self, llm_request, stream)

    monkeypatch.setattr(fakes._StackLlm, "generate_content_async", recording)
    spec = _spec(tmp_path, tasks=["position"], position=PositionSpec(playbook=True),
                 disabled=["look:positioning", "zoom", "grep_atlas_view"])
    state = asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS))
    assert state.submitted is True

    first = requests[0]
    instruction = str(first.config.system_instruction)
    tools = first.config.tools[0].function_declarations
    declared = {tool.name: tool for tool in tools}
    assert "zoom" not in declared and "grep_atlas_view" not in declared
    text = instruction + " ".join(tool.description for tool in tools)
    for gone in ("`zoom`", "grep_atlas_view", "mode positioning", "positioning mode",
                 "zoom takes"):
        assert gone not in text
    mode = declared["look"].parameters_json_schema["properties"]["mode"]
    assert mode["enum"] == ["section", "atlas", "overlay"]

    def images(request) -> list[bytes]:
        found = []
        for content in request.contents or []:
            for part in content.parts or []:
                response = part.function_response
                if response is not None and response.name == "position_sections":
                    found += [item.inline_data.data for item in response.parts or []
                              if item.inline_data is not None]
        return found

    assert len(images(requests[1])) == 2  # an overlay for each written section


def test_run_calls_on_write_with_the_initial_state_and_every_checkpoint(
    tmp_path: Path, monkeypatch
):
    names = _make_stack(tmp_path, n=3)
    positions = {name: 2.0 + index for index, name in enumerate(names)}
    install_fake_adk_model_stack(monkeypatch, positions=positions)
    seen: list[bool] = []  # snapshot of "has any position of its own yet" per call

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


def test_a_session_that_fails_before_submitting_keeps_its_writes_and_its_spend(
    tmp_path: Path, monkeypatch
):
    names = _make_stack(tmp_path, n=3)
    positions = {names[0]: 4.0}  # incomplete: submit is refused every turn
    install_fake_adk_model_stack(monkeypatch, positions=positions, input_tokens_per_call=1000,
                                 fail_after_calls=3)

    spec = _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500))
    with pytest.raises(RuntimeError, match="fake model failed"):
        asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS))

    exports = tmp_path / "langslice" / "exports"
    spend = json.loads((exports / "spend.json").read_text())
    assert spend["tokens"]["input_uncached"] == 3000  # the three calls that were answered
    saved = load_checkpoint(tmp_path / "langslice" / "state.json")
    assert saved.by_id(names[0]).position_mm == 4.0


def test_repeated_inputs_are_not_cumulative_context(tmp_path: Path, monkeypatch):
    names = _make_stack(tmp_path, n=3)
    install_fake_adk_model_stack(
        monkeypatch, positions=dict(zip(names, [4.0, 4.5, 5.0], strict=True)),
        input_tokens_per_call=1000,
    )
    lines: list[str] = []
    state = asyncio.run(run(
        _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500)),
        emit=lines.append, atlas_loader=lambda _n: _ATLAS,
    ))
    assert state.submitted
    assert any("cumulative_input=3000" in line and "peak_input=1000" in line for line in lines)


def test_no_budget_setting_exists(tmp_path):
    for key in ("max_input_tokens", "max_quota_percent"):
        with pytest.raises(ValueError, match="Unknown job setting"):
            JobSpec.from_dict({"image_folder": str(tmp_path), key: 25})


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


def test_window_share_warnings_go_to_the_operator_every_ten_points(
    tmp_path: Path, monkeypatch
):
    """The window stands at 50% on the run's first call and rises 10 points per
    call: the run is told at 10% and 20%, measured from its own first call, and
    nothing stops it; the account lands in exports/spend.json and the trace."""
    names = _make_stack(tmp_path, n=3)
    install_fake_adk_model_stack(
        monkeypatch, positions=dict(zip(names, [4.0, 4.5, 5.0], strict=True)),
        quota_percent_per_call=10)
    lines: list[str] = []
    events: list[dict] = []
    monkeypatch.setenv("LANGSLICE_TRACE_DIR", str(tmp_path / "trace"))

    spec = _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500))
    state = asyncio.run(run(spec, emit=lines.append, atlas_loader=lambda _n: _ATLAS,
                            on_event=events.append))

    assert state.submitted is True
    spent = [line for line in lines if line.startswith("[spend] this run has used")]
    assert spent == [
        "[spend] this run has used 10% of the usage window (passed 10%)",
        "[spend] this run has used 20% of the usage window (passed 20%)",
    ]
    assert [e["threshold"] for e in events if e["kind"] == "spend"
            and "used_percent" in e] == [10, 20]
    assert any("window used by this run 10%" in line for line in lines)
    assert any(line == "[spend] run total: 20% of the usage window" for line in lines)
    spend = json.loads((tmp_path / "langslice" / "exports" / "spend.json").read_text())
    assert spend["quota_used_percent"] == 20 and spend["usd_estimate"] is None
    records = [json.loads(row) for path in (tmp_path / "trace").glob("*.jsonl")
               for row in path.read_text().splitlines()]
    assert [r["threshold"] for r in records if r["kind"] == "spend"
            and "used_percent" in r] == [10, 20]
    assert next(r for r in records if r["kind"] == "summary")["spend"]["quota_used_percent"] == 20
    assert next(e for e in events if e["kind"] == "complete")["spend"]["quota_used_percent"] == 20


def test_an_api_key_lane_is_told_its_dollar_estimate_through_the_whole_session(
    tmp_path: Path, monkeypatch
):
    """Three calls of 4M uncached input tokens on Gemini 2.5 Pro ($1.25 per
    million) cost $5 each: the operator hears at $5 and $10, the model hears
    nothing, and the account has the estimate."""
    names = _make_stack(tmp_path, n=3)
    install_fake_adk_model_stack(
        monkeypatch, positions=dict(zip(names, [4.0, 4.5, 5.0], strict=True)),
        input_tokens_per_call=4_000_000)
    lines: list[str] = []
    events: list[dict] = []

    spec = _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500),
                 model="gemini-2.5-pro")
    state = asyncio.run(run(spec, emit=lines.append, atlas_loader=lambda _n: _ATLAS,
                            on_event=events.append))

    assert state.submitted is True
    assert [line for line in lines if line.startswith("[spend] this run's")] == [
        "[spend] this run's estimated API cost is $5.00 (passed $5; gemini-api/gemini-2.5-pro, "
        "list prices)",
        "[spend] this run's estimated API cost is $10.00 (passed $10; gemini-api/gemini-2.5-pro, "
        "list prices)",
    ]
    assert [e["threshold"] for e in events if e["kind"] == "spend"] == [5.0, 10.0]
    spend = json.loads((tmp_path / "langslice" / "exports" / "spend.json").read_text())
    assert spend["usd_estimate"] == 15.0 and spend["quota_used_percent"] is None
