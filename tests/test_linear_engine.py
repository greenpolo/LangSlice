"""The run: ingest, host inputs, a fake-model session, results.

No live model calls: the session is driven by a scripted fake BaseLlm swapped
in through ``LLMRegistry.new_llm`` (tests/fakes.py).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from langslice.linear import JobSpec, run
from langslice.linear.checkpoint import load_checkpoint
from langslice.linear.engine import (
    apply_host_inputs,
    build_context,
    emit_results,
    ingest,
)
from langslice.linear.prompt import build_job_statement
from langslice.linear.spec import PositionSpec
from langslice.linear.toolbox import build_tools
from tests.fakes import SlabAtlas, install_fake_adk_model_stack

_ATLAS = SlabAtlas()

#: The folder of real sections the no-model smoke test uses when it is there.
_REAL_STACK = Path.home() / "LSD_910" / "images" / "M04"


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


# --- end to end ----------------------------------------------------------


def test_run_places_the_stack_and_writes_results(tmp_path: Path, monkeypatch):
    names = _make_stack(tmp_path, n=4)
    positions = {name: 2.0 + index for index, name in enumerate(names)}
    install_fake_adk_model_stack(monkeypatch, positions=positions)

    spec = _spec(tmp_path, tasks=["position"])
    state = asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS))

    assert state.submitted is True
    assert [s.position_mm for s in state.in_order()] == list(positions.values())

    results = json.loads((tmp_path / "linear_results.json").read_text())
    assert [row["id"] for row in results["slices"]] == names
    assert results["submitted"] is True

    checkpoint = load_checkpoint(str(tmp_path / "linear_state.json"))
    assert checkpoint is not None and checkpoint.submitted is True


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
    assert (tmp_path / "linear_results.json").exists()


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

    from langslice.linear.session import TokenTally

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


def test_the_seed_carries_the_atlas_strip_after_the_sections(tmp_path: Path):
    from langslice.linear.engine import build_context, build_seed_message, ingest

    _make_stack(tmp_path, n=3)
    spec = _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500))
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
    state = ingest(spec, ctx)
    parts = build_seed_message(state, ctx).parts or []
    texts = [p.text for p in parts if p.text]
    intro = next(t for t in texts if t.startswith("The atlas follows at"))
    n_atlas = sum(1 for t in texts if t.startswith("atlas ") and t.endswith(" mm"))
    assert f"at {n_atlas} positions" in intro and 1 < n_atlas <= 48
    images = [p for p in parts if p.inline_data is not None]
    assert len(images) == 3 + n_atlas
    # sections first, then the atlas, then the table
    assert texts.index(intro) > texts.index("0: " + state.in_order()[0].id)
    assert texts[-1].startswith("Status table")


# --- a real folder, no model ---------------------------------------------


@pytest.mark.skipif(not _REAL_STACK.is_dir(), reason="LSD_910/M04 is not on this machine")
def test_ingest_tools_and_emit_on_a_real_folder(tmp_path: Path):
    """ingest -> build_tools -> a tool sequence -> emit, on real TIFFs."""
    for path in sorted(_REAL_STACK.iterdir()):
        if path.suffix.lower() in {".tif", ".tiff", ".png"}:
            os.symlink(path, tmp_path / path.name)

    spec = _spec(tmp_path, tasks=["position"], out=str(tmp_path / "out.json"))
    ctx = _ctx(spec)
    state = ingest(spec, ctx)
    assert len(state.slices) > 1

    box = build_tools(state, ctx, spec)
    tools = {tool.__name__: tool for tool in box.tools}
    first, last = state.in_order()[0].id, state.in_order()[-1].id
    assert tools["set_positions"](
        [{"id": first, "position_mm": 3.0}, {"id": last, "position_mm": 9.0}]
    )["status"] == "ok"
    assert tools["status"]()["rows"][0]["position_mm"] == 3.0

    emit_results(state, ctx)
    written = json.loads(Path(spec.out).read_text())
    assert len(written["slices"]) == len(state.slices)


def test_the_job_statement_states_the_alignment_frame_when_transforms_are_on(tmp_path: Path):
    names = _make_stack(tmp_path, n=2)
    del names
    on = _spec(tmp_path, tasks=["transform"])
    off = _spec(tmp_path, tasks=["position"])
    from langslice.linear.engine import build_context, ingest
    from langslice.linear.prompt import build_job_statement
    from langslice.linear.toolbox import build_tools

    for spec, expected in ((on, True), (off, False)):
        ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS)
        state = ingest(spec, ctx)
        box = build_tools(state, ctx, spec)
        text = build_job_statement(
            spec, state, tool_names=box.names, species="mouse", pos_lo=0.0, pos_hi=10.0,
            axis_ends=("anterior", "posterior"),
        )
        assert ("TRUE physical size" in text) is expected
