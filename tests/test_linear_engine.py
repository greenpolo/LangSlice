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


def test_the_input_token_budget_ends_a_run_before_the_account_does(
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
        max_input_tokens=2500,
    )
    state = asyncio.run(run(spec, emit=lines.append, atlas_loader=lambda _n: _ATLAS))

    assert state.submitted is False
    assert state.by_id(names[0]).position_mm == 4.0  # the write survived the stop
    budget = [line for line in lines if "passed the budget" in line]
    assert len(budget) == 1 and "3000" in budget[0]
    assert any(line.startswith("[tokens] call 1: in=1000") for line in lines)
    assert any("run total: 3 calls, in=3000" in line for line in lines)


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
