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
from langslice.core.spec import JobSpec, PositionSpec
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


@pytest.mark.parametrize("inputs", [None, {"positions": {"ghost.png": 1.0}}])
def test_ingest_and_host_inputs_refuse_an_empty_folder_and_an_unknown_filename(
    tmp_path: Path, inputs
):
    if inputs is not None:
        _make_stack(tmp_path, n=2)
    spec = _spec(tmp_path, **({"inputs": inputs} if inputs else {}))
    if inputs is None:
        with pytest.raises(ValueError, match="No slice images"):
            ingest(spec, _ctx(spec))
        return
    state = ingest(spec, _ctx(spec))
    with pytest.raises(ValueError, match="ghost.png"):
        apply_host_inputs(state, spec)


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
    model's next request carries one overlay per written section as an image in
    the function response."""
    names = _make_stack(tmp_path, n=2)
    install_fake_adk_model_stack(monkeypatch, positions={names[0]: 2.0, names[1]: 3.0})
    requests: list = []
    generate = fakes._StackLlm.generate_content_async

    def recording(self, llm_request, stream=False):
        requests.append(llm_request.model_copy(deep=True))
        return generate(self, llm_request, stream)

    monkeypatch.setattr(fakes._StackLlm, "generate_content_async", recording)
    spec = _spec(tmp_path, tasks=["position"], disabled=["look:positioning"])
    state = asyncio.run(run(spec, emit=lambda _m: None, atlas_loader=lambda _n: _ATLAS))
    assert state.submitted is True

    found = []
    for content in requests[1].contents or []:
        for part in content.parts or []:
            response = part.function_response
            if response is not None and response.name == "position_sections":
                found += [item.inline_data.data for item in response.parts or []
                          if item.inline_data is not None]
    assert len(found) == 2 and all(data[:2] == b"\xff\xd8" for data in found)


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


def test_a_session_that_never_submits_keeps_its_writes_and_stops_at_the_context_limit(
    tmp_path: Path, monkeypatch
):
    names = _make_stack(tmp_path, n=3)
    positions = {names[0]: 4.0}  # incomplete: submit is refused every turn
    install_fake_adk_model_stack(monkeypatch, positions=positions, input_tokens_per_call=1000)
    lines: list[str] = []

    spec = _spec(tmp_path, tasks=["position"], position=PositionSpec(interval_um=500),
                 max_input_tokens=999)
    state = asyncio.run(run(spec, emit=lines.append, atlas_loader=lambda _n: _ATLAS))

    assert state.submitted is False
    assert state.by_id(names[0]).position_mm == 4.0  # the write survived the stop
    assert any("no submission" in note for note in state.notes)
    assert (tmp_path / "langslice" / "exports" / "linear_results.json").exists()
    budget = [line for line in lines if "exceeded the context limit" in line]
    assert len(budget) == 1 and "request input 1000" in budget[0]
    assert any(line.startswith("[tokens] call 1: request_input=1000") for line in lines)
    # the stop, then ONE grace call to submit (refused: the stack is incomplete)
    assert any("run total: 2 calls, cumulative_input=2000" in line for line in lines)


@pytest.mark.parametrize("limit", [None, 2500])
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
    positions = {names[0]: 4.0}  # incomplete: submit keeps being refused
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


# --- background work between turns ------------------------------------------


def _job(tmp_path: Path):
    from langslice.job.job import Job

    _make_stack(tmp_path, n=2)
    spec = _spec(tmp_path, tasks=["position"])
    ctx = _ctx(spec)
    return Job.open(spec, ctx, folder=ctx.job_folder, results_path=ctx.results_path)


def test_background_message_waits_for_running_work_and_hands_it_out_once(tmp_path: Path):
    import threading

    from langslice.agent.engine import background_message
    from langslice.job.background import DONE, Landed

    job = _job(tmp_path)
    release = threading.Event()
    picture = Image.new("RGB", (32, 24), (200, 30, 30))

    def land() -> Landed:
        release.wait(5)
        return Landed(status=DONE, text="fitted.", pictures=[picture])

    work = job.background.start("trace_borders", ["slice_1.png"], land,
                                wait_for_images=False)
    threading.Timer(0.02, release.set).start()
    assert job.background.running()  # still running when the turn ends

    items = background_message(job)
    assert items is not None and not job.background.running()
    assert items[0] == ("Background work finished:\n"
                        f"{work.id} trace_borders of slice_1.png finished: fitted. "
                        f"Pictures #{work.pictures[0]}.")
    assert items[1] == f"picture {work.pictures[0]} ({work.id})"
    assert isinstance(items[2], Image.Image) and items[2].size == picture.size
    assert len(items) == 3
    # Each finished piece is handed out once.
    assert background_message(job) is None
    job.background.close()
