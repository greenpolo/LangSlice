"""Anchor seeding: anchor choice, interpolation fill, failure handling.

The single-slice worker is monkeypatched throughout — seeding would otherwise
run a live estimation session per anchor.
"""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from langslice.linear import APResult
from langslice.linear.whole_brain.deepslice import deepslice_available, run_deepslice
from langslice.linear.whole_brain.engine import build_context
from langslice.linear.whole_brain.nodes import ingest, seed
from langslice.linear.whole_brain.seeding import (
    DAMAGED_CAVEAT,
    anchor_budget,
    seed_positions,
    select_anchor_indices,
)
from langslice.linear.whole_brain.state import BrainConfig, StackState

_SEEDING = "langslice.linear.whole_brain.seeding.run_slice_estimation"


class _FakeVolume:
    shape = (528, 320, 456)


class _FakeAtlas:
    atlas_name = "fake_mouse_25um"
    orientation = "asr"
    reference = _FakeVolume()
    resolution = (25.0, 25.0, 25.0)
    metadata = {"species": "mouse"}


def _ingested(folder: Path, n: int = 9, **kwargs) -> tuple[StackState, object]:
    for index in range(n):
        Image.new("RGB", (40, 30), (10 * index, 60, 120)).save(
            folder / f"slice_{index:02d}.png"
        )
    ctx = build_context(
        BrainConfig(image_folder=str(folder), **kwargs),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    state = StackState()
    asyncio.run(ingest(state, ctx))
    return state, ctx


def _worker(fail_ids: tuple[str, ...] = (), spacing_mm: float = 0.5):
    """Fake worker: position = corrected index * spacing, or a failure."""
    calls: list[str] = []

    async def run(*, image_path: str, atlas_name: str, plane: str, model_name):
        del atlas_name, plane, model_name
        slice_id = Path(image_path).name
        calls.append(slice_id)
        if slice_id in fail_ids:
            raise RuntimeError(f"quota exhausted for {slice_id}")
        index = int(Path(image_path).stem.split("_")[-1])
        return APResult(position_mm=index * spacing_mm, reasoning="fake")

    run.calls = calls  # type: ignore[attr-defined]
    return run


# --- anchor selection ----------------------------------------------------


def test_anchor_budget_grows_with_the_stack():
    assert anchor_budget(8) == 4
    assert anchor_budget(20) == 4
    assert anchor_budget(21) == 6


def test_anchors_spread_across_the_stack_and_avoid_the_extremes():
    picks = select_anchor_indices(24, 6)

    assert len(picks) == 6
    assert picks == sorted(set(picks))
    assert 0 not in picks and 23 not in picks
    gaps = [b - a for a, b in zip(picks, picks[1:], strict=False)]
    assert max(gaps) - min(gaps) <= 1  # evenly spread


def test_anchors_skip_damaged_sections():
    picks = select_anchor_indices(12, 4, damaged={2, 5, 7})

    assert len(picks) == 4
    assert not {2, 5, 7} & set(picks)


def test_anchor_selection_edges():
    assert select_anchor_indices(0, 4) == []
    assert select_anchor_indices(3, 0) == []
    # Budget covering every usable section takes them all.
    assert select_anchor_indices(3, 4) == [0, 1, 2]
    assert select_anchor_indices(4, 4, damaged={1}) == [0, 2, 3]
    # Nothing usable at all.
    assert select_anchor_indices(3, 2, damaged={0, 1, 2}) == []


# --- seeding -------------------------------------------------------------


def test_seed_estimates_anchors_and_interpolates_the_rest(tmp_path: Path, monkeypatch):
    state, ctx = _ingested(tmp_path)
    worker = _worker()
    monkeypatch.setattr(_SEEDING, worker)

    anchored, interpolated = asyncio.run(
        seed_positions(state, ctx, pos_lo=0.0, pos_hi=13.18)
    )

    assert anchored == 4
    assert interpolated == 5
    assert len(worker.calls) == 4  # one estimate per anchor, no retries
    ordered = state.in_order()
    assert [s.position_source for s in ordered].count("anchor") == 4
    assert all(s.position_mm is not None for s in ordered)
    positions = [s.position_mm for s in ordered]
    assert positions == sorted(positions)
    # Between anchors the fill is exactly the worker's 0.5 mm ladder.
    assert ordered[5].position_mm == pytest.approx(2.5)
    assert ordered[5].position_source == "interpolated"
    # Outside the outermost anchors it steps by the nominal interval.
    assert ordered[0].position_mm == pytest.approx(ordered[1].position_mm - 0.2)


def test_seed_runs_anchor_estimates_one_at_a_time(tmp_path: Path, monkeypatch):
    """Rate-limit friendliness: no anchor starts before the previous finishes."""
    state, ctx = _ingested(tmp_path)
    in_flight = 0
    peak = 0

    async def run(*, image_path: str, **_kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return APResult(position_mm=1.0, reasoning="fake")

    monkeypatch.setattr(_SEEDING, run)
    asyncio.run(seed_positions(state, ctx, pos_lo=0.0, pos_hi=13.18))

    assert peak == 1


def test_seed_clamps_to_the_atlas_range(tmp_path: Path, monkeypatch):
    state, ctx = _ingested(tmp_path, n=5)
    monkeypatch.setattr(_SEEDING, _worker(spacing_mm=10.0))

    asyncio.run(seed_positions(state, ctx, pos_lo=0.0, pos_hi=13.18))

    assert all(0.0 <= s.position_mm <= 13.18 for s in state.slices)


def test_seed_retries_a_failed_anchor_on_its_nearest_neighbour(
    tmp_path: Path, monkeypatch
):
    state, ctx = _ingested(tmp_path)
    planned = select_anchor_indices(9, 4)
    doomed = state.in_order()[planned[0]].id
    worker = _worker(fail_ids=(doomed,))
    monkeypatch.setattr(_SEEDING, worker)

    anchored, _ = asyncio.run(seed_positions(state, ctx, pos_lo=0.0, pos_hi=13.18))

    assert anchored == 4  # the replacement covered for the failure
    assert len(worker.calls) == 5  # four planned plus one retry
    failed = state.by_id(doomed)
    assert failed is not None and failed.position_source == "interpolated"
    assert any("failed" in note for note in state.notes)
    assert any("retrying anchor" in note for note in state.notes)


def test_seed_raises_when_every_anchor_fails(tmp_path: Path, monkeypatch):
    state, ctx = _ingested(tmp_path, n=5)
    names = tuple(s.id for s in state.slices)
    monkeypatch.setattr(_SEEDING, _worker(fail_ids=names))

    with pytest.raises(RuntimeError, match="anchor estimates failed"):
        asyncio.run(seed_positions(state, ctx, pos_lo=0.0, pos_hi=13.18))


def test_seed_positions_damaged_sections_with_a_caveat(tmp_path: Path, monkeypatch):
    state, ctx = _ingested(tmp_path)
    monkeypatch.setattr(_SEEDING, _worker())
    damaged = state.in_order()[3]
    damaged.damaged = True

    asyncio.run(seed_positions(state, ctx, pos_lo=0.0, pos_hi=13.18))

    assert damaged.position_mm is not None
    assert damaged.position_source == "interpolated"
    assert DAMAGED_CAVEAT in damaged.caveats


def test_seed_anchors_a_fully_damaged_stack_anyway(tmp_path: Path, monkeypatch):
    state, ctx = _ingested(tmp_path, n=5)
    monkeypatch.setattr(_SEEDING, _worker())
    for record in state.slices:
        record.damaged = True

    anchored, _ = asyncio.run(seed_positions(state, ctx, pos_lo=0.0, pos_hi=13.18))

    assert anchored > 0
    assert all(s.position_mm is not None for s in state.slices)
    assert any("every section is marked damaged" in note for note in state.notes)


def test_seed_extrapolates_backwards_for_a_reversed_stack(tmp_path: Path, monkeypatch):
    """A stack cut back-to-front seeds decreasing positions, ends included."""
    state, ctx = _ingested(tmp_path)
    monkeypatch.setattr(_SEEDING, _worker(spacing_mm=-0.5))

    asyncio.run(seed_positions(state, ctx, pos_lo=-100.0, pos_hi=100.0))

    positions = [s.position_mm for s in state.in_order()]
    assert positions == sorted(positions, reverse=True)


# --- the seed node -------------------------------------------------------


def test_seed_node_logs_the_deepslice_fallback_and_routes_on(
    tmp_path: Path, monkeypatch
):
    messages: list[str] = []
    state, ctx = _ingested(tmp_path, n=5)
    ctx.emit = messages.append  # type: ignore[attr-defined]
    monkeypatch.setattr(_SEEDING, _worker())

    assert asyncio.run(seed(state, ctx)) == ""

    assert any("deepslice unavailable" in m for m in messages)
    assert all(s.position_mm is not None for s in state.slices)
    assert any("anchor estimate" in note for note in state.notes)


def test_seed_node_is_a_no_op_on_an_empty_stack(tmp_path: Path):
    ctx = build_context(
        BrainConfig(image_folder=str(tmp_path)),
        emit=lambda _m: None,
        atlas_loader=lambda _name: _FakeAtlas(),
    )
    assert asyncio.run(seed(StackState(), ctx)) == ""


# --- the DeepSlice seam --------------------------------------------------


def test_deepslice_is_not_installed_and_not_implemented():
    assert deepslice_available() is False
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        run_deepslice(StackState(), None)  # type: ignore[arg-type]
