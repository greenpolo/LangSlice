from unittest.mock import MagicMock

from langslice.harness.estimation.session import build_initial_state
from langslice.harness.estimation.validators import gate_submit_tool


def _make_state(**overrides):
    state = build_initial_state(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, n_slices=1,
        interval_mm=0.0, thickness_um=50, max_iterations=20,
    )
    state.update(overrides)
    return state


def _fake_ctx(state):
    ctx = MagicMock()
    ctx.state = state
    ctx.actions = MagicMock()
    return ctx


class _Tool:
    def __init__(self, name): self.name = name


def test_gate_rejects_submit_without_broad_sweep():
    state = _make_state(saw_broad_sweep=False)
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_estimate"), {"position_mm": 5.0, "reasoning": "x"}, ctx
    )
    assert out is not None and out["status"] == "error"
    assert state["submit_attempts"] == 1


def test_gate_relaxes_after_two_rejections():
    state = _make_state(saw_broad_sweep=False, submit_attempts=2)
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_estimate"), {"position_mm": 5.0, "reasoning": "x"}, ctx
    )
    # After 2+ attempts, relaxation lets the submit through.
    assert out is None


def test_gate_passes_when_all_checks_satisfied():
    state = _make_state(
        saw_broad_sweep=True, saw_narrow_sweep=True,
        fetched_positions=[4.8, 5.2],
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_estimate"), {"position_mm": 5.0, "reasoning": "x"}, ctx
    )
    assert out is None


def test_gate_rejects_group_non_monotonic():
    state = _make_state(
        n_slices=3, interval_mm=0.200,
        saw_broad_sweep=True, saw_narrow_sweep=True,
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_group_estimate"),
        {"positions_mm": [5.0, 4.8, 5.4], "reasoning": "x"},
        ctx,
    )
    assert out is not None and "monotonic" in out["error"].lower()


def test_gate_rejects_group_bad_interval():
    state = _make_state(
        n_slices=3, interval_mm=0.200,
        saw_broad_sweep=True, saw_narrow_sweep=True,
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_group_estimate"),
        {"positions_mm": [4.0, 4.2, 5.5], "reasoning": "x"},  # 1.3mm gap
        ctx,
    )
    assert out is not None
    assert "interval" in out["error"].lower()


def test_gate_ignores_non_submit_tools():
    state = _make_state()
    ctx = _fake_ctx(state)
    out = gate_submit_tool(_Tool("fetch_atlas"), {"positions_mm": [1.0]}, ctx)
    assert out is None
