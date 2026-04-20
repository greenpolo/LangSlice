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


# --- Phase 4 Task 4.1: gate ordering + submit_attempts fixes ---


def test_gate_group_broad_sweep_error_precedes_monotonicity():
    """When broad sweep hasn't happened AND positions are non-monotonic,
    the agent should see the broad-sweep nudge first (keep exploring) rather
    than a monotonicity complaint about positions it hasn't yet verified.
    """
    state = _make_state(
        n_slices=3, interval_mm=0.200,
        saw_broad_sweep=False, saw_narrow_sweep=True,
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_group_estimate"),
        {"positions_mm": [5.0, 4.8, 5.4], "reasoning": "x"},  # non-monotonic
        ctx,
    )
    assert out is not None
    assert "broad" in out["error"].lower()
    assert "monotonic" not in out["error"].lower()


def test_gate_group_narrow_sweep_error_precedes_interval():
    """Narrow sweep nudge should precede interval complaint for the same reason."""
    state = _make_state(
        n_slices=3, interval_mm=0.200,
        saw_broad_sweep=True, saw_narrow_sweep=False,
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_group_estimate"),
        {"positions_mm": [4.0, 4.2, 5.5], "reasoning": "x"},  # 1.3mm gap
        ctx,
    )
    assert out is not None
    assert "narrow" in out["error"].lower()
    assert "interval" not in out["error"].lower()


def test_gate_group_length_mismatch_does_not_count_toward_submit_attempts():
    """Length mismatch is a HARD rejection - the agent cannot fix it by
    exploring more, so it should not burn the relaxation budget.
    """
    state = _make_state(
        n_slices=4, interval_mm=0.200,
        saw_broad_sweep=True, saw_narrow_sweep=True,
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_group_estimate"),
        {"positions_mm": [5.0, 5.2, 5.4, 5.6, 5.8], "reasoning": "x"},  # 5 != 4
        ctx,
    )
    assert out is not None
    assert "expected" in out["error"].lower() and "got" in out["error"].lower()
    assert state["submit_attempts"] == 0


def test_gate_group_soft_rejection_counts_toward_submit_attempts():
    """A relaxable rejection (e.g. missing broad sweep) DOES count toward
    the retry budget so the agent can eventually submit after enough nudges.
    """
    state = _make_state(
        n_slices=3, interval_mm=0.200,
        saw_broad_sweep=False, saw_narrow_sweep=True,
    )
    ctx = _fake_ctx(state)
    out = gate_submit_tool(
        _Tool("submit_group_estimate"),
        {"positions_mm": [5.0, 5.2, 5.4], "reasoning": "x"},
        ctx,
    )
    assert out is not None
    assert state["submit_attempts"] == 1
