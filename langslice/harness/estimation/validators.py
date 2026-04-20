"""before_tool_callback: gate the submit tools on broad/narrow sweep + bracket + monotonicity."""

from __future__ import annotations

from typing import Any

_RELAXATION_AFTER_ATTEMPTS = 2


def _has_neighbor_bracket(
    fetched: list[float], center: float, *, pos_lo: float, pos_hi: float, tol: float = 0.25,
    edge_margin: float = 0.25,
) -> bool:
    needs_lower = center > pos_lo + edge_margin
    needs_upper = center < pos_hi - edge_margin
    has_lower = any(center - tol <= p < center for p in fetched)
    has_upper = any(center < p <= center + tol for p in fetched)
    return (has_lower or not needs_lower) and (has_upper or not needs_upper)


def _gate_single(
    args: dict[str, Any], state: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    """Gate a single-slice submit.

    Returns (error_or_None, counts_toward_submit_attempts). Every gate below
    (broad_sweep, narrow_sweep, neighbor_bracket) is soft/relaxable — the
    agent can fix any of them by making more ``fetch_atlas`` calls — so every
    rejection counts toward ``submit_attempts`` and auto-relaxes after
    ``_RELAXATION_AFTER_ATTEMPTS``. Contrast with ``_gate_group``, which has
    a hard length-mismatch gate that does NOT count (the agent can't fix the
    wrong number of positions by fetching more atlas slices).
    """
    relaxed = state.get("submit_attempts", 0) >= _RELAXATION_AFTER_ATTEMPTS
    if not state.get("saw_broad_sweep") and not relaxed:
        return (
            {"status": "error", "error": "Run a broad `fetch_atlas` sweep before submitting."},
            True,
        )
    if not state.get("saw_narrow_sweep") and not relaxed:
        return (
            {
                "status": "error",
                "error": (
                    "Run a narrow `fetch_atlas` sweep around your best candidate "
                    "before submitting."
                ),
            },
            True,
        )
    pos = float(args.get("position_mm", 0.0))
    if not _has_neighbor_bracket(
        state.get("fetched_positions", []), pos,
        pos_lo=float(state["pos_lo"]), pos_hi=float(state["pos_hi"]),
    ) and not relaxed:
        return (
            {
                "status": "error",
                "error": (
                    f"Verify at least one lower and one higher neighboring atlas "
                    f"position around {pos:.2f} mm before submitting."
                ),
            },
            True,
        )
    return (None, False)


def _gate_group(
    args: dict[str, Any], state: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    """Gate a multi-slice group submit.

    Returns (error_or_None, counts_toward_submit_attempts).

    Order:
      1. length (hard, does NOT count toward submit_attempts - agent cannot
         fix this by calling `fetch_atlas` more)
      2. broad sweep (soft, relaxable, counts)
      3. narrow sweep (soft, relaxable, counts)
      4. monotonicity (soft, relaxable, counts)
      5. interval (soft, relaxable, counts)

    The sweep gates run before monotonicity/interval so the agent is nudged to
    keep exploring context rather than being told "your positions are wrong"
    before it has a chance to verify them.
    """
    positions = list(args.get("positions_mm", []))
    n_expected = int(state["n_slices"])
    if len(positions) != n_expected:
        return (
            {
                "status": "error",
                "error": f"Expected {n_expected} positions, got {len(positions)}.",
            },
            False,
        )

    relaxed = state.get("submit_attempts", 0) >= _RELAXATION_AFTER_ATTEMPTS
    if not state.get("saw_broad_sweep") and not relaxed:
        return (
            {"status": "error", "error": "Run a broad `fetch_atlas` sweep before submitting."},
            True,
        )
    if not state.get("saw_narrow_sweep") and not relaxed:
        return (
            {"status": "error", "error": "Run a narrow `fetch_atlas` sweep before submitting."},
            True,
        )
    if not relaxed and not all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1)):
        return (
            {"status": "error", "error": "Positions must be monotonically increasing."},
            True,
        )

    interval = float(state["interval_mm"])
    intervals = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    tolerance = max(0.5 * interval, 0.25)
    if not relaxed:
        bad = [(i, iv) for i, iv in enumerate(intervals) if abs(iv - interval) > tolerance]
        if bad:
            detail = "; ".join(f"{i + 1}->{i + 2}: {iv:.3f}mm" for i, iv in bad)
            return (
                {
                    "status": "error",
                    "error": (
                        f"Intervals deviate >50% from expected {interval:.3f}mm: {detail}."
                    ),
                },
                True,
            )
    return (None, False)


def gate_submit_tool(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """ADK before_tool_callback: short-circuit submit tools that fail gating.

    Public contract unchanged: returns an error dict (tool short-circuits) or
    None (tool runs normally). Internally dispatches by tool name to the
    appropriate gate, which reports whether the rejection should count toward
    the soft-relaxation budget.
    """
    name = getattr(tool, "name", None)
    if name == "submit_estimate":
        err, should_count = _gate_single(args, tool_context.state)
    elif name == "submit_group_estimate":
        err, should_count = _gate_group(args, tool_context.state)
    else:
        return None  # Pass through all non-submit tools untouched.

    if err is not None and should_count:
        tool_context.state["submit_attempts"] = int(
            tool_context.state.get("submit_attempts", 0)
        ) + 1
    return err
