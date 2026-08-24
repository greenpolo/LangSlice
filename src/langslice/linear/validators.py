"""before_tool_callback: gate the single-slice submit tool on broad/narrow sweeps."""

from __future__ import annotations

from typing import Any

_RELAXATION_AFTER_ATTEMPTS = 2


def _gate_single(
    args: dict[str, Any], state: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    """Gate a single-slice submit.

    Returns (error_or_None, counts_toward_submit_attempts). Both gates
    (broad_sweep, narrow_sweep) are soft/relaxable — the agent can fix either
    by making more ``fetch_atlas`` calls — so every rejection counts toward
    ``submit_attempts`` and auto-relaxes after ``_RELAXATION_AFTER_ATTEMPTS``.
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
    return (None, False)


def gate_submit_tool(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """ADK before_tool_callback: short-circuit a submit that fails gating.

    Public contract unchanged: returns an error dict (tool short-circuits) or
    None (tool runs normally). The gate reports whether the rejection should
    count toward the soft-relaxation budget.
    """
    if getattr(tool, "name", None) != "submit_estimate":
        return None  # Pass through all non-submit tools untouched.

    err, should_count = _gate_single(args, tool_context.state)
    if err is not None and should_count:
        tool_context.state["submit_attempts"] = int(
            tool_context.state.get("submit_attempts", 0)
        ) + 1
    return err
