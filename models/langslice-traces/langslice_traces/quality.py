"""SFT-quality classification for a collected trace."""

from __future__ import annotations

import logging

from .constants import MODEL_POWER_TIER

logger = logging.getLogger(__name__)


def classify_quality(
    *,
    model: str | None,
    err: float,
    tol: float,
    rescue: float | None = None,
    sub_canonical_mm: float,
    events: list[dict],
) -> dict:
    """Three orthogonal quality axes per training example.

    1. model_power     — teacher tier ("pro" > "flash3" > "flash2_5")
    2. accuracy        — "tight" (err <= 0.5*tol) | "in_tolerance" (err <= tol)
    3. struggle_then_success — bool, set when the trace has a winding path
       before landing in tolerance. Heuristic from the SFT spec:
         turn_count > 5 OR max(|fetched_pos - submitted|) > 2 * tol.
       These traces are valuable hard-negative-recovered training signal.
    """
    fetched: list[float] = []
    fetch_calls = 0
    for ev in events:
        if ev.get("role") == "assistant":
            for tc in ev.get("tool_calls") or []:
                if tc.get("name") == "fetch_atlas":
                    fetch_calls += 1
                    for pos in (tc.get("args") or {}).get("positions_mm") or []:
                        try:
                            fetched.append(float(pos))
                        except (TypeError, ValueError) as exc:
                            logger.debug("dropping non-numeric fetched position %r: %s", pos, exc)

    excursion = max(
        (abs(p - sub_canonical_mm) for p in fetched), default=0.0
    )
    # Match the production categorize_trace heuristic: median is 6 fetch
    # calls (a clean sweep is broad -> narrow -> optional finer = ~3 calls).
    # 2× tolerance excursion fires on any broad sweep, so we don't use it
    # alone — only when paired with above-median calls.
    struggle = fetch_calls > 6

    if err <= 0.5 * tol:
        accuracy = "tight"
    elif err <= tol:
        accuracy = "in_tolerance"
    elif rescue is not None and err <= rescue:
        accuracy = "rescued"
    else:
        accuracy = "out_of_tolerance"
    return {
        "model_power": MODEL_POWER_TIER.get(model or "", "unknown"),
        "model": model,
        "accuracy": accuracy,
        "max_error_mm": round(err, 4),
        "tolerance_mm": round(tol, 4),
        "err_over_tol": round(err / tol if tol > 0 else 0.0, 3),
        "struggle_then_success": struggle,
        "fetch_calls": fetch_calls,
        "trajectory_excursion_mm": round(excursion, 3),
    }
