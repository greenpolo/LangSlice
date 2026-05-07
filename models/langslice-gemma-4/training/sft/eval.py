"""SFT-time evaluation: agent-loop callbacks (baseline + periodic) and metric utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSubmit:
    is_parseable: bool
    position_mm: float | None = None


def parse_submit_call(arguments_json: str, *, expected_kind: str) -> ParsedSubmit:
    """Parse a submit_*_estimate call's arguments string. Returns is_parseable=False on any failure."""
    try:
        args = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return ParsedSubmit(is_parseable=False)
    if expected_kind == "single_slice":
        pos = args.get("position_mm")
        reasoning = args.get("reasoning")
        if not isinstance(pos, (int, float)):
            return ParsedSubmit(is_parseable=False)
        if not isinstance(reasoning, str) or not reasoning.strip():
            return ParsedSubmit(is_parseable=False)
        return ParsedSubmit(is_parseable=True, position_mm=float(pos))
    raise ValueError(f"unknown expected_kind: {expected_kind!r}")


def compute_position_mae_mm(predicted: list[float], truth: list[float]) -> float:
    """Mean absolute error in mm. predicted and truth must be the same length."""
    if len(predicted) != len(truth):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(truth)}")
    if not predicted:
        raise ValueError("empty prediction list")
    errors = [abs(p - t) for p, t in zip(predicted, truth)]
    return sum(errors) / len(errors)


@dataclass
class EvalRun:
    subject_id: str
    predicted_mm: list[float] | None  # None if not parseable / no submit
    truth_mm: list[float]
    parseable: bool
    n_turns: int


def summarize_eval_runs(runs: list[EvalRun]) -> dict[str, float]:
    """Aggregate metrics over a held-out eval run."""
    if not runs:
        raise ValueError("no eval runs to summarize")
    # Collect (predicted, truth) pairs from runs that both parsed and emitted a submit.
    # This filtering also narrows predicted_mm from list[float] | None to list[float]
    # without needing a type cast or ignore.
    parseable_pairs: list[tuple[list[float], list[float]]] = [
        (r.predicted_mm, r.truth_mm)
        for r in runs
        if r.parseable and r.predicted_mm is not None
    ]
    n_submits = sum(1 for r in runs if r.predicted_mm is not None)
    if parseable_pairs:
        per_run_maes = [compute_position_mae_mm(pred, truth) for pred, truth in parseable_pairs]
        position_mae = sum(per_run_maes) / len(per_run_maes)
    else:
        position_mae = float("nan")
    return {
        "position_mae_mm": position_mae,
        "tool_call_parseability_rate": len(parseable_pairs) / len(runs),
        "no_submit_rate": 1.0 - (n_submits / len(runs)),
        "mean_trace_length": sum(r.n_turns for r in runs) / len(runs),
    }
