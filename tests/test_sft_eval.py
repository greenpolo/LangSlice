"""Tests for models/langslice-gemma-4/training/sft/eval.py."""

from __future__ import annotations

import pytest

from sft.eval import (
    EvalRun,
    compute_position_mae_mm,
    parse_submit_call,
    summarize_eval_runs,
)


def test_parse_submit_call_extracts_position_from_valid_json():
    raw = '{"position_mm": 5.2, "reasoning": "looks like AC level"}'
    parsed = parse_submit_call(raw, expected_kind="single_slice")
    assert parsed.position_mm == pytest.approx(5.2)
    assert parsed.is_parseable is True


def test_parse_submit_call_handles_malformed_json():
    parsed = parse_submit_call("{not json}", expected_kind="single_slice")
    assert parsed.is_parseable is False
    assert parsed.position_mm is None


def test_parse_submit_call_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown expected_kind"):
        parse_submit_call('{"position_mm": 5.0, "reasoning": "x"}', expected_kind="group")


def test_parse_submit_call_rejects_non_numeric_position():
    parsed = parse_submit_call(
        '{"position_mm": "x", "reasoning": "ok"}', expected_kind="single_slice"
    )
    assert parsed.is_parseable is False


def test_parse_submit_call_rejects_empty_reasoning():
    parsed = parse_submit_call(
        '{"position_mm": 5.0, "reasoning": "  "}', expected_kind="single_slice"
    )
    assert parsed.is_parseable is False


def test_compute_position_mae_mm_simple():
    pred = [1.0, 2.0, 3.0]
    truth = [1.5, 2.0, 4.0]
    mae = compute_position_mae_mm(pred, truth)
    # Mean of |0.5|, |0|, |1.0| = 0.5
    assert mae == pytest.approx(0.5)


def test_compute_position_mae_mm_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_position_mae_mm([1.0, 2.0], [1.0])


def test_compute_position_mae_mm_rejects_empty():
    with pytest.raises(ValueError, match="empty prediction list"):
        compute_position_mae_mm([], [])


def test_summarize_eval_runs():
    runs = [
        EvalRun(subject_id="M01", predicted_mm=[5.0], truth_mm=[5.5], parseable=True, n_turns=4),
        EvalRun(subject_id="M02", predicted_mm=[7.2], truth_mm=[7.0], parseable=True, n_turns=6),
        EvalRun(subject_id="M03", predicted_mm=None, truth_mm=[3.0], parseable=False, n_turns=12),
    ]
    summary = summarize_eval_runs(runs)
    # Only the 2 parseable runs contribute to MAE
    assert summary["position_mae_mm"] == pytest.approx((0.5 + 0.2) / 2)
    assert summary["tool_call_parseability_rate"] == pytest.approx(2 / 3)
    assert summary["no_submit_rate"] == pytest.approx(1 / 3)
    assert summary["mean_trace_length"] == pytest.approx((4 + 6 + 12) / 3)


def test_summarize_eval_runs_rejects_empty():
    with pytest.raises(ValueError, match="no eval runs"):
        summarize_eval_runs([])
