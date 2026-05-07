"""Tests for models/langslice-gemma-4/training/sft/eval.py."""

from __future__ import annotations

import sys
import types

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


def test_parse_submit_call_rejects_bool_position():
    parsed = parse_submit_call(
        '{"position_mm": true, "reasoning": "ok"}', expected_kind="single_slice"
    )
    assert parsed.is_parseable is False


def test_parse_submit_call_rejects_bool_position_false():
    parsed = parse_submit_call(
        '{"position_mm": false, "reasoning": "ok"}', expected_kind="single_slice"
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


def test_summarize_eval_runs_all_no_submit():
    runs = [
        EvalRun(subject_id="M01", predicted_mm=None, truth_mm=[5.0], parseable=False, n_turns=8),
        EvalRun(subject_id="M02", predicted_mm=None, truth_mm=[7.0], parseable=False, n_turns=10),
    ]
    summary = summarize_eval_runs(runs)
    import math
    assert math.isnan(summary["position_mae_mm"])
    assert summary["tool_call_parseability_rate"] == 0.0
    assert summary["no_submit_rate"] == 1.0
    assert summary["mean_trace_length"] == pytest.approx(9.0)


def test_agent_loop_eval_callback_signature_compiles():
    """Smoke check that the callback class is importable + has the expected attrs."""
    from sft.eval import AgentLoopEvalCallback, BaselineEvalCallback
    assert hasattr(AgentLoopEvalCallback, "on_step_end")
    assert hasattr(BaselineEvalCallback, "on_train_begin")


def test_run_agent_loop_for_one_uses_rlvr_env_single_slice(monkeypatch, tmp_path):
    """Stub model/processor path verifies env.reset/fetch wiring without loading Gemma."""
    from PIL import Image
    from sft import eval as eval_mod

    image_path = tmp_path / "query.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(image_path)

    class StubMeta:
        pos_lo = 0.0
        pos_hi = 13.2
        species = "mouse"

    class StubCache:
        def get(self, atlas_name, plane):  # noqa: ANN001
            return StubMeta()

    class StubEnv:
        reset_kwargs = None

        def __init__(self, atlas_grid):  # noqa: ANN001
            self._state = type("State", (), {"turns": 0})()

        def reset(self, **kwargs):  # noqa: ANN003
            StubEnv.reset_kwargs = kwargs

        def fetch_atlas(self, positions_mm):  # noqa: ANN001
            self._state.turns += 1
            return {"content": [{"type": "text", "text": "Atlas at 5.00 mm."}]}

    fetch_args = '{"positions_mm":[5.0]}'
    submit_args = '{"position_mm":5.2,"reasoning":"matched"}'
    calls = iter([
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "fetch_atlas", "arguments": fetch_args},
        },
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "submit_estimate", "arguments": submit_args},
        },
    ])
    monkeypatch.setattr(eval_mod, "_extract_tool_call_from_decoded", lambda text: next(calls, None))
    rlvr_pkg = types.ModuleType("rlvr")
    rlvr_env_mod = types.ModuleType("rlvr.env")
    rlvr_env_mod.LangSliceEstimateEnv = StubEnv
    monkeypatch.setitem(sys.modules, "rlvr", rlvr_pkg)
    monkeypatch.setitem(sys.modules, "rlvr.env", rlvr_env_mod)

    class StubProcessor:
        def apply_chat_template(self, *args, **kwargs):  # noqa: ANN002, ANN003
            class Batch(dict):
                def to(self, device):  # noqa: ANN001
                    return self
            return Batch()
        def decode(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return "<tool_call>{}</tool_call>"

    class StubModel:
        device = "cpu"
        def generate(self, **kwargs):  # noqa: ANN003
            return [[0]]

    run = eval_mod._run_agent_loop_for_one(
        model=StubModel(),
        processor=StubProcessor(),
        eval_row={
            "subject_id": "M01",
            "image_path": image_path,
            "atlas_name": "allen_mouse_25um",
            "plane": "coronal",
            "ground_truth_position_mm": 5.0,
        },
        atlas_grid=object(),
        atlas_meta_cache=StubCache(),
    )
    assert run.predicted_mm == [5.2]
    assert run.parseable is True
    assert StubEnv.reset_kwargs["kind"] == "single"
    assert StubEnv.reset_kwargs["valid_range_mm"] == (0.0, 13.2)
