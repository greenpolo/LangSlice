"""Pre-run cost estimate: table lookups, honest refusals, and the worker method."""
from __future__ import annotations

import io
import json

import pytest

from langslice.agent.cost import estimate
from langslice.hosts.api.service import run_stdio

FULL = {"tasks": ["reorder", "position", "transform"], "model": "openai-oauth/gpt-5.6-sol",
        "reasoning": "medium"}


def test_a_well_measured_setting_uses_its_band_times_sections():
    result = estimate(FULL, 10)
    assert (result["low"], result["high"]) == (2.2, 5.5)
    assert result["basis"].startswith("23 measured runs")
    assert result["unit"] == "percent_of_usage_window"


def test_the_high_end_stops_at_the_run_cap():
    result = estimate({**FULL, "max_quota_percent": 10}, 38)
    assert result["high"] == 10.0 and "cap" in result["basis"]
    assert result["low"] <= result["high"]


def test_a_single_run_or_unmeasured_setting_is_widened():
    single = estimate({**FULL, "model": "openai-oauth/gpt-6-luna"}, 38)
    assert "widened" in single["basis"]
    nearest = estimate({**FULL, "reasoning": "high"}, 38)
    assert "nearest measured setting" in nearest["basis"]
    unknown = estimate({**FULL, "model": "openai-oauth/some-future-model"}, 38)
    assert "no runs with this model" in unknown["basis"]


def test_default_reasoning_is_priced_as_medium():
    unset = {key: value for key, value in FULL.items() if key != "reasoning"}
    assert estimate(unset, 38) == estimate(FULL, 38)


def test_locked_sections_cost_nothing_without_positioning():
    transform_only = {**FULL, "tasks": ["transform"]}
    assert estimate(transform_only, 10, locked=4)["high"] < estimate(transform_only, 10)["high"]
    assert estimate(transform_only, 3, locked=3)["high"] == 0.0
    # With positioning on every section is still positioned.
    assert estimate(FULL, 10, locked=4) == estimate(FULL, 10)


@pytest.mark.parametrize("resolution", ["medium", "high", "auto"])
def test_unmeasured_resolutions_say_why_instead_of_refusing(resolution):
    result = estimate({**FULL, "image_resolution": resolution}, 38)
    assert result["available"] is False and result["low"] is None and result["high"] is None
    assert resolution in result["basis"] and "low" in result["basis"]


def test_a_nonlinear_only_run_says_why_it_has_no_estimate():
    result = estimate({**FULL, "tasks": ["nonlinear"],
                       "nonlinear": {"provider": "openai-oauth"}}, 38)
    assert result["available"] is False and "Nonlinear" in result["basis"]
    assert estimate({**FULL, "tasks": []}, 38)["available"] is False


def test_nonlinear_beside_the_priced_tasks_is_estimated_without_it():
    on = estimate({**FULL, "tasks": [*FULL["tasks"], "nonlinear"],
                   "nonlinear": {"provider": "openai-oauth"}}, 10)
    off = estimate(FULL, 10)
    assert on["available"] and (on["low"], on["high"]) == (off["low"], off["high"])
    assert "not included" in on["basis"] and "not included" not in off["basis"]


def test_stack_sizes_far_from_the_measured_ones_are_flagged():
    assert "36-38" in estimate(FULL, 4)["basis"]
    assert "36-38" not in estimate(FULL, 38)["basis"]


def _request(line: dict[str, object]) -> dict[str, object]:
    output = io.StringIO()
    run_stdio(input_stream=io.StringIO(json.dumps(line) + "\n"), output_stream=output)
    [message] = [json.loads(text) for text in output.getvalue().splitlines() if text.strip()]
    return message


def test_the_worker_method_returns_the_estimate():
    message = _request({"id": "e", "method": "linear.estimate",
                        "params": {"spec": FULL, "n_slices": 10, "locked": 0}})
    assert message["result"] == estimate(FULL, 10)


def test_an_unestimable_setting_is_a_result_with_its_reason():
    # Never a bare refusal: the dialog shows the reason (`basis`).
    message = _request({"id": "e", "method": "linear.estimate", "params": {
        "spec": {**FULL, "image_resolution": "medium"}, "n_slices": 10}})
    assert message["result"]["available"] is False
    assert message["result"]["low"] is None and "medium" in message["result"]["basis"]


def test_a_malformed_setting_is_a_runtime_error_not_a_validation_error():
    # The connector treats validation errors as "old worker without this method"
    # and stops asking; a setting it cannot read must stay retryable.
    message = _request({"id": "e", "method": "linear.estimate", "params": {
        "spec": {**FULL, "reasoning": "galactic"}, "n_slices": 10}})
    assert message["error"]["code"] == "runtime_error"
