"""Pre-run cost estimate: table lookups, honest refusals, and the worker method."""
from __future__ import annotations

import io
import json

import pytest

from langslice.agent.cost import estimate
from langslice.hosts.api.service import run_stdio

FULL = {"tasks": ["position", "transform"], "model": "openai-oauth/gpt-5.6-sol",
        "reasoning": "medium"}


def test_a_well_measured_setting_uses_its_band_times_sections():
    result = estimate(FULL, 10)
    assert 0 < result["low"] < result["high"]
    assert result["unit"] == "percent_of_usage_window"
    double = estimate(FULL, 20)
    assert double["low"] > result["low"] and double["high"] > result["high"]


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


def test_frozen_sections_cost_nothing_without_positioning():
    transform_only = {**FULL, "tasks": ["transform"]}
    four = {name: ["orientation", "affine"] for name in "abcd"}
    assert estimate({**transform_only, "frozen": four}, 10)["high"] < estimate(
        transform_only, 10)["high"]
    every = {"*": ["orientation", "affine"]}
    assert estimate({**transform_only, "frozen": every}, 3)["available"] is False
    # Only the orientation frozen leaves the affine to fit: the full price.
    assert estimate({**transform_only, "frozen": {"*": ["orientation"]}}, 3) == estimate(
        transform_only, 3)
    # With positioning on every section is still positioned.
    assert estimate({**FULL, "frozen": four}, 10) == estimate(FULL, 10)


@pytest.mark.parametrize("setting, word", [
    ({"image_resolution": "medium"}, "medium"),
    ({"image_resolution": "auto"}, "auto"),
    ({"tasks": ["nonlinear"], "nonlinear": {"provider": "openai-oauth"}}, "Nonlinear"),
    ({"tasks": []}, ""),
])
def test_unavailable_settings_say_why_instead_of_refusing(setting, word):
    result = estimate({**FULL, **setting}, 38)
    assert result["available"] is False and result["low"] is None and result["high"] is None
    assert word in result["basis"]


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
                        "params": {"spec": FULL, "n_slices": 10}})
    assert message["result"] == estimate(FULL, 10)


def test_a_malformed_setting_is_refused_with_its_reason():
    # The dialog shows the reason (`details.error`).
    message = _request({"id": "e", "method": "linear.estimate", "params": {
        "spec": {**FULL, "reasoning": "galactic"}, "n_slices": 10}})
    assert message["error"]["code"] == "runtime_error"
    assert "galactic" in message["error"]["details"]["error"]
