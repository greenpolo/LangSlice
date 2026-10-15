"""Operator spend warnings: the dollar meter and the price table (agent/spend.py).

The window-share warnings run through a whole fake session in
tests/test_linear_engine.py.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from langslice.agent.spend import PRICES, SpendMonitor, lane_of, rate_for


def usage(prompt=0, cached=0, out=0, thoughts=0):
    return SimpleNamespace(prompt_token_count=prompt, cached_content_token_count=cached,
                           candidates_token_count=out, thoughts_token_count=thoughts)


class AnthropicApiLlm:
    model = "claude-opus-5-5"


def test_the_lane_comes_from_the_model_object():
    assert lane_of("gemini-3.1-pro-preview") == ("gemini-api", "gemini-3.1-pro-preview")
    assert lane_of(AnthropicApiLlm()) == ("anthropic-api", "claude-opus-5-5")
    assert lane_of(SimpleNamespace())[0] is None


def test_a_claude_run_is_estimated_with_cache_writes_and_warned_at_five_dollars():
    monitor = SpendMonitor(AnthropicApiLlm())
    # 200K prompt, 150K read from the cache, 50K written, 4K output + 6K thinking:
    # 50K * $5 + 150K * $0.20 + 10K * $20 per million = $0.48 a call.
    said = []
    for _ in range(10):
        said += monitor.add(usage(prompt=200_000, cached=150_000, out=4_000, thoughts=6_000))
    assert said == [] and round(monitor.usd, 2) == 4.80
    said = monitor.add(usage(prompt=200_000, cached=150_000, out=4_000, thoughts=6_000))
    assert [w["text"] for w in said] == [
        "[spend] this run's estimated API cost is $5.28 "
        "(passed $5; anthropic-api/claude-opus-5-5, list prices)"]


def test_dollar_warnings_double():
    monitor = SpendMonitor("gemini-2.5-pro")
    said = monitor.add(usage(prompt=0, out=1_000_000))  # $10 of output: crosses 5 and 10
    assert [w["threshold"] for w in said] == [5.0, 10.0]
    assert [w["threshold"] for w in monitor.add(usage(out=1_000_000))] == [20.0]
    assert monitor.add(usage(out=1)) == []


def test_openai_reasoning_is_already_inside_the_output_count_and_gemini_thinking_is_not():
    openai = SpendMonitor(SimpleNamespace(model="openai/gpt-5.1"))
    openai.lane, openai.model = "openai-api", "gpt-5.1"
    openai.add(usage(out=1_000_000, thoughts=900_000))
    gemini = SpendMonitor("gemini-2.5-pro")
    gemini.add(usage(out=100_000, thoughts=900_000))
    assert round(openai.usd, 2) == 10.0 and round(gemini.usd, 2) == 10.0


def test_an_unpriced_model_is_said_once_and_guessed_never():
    monitor = SpendMonitor("gemini-9-imaginary")
    first = monitor.add(usage(out=10))
    assert len(first) == 1
    assert "no known price for gemini-api/gemini-9-imaginary" in first[0]["text"]
    assert monitor.add(usage(out=10)) == []
    assert monitor.summary()["usd_estimate"] is None
    assert monitor.line() == "[spend] run total: no cost meter for this model"
    local = SpendMonitor(SimpleNamespace(model="ollama_chat/x"))
    assert "no price table" in local.add(usage(out=1))[0]["text"]


def test_dated_prices_follow_the_calendar():
    assert rate_for("gemini-api", "gemini-4-argon", today=date(2026, 11, 30)).input == 2
    assert rate_for("gemini-api", "gemini-4-argon", today=date(2026, 12, 1)).input == 4


def test_every_row_is_complete():
    for lane, rows in PRICES.items():
        for model, dated in rows.items():
            assert dated and all(r.input > 0 and r.output > 0 and r.cached >= 0
                                 for _, r in dated), (lane, model)
