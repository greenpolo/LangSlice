"""What a run has spent, told to the operator, never to the model.

LangSlice sets no limit on tokens, time or tool calls, so nothing here stops a
run or reaches the agent: a warning goes to the progress log, the live event
stream (kind ``spend``), and the trace, for the person or agent running
LangSlice. Two meters:

- **Usage window** (the ChatGPT OAuth lane, whose responses carry the account's
  usage-window reading): one warning each time this run's share passes another
  10 percentage points. The share is the window's rise since the run's first
  reading, in whole percent.
- **Dollars** (the API-key lanes, ``anthropic-api``, ``openai-api``,
  ``gemini-api``): a running estimate from per-token list prices
  (:data:`PRICES`), one warning each time it passes $5, $10, $20, $40 and so
  on, doubling. A model with no price in the table is said so once.

:meth:`SpendMonitor.summary` is the run's final account (tokens by kind, the
window share, the dollar estimate), written to ``exports/spend.json`` by the
engine. The provider modules only supply raw usage and headers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

#: Window-share warnings come every this many percentage points.
QUOTA_STEP = 10
#: The first dollar warning; each later one doubles it.
FIRST_DOLLAR_WARNING = 5.0


@dataclass(frozen=True)
class Rate:
    """USD per million tokens. ``cache_write`` is what an input token that is
    not a cache read costs when the provider bills it as a cache write
    (None: it is plain input)."""

    input: float
    cached: float
    output: float
    cache_write: float | None = None


def _r(input: float, cached: float, output: float, write: float | None = None) -> Rate:
    return Rate(input, cached, output, write)


#: Per-token list prices, USD per million tokens, standard tier, short context.
#: Sources, read 2026-10-15:
#: - anthropic-api: https://platform.claude.com/docs/en/about-claude/pricing
#:   (cache write = the 5-minute rate; LangSlice marks the newest message block
#:   on every call, providers/anthropic_api.py, so the input that is not a
#:   cache read is billed as a cache write).
#: - openai-api: https://developers.openai.com/api/docs/pricing (reasoning
#:   tokens are inside the output count, as OpenAI bills them).
#: - gemini-api: https://ai.google.dev/gemini-api/docs/pricing (output includes
#:   thinking tokens; prompts over 200K tokens on 3.1 Pro and 2.5 Pro, and
#:   over 272K / 100K tokens where noted below, bill higher than listed here).
#: Entries with several rows are dated: the latest row whose date has come.
#: Not priced: batch, flex, fast or regional tiers, the Claude Sonnet 5.5
#: and Opus 5.5 long-context bands (none listed), Gemini audio input.
_ALWAYS = date.min
PRICES: dict[str, dict[str, list[tuple[date, Rate]]]] = {
    "anthropic-api": {
        "claude-fable-5-1": [(_ALWAYS, _r(10, 0.25, 50, 12.5))],
        "claude-fable-5": [(_ALWAYS, _r(10, 1, 50, 12.5))],
        "claude-opus-5-5": [(_ALWAYS, _r(4, 0.20, 20, 5))],
        "claude-opus-5": [(_ALWAYS, _r(5, 0.50, 25, 6.25))],
        "claude-opus-4-8": [(_ALWAYS, _r(5, 0.50, 25, 6.25))],
        "claude-opus-4-7": [(_ALWAYS, _r(5, 0.50, 25, 6.25))],
        "claude-opus-4-6": [(_ALWAYS, _r(5, 0.50, 25, 6.25))],
        "claude-sonnet-5-5": [(_ALWAYS, _r(2, 0.10, 10, 2.5))],
        "claude-sonnet-5": [(_ALWAYS, _r(2, 0.20, 10, 2.5))],
        "claude-sonnet-4-6": [(_ALWAYS, _r(3, 0.30, 15, 3.75))],
        "claude-haiku-5-5": [(_ALWAYS, _r(0.10, 0.01, 0.50, 0.125))],
        "claude-haiku-4-5": [(_ALWAYS, _r(1, 0.10, 5, 1.25))],
    },
    "openai-api": {
        "gpt-6-astra": [(_ALWAYS, _r(10, 1, 50))],
        "gpt-6.1-sol": [(_ALWAYS, _r(2, 0.10, 10))],
        "gpt-6-sol": [(_ALWAYS, _r(2, 0.20, 10))],
        "gpt-6-luna": [(_ALWAYS, _r(0.10, 0.01, 0.50))],
        "gpt-5.6-sol": [(_ALWAYS, _r(4, 0.40, 20))],
        "gpt-5.6-terra": [(_ALWAYS, _r(2, 0.20, 12))],
        "gpt-5.6-luna": [(_ALWAYS, _r(0.20, 0.02, 1.20))],
        "gpt-5.5": [(_ALWAYS, _r(5, 0.50, 30))],
        "gpt-5.4": [(_ALWAYS, _r(2.5, 0.25, 15))],
        "gpt-5.4-mini": [(_ALWAYS, _r(0.75, 0.075, 4.5))],
        "gpt-5.2": [(_ALWAYS, _r(1.75, 0.175, 14))],
        "gpt-5.1": [(_ALWAYS, _r(1.25, 0.125, 10))],
        "gpt-5": [(_ALWAYS, _r(1.25, 0.125, 10))],
    },
    "gemini-api": {
        # Google announced these increases on the pricing page.
        "gemini-4-argon": [(_ALWAYS, _r(2, 0.10, 10)), (date(2026, 12, 1), _r(4, 0.20, 20))],
        "gemini-3.8-flash": [(_ALWAYS, _r(0.75, 0.075, 3.75)),
                             (date(2027, 1, 1), _r(1.5, 0.15, 7.5))],
        "gemini-3.6-flash": [(_ALWAYS, _r(0.75, 0.075, 3.75)),
                             (date(2027, 1, 1), _r(1.5, 0.15, 7.5))],
        "gemini-3.5-flash-lite": [(_ALWAYS, _r(0.30, 0.03, 2.5))],
        "gemini-3.1-flash-lite": [(_ALWAYS, _r(0.25, 0.025, 1.5))],
        "gemini-3.1-pro-preview": [(_ALWAYS, _r(2, 0.20, 12))],
        "gemini-3-flash-preview": [(_ALWAYS, _r(0.50, 0.05, 3))],
        "gemini-2.5-pro": [(_ALWAYS, _r(1.25, 0.125, 10))],
        "gemini-2.5-flash": [(_ALWAYS, _r(0.30, 0.03, 2.5))],
        "gemini-2.5-flash-lite": [(_ALWAYS, _r(0.10, 0.01, 0.4))],
    },
}
#: Haiku 5.5 bills a whole prompt over 100,000 tokens at a higher band.
_HAIKU_LONG = (100_000, _r(0.50, 0.05, 2.5, 0.625))
#: Lanes whose usage metadata keeps thinking tokens apart from the output count
#: (ADK reports Gemini and Claude thoughts separately; both bill them as output).
_THOUGHTS_SEPARATE = {"anthropic-api", "gemini-api"}


def lane_of(model: Any) -> tuple[str | None, str]:
    """``(lane, model name)`` for the model object an agent runs on.

    The lane is the canonical provider that bills it (``openai-oauth``,
    ``anthropic-api``, ``openai-api``, ``gemini-api``), or None for a model
    served another way (Gemma, OpenRouter, a local server, a custom
    endpoint). Read from the object ADK holds, never from its repr: a
    provider wrapper's repr can hold a key.
    """
    if isinstance(model, str):
        return "gemini-api", model.removeprefix("models/")
    name = getattr(model, "model", None)
    name = name if isinstance(name, str) else type(model).__name__
    kind = type(model).__name__
    if kind == "OpenAIOAuthLlm":
        return "openai-oauth", name
    if kind == "AnthropicApiLlm":
        return "anthropic-api", name
    if kind == "LiteLlm":
        extra = getattr(model, "_additional_args", None) or {}
        if name.startswith("openai/") and not extra.get("api_base"):
            return "openai-api", name.removeprefix("openai/")
        return None, name
    return None, name


def rate_for(lane: str, model: str, *, today: date | None = None,
             prompt_tokens: int = 0) -> Rate | None:
    """The price of *model* on *lane*, or None when the table has none."""
    rows = PRICES.get(lane, {}).get(model)
    if not rows:
        return None
    if model == "claude-haiku-5-5" and prompt_tokens > _HAIKU_LONG[0]:
        return _HAIKU_LONG[1]
    now = today or date.today()
    current = [rate for since, rate in sorted(rows, key=lambda row: row[0]) if since <= now]
    return current[-1] if current else rows[0][1]


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


class SpendMonitor:
    """Tallies the spend of one session and returns the operator's warnings."""

    def __init__(self, model: Any, *, today: date | None = None) -> None:
        self.lane, self.model = lane_of(model)
        self.today = today
        self.uncached = 0
        self.cached = 0
        self.output = 0
        self.reasoning = 0
        self.usd = 0.0
        self.priced = False
        self._unpriced_said = False
        self.quota_start: int | None = None
        self.quota_used: int | None = None
        self.quota_reading: dict[str, Any] | None = None
        self._quota_said = 0
        self._next_usd = FIRST_DOLLAR_WARNING

    def add(self, usage: Any, quota: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Count one model call; the warnings it triggers, each a dict with
        ``text`` and the numbers it states (``used_percent`` or ``usd``,
        ``threshold``)."""
        prompt = _int(getattr(usage, "prompt_token_count", None))
        cached = min(_int(getattr(usage, "cached_content_token_count", None)), prompt)
        candidates = _int(getattr(usage, "candidates_token_count", None))
        thoughts = _int(getattr(usage, "thoughts_token_count", None))
        self.uncached += prompt - cached
        self.cached += cached
        self.output += candidates
        self.reasoning += thoughts
        warnings: list[dict[str, Any]] = []
        if quota:
            warnings += self._quota(quota)
        if self.lane in PRICES:
            warnings += self._dollars(prompt, cached, candidates, thoughts)
        elif self.lane is None and not self._unpriced_said:
            self._unpriced_said = True
            warnings.append({"text": (
                f"[spend] no price table for {self.model}: no dollar estimate for this "
                "run (tokens are still logged)")})
        return warnings

    def _quota(self, quota: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            percent = int(str(quota.get("primary_used_percent")))
        except ValueError:
            return []
        self.quota_reading = dict(quota)
        if self.quota_start is None:
            self.quota_start = percent
        self.quota_used = used = percent - self.quota_start
        out = []
        while (self._quota_said + 1) * QUOTA_STEP <= used:
            self._quota_said += 1
            step = self._quota_said * QUOTA_STEP
            out.append({"text": (
                f"[spend] this run has used {used}% of the usage window "
                f"(passed {step}%)"), "used_percent": used, "threshold": step})
        return out

    def _dollars(self, prompt: int, cached: int, candidates: int,
                 thoughts: int) -> list[dict[str, Any]]:
        assert self.lane is not None
        rate = rate_for(self.lane, self.model, today=self.today, prompt_tokens=prompt)
        if rate is None:
            if self._unpriced_said:
                return []
            self._unpriced_said = True
            return [{"text": (
                f"[spend] no known price for {self.lane}/{self.model}: no dollar estimate "
                "for this run (tokens are still logged)")}]
        self.priced = True
        output = candidates + (thoughts if self.lane in _THOUGHTS_SEPARATE else 0)
        plain = rate.cache_write if rate.cache_write is not None else rate.input
        self.usd += ((prompt - cached) * plain + cached * rate.cached
                     + output * rate.output) / 1e6
        out = []
        while self.usd >= self._next_usd:
            step = self._next_usd
            self._next_usd *= 2
            out.append({"text": (
                f"[spend] this run's estimated API cost is ${self.usd:.2f} "
                f"(passed ${step:g}; {self.lane}/{self.model}, list prices)"),
                "usd": round(self.usd, 4), "threshold": step})
        return out

    def summary(self) -> dict[str, Any]:
        """The run's final account: tokens by kind, the window share when the
        provider reported one, the dollar estimate when the model is priced."""
        return {
            "lane": self.lane,
            "model": self.model,
            "tokens": {
                "input_uncached": self.uncached,
                "input_cached": self.cached,
                "output": self.output,
                "reasoning": self.reasoning,
            },
            "quota_used_percent": self.quota_used,
            "quota_reading": self.quota_reading,
            "usd_estimate": round(self.usd, 4) if self.priced else None,
            "note": (
                "usd_estimate is list prices times the reported tokens (output = "
                "candidates, plus thinking tokens where the lane reports them apart); "
                "quota_used_percent is the usage window's rise since this run's first reading"),
        }

    def line(self) -> str:
        """The run-total line for the operator's log."""
        parts = []
        if self.quota_used is not None:
            parts.append(f"{self.quota_used}% of the usage window")
        if self.priced:
            parts.append(f"~${self.usd:.2f} at list prices")
        if not parts:
            parts.append("no cost meter for this model")
        return "[spend] run total: " + ", ".join(parts)
