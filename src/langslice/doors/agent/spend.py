"""What a run has used and spent, told to the operator, never to the model.

LangSlice sets no limit on tokens, time or tool calls, so nothing here stops a
run or reaches the agent: a warning goes to the progress log, the live event
stream (kind ``spend``), and the trace, for the person or agent running
LangSlice. :class:`UsageMeter` is the one meter of a run: it receives every
model call's usage and keeps the tokens, the cost and the usage window.

- **Usage window** (the ChatGPT OAuth lane, whose responses carry the account's
  usage-window reading): one warning each time this run's share passes another
  10 percentage points. The share is the window's rise since the run's first
  reading, in whole percent.
- **Dollars** (the API-key lanes, ``anthropic-api``, ``openai-api``,
  ``gemini-api``): a running estimate from the per-token list prices of the
  provider table (:mod:`langslice.providers.table`), one warning each time it
  passes $5, $10, $20, $40 and so on, doubling. A model with no price in the
  table is said so once.

:meth:`UsageMeter.summary` is the run's final account (tokens by kind, the
window share, the dollar estimate), written to ``exports/spend.json`` by the
engine. The provider modules only supply raw usage and headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from langslice.providers.table import Provider, find

QUOTA_STEP = 10
FIRST_DOLLAR_WARNING = 5.0
CACHED_TOKEN_WEIGHT = 0.13
_OPENAI_LITELLM_PREFIX = "openai/"


@dataclass(frozen=True)
class SpendWarning:
    """One warning for the operator: its text and the numbers it states
    (``used_percent`` for the usage window or ``usd`` for the dollar
    estimate, and the ``threshold`` passed)."""

    text: str
    used_percent: int | None = None
    usd: float | None = None
    threshold: float | None = None

    def record(self) -> dict[str, Any]:
        """The warning as the trace and the live stream carry it."""
        out: dict[str, Any] = {"text": self.text}
        if self.used_percent is not None:
            out["used_percent"] = self.used_percent
        if self.usd is not None:
            out["usd"] = self.usd
        if self.threshold is not None:
            out["threshold"] = self.threshold
        return out


@dataclass(frozen=True)
class Counted:
    """What one model call added to the meter: its one-line report and the
    warnings it triggered."""

    line: str
    warnings: list[SpendWarning]


def billing_of(model: Any) -> tuple[Provider | None, str]:
    """``(provider, model id)`` for the model an agent runs on.

    The provider bills it: a bare model string is native Gemini, a model
    object names its provider in a ``provider`` class attribute (the OpenAI
    OAuth and Anthropic clients, and the OpenAI API client built by
    :func:`langslice.doors.agent.models.resolve_adk_model`). None is a model
    served another way (Gemma, OpenRouter, a local server, a custom endpoint).
    Read from the object ADK holds, never from its repr: a provider
    wrapper's repr can hold a key.
    """
    if isinstance(model, str):
        return find("gemini-api"), model.removeprefix("models/")
    name = getattr(model, "model", None)
    name = name if isinstance(name, str) else type(model).__name__
    declared = getattr(model, "provider", None)
    billed = find(declared) if isinstance(declared, str) else None
    if billed is not None and billed.name == "openai-api":
        name = name.removeprefix(_OPENAI_LITELLM_PREFIX)
    return billed, name


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


class UsageMeter:
    """The tokens and cost of one session, counted call by call."""

    def __init__(self, model: Any, *, today: date | None = None) -> None:
        self.provider, self.model = billing_of(model)
        self.today = today
        self.calls = 0
        self.uncached = 0
        self.cached = 0
        self.output = 0
        self.reasoning = 0
        self.latest_input = 0
        self.peak_input = 0
        self.usd = 0.0
        self.priced = False
        self._unpriced_said = False
        self.quota_start: int | None = None
        self.quota_used: int | None = None
        self.quota_reading: dict[str, Any] | None = None
        self._quota_said = 0
        self._next_usd = FIRST_DOLLAR_WARNING

    @property
    def input(self) -> int:
        """Cumulative input tokens over every call, cached ones included."""
        return self.uncached + self.cached

    @property
    def paid(self) -> int:
        """An input-only proxy: uncached input plus cached input at
        :data:`CACHED_TOKEN_WEIGHT`. It excludes output and is not an
        all-token API-equivalent cost (that is ``usd``)."""
        return int(self.uncached + CACHED_TOKEN_WEIGHT * self.cached)

    def add(self, usage: Any, quota: dict[str, Any] | None = None) -> Counted:
        """Count one model call: its one-line report and the warnings it
        triggers."""
        prompt = _int(getattr(usage, "prompt_token_count", None))
        cached = min(_int(getattr(usage, "cached_content_token_count", None)), prompt)
        candidates = _int(getattr(usage, "candidates_token_count", None))
        thoughts = _int(getattr(usage, "thoughts_token_count", None))
        self.calls += 1
        self.uncached += prompt - cached
        self.cached += cached
        self.output += candidates
        self.reasoning += thoughts
        self.latest_input = prompt
        self.peak_input = max(self.peak_input, prompt)
        warnings: list[SpendWarning] = []
        if quota:
            warnings += self._quota(quota)
        if self.provider is not None and self.provider.prices:
            warnings += self._dollars(prompt, cached, candidates, thoughts)
        elif self.provider is None and not self._unpriced_said:
            self._unpriced_said = True
            warnings.append(
                SpendWarning(
                    f"[spend] no price table for {self.model}: no dollar estimate for this "
                    "run (tokens are still logged)"
                )
            )
        line = (
            f"call {self.calls}: request_input={prompt} (cached {cached}) out={candidates}; "
            f"cumulative_input={self.input} peak_input={self.peak_input} paid~{self.paid}"
        )
        if quota and self.quota_used is not None:
            line += f"; quota {quota}; window used by this run {self.quota_used}%"
        elif quota:
            line += f"; quota {quota}"
        return Counted(line, warnings)

    def _quota(self, quota: dict[str, Any]) -> list[SpendWarning]:
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
            out.append(
                SpendWarning(
                    f"[spend] this run has used {used}% of the usage window (passed {step}%)",
                    used_percent=used,
                    threshold=step,
                )
            )
        return out

    def _dollars(
        self, prompt: int, cached: int, candidates: int, thoughts: int
    ) -> list[SpendWarning]:
        assert self.provider is not None
        price = self.provider.price(self.model)
        if price is None:
            if self._unpriced_said:
                return []
            self._unpriced_said = True
            return [
                SpendWarning(
                    f"[spend] no known price for {self.provider.name}/{self.model}: no dollar "
                    "estimate for this run (tokens are still logged)"
                )
            ]
        rate = price.rate(today=self.today or date.today(), prompt_tokens=prompt)
        self.priced = True
        output = candidates + (thoughts if self.provider.thoughts_separate else 0)
        plain = rate.cache_write if rate.cache_write is not None else rate.input
        self.usd += ((prompt - cached) * plain + cached * rate.cached + output * rate.output) / 1e6
        out = []
        while self.usd >= self._next_usd:
            step = self._next_usd
            self._next_usd *= 2
            out.append(
                SpendWarning(
                    f"[spend] this run's estimated API cost is ${self.usd:.2f} "
                    f"(passed ${step:g}; {self.provider.name}/{self.model}, list prices)",
                    usd=round(self.usd, 4),
                    threshold=step,
                )
            )
        return out

    def tokens(self) -> dict[str, int]:
        """The token counts the trace summary and the live stream carry."""
        return {
            "calls": self.calls,
            "input": self.input,
            "latest_input": self.latest_input,
            "peak_input": self.peak_input,
            "cached": self.cached,
            "output": self.output,
            "reasoning": self.reasoning,
            "paid": self.paid,
        }

    def totals(self) -> str:
        """The run's token total line for the operator's log."""
        return (
            f"{self.calls} calls, cumulative_input={self.input} (cached {self.cached}, "
            f"paid~{self.paid}), out={self.output}, peak_input={self.peak_input}"
        )

    def summary(self) -> dict[str, Any]:
        """The run's final account: tokens by kind, the window share when the
        provider reported one, the dollar estimate when the model is priced."""
        return {
            "lane": self.provider.name if self.provider else None,
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
                "quota_used_percent is the usage window's rise since this run's first reading"
            ),
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
