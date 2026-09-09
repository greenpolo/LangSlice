"""The ADK session: one agent, its tools, and the loop that drives it.

Generic on purpose — the main stack session and the per-section alignment
sub-session are the same loop with different tools, prompts and turn budgets.
Tools mutate the stack state as they are called, so a pass that never submits
still leaves its writes behind.

With ``LANGSLICE_TRACE_DIR`` set, everything the agent is shown, says, calls
and gets back is appended to a JSONL trace (:mod:`langslice.linear.trace`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

from langslice.adk.model_resolver import (
    _env,
    _env_float,
    default_http_options,
    resolve_adk_model,
)
from langslice.adk.plugins import (
    ModelCallPacingPlugin,
    RequestCapturePlugin,
    WorkingSetImages,
)
from langslice.linear.trace import open_trace

logger = logging.getLogger(__name__)

_APP_NAME = "langslice"
_USER_ID = "langslice-user"

#: Model turns one main stack session may spend.
DEFAULT_MAX_ITERATIONS = 60


def build_plugins(run_label: str) -> list[BasePlugin]:
    """The ADK plugins every LangSlice session runs with."""
    plugins: list[BasePlugin] = [
        # One working set per session: the instance remembers its cut.
        ContextFilterPlugin(custom_filter=WorkingSetImages())
    ]
    model_call_delay_s = _env_float("LANGSLICE_ADK_MODEL_CALL_DELAY_S")
    if model_call_delay_s is not None and model_call_delay_s > 0:
        plugins.append(ModelCallPacingPlugin(model_call_delay_s))
    capture_dir = _env("LANGSLICE_ADK_CAPTURE_REQUESTS_DIR")
    if capture_dir is not None:
        plugins.append(RequestCapturePlugin(capture_dir, run_label=run_label))
    return plugins


def _with_reasoning(model: str | object, reasoning: str | None) -> str | object:
    """Set a resolved model's reasoning effort, when it has one to set.

    Providers that expose the knob carry a ``reasoning_effort`` field (the
    OAuth ``ChatGptLlm`` is a pydantic model, hence ``model_copy``); every
    other backend is left exactly as it was.
    """
    if not reasoning or not hasattr(model, "reasoning_effort"):
        return model
    copier = getattr(model, "model_copy", None)
    if callable(copier):  # pydantic BaseLlm, e.g. the OAuth ChatGptLlm
        return copier(update={"reasoning_effort": reasoning})
    object.__setattr__(model, "reasoning_effort", reasoning)
    return model


def build_agent(
    *,
    model: str | object,
    name: str,
    instruction: str,
    tools: list[Any],
    max_output_tokens: int = 8000,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
    reasoning: str | None = None,
) -> LlmAgent:
    """Construct an :class:`LlmAgent` with the harness's standard config."""
    # kwargs dict so the enum-typed media_resolution string is accepted as-is.
    config_kwargs: dict[str, Any] = {
        "temperature": 1.0,
        "max_output_tokens": max_output_tokens,
        "media_resolution": media_resolution,
        "http_options": default_http_options(),
    }
    return LlmAgent(
        model=_with_reasoning(resolve_adk_model(model), reasoning),  # type: ignore[arg-type]
        name=name,
        instruction=instruction,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
    )


#: What a cached input token costs relative to an uncached one. Measured on
#: the OAuth lane's quota headers (run 5, 2026-09-09: 96%/M uncached,
#: 13%/M cached, fit error half a point); OpenAI's published API rate is 0.1x.
CACHED_TOKEN_WEIGHT = 0.13


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class TokenTally:
    """Summed token usage over one session, one line per model call.

    ``paid`` is the uncached-equivalent count: uncached input plus cached
    input at :data:`CACHED_TOKEN_WEIGHT`. That, not ``input``, is what a
    run costs.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input = 0
        self.cached = 0
        self.output = 0

    @property
    def paid(self) -> int:
        return int(self.input - self.cached + CACHED_TOKEN_WEIGHT * self.cached)

    def add(self, usage: Any) -> str:
        """Add one call's usage; return the one-line report for it."""
        prompt = int(getattr(usage, "prompt_token_count", None) or 0)
        cached = int(getattr(usage, "cached_content_token_count", None) or 0)
        output = int(getattr(usage, "candidates_token_count", None) or 0)
        self.calls += 1
        self.input += prompt
        self.cached += cached
        self.output += output
        return (
            f"call {self.calls}: in={prompt} (cached {cached}) out={output}; "
            f"run in={self.input} paid~{self.paid}"
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input": self.input,
            "cached": self.cached,
            "output": self.output,
            "paid": self.paid,
        }

    def totals(self) -> str:
        return (
            f"{self.calls} calls, in={self.input} (cached {self.cached}, "
            f"paid~{self.paid}), out={self.output}"
        )


async def run_agent_session(
    *,
    agent: LlmAgent,
    seed_message: types.Content,
    done: Callable[[], bool],
    nudge_no_tool: str,
    nudge_continue: str,
    max_iterations: int,
    run_label: str,
    debrief: str | None = None,
    debrief_sink: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
    max_input_tokens: int | None = None,
    max_quota_percent: int | None = None,
) -> tuple[int, int]:
    """Drive one agent pass; return ``(tool_calls, turns)``.

    Ends when *done* reports the session's submit tool has fired, or when the
    turn/tool-call budget runs out, or when the run's summed input tokens
    pass *max_input_tokens* — the history is resent on every call, so input
    grows with the square of the call count and the account, not the job,
    would otherwise end the run.
    """
    trace = open_trace(run_label, agent=agent)
    app = App(name=_APP_NAME, root_agent=agent, plugins=build_plugins(run_label))
    runner = InMemoryRunner(app=app)
    assert runner.session_service is not None
    await runner.session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID, session_id=run_label, state={}
    )

    message = seed_message
    if trace is not None:
        trace.seed(seed_message)
    # The nudge is traced where it is sent, not where it is written: the last
    # one built before the budget runs out never reaches the model.
    nudge: str | None = None
    tool_calls = 0
    turns = 0
    tokens = TokenTally()
    quota_start: int | None = None
    stopped: str | None = None
    while turns < max_iterations and not done() and stopped is None:
        turns += 1
        if trace is not None and nudge is not None:
            trace.nudge(nudge, turn=turns)
        saw_tool_call = False
        async for event in runner.run_async(
            user_id=_USER_ID, session_id=run_label, new_message=message
        ):
            if trace is not None:
                trace.event(event, turn=turns)
            if getattr(event, "error_message", None):
                # A model/provider error is a stop, not a turn to nudge past:
                # looping would re-pay the same failure until the budget ran out.
                raise RuntimeError(
                    f"{run_label}: model error on turn {turns} "
                    f"({getattr(event, 'error_code', None)}): {event.error_message}"
                )
            usage = getattr(event, "usage_metadata", None)
            if usage is not None and not getattr(event, "partial", False):
                line = tokens.add(usage)
                quota = (getattr(event, "custom_metadata", None) or {}).get("quota")
                used: int | None = None
                if quota:
                    line += f"; quota {quota}"
                    percent = _int_or_none(quota.get("primary_used_percent"))
                    if percent is not None:
                        if quota_start is None:
                            quota_start = percent
                        used = percent - quota_start
                        line += f"; window used by this run {used}%"
                (progress or logger.info)(f"[tokens] {line}")
                if (
                    max_quota_percent is not None
                    and used is not None
                    and used >= max_quota_percent
                ):
                    stopped = "quota_budget"
                    (progress or logger.warning)(
                        f"[tokens] this run has used {used}% of the usage window, "
                        f"the budget of {max_quota_percent}%; ending the session"
                    )
                    break
                if max_input_tokens is not None and tokens.input > max_input_tokens:
                    stopped = "input_budget"
                    (progress or logger.warning)(
                        f"[tokens] run input {tokens.input} passed the budget of "
                        f"{max_input_tokens}; ending the session"
                    )
                    break
            calls = event.get_function_calls() or []
            if calls:
                saw_tool_call = True
                tool_calls += len(calls)
                logger.info(
                    "%s turn %d: %s (tool calls=%d)",
                    run_label,
                    turns,
                    [getattr(call, "name", "?") for call in calls],
                    tool_calls,
                )
            if done() or tool_calls > max_iterations:
                break
        if done():
            break
        if tool_calls > max_iterations:
            logger.warning(
                "%s hit max_iterations=%d; ending pass", run_label, max_iterations
            )
            break
        nudge = nudge_continue if saw_tool_call else nudge_no_tool
        message = types.Content(role="user", parts=[types.Part.from_text(text=nudge)])

    if debrief and done():
        # One more user message in the SAME context, after the job is over:
        # what did the agent reach for that was not there? Its answer is data
        # for the people building the environment, never for the run.
        if trace is not None:
            trace.nudge(debrief, turn=turns + 1)
        answer: list[str] = []
        # The debrief's own usage is counted too: it is the biggest single
        # request of the run (the whole history plus a long answer).
        try:
            async for event in runner.run_async(
                user_id=_USER_ID,
                session_id=run_label,
                new_message=types.Content(
                    role="user", parts=[types.Part.from_text(text=debrief)]
                ),
            ):
                if trace is not None:
                    trace.event(event, turn=turns + 1)
                usage = getattr(event, "usage_metadata", None)
                if usage is not None and not getattr(event, "partial", False):
                    (progress or logger.info)(f"[tokens] {tokens.add(usage)}")
                for part in getattr(getattr(event, "content", None), "parts", None) or []:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text and not getattr(part, "thought", False):
                        answer.append(text)
        except Exception as exc:  # the debrief is a courtesy; it never fails a run
            logger.warning("%s: debrief failed: %s", run_label, exc)
        if debrief_sink is not None and answer:
            debrief_sink.append("\n".join(answer).strip())

    (progress or logger.info)(f"[tokens] run total: {tokens.totals()}")
    if trace is not None:
        trace.summary(
            tool_calls=tool_calls,
            turns=turns,
            submitted=done(),
            tokens=tokens.as_dict(),
            stopped=stopped,
        )
    return tool_calls, turns
