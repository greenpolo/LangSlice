"""The ADK session: one agent, its tools, and the loop that drives it.

Tools mutate the stack state as they are called, so a session that never
submits still leaves its writes behind; a post-submit debrief, when asked
for, runs on the same loop.

With ``LANGSLICE_TRACE_DIR`` set, everything the agent is shown, says, calls
and gets back is appended to a JSONL trace (:mod:`langslice.agent.trace`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import aclosing
from typing import Any, cast

from google.adk.agents import LlmAgent
from google.adk.agents._streaming_mode import StreamingMode
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

from langslice.agent.live import LiveCallback, LiveEvents
from langslice.agent.model_resolver import (
    default_http_options,
    env_float,
    env_value,
    resolve_adk_model,
)
from langslice.agent.plugins import (
    ModelCallPacingPlugin,
    RequestCapturePlugin,
    RetiredToolsPlugin,
    StrictArgumentsPlugin,
    WorkingSetImages,
)
from langslice.agent.trace import open_trace

logger = logging.getLogger(__name__)

_APP_NAME = "langslice"
_USER_ID = "langslice-user"


def build_plugins(run_label: str) -> list[BasePlugin]:
    """The ADK plugins every LangSlice session runs with."""
    plugins: list[BasePlugin] = [
        # One working set per session: the instance remembers its cut.
        ContextFilterPlugin(custom_filter=WorkingSetImages()),
    ]
    model_call_delay_s = env_float("LANGSLICE_ADK_MODEL_CALL_DELAY_S")
    if model_call_delay_s is not None and model_call_delay_s > 0:
        plugins.append(ModelCallPacingPlugin(model_call_delay_s))
    capture_dir = env_value("LANGSLICE_ADK_CAPTURE_REQUESTS_DIR")
    if capture_dir is not None:
        plugins.append(RequestCapturePlugin(capture_dir, run_label=run_label))
    # A retired tool's name answers with the tool to use instead.
    plugins.append(RetiredToolsPlugin())
    # Unknown or misplaced arguments are refused, never silently dropped.
    plugins.append(StrictArgumentsPlugin())
    return plugins


def _with_reasoning(model: str | object, reasoning: str | None) -> str | object:
    """Set a resolved model's reasoning effort, when it has one to set.

    Providers that expose the knob carry a ``reasoning_effort`` field (the
    OAuth ``OpenAIOAuthLlm`` is a pydantic model, hence ``model_copy``); every
    other backend is left exactly as it was.
    """
    if not reasoning or not hasattr(model, "reasoning_effort"):
        return model
    copier = getattr(model, "model_copy", None)
    if callable(copier):  # pydantic BaseLlm, e.g. the OAuth OpenAIOAuthLlm
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


#: What a cached input token costs relative to an uncached one, measured on
#: the OAuth lane's quota headers; OpenAI's published API rate is 0.1x.
CACHED_TOKEN_WEIGHT = 0.13


_BUDGET_GRACE = (
    "The run's budget is spent. Call `submit` now with the stack as it stands; "
    "no other tool."
)
_CONTEXT_GRACE = (
    "The last request exceeded the configured input-context limit. "
    "Call `submit` now with the stack as it stands; no other tool."
)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class TokenTally:
    """Summed token usage over one session, one line per model call.

    ``paid`` is an input-only proxy: uncached input plus cached input at
    :data:`CACHED_TOKEN_WEIGHT`. It excludes output and is not an all-token
    API-equivalent cost. ``input`` remains cumulative;
    ``latest_input`` and ``peak_input`` measure individual request context.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input = 0
        self.latest_input = 0
        self.peak_input = 0
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
        self.latest_input = prompt
        self.peak_input = max(self.peak_input, prompt)
        self.cached += cached
        self.output += output
        return (
            f"call {self.calls}: request_input={prompt} (cached {cached}) out={output}; "
            f"cumulative_input={self.input} peak_input={self.peak_input} paid~{self.paid}"
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input": self.input,
            "latest_input": self.latest_input,
            "peak_input": self.peak_input,
            "cached": self.cached,
            "output": self.output,
            "paid": self.paid,
        }

    def totals(self) -> str:
        return (
            f"{self.calls} calls, cumulative_input={self.input} (cached {self.cached}, "
            f"paid~{self.paid}), out={self.output}, peak_input={self.peak_input}"
        )


async def run_agent_session(
    *,
    agent: LlmAgent,
    seed_message: types.Content,
    done: Callable[[], bool],
    nudge_no_tool: str,
    nudge_continue: str,
    run_label: str,
    debrief: str | None = None,
    debrief_sink: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
    max_input_tokens: int | None = None,
    max_quota_percent: int | None = None,
    on_event: LiveCallback | None = None,
    background: Callable[[], list[Any] | None] | None = None,
) -> tuple[int, int]:
    """Drive one agent pass; return ``(tool_calls, turns)``.

    Ends when *done* reports the session's submit tool has fired, or a configured
    quota/input-context budget is reached. There is no turn or tool-call limit.
    Cached input still occupies context. Budget checks happen after the
    response, not before sending it;
    cumulative usage remains available separately for cost accounting.

    *background*, when the model ends its turn without submitting: waits
    for the job's background work still running and returns what to say
    (texts and pictures: the notices of the work that finished), or None
    when there is nothing; that is then the next message instead of a
    nudge.
    """
    live = LiveEvents(on_event) if on_event is not None else None
    run_config = RunConfig(
        max_llm_calls=0,
        streaming_mode=StreamingMode.SSE if live else StreamingMode.NONE,
    )
    trace = open_trace(run_label, agent=agent)
    if trace is not None and hasattr(agent.model, "capture_usage_details"):
        cast(Any, agent.model).capture_usage_details = True
    app = App(
        name=_APP_NAME,
        root_agent=agent,
        plugins=build_plugins(run_label),
    )
    runner = InMemoryRunner(app=app)
    assert runner.session_service is not None
    await runner.session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID, session_id=run_label, state={}
    )

    message = seed_message
    if live is not None:
        live.start(agent, seed_message)
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
    # A budget stop gets ONE more call, to submit: a run stopped one call
    # short of its submit loses all its work for a few hundred tokens.
    grace_left = 1
    while not done() and (stopped is None or grace_left):
        turns += 1
        if stopped is not None:
            grace_left -= 1
            nudge = _CONTEXT_GRACE if stopped == "input_context_limit" else _BUDGET_GRACE
            message = types.Content(role="user", parts=[types.Part.from_text(text=nudge)])
        if trace is not None and nudge is not None:
            trace.nudge(nudge, turn=turns)
        saw_tool_call = False
        # Close inside this task before submit/budget breaks leave the loop.
        # Deferred finalization detaches ADK telemetry in a different context.
        async with aclosing(runner.run_async(
            user_id=_USER_ID, session_id=run_label, new_message=message, run_config=run_config
        )) as events:
            async for event in events:
                if trace is not None:
                    trace.event(event, turn=turns)
                if live is not None:
                    live.event(event)
                if getattr(event, "error_message", None):
                    # A model/provider error is a stop, not a turn to nudge past:
                    # looping would re-pay the same failure until the budget ran out.
                    if live is not None:
                        live.emit("error", text=str(event.error_message))
                    raise RuntimeError(
                        f"{run_label}: model error on turn {turns} "
                        f"({getattr(event, 'error_code', None)}): {event.error_message}"
                    )
                if getattr(event, "partial", False):
                    continue
                usage = getattr(event, "usage_metadata", None)
                if usage is not None:
                    line = tokens.add(usage)
                    if live is not None:
                        live.emit("usage", tokens=tokens.as_dict())
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
                        # No break here: the pending tool call is answered first
                        # (see below), so the history stays well-formed for the
                        # grace call.
                        if stopped is None:
                            (progress or logger.warning)(
                                f"[tokens] this run has used {used}% of the usage window, "
                                f"the budget of {max_quota_percent}%; one more call to submit"
                            )
                        stopped = stopped or "quota_budget"
                    if max_input_tokens is not None and tokens.latest_input > max_input_tokens:
                        if stopped is None:
                            (progress or logger.warning)(
                                f"[tokens] request input {tokens.latest_input} exceeded the "
                                f"context limit of {max_input_tokens}; one more call to submit"
                            )
                        stopped = stopped or "input_context_limit"
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
                if done():
                    break
                if stopped is not None and (event.get_function_responses() or not calls):
                    # ADK runs the WHOLE tool loop inside one run_async: a model
                    # that never stops calling tools never ends the turn on its
                    # own. Leave once the budget-tripping call is answered.
                    break
        if done():
            break
        if background is not None and stopped is None:
            # Work still running when the turn ended: wait for it, and its
            # notice is the next message.
            items = await asyncio.to_thread(background)
            if items:
                from langslice.doors.tools.media import items_to_parts

                nudge = "\n".join(item for item in items if isinstance(item, str))
                message = types.Content(role="user", parts=items_to_parts(items))
                continue
        nudge = nudge_continue if saw_tool_call else nudge_no_tool
        message = types.Content(role="user", parts=[types.Part.from_text(text=nudge)])

    if debrief and done() and stopped is None:
        # One more user message in the SAME context, after the job is over:
        # what did the agent reach for that was not there? Its answer is data
        # for the people building the environment, never for the run.
        if trace is not None:
            trace.nudge(debrief, turn=turns + 1)
        answer: list[str] = []
        # The debrief's own usage is counted too: it is the biggest single
        # request of the run (the whole history plus a long answer).
        try:
            async with aclosing(runner.run_async(
                user_id=_USER_ID,
                session_id=run_label,
                new_message=types.Content(
                    role="user", parts=[types.Part.from_text(text=debrief)]
                ),
                run_config=run_config,
            )) as events:
                async for event in events:
                    if trace is not None:
                        trace.event(event, turn=turns + 1)
                    if live is not None:
                        live.event(event)
                    if getattr(event, "partial", False):
                        continue
                    usage = getattr(event, "usage_metadata", None)
                    if usage is not None:
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
    if live is not None:
        live.emit("complete", submitted=done(), tokens=tokens.as_dict())
    return tool_calls, turns
