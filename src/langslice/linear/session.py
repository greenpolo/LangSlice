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
    trim_stale_tool_images,
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
        ContextFilterPlugin(custom_filter=trim_stale_tool_images)
    ]
    model_call_delay_s = _env_float("LANGSLICE_ADK_MODEL_CALL_DELAY_S")
    if model_call_delay_s is not None and model_call_delay_s > 0:
        plugins.append(ModelCallPacingPlugin(model_call_delay_s))
    capture_dir = _env("LANGSLICE_ADK_CAPTURE_REQUESTS_DIR")
    if capture_dir is not None:
        plugins.append(RequestCapturePlugin(capture_dir, run_label=run_label))
    return plugins


def build_agent(
    *,
    model: str | object,
    name: str,
    instruction: str,
    tools: list[Any],
    max_output_tokens: int = 8000,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
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
        model=resolve_adk_model(model),  # type: ignore[arg-type]
        name=name,
        instruction=instruction,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(**config_kwargs),
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
) -> tuple[int, int]:
    """Drive one agent pass; return ``(tool_calls, turns)``.

    Ends when *done* reports the session's submit tool has fired, or when the
    turn/tool-call budget runs out.
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
    while turns < max_iterations and not done():
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

    if trace is not None:
        trace.summary(tool_calls=tool_calls, turns=turns, submitted=done())
    return tool_calls, turns
