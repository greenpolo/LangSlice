"""Test doubles for ADK LlmAgent model invocations.

The ADK `BaseLlm` abstraction (``google.adk.models.BaseLlm``) exposes a single
entry point for a model turn: ``generate_content_async(llm_request, stream)``.
`LlmAgent.canonical_model` returns either an explicit ``BaseLlm`` passed via
``model=...`` or delegates to ``LLMRegistry.new_llm(model_name)`` when a string
is provided. Monkeypatching ``LLMRegistry.new_llm`` lets us swap in a scripted
fake without touching the runner or agent builder.

The fake below emits a fixed sequence of turns. It reads the `contents` of the
incoming `LlmRequest` to figure out what step it is on (it counts how many
function-response parts it has received so far) and yields the corresponding
scripted response. Each response is a single ``LlmResponse`` with
``partial=False`` and a ``types.Content`` containing one function-call part.

This matches the non-streaming contract documented in ``BaseLlm``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class _ScriptedSubmitLlm(BaseLlm):
    """A fake BaseLlm that scripts broad sweep -> narrow sweep -> submit_estimate.

    The step is inferred from how many function-response parts are already
    present in the request. This avoids needing instance state on the fake and
    keeps behaviour deterministic across retries.
    """

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream  # unused; we always emit one response
        n_responses = _count_function_responses(llm_request)

        if n_responses == 0:
            # Turn 1: broad sweep
            call = types.Part.from_function_call(
                name="fetch_atlas",
                args={"positions_mm": [2.0, 4.0, 6.0, 8.0, 10.0]},
            )
        elif n_responses == 1:
            # Turn 2: narrow sweep around a plausible candidate
            call = types.Part.from_function_call(
                name="fetch_atlas",
                args={"positions_mm": [5.6, 5.8, 6.0, 6.2, 6.4]},
            )
        else:
            # Turn 3+: submit. The validator may reject the first few attempts
            # if gating is strict, but will relax after enough attempts.
            call = types.Part.from_function_call(
                name="submit_estimate",
                args={
                    "position_mm": 6.0,
                    "reasoning": "Broad sweep placed slice near 6mm; narrow sweep confirmed.",
                },
            )

        yield LlmResponse(
            content=types.Content(role="model", parts=[call]),
            partial=False,
            turn_complete=True,
        )


def _count_function_responses(llm_request: LlmRequest) -> int:
    """Count how many function-response parts appear across request contents."""
    count = 0
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if getattr(part, "function_response", None) is not None:
                count += 1
    return count


def install_fake_adk_model_scripted_submit(monkeypatch: Any) -> None:
    """Patch LLMRegistry.new_llm so any ``model=<string>`` resolves to the fake.

    The resulting agent will, over the course of a session:
      1. call fetch_atlas with a broad sweep
      2. call fetch_atlas with a narrow sweep
      3. call submit_estimate
    """
    from google.adk.models.registry import LLMRegistry

    def _fake_new_llm(model: str) -> BaseLlm:
        return _ScriptedSubmitLlm(model=model)

    monkeypatch.setattr(LLMRegistry, "new_llm", staticmethod(_fake_new_llm))
