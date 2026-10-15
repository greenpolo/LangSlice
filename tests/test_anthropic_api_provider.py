"""``anthropic-api/<model>``: resolver route, effort mapping and cache breakpoints (no network)."""

import pytest

pytest.importorskip("anthropic")

from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.genai import types  # noqa: E402

from langslice.agent.model_resolver import resolve_adk_model  # noqa: E402
from langslice.agent.session import _with_reasoning  # noqa: E402
from langslice.providers.anthropic_api import AnthropicApiLlm, _mark_last_block  # noqa: E402


def _kwargs(model):
    req = LlmRequest(
        model=model.model,
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
        config=types.GenerateContentConfig(system_instruction="system text", temperature=0.2))
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    from anthropic import NOT_GIVEN
    return model._build_anthropic_kwargs(req, messages, NOT_GIVEN, NOT_GIVEN, NOT_GIVEN)


def test_route_and_reasoning():
    model = _with_reasoning(resolve_adk_model("anthropic-api/claude-opus-5-5"), "high")
    assert isinstance(model, AnthropicApiLlm) and model.model == "claude-opus-5-5"
    kw = _kwargs(model)
    assert kw["output_config"] == {"effort": "high"}
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "temperature" not in kw
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_no_reasoning_keeps_api_defaults():
    kw = _kwargs(resolve_adk_model("anthropic-api/claude-haiku-5-5"))
    assert "output_config" not in kw and kw["temperature"] == 0.2


def test_unknown_effort_is_refused():
    with pytest.raises(ValueError, match="effort"):
        _kwargs(_with_reasoning(resolve_adk_model("anthropic-api/claude-haiku-5-5"), "turbo"))


def test_thinking_blocks_are_never_marked():
    msgs = [{"role": "assistant",
             "content": [{"type": "thinking", "thinking": "x", "signature": "s"}]}]
    assert "cache_control" not in _mark_last_block(msgs)[-1]["content"][-1]
    plain = _mark_last_block([{"role": "user", "content": "plain"}])
    assert plain[-1]["content"][0]["cache_control"]
