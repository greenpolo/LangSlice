"""``anthropic-api``: Claude models through the Anthropic API with an API key.

A thin layer over ADK's own ``AnthropicLlm`` (google.adk.models.anthropic_llm),
which converts ADK requests to the Messages API. Three things it adds:

- **The key**: ``ANTHROPIC_API_KEY`` when set, else the file
  ``~/.langslice/anthropic_api_key`` (beside the openai-oauth login files).
- **Reasoning effort**: LangSlice sets ``reasoning_effort`` on a resolved model
  (``agent.session._with_reasoning``); here it becomes Claude's
  ``output_config.effort`` with adaptive thinking, the API's way to set effort.
- **Readable reasoning**: adaptive thinking asks for ``display: "summarized"``.
  On Opus 4.7 and later the API default is ``"omitted"`` (empty thinking text,
  only an encrypted signature), which leaves run traces with no reasoning.
- **Prompt caching**: the Messages API caches only up to blocks marked
  ``cache_control``; ADK marks none. The system prompt and the newest message's
  last block are marked, so each turn reads the previous turn's prefix
  (tools, system, history) from the cache instead of paying for it again.
"""

from __future__ import annotations

import copy
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

from google.adk.models.anthropic_llm import AnthropicLlm

#: Where the key is kept when ANTHROPIC_API_KEY is not set.
KEY_FILE = Path.home() / ".langslice" / "anthropic_api_key"
#: Claude's effort levels; LangSlice's --reasoning values map onto them as named.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
#: Output cap per call. Thinking counts against it; the SDK refuses a
#: non-streaming call whose cap implies more than ten minutes, so stay below that.
MAX_TOKENS = 16000
_EPHEMERAL = {"type": "ephemeral"}


def api_key() -> str:
    """The Anthropic API key: the environment's, else :data:`KEY_FILE`'s."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(
            f"No Anthropic API key: set ANTHROPIC_API_KEY or write the key to {KEY_FILE}"
        )
    return key


def _mark_last_block(messages: list[Any]) -> list[Any]:
    """*messages* with a cache breakpoint on the newest message's last block."""
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        if isinstance(blocks[-1], dict) and blocks[-1].get("type") not in ("thinking", "redacted_thinking"):
            blocks[-1] = {**blocks[-1], "cache_control": _EPHEMERAL}
        last["content"] = blocks
    out[-1] = last
    return out


class AnthropicApiLlm(AnthropicLlm):
    """ADK ``AnthropicLlm`` with LangSlice's key lookup, effort and prompt caching."""

    max_tokens: int = MAX_TOKENS
    #: Set by ``agent.session._with_reasoning`` from ``--reasoning``; None keeps the API default.
    reasoning_effort: Optional[str] = None

    def _build_anthropic_kwargs(self, llm_request, messages, tools, tool_choice, thinking):  # type: ignore[override]
        kwargs = super()._build_anthropic_kwargs(llm_request, messages, tools, tool_choice, thinking)
        effort = self.reasoning_effort
        if effort and "output_config" not in kwargs:
            if effort not in EFFORTS:
                raise ValueError(f"Unknown reasoning effort {effort!r} for Claude: use one of {EFFORTS}")
            kwargs["output_config"] = {"effort": effort}
            if not isinstance(kwargs.get("thinking"), dict):
                kwargs["thinking"] = {"type": "adaptive"}
            for name in ("temperature", "top_p", "top_k"):  # not accepted with effort set
                kwargs.pop(name, None)
        thinking_param = kwargs.get("thinking")
        if isinstance(thinking_param, dict) and thinking_param.get("type") == "adaptive" and "display" not in thinking_param:
            kwargs["thinking"] = {**thinking_param, "display": "summarized"}
        system = kwargs.get("system")
        if isinstance(system, str) and system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]
        kwargs["messages"] = _mark_last_block(copy.copy(kwargs.get("messages") or []))
        return kwargs

    @cached_property
    def _anthropic_client(self):  # type: ignore[override]
        from anthropic import AsyncAnthropic
        import httpx2 as httpx  # the HTTP library the anthropic SDK (>=1.13) is built on

        # A plain client, not the SDK's default wrapper: that wrapper turns every
        # NO_PROXY entry into a URL pattern and fails on a bracketed IPv6 one
        # ("[::1]", set on the lab machine). An explicit proxy is still honoured.
        proxy = next((os.environ[n] for n in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
                      if os.environ.get(n)), None)
        http_client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(600.0, connect=15.0),
                                        **({"proxy": proxy} if proxy else {}))
        return AsyncAnthropic(api_key=api_key(), max_retries=4, http_client=http_client)
