"""OpenAI subscription-OAuth backend (Codex Responses).

NOT the OpenAI API: this transport talks to ``chatgpt.com/backend-api/codex``
— the private backend behind the ChatGPT app and Codex CLI — with its own
wire format and behaviors (e.g. its image endpoint ignores ``size`` and
matches the input image's aspect ratio).

One user's ChatGPT Plus/Pro login powers both of LangSlice's model needs with
no API key:

* chat/vision/tool-use through :class:`OpenAIOAuthLlm`, a native ``google-adk``
  model backend registered for ``openai-oauth/*`` model strings;
* image edits (``gpt-image-2.5-sunburst`` by default) through :func:`edit_image`, the Codex
  ``images/edits`` endpoint, used by the image-model trace.

Credentials come only from LangSlice's own file, ``~/.langslice/openai_auth.json``
(written by :func:`login`); the Codex CLI's login is never read, so its account
and rotating refresh token stay its own. ``LANGSLICE_OPENAI_AUTH`` names another
file for one process (a second account). Access tokens are refreshed
automatically.

Wire format (Codex Responses, ``https://chatgpt.com/backend-api/codex``)
matches the openai/codex client and RayBytes/ChatMock:

* user images are ``{"type": "input_image", "image_url": "<data-uri>"}`` with a
  bare string URL, not ``{"url": ...}``;
* function tools are flat: ``{"type": "function", "name", "description",
  "parameters", "strict"}``;
* model tool calls arrive on ``response.output_item.done`` with
  ``item.type == "function_call"`` (``name``/``arguments``/``call_id``);
* reasoning items (``encrypted_content``, ``store`` is false) are kept on the
  model turn and replayed verbatim ahead of it, and the request asks for
  ``reasoning.context = all_turns`` so the model actually renders them (the
  default, ``current_turn``, drops everything before the last user message);
* tool results go back as ``{"type": "function_call_output", "call_id",
  "output"}`` input items; a tool's images ride inside ``output`` as
  ``input_image`` parts, never as a separate user message.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
import math
import random
import time
import uuid
from collections.abc import AsyncGenerator, Iterator, Sequence
from email.utils import parsedate_to_datetime
from typing import Any

import requests
from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.genai import types
from pydantic import Field, field_validator

from langslice.core.pictures.media_keys import MEDIA_LAYOUT_ATTR
from langslice.providers.connections import (
    OPENAI_OAUTH_DEFAULT_AGENT_MODEL,
    OPENAI_OAUTH_DEFAULT_IMAGE_MODEL,
)
from langslice.providers.oauth.login import (
    ORIGINATOR,
    Creds,
    load_credentials,
    refresh,
)
from langslice.providers.usage import (
    item_descriptor,
    request_descriptor,
    usage_diagnostics,
)

logger = logging.getLogger(__name__)
RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"

IMAGES_EDITS_URL = "https://chatgpt.com/backend-api/codex/images/edits"
MODEL_PREFIX = "openai-oauth/"

#: The image model :func:`edit_image` calls when none is named.
DEFAULT_IMAGE_MODEL = OPENAI_OAUTH_DEFAULT_IMAGE_MODEL

#: The default agent model on this lane.
DEFAULT_REVIEW_MODEL = OPENAI_OAUTH_DEFAULT_AGENT_MODEL


# --- request plumbing --------------------------------------------------------
def _headers(creds: Creds, session_id: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {creds.access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": "responses=experimental",
        "originator": ORIGINATOR,
        "session_id": session_id,
        "User-Agent": "langslice/0.1",
    }
    if creds.account_id:
        headers["chatgpt-account-id"] = creds.account_id
    return headers


def data_uri(payload: bytes, mime: str = "image/png") -> str:
    """Return a ``data:`` URI for raw image bytes."""
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def iter_sse(response: requests.Response) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON events from a Codex Responses SSE stream.

    Lines arrive as bytes with no charset on this endpoint, so decode
    explicitly rather than relying on ``requests`` to do it.
    """
    for raw in response.iter_lines():
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


#: Quota headers the Codex backend may send (the CLI's ``/status`` reads
#: them); ``primary`` is the short window, ``secondary`` the long one.
QUOTA_HEADERS = (
    "x-codex-primary-used-percent",
    "x-codex-primary-reset-at",
    "x-codex-secondary-used-percent",
    "x-codex-secondary-reset-at",
)


def quota_from_headers(headers: Any) -> dict[str, str]:
    """The quota headers present on a response, short keys, as strings."""
    return {
        name.removeprefix("x-codex-").replace("-", "_"): str(headers[name])
        for name in QUOTA_HEADERS
        if name in headers
    }


def stream_events(
    body: dict[str, Any], *, session_id: str | None = None
) -> Iterator[dict[str, Any]]:
    """POST a Responses request with bounded retries, refreshing once on 401.

    The first event is a synthetic ``langslice.quota`` carrying any quota
    headers the backend sent. *session_id* rides in the headers; keep it
    stable across one agent loop so the backend routes to a warm cache.
    Early transient failure events are also retried, before any model output.
    A stream containing model output is never replayed.
    """
    creds = load_credentials()
    session_id = session_id or str(uuid.uuid4())
    retries = 0
    refreshed = False
    while True:
        try:
            response = requests.post(
                RESPONSES_URL,
                headers=_headers(creds, session_id),
                json=body,
                stream=True,
                timeout=600,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            if isinstance(exc, requests.exceptions.SSLError) or retries >= 3:
                raise
            delay = _response_retry_delay(retries, None)
            assert delay is not None
            reason = type(exc).__name__
        else:
            if response.status_code == 401 and creds.refresh_token and not refreshed:
                response.close()
                creds = refresh(creds)
                refreshed = True
                continue
            if response.status_code < 400:
                events = iter_sse(response)
                prefix: list[dict[str, Any]] = []
                try:
                    for event in events:
                        prefix.append(event)
                        if event.get("type") not in (
                            "response.created", "response.queued", "response.in_progress",
                        ):
                            break
                except Exception:
                    response.close()
                    raise
                delay = _response_retry_delay(retries, response.headers.get("Retry-After"))
                if (not prefix or not _retryable_failure(prefix[-1])
                        or retries >= 3 or delay is None):
                    break
                response.close()
                reason = "early response.failed"
            else:
                try:
                    message = response.text
                    quota = quota_from_headers(response.headers)
                    delay = _response_retry_delay(retries, response.headers.get("Retry-After"))
                finally:
                    response.close()
                retryable = (
                    response.status_code in (408, 409, 429)
                    or 500 <= response.status_code < 600
                )
                if retries >= 3 or not retryable or delay is None or _hard_quota_error(message):
                    raise RuntimeError(
                        f"Codex Responses request failed ({response.status_code}): {message[:500]}"
                        + (f" quota={quota}" if quota else "")
                    )
                reason = f"HTTP {response.status_code}"
        retries += 1
        logger.warning("Codex Responses %s; retry %d/3 in %.1fs", reason, retries, delay)
        time.sleep(delay)
    quota = quota_from_headers(response.headers)
    return itertools.chain(
        [{"type": "langslice.quota", "quota": quota}], prefix, _closing(response, events),
    )


def _retryable_failure(event: dict[str, Any]) -> bool:
    """Only known transient failures, with no quota exhaustion, can be replayed."""
    if event.get("type") != "response.failed":
        return False
    error = event.get("response", {}).get("error") or {}
    if _hard_quota_error(json.dumps({"error": error})):
        return False
    return any(error.get(key) in (
        "server_error", "service_unavailable", "overloaded", "temporarily_unavailable",
        "rate_limit_exceeded",
    ) for key in ("code", "type")) or (
        "servers are currently overloaded" in str(error.get("message", "")).lower()
    )


def _response_retry_delay(retries: int, retry_after: str | None) -> float | None:
    """Respect server delay; defer to the caller if it exceeds our one-minute wait."""
    delay = float(2 ** (retries + 1))
    if retry_after:
        try:
            server_delay = float(retry_after)
        except ValueError:
            try:
                server_delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                server_delay = 0.0
        if math.isfinite(server_delay):
            delay = max(delay, server_delay)
    if delay > 60:
        return None
    return delay + random.uniform(0, min(0.5, 60 - delay))


def _hard_quota_error(message: str) -> bool:
    try:
        payload = json.loads(message)
    except ValueError:
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    return isinstance(error, dict) and any(
        error.get(key)
        in ("insufficient_quota", "usage_limit_reached", "billing_hard_limit_reached")
        for key in ("code", "type")
    )


def _closing(
    response: requests.Response, events: Iterator[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    try:
        yield from events
    finally:
        response.close()


def _failure_message(event: dict[str, Any]) -> str:
    error = event.get("response", {}).get("error") or {}
    return str(error.get("message", "unknown error"))


# --- image generation --------------------------------------------------------
def edit_image(
    prompt: str,
    reference_images: Sequence[str],
    *,
    image_model: str = DEFAULT_IMAGE_MODEL,
    quality: str = "high",
    size: str | None = None,
    n: int = 1,
) -> bytes:
    """Edit an image via the direct Codex ``images/edits`` endpoint.

    ``prompt`` goes into the request body verbatim and reaches the image
    model untouched. The endpoint mirrors the public ``images.edit`` shape
    (the Codex CLI's own image edits use it); it is undocumented and carries
    no compatibility guarantee.
    """
    creds = load_credentials()
    body: dict[str, Any] = {
        "model": image_model,
        "prompt": prompt,
        "images": [{"image_url": uri} for uri in reference_images],
        "quality": quality,
        "n": n,
    }
    if size:
        body["size"] = size

    def _post(c: Creds) -> requests.Response:
        headers = _headers(c, str(uuid.uuid4()))
        headers["Accept"] = "application/json"
        return requests.post(IMAGES_EDITS_URL, headers=headers, json=body, timeout=600)

    response = _post(creds)
    if response.status_code == 401 and creds.refresh_token:
        response = _post(refresh(creds))
    if response.status_code >= 400:
        raise RuntimeError(
            f"Codex images/edits request failed ({response.status_code}): {response.text[:500]}"
        )
    data = response.json().get("data") or []
    b64 = data[0].get("b64_json") if data else None
    if not b64:
        raise RuntimeError("Codex images/edits returned no image data")
    return base64.b64decode(b64)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _part_data_uri(blob: Any) -> str | None:
    data = getattr(blob, "data", None)
    mime = getattr(blob, "mime_type", None)
    if not data or not isinstance(mime, str) or not mime.startswith("image/"):
        return None
    return data_uri(bytes(data), mime)


def content_to_input_items(content: types.Content) -> list[dict[str, Any]]:
    """Convert one ADK ``Content`` into Codex Responses ``input`` items."""
    items: list[dict[str, Any]] = []
    # User content keeps PART ORDER (label text directly before its image —
    # interleaving is load-bearing for multi-image seed messages); model text
    # is joined as one assistant message.
    user_content: list[dict[str, Any]] = []
    model_texts: list[str] = []

    reasoning_items: list[dict[str, Any]] = []
    for part in content.parts or []:
        if part.thought_signature:
            # A previous turn's reasoning, replayed verbatim (encrypted) so the
            # model continues its own chain of thought instead of restarting.
            reasoning_items.append(json.loads(part.thought_signature))
        elif part.function_response is not None:
            items.append(_function_call_output(part.function_response))
        elif part.function_call is not None:
            call = part.function_call
            items.append(
                {
                    "type": "function_call",
                    "name": call.name or "",
                    "arguments": _json_dumps(call.args or {}),
                    "call_id": call.id or "",
                }
            )
        elif part.inline_data is not None:
            uri = _part_data_uri(part.inline_data)
            if uri:
                user_content.append({"type": "input_image", "image_url": uri})
        elif part.text and not part.thought:
            if content.role == "model":
                model_texts.append(part.text)
            else:
                user_content.append({"type": "input_text", "text": part.text})

    if content.role == "model":
        # Reasoning leads the turn it produced.
        items[:0] = reasoning_items
        if model_texts:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "\n".join(model_texts)}],
                }
            )
    elif user_content:
        items.append({"type": "message", "role": "user", "content": user_content})
    return items


def _function_call_output(response: types.FunctionResponse) -> dict[str, Any]:
    """One tool result as a ``function_call_output`` item.

    A tool's images ride INSIDE the output as ``input_image`` parts (the
    Codex CLI's ``view_image`` does the same), each preceded by an
    ``input_text`` label naming the call and its index, so the model can tell
    which call any image answers. Images sent as a separate user message
    would open a new turn, and the backend drops the replayed reasoning of
    earlier turns at each one.
    """
    payload = response.response
    output = payload if isinstance(payload, str) else _json_dumps(payload)
    uris = [
        uri
        for response_part in response.parts or []
        if (uri := _part_data_uri(response_part.inline_data))
    ]
    item: dict[str, Any] = {
        "type": "function_call_output",
        "call_id": response.id or "",
        "output": output,
    }
    layout = getattr(response, MEDIA_LAYOUT_ATTR, None)
    if uris or layout is not None:
        name = response.name or "tool"
        content: list[dict[str, Any]] = [{"type": "input_text", "text": output}]
        total, slots = layout if layout is not None else (len(uris), list(range(len(uris))))
        surviving = dict(zip(slots, uris, strict=True))
        for index in range(total):
            content.append(
                {"type": "input_text", "text": f"{name} image {index + 1} of {total}"}
            )
            if index in surviving:
                content.append({
                    "type": "input_image", "image_url": surviving[index], "detail": "high",
                })
        item["output"] = content
    return item


def _reasoning_part(item: dict[str, Any]) -> types.Part:
    """Keep a reasoning output item for replay.

    The whole item (id, summary, ``encrypted_content``) rides in
    ``thought_signature`` — the Gemini field for exactly this job — and the
    summary text, when the backend sent one, in ``text`` so traces show it.
    """
    summary = "\n".join(
        text
        for entry in item.get("summary") or []
        if isinstance(text := entry.get("text"), str) and text
    )
    return types.Part(
        thought=True, text=summary or None, thought_signature=_json_dumps(item).encode()
    )


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """*schema* with every local ``#/$defs/...`` reference replaced by its definition.

    A typed-dict argument (``view``, ``entries``) reaches ADK's declaration
    as a ``$ref`` into ``$defs``; the tool is sent with each object written
    out where it is used, so every key and type sits next to its argument.
    Recursive definitions are left as references.
    """
    definitions = schema.get("$defs") or {}
    if not isinstance(definitions, dict) or not definitions:
        return schema

    def resolve(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if name in definitions and name not in seen:
                merged = {**definitions[name], **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged, (*seen, name))
        return {key: resolve(value, seen) for key, value in node.items() if key != "$defs"}

    inlined = resolve(schema, ())
    if any(isinstance(ref, str) for ref in _refs(inlined)):
        inlined["$defs"] = definitions  # a recursive definition still needs its target
    return inlined


def _refs(node: Any) -> list[Any]:
    if isinstance(node, list):
        return [ref for item in node for ref in _refs(item)]
    if isinstance(node, dict):
        return [node.get("$ref"), *[ref for value in node.values() for ref in _refs(value)]]
    return []


def _json_schema_dict(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    if declaration.parameters_json_schema is not None:
        schema = declaration.parameters_json_schema
        return (_inline_refs(dict(schema)) if isinstance(schema, dict)
                else {"type": "object", "properties": {}})
    if declaration.parameters is not None:
        dumped = declaration.parameters.json_schema.model_dump(
            mode="json", exclude_none=True, by_alias=True
        )
        return _inline_refs(dumped)
    return {"type": "object", "properties": {}}


def tools_to_wire(config: types.GenerateContentConfig) -> list[dict[str, Any]]:
    """Convert ADK function declarations into Responses function tools."""
    wire: list[dict[str, Any]] = []
    for tool in config.tools or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            if not declaration.name:
                continue
            wire.append(
                {
                    "type": "function",
                    "name": declaration.name,
                    "description": declaration.description or "",
                    "strict": False,
                    "parameters": _json_schema_dict(declaration),
                }
            )
    return wire


def _system_instruction(config: types.GenerateContentConfig) -> str | None:
    instruction = config.system_instruction
    if instruction is None:
        return None
    if isinstance(instruction, str):
        return instruction
    parts = getattr(instruction, "parts", None)
    if parts:
        return "\n".join(part.text for part in parts if part.text)
    if isinstance(instruction, list):
        return "\n".join(str(item) for item in instruction)
    return str(instruction)


def _usage_metadata(event: dict[str, Any]) -> types.GenerateContentResponseUsageMetadata | None:
    usage = event.get("response", {}).get("usage")
    if not isinstance(usage, dict):
        return None
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage.get("input_tokens"),
        cached_content_token_count=cached,
        candidates_token_count=usage.get("output_tokens"),
        thoughts_token_count=reasoning,
        total_token_count=usage.get("total_tokens"),
    )


async def _aiter(events: Iterator[dict[str, Any]]) -> AsyncGenerator[dict[str, Any], None]:
    """Drive a blocking SSE iterator from the event loop, one event per hop."""
    while True:
        event = await asyncio.to_thread(next, events, None)
        if event is None:
            return
        yield event


class OpenAIOAuthLlm(BaseLlm):
    """ADK model backed by a ChatGPT subscription (Codex Responses backend).

    Registered for ``openai-oauth/<model>`` strings, e.g.
    ``openai-oauth/gpt-5.6-luna``.
    Supports vision input, function calling, and media returned by tools.
    """

    reasoning_effort: str = "medium"
    """Reasoning effort: low | medium | high | xhigh | max (the GPT-6 models
    answer ``none`` and ``minimal`` with HTTP 400)."""

    prompt_cache_key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Stable across the turns of one agent loop, for upstream prompt caching."""

    capture_usage_details: bool = False
    """Capture content-free request/usage diagnostics in host metadata, not the prompt."""

    @field_validator("model")
    @classmethod
    def _strip_prefix(cls, value: str) -> str:
        return value[len(MODEL_PREFIX):] if value.startswith(MODEL_PREFIX) else value

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"openai-oauth/.*"]

    @property
    def capabilities(self) -> LlmCapabilities:
        # No structured-output-with-tools support on the subscription backend.
        return LlmCapabilities(output_schema_and_tools=False)

    def build_request_body(self, llm_request: LlmRequest) -> dict[str, Any]:
        """Convert an ``LlmRequest`` into a Codex Responses request body."""
        input_items: list[dict[str, Any]] = []
        for content in llm_request.contents:
            input_items.extend(content_to_input_items(content))

        body: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": tools_to_wire(llm_request.config),
            "tool_choice": "auto",
            # Several tool calls in one model turn: one call's history cost
            # instead of one per tool. The toolbox serializes them.
            "parallel_tool_calls": True,
            "store": False,
            "stream": True,
            "prompt_cache_key": self.prompt_cache_key,
            # context=all_turns: render the replayed reasoning of EARLIER turns
            # into this sample; the default, current_turn, drops it at every
            # user message.
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": "auto",
                "context": "all_turns",
            },
            "include": ["reasoning.encrypted_content"],
        }
        instructions = _system_instruction(llm_request.config)
        if instructions:
            body["instructions"] = instructions
        return body

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self._maybe_append_user_content(llm_request)
        body = self.build_request_body(llm_request)
        request_usage = request_descriptor(body) if self.capture_usage_details else None
        output_usage: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] | None = None
        events = await asyncio.to_thread(
            stream_events, body, session_id=self.prompt_cache_key
        )

        text_chunks: list[str] = []
        reasoning_parts: list[types.Part] = []
        last_summary_key: tuple[Any, Any] | None = None
        calls: list[types.Part] = []
        usage: types.GenerateContentResponseUsageMetadata | None = None
        quota: dict[str, str] = {}

        async for event in _aiter(events):
            kind = event.get("type")
            if kind == "response.output_text.delta":
                delta = event.get("delta", "")
                if not delta:
                    continue
                text_chunks.append(delta)
                if stream:
                    yield LlmResponse(
                        content=types.Content(
                            role="model", parts=[types.Part.from_text(text=delta)]
                        ),
                        partial=True,
                    )
            elif kind == "response.reasoning_summary_text.delta":
                # Only the backend's public summary is displayable. Encrypted
                # reasoning remains on the final item's replay signature below.
                delta = event.get("delta", "")
                if stream and isinstance(delta, str) and delta:
                    # Summary parts have independent text streams. Preserve
                    # their boundaries to match the finalized summary text.
                    key = (event.get("item_id"), event.get("summary_index", 0))
                    if last_summary_key is not None and key != last_summary_key:
                        delta = "\n" + delta
                    last_summary_key = key
                    yield LlmResponse(
                        content=types.Content(
                            role="model", parts=[types.Part(thought=True, text=delta)]
                        ),
                        partial=True,
                    )
            elif kind == "response.output_item.done":
                item = event.get("item", {})
                if self.capture_usage_details:
                    output_usage.append(item_descriptor(item))
                if item.get("type") == "reasoning":
                    reasoning_part = _reasoning_part(item)
                    if reasoning_part.text and any(part.text for part in reasoning_parts):
                        reasoning_part.text = "\n" + reasoning_part.text
                    reasoning_parts.append(reasoning_part)
                    continue
                if item.get("type") != "function_call":
                    continue
                part = types.Part(
                    function_call=types.FunctionCall(
                        id=item.get("call_id") or item.get("id"),
                        name=item.get("name") or "",
                        args=_parse_arguments(item.get("arguments")),
                    )
                )
                calls.append(part)
                if stream:
                    yield LlmResponse(
                        content=types.Content(role="model", parts=[part]), partial=True
                    )
            elif kind == "response.failed":
                raise RuntimeError(f"Codex response.failed: {_failure_message(event)}")
            elif kind == "langslice.quota":
                quota = event.get("quota") or {}
            elif kind == "response.completed":
                usage = _usage_metadata(event)
                diagnostics = usage_diagnostics(
                    event.get("response", {}).get("usage"), request_usage, output_usage,
                )
                break

        parts: list[types.Part] = list(reasoning_parts)
        text = "".join(text_chunks)
        if text:
            parts.append(types.Part.from_text(text=text))
        parts.extend(calls)
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            usage_metadata=usage,
            custom_metadata={
                **({"quota": quota} if quota else {}),
                **({"usage_diagnostics": diagnostics} if diagnostics is not None else {}),
            } or None,
            model_version=self.model,
        )


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


LLMRegistry.register(OpenAIOAuthLlm)
