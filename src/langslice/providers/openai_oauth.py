"""OpenAI subscription-OAuth backend (Codex Responses).

NOT the OpenAI API: this transport talks to ``chatgpt.com/backend-api/codex``
— the private backend behind the ChatGPT app and Codex CLI — with its own
wire format and behaviors (e.g. its image tool ignores ``size`` and matches
the input image's aspect ratio; probed 2026-08-25).

One user's ChatGPT Plus/Pro login powers both of LangSlice's model needs with
no API key:

* chat/vision/tool-use through :class:`ChatGptLlm`, a native ``google-adk``
  model backend registered for ``openai-oauth/*`` (and legacy ``chatgpt/*``) model strings;
* image generation (``gpt-image-2``) through :func:`edit_image`, used by
  the nonlinear registration provider (the router session drives the hosted
  ``image_generation`` tool through the Responses body directly).

Credentials come from ``~/.langslice/openai_auth.json`` (written by
:func:`login`), falling back to the Codex CLI's ``~/.codex/auth.json`` and then
its OS-keyring entry. Access tokens are refreshed automatically.

Wire format (Codex Responses, ``https://chatgpt.com/backend-api/codex``)
matches the openai/codex client and RayBytes/ChatMock:

* user images are ``{"type": "input_image", "image_url": "<data-uri>"}`` with a
  bare string URL, not ``{"url": ...}``;
* function tools are flat: ``{"type": "function", "name", "description",
  "parameters", "strict"}``;
* model tool calls arrive on ``response.output_item.done`` with
  ``item.type == "function_call"`` (``name``/``arguments``/``call_id``);
* tool results go back as ``{"type": "function_call_output", "call_id",
  "output"}`` input items.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import itertools
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import uuid
import webbrowser
from collections.abc import AsyncGenerator, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.genai import types
from pydantic import Field, field_validator

logger = logging.getLogger(__name__)

# --- constants ---------------------------------------------------------------
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
IMAGES_EDITS_URL = "https://chatgpt.com/backend-api/codex/images/edits"
ORIGINATOR = "langslice"
MODEL_PREFIX = "openai-oauth/"
LEGACY_MODEL_PREFIX = "chatgpt/"  # accepted alias; older configs and docs use it
REFRESH_SKEW_S = 5 * 60  # refresh when the access token expires within 5 min

#: Where ``langslice login`` stores its token (mode 600).
CREDENTIALS_PATH = Path.home() / ".langslice" / "openai_auth.json"
_CODEX_AUTH = Path.home() / ".codex" / "auth.json"

#: Routing model for image generation; the image itself is always rendered by
#: ``gpt-image-2`` server-side.
DEFAULT_IMAGE_MODEL = "gpt-image-2"

#: Default review/mainline model for openai-oauth registration paths.
DEFAULT_REVIEW_MODEL = "openai-oauth/gpt-5.6-sol"
MAX_REFERENCE_IMAGES = 8

# OAuth callback (the Codex client_id only whitelists this redirect).
_REDIRECT_URI = "http://localhost:1455/auth/callback"
_LOGIN_PORT = 1455
_LOGIN_SCOPE = "openid profile email offline_access"


# --- credentials -------------------------------------------------------------
@dataclass
class Creds:
    """A ChatGPT-subscription access token and the metadata needed to use it."""

    access_token: str
    refresh_token: str | None = None
    id_token: str | None = None
    account_id: str | None = None
    source: str = "?"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verifying the signature (client-side only)."""
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)  # fix padding
    decoded = json.loads(base64.urlsafe_b64decode(payload_b64))
    if not isinstance(decoded, dict):
        raise ValueError("JWT payload is not an object")
    return decoded


def _auth_claims(token: str | None) -> dict[str, Any]:
    if not token:
        return {}
    try:
        claims = _decode_jwt_payload(token).get("https://api.openai.com/auth", {})
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def account_id_from_id_token(id_token: str | None) -> str | None:
    """Return the ChatGPT account id embedded in an OAuth id_token, if present."""
    value = _auth_claims(id_token).get("chatgpt_account_id")
    return value if isinstance(value, str) else None


def plan_from_id_token(id_token: str | None) -> str | None:
    """Return the subscription plan claim from an id_token, if present."""
    value = _auth_claims(id_token).get("chatgpt_plan_type")
    return value if isinstance(value, str) else None


def _token_expiry(access_token: str) -> float | None:
    try:
        return float(_decode_jwt_payload(access_token)["exp"])
    except Exception:
        return None


def creds_from_doc(doc: dict[str, Any], source: str) -> Creds:
    """Build :class:`Creds` from an ``auth.json``-shaped document.

    Accepts both the nested Codex shape (``{"tokens": {...}}``) and a flat one.
    """
    tokens = doc.get("tokens", doc)
    access = tokens.get("access_token")
    if not access:
        raise ValueError(f"{source}: no access_token in credential document")
    id_token = tokens.get("id_token")
    return Creds(
        access_token=access,
        refresh_token=tokens.get("refresh_token"),
        id_token=id_token,
        account_id=tokens.get("account_id") or account_id_from_id_token(id_token),
        source=source,
        raw=doc,
    )


def _try_keyring() -> Creds | None:
    """Best-effort read of the Codex CLI's OS-keyring credential."""
    try:
        import keyring  # type: ignore[import-untyped]
    except Exception:
        return None
    codex_home = os.path.realpath(str(Path.home() / ".codex"))
    key = "cli|" + hashlib.sha256(codex_home.encode()).hexdigest()[:16]
    try:
        secret = keyring.get_password("Codex Auth", key)
    except Exception:
        return None
    if not secret:
        return None
    try:
        return creds_from_doc(json.loads(secret), source="keyring")
    except Exception:
        return None


def _write_creds(creds: Creds) -> None:
    """Persist tokens to :data:`CREDENTIALS_PATH` with owner-only permissions."""
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": creds.access_token,
                    "refresh_token": creds.refresh_token,
                    "id_token": creds.id_token,
                    "account_id": creds.account_id,
                },
                "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    CREDENTIALS_PATH.chmod(0o600)


def refresh(creds: Creds) -> Creds:
    """Exchange the refresh token for a fresh access token (JSON body)."""
    if not creds.refresh_token:
        return creds
    resp = requests.post(
        TOKEN_URL,
        json={
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": creds.refresh_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    id_token = data.get("id_token") or creds.id_token
    new = Creds(
        access_token=data.get("access_token") or creds.access_token,
        refresh_token=data.get("refresh_token") or creds.refresh_token,
        id_token=id_token,
        account_id=creds.account_id or account_id_from_id_token(id_token),
        source=creds.source,
        raw=creds.raw,
    )
    # Persist only to our own file; never rewrite the Codex CLI's auth.json.
    if creds.source == str(CREDENTIALS_PATH):
        _write_creds(new)
    return new


def load_credentials() -> Creds:
    """Load subscription credentials, refreshing when near expiry.

    Search order: ``~/.langslice/openai_auth.json``, the Codex CLI's
    ``~/.codex/auth.json``, then the Codex keyring entry.
    """
    creds: Creds | None = None
    for path in (CREDENTIALS_PATH, _CODEX_AUTH):
        if path.exists():
            try:
                creds = creds_from_doc(json.loads(path.read_text()), source=str(path))
                break
            except Exception:
                continue
    if creds is None:
        creds = _try_keyring()
    if creds is None:
        raise RuntimeError(
            "No ChatGPT-subscription credentials found. Run `langslice login` "
            f"(writes {CREDENTIALS_PATH}) or `codex login`, then retry."
        )

    expiry = _token_expiry(creds.access_token)
    if expiry is not None and expiry <= time.time() + REFRESH_SKEW_S:
        creds = refresh(creds)
    return creds


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


def image_data_uri(path: str | Path, mime: str = "image/png") -> str:
    """Return a ``data:`` URI for an image file on disk."""
    return data_uri(Path(path).read_bytes(), mime)


def user_message(text: str | None, image_uris: Sequence[str] = ()) -> dict[str, Any]:
    """Build a Responses ``input`` user message.

    ``image_url`` is a bare data-URI string here — the Responses input shape,
    not the chat-completions ``{"url": ...}`` object.
    """
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "input_text", "text": text})
    content.extend({"type": "input_image", "image_url": uri} for uri in image_uris)
    return {"type": "message", "role": "user", "content": content}


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
    """POST a Responses request and yield its SSE events, refreshing on 401.

    The first event is a synthetic ``langslice.quota`` carrying any quota
    headers the backend sent. *session_id* rides in the headers; keep it
    stable across one agent loop so the backend routes to a warm cache.
    """
    creds = load_credentials()
    session_id = session_id or str(uuid.uuid4())
    response = requests.post(
        RESPONSES_URL, headers=_headers(creds, session_id), json=body, stream=True, timeout=600
    )
    if response.status_code == 401 and creds.refresh_token:
        response.close()
        creds = refresh(creds)
        response = requests.post(
            RESPONSES_URL,
            headers=_headers(creds, session_id),
            json=body,
            stream=True,
            timeout=600,
        )
    if response.status_code >= 400:
        message = response.text[:500]
        quota = quota_from_headers(response.headers)
        response.close()
        raise RuntimeError(
            f"Codex Responses request failed ({response.status_code}): {message}"
            + (f" quota={quota}" if quota else "")
        )
    quota = quota_from_headers(response.headers)
    return itertools.chain([{"type": "langslice.quota", "quota": quota}], _closing(response))


def _closing(response: requests.Response) -> Iterator[dict[str, Any]]:
    try:
        yield from iter_sse(response)
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

    There is NO mainline routing model in this path: ``prompt`` goes into the
    request body verbatim and reaches the image model untouched. Same OAuth
    credential; the endpoint mirrors the public ``images.edit`` shape (used by
    the Codex CLI's own ``imagegenext``), but it is undocumented and carries
    no compatibility guarantee — on breakage, use the hosted
    ``image_generation`` tool on a Responses request instead.
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
    tool_media: list[str] = []
    tool_names: list[str] = []

    for part in content.parts or []:
        if part.function_response is not None:
            response = part.function_response
            payload = response.response
            output = payload if isinstance(payload, str) else _json_dumps(payload)
            # A tool can attach media to its result. A function_call_output
            # carries text only, so the media follows as its own user message,
            # and the output says so: GPT-6 Astra read a result that said
            # "attached" and found no image in it (2026-09-06 debrief).
            uris = [
                uri
                for response_part in response.parts or []
                if (uri := _part_data_uri(response_part.inline_data))
            ]
            if uris:
                name = response.name or "tool"
                output += (
                    f"\n\n[{len(uris)} image(s) from this {name} call follow "
                    "in the next user message, each preceded by "
                    f"'{name} image k of {len(uris)}'.]"
                )
                tool_media.extend(uris)
                tool_names.extend([name] * len(uris))
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": response.id or "",
                    "output": output,
                }
            )
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

    if tool_media:
        label = ", ".join(dict.fromkeys(tool_names))
        # One user message, but each image gets its own text item right before
        # it: without that the model sees N anonymous attachments and cannot
        # tell which tool call, or which argument, any one of them answers.
        media_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": f"Image(s) returned by tool: {label}."}
        ]
        total = len(tool_media)
        pairs = enumerate(zip(tool_media, tool_names, strict=True), start=1)
        for index, (uri, name) in pairs:
            media_content.append(
                {"type": "input_text", "text": f"{name} image {index} of {total}"}
            )
            media_content.append({"type": "input_image", "image_url": uri})
        items.append({"type": "message", "role": "user", "content": media_content})
    return items


def _json_schema_dict(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    if declaration.parameters_json_schema is not None:
        schema = declaration.parameters_json_schema
        return dict(schema) if isinstance(schema, dict) else {"type": "object", "properties": {}}
    if declaration.parameters is not None:
        dumped = declaration.parameters.json_schema.model_dump(
            mode="json", exclude_none=True, by_alias=True
        )
        return dumped
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


class ChatGptLlm(BaseLlm):
    """ADK model backed by a ChatGPT subscription (Codex Responses backend).

    Registered for ``openai-oauth/<model>`` strings (legacy ``chatgpt/<model>``
    accepted), e.g. ``openai-oauth/gpt-5.6-luna``.
    Supports vision input, function calling, and media returned by tools.
    """

    reasoning_effort: str = "medium"
    """Codex reasoning effort: none | minimal | low | medium | high."""

    prompt_cache_key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Stable across the turns of one agent loop, for upstream prompt caching."""

    @field_validator("model")
    @classmethod
    def _strip_prefix(cls, value: str) -> str:
        for prefix in (MODEL_PREFIX, LEGACY_MODEL_PREFIX):
            if value.startswith(prefix):
                return value[len(prefix):]
        return value

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"openai-oauth/.*", r"chatgpt/.*"]

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
            "parallel_tool_calls": False,
            "store": False,
            "stream": True,
            "prompt_cache_key": self.prompt_cache_key,
            "reasoning": {"effort": self.reasoning_effort, "summary": "auto"},
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
        events = await asyncio.to_thread(
            stream_events, body, session_id=self.prompt_cache_key
        )

        text_chunks: list[str] = []
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
            elif kind == "response.output_item.done":
                item = event.get("item", {})
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
                break

        parts: list[types.Part] = []
        text = "".join(text_chunks)
        if text:
            parts.append(types.Part.from_text(text=text))
        parts.extend(calls)
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            usage_metadata=usage,
            custom_metadata={"quota": quota} if quota else None,
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


LLMRegistry.register(ChatGptLlm)


# --- "Sign in with ChatGPT" (PKCE OAuth) -------------------------------------
def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _callback_handler(result: dict[str, str]) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.startswith("/auth/callback"):
                self.send_response(404)
                self.end_headers()
                return
            result.update(
                {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            message = (
                "Signed in. You can close this tab."
                if "code" in result
                else "Sign-in failed."
            )
            self.wfile.write(f"<html><body><h2>{message}</h2></body></html>".encode())

        def log_message(self, *args: object) -> None:
            pass

    return Handler


def login(timeout_s: float = 300.0, open_browser: bool = True) -> Path:
    """Run the "Sign in with ChatGPT" PKCE flow and store the token.

    Returns the path the credentials were written to.
    """
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    url = AUTHORIZE_URL + "?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": _REDIRECT_URI,
            "scope": _LOGIN_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": ORIGINATOR,
        }
    )

    result: dict[str, str] = {}
    server = http.server.HTTPServer(("127.0.0.1", _LOGIN_PORT), _callback_handler(result))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        print("Opening browser to sign in with ChatGPT...")
        print(f"If it doesn't open, visit:\n  {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        deadline = time.time() + timeout_s
        while "code" not in result and time.time() < deadline:
            if "error" in result:
                raise RuntimeError(f"OAuth error: {result.get('error')}")
            time.sleep(0.5)
    finally:
        server.shutdown()
        server.server_close()

    if "code" not in result:
        raise RuntimeError("Timed out waiting for sign-in.")
    if result.get("state") != state:
        raise RuntimeError("OAuth state mismatch - aborting.")

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": result["code"],
            "redirect_uri": _REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    id_token = data.get("id_token")
    _write_creds(
        Creds(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            id_token=id_token,
            account_id=account_id_from_id_token(id_token),
            source=str(CREDENTIALS_PATH),
        )
    )
    return CREDENTIALS_PATH
