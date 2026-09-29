"""Offline tests for the ChatGPT-subscription backend (no network)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from PIL import Image

from langslice.providers import openai_oauth as chatgpt


# --- helpers -----------------------------------------------------------------
def _fake_jwt(claims: dict[str, Any]) -> str:
    def part(payload: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


def _id_token(account_id: str = "acct_9", plan: str = "plus") -> str:
    return _fake_jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account_id,
                "chatgpt_plan_type": plan,
            }
        }
    )


class _FakeResponse:
    """Stand-in for requests.Response streaming bytes lines (no charset)."""

    status_code = 200
    text = ""

    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode() for line in lines]

    def iter_lines(self) -> Any:
        return iter(self._lines)

    def close(self) -> None:
        pass


def _sse(events: list[dict[str, Any]]) -> list[str]:
    return [f"data: {json.dumps(event)}" for event in events] + ["data: [DONE]"]


# --- credentials -------------------------------------------------------------
def test_jwt_claims_and_account_id():
    token = _id_token("acct_123", "pro")
    assert chatgpt.account_id_from_id_token(token) == "acct_123"
    assert chatgpt.plan_from_id_token(token) == "pro"
    assert chatgpt.account_id_from_id_token(None) is None
    assert chatgpt.account_id_from_id_token("garbage") is None


def test_creds_from_doc_nested_and_flat():
    nested = {"tokens": {"access_token": "A", "refresh_token": "R", "id_token": _id_token()}}
    creds = chatgpt.creds_from_doc(nested, "test")
    assert (creds.access_token, creds.refresh_token) == ("A", "R")
    assert creds.account_id == "acct_9"  # derived from id_token

    flat = {"access_token": "A2", "account_id": "acct_explicit"}
    assert chatgpt.creds_from_doc(flat, "test").account_id == "acct_explicit"

    with pytest.raises(ValueError):
        chatgpt.creds_from_doc({"tokens": {}}, "test")


def test_refresh_persists_only_to_our_own_file(tmp_path, monkeypatch):
    creds_path = tmp_path / "openai_auth.json"
    monkeypatch.setattr(chatgpt, "CREDENTIALS_PATH", creds_path)

    posted: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any):
        posted["url"] = url
        posted["json"] = kwargs.get("json")
        return type(
            "R",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"access_token": "NEW", "refresh_token": "NEWR"},
            },
        )()

    monkeypatch.setattr(chatgpt.requests, "post", fake_post)

    codex = chatgpt.Creds("OLD", "R", None, "acct_1", source="/home/x/.codex/auth.json")
    refreshed = chatgpt.refresh(codex)
    assert refreshed.access_token == "NEW"
    assert not creds_path.exists()  # never rewrite the Codex CLI's file
    assert posted["url"] == chatgpt.TOKEN_URL
    assert posted["json"]["grant_type"] == "refresh_token"
    assert posted["json"]["client_id"] == chatgpt.CLIENT_ID

    ours = chatgpt.Creds("OLD", "R", None, "acct_1", source=str(creds_path))
    chatgpt.refresh(ours)
    saved = json.loads(creds_path.read_text())
    assert saved["tokens"]["access_token"] == "NEW"
    assert saved["tokens"]["refresh_token"] == "NEWR"
    assert creds_path.stat().st_mode & 0o777 == 0o600


def test_load_credentials_prefers_our_file_and_refreshes_when_expired(tmp_path, monkeypatch):
    creds_path = tmp_path / "openai_auth.json"
    expired = _fake_jwt({"exp": 0})
    creds_path.write_text(json.dumps({"tokens": {"access_token": expired, "refresh_token": "R"}}))
    monkeypatch.setattr(chatgpt, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(chatgpt, "refresh", lambda creds: chatgpt.Creds("FRESH"))

    assert chatgpt.load_credentials().access_token == "FRESH"


def test_load_credentials_without_any_source_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(chatgpt, "CREDENTIALS_PATH", tmp_path / "none.json")

    with pytest.raises(RuntimeError, match="langslice login"):
        chatgpt.load_credentials()


# --- wire shaping ------------------------------------------------------------
def test_headers():
    headers = chatgpt._headers(chatgpt.Creds("ACCESS", account_id="acct_5"), "sess-1")
    assert headers["Authorization"] == "Bearer ACCESS"
    assert headers["chatgpt-account-id"] == "acct_5"
    assert headers["OpenAI-Beta"] == "responses=experimental"
    assert headers["Accept"] == "text/event-stream"
    assert headers["originator"] == "langslice"
    assert headers["session_id"] == "sess-1"
    assert "chatgpt-account-id" not in chatgpt._headers(chatgpt.Creds("A"), "sess-1")


def test_user_message_image_url_is_a_bare_string():
    message = chatgpt.user_message("hello", ["data:image/png;base64,AAAA"])
    assert message["role"] == "user"
    text, image = message["content"]
    assert text == {"type": "input_text", "text": "hello"}
    # CRITICAL: Responses input uses a bare string here, not {"url": ...}
    assert image == {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}


def test_iter_sse_decodes_bytes_lines_and_stops_at_done():
    lines = _sse(
        [
            {"type": "response.output_text.delta", "delta": "Hel"},
            {"type": "response.output_text.delta", "delta": "lo"},
            {"type": "response.completed"},
        ]
    )
    lines.insert(1, "")  # blank keepalive line
    lines.append("data: " + json.dumps({"type": "never.read"}))
    events = list(chatgpt.iter_sse(_FakeResponse(lines)))
    assert [event["type"] for event in events] == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]


# --- ADK request conversion --------------------------------------------------
def _png_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_content_conversion_text_image_and_function_calls():
    user = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="describe"),
            types.Part(
                inline_data=types.Blob(data=_png_bytes(), mime_type="image/png")
            ),
        ],
    )
    items = chatgpt.content_to_input_items(user)
    assert len(items) == 1
    assert items[0]["role"] == "user"
    assert items[0]["content"][0] == {"type": "input_text", "text": "describe"}
    assert items[0]["content"][1]["type"] == "input_image"
    assert items[0]["content"][1]["image_url"].startswith("data:image/png;base64,")

    model = types.Content(
        role="model",
        parts=[
            types.Part.from_text(text="calling"),
            types.Part(
                function_call=types.FunctionCall(
                    id="call_1", name="get_slice", args={"position_mm": 5.0}
                )
            ),
        ],
    )
    items = chatgpt.content_to_input_items(model)
    assert items[0] == {
        "type": "function_call",
        "name": "get_slice",
        "arguments": json.dumps({"position_mm": 5.0}),
        "call_id": "call_1",
    }
    assert items[1] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "calling"}],
    }


def test_function_response_media_is_forwarded_as_input_image():
    response = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id="call_1",
                    name="get_slice",
                    response={"status": "ok"},
                    parts=[
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                data=_png_bytes(), mime_type="image/png"
                            )
                        )
                    ],
                )
            )
        ],
    )
    items = chatgpt.content_to_input_items(response)
    # ONE item: the images ride inside the tool result. A separate user
    # message would open a new turn and lose the replayed reasoning.
    assert len(items) == 1
    item = items[0]
    assert item["type"] == "function_call_output"
    assert item["call_id"] == "call_1"
    content = item["output"]
    assert content[0] == {"type": "input_text", "text": json.dumps({"status": "ok"})}
    assert content[1] == {"type": "input_text", "text": "get_slice image 1 of 1"}
    assert content[2]["type"] == "input_image"
    assert content[2]["detail"] == "high"
    assert content[2]["image_url"].startswith("data:image/png;base64,")


def test_each_tool_image_is_labelled_before_it():
    """N images from one tool arrive interleaved with 'image k of N' text."""
    blob = types.FunctionResponsePart(
        inline_data=types.FunctionResponseBlob(data=_png_bytes(), mime_type="image/png")
    )
    response = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id="call_1",
                    name="fetch_atlas",
                    response={"status": "ok"},
                    parts=[blob, blob, blob],
                )
            )
        ],
    )
    content = chatgpt.content_to_input_items(response)[0]["output"]
    assert [item["type"] for item in content] == [
        "input_text",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
    ]
    assert [item["text"] for item in content[1::2]] == [
        "fetch_atlas image 1 of 3",
        "fetch_atlas image 2 of 3",
        "fetch_atlas image 3 of 3",
    ]


def test_tools_to_wire_uses_flat_responses_function_shape():
    declaration = types.FunctionDeclaration(
        name="get_atlas_slice",
        description="Fetch an atlas slice",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"position_mm": types.Schema(type=types.Type.NUMBER)},
            required=["position_mm"],
        ),
    )
    config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[declaration])])
    wire = chatgpt.tools_to_wire(config)
    assert wire == [
        {
            "type": "function",
            "name": "get_atlas_slice",
            "description": "Fetch an atlas slice",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {"position_mm": {"type": "number"}},
                "required": ["position_mm"],
            },
        }
    ]


def test_build_request_body_carries_instructions_and_model():
    llm = chatgpt.ChatGptLlm(model="chatgpt/gpt-5.6-luna")
    assert llm.model == "gpt-5.6-luna"  # prefix stripped
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])],
        config=types.GenerateContentConfig(system_instruction="be terse"),
    )
    body = llm.build_request_body(request)
    assert body["model"] == "gpt-5.6-luna"
    assert body["instructions"] == "be terse"
    assert body["stream"] is True
    assert body["store"] is False
    assert body["reasoning"] == {
        "effort": "medium",
        "summary": "auto",
        "context": "all_turns",
    }
    assert body["input"][0]["content"][0]["text"] == "hi"


# --- streaming ---------------------------------------------------------------
def _run_turn(monkeypatch, events: list[dict[str, Any]], *, stream: bool):
    captured: dict[str, Any] = {}

    def fake_stream_events(body: dict[str, Any], **_kw: Any):
        captured["body"] = body
        return iter(events)

    monkeypatch.setattr(chatgpt, "stream_events", fake_stream_events)
    llm = chatgpt.ChatGptLlm(model="chatgpt/gpt-5.6-luna")
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="hi")])]
    )

    async def collect():
        return [response async for response in llm.generate_content_async(request, stream=stream)]

    return asyncio.run(collect()), captured


def test_streaming_yields_partials_then_aggregated_final(monkeypatch):
    events = [
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 7,
                    "input_tokens_details": {"cached_tokens": 5},
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 2},
                }
            },
        },
    ]
    responses, _ = _run_turn(monkeypatch, events, stream=True)
    assert [r.partial for r in responses] == [True, True, False]
    assert responses[-1].content is not None
    assert responses[-1].content.parts[0].text == "Hello"
    usage = responses[-1].usage_metadata
    assert usage.prompt_token_count == 7
    assert usage.cached_content_token_count == 5
    assert usage.candidates_token_count == 3
    assert usage.thoughts_token_count == 2


@pytest.mark.parametrize("stream", [False, True])
def test_summary_streaming_preserves_final_reasoning_and_ignores_raw_reasoning(monkeypatch, stream):
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Compare the sections."}],
        "encrypted_content": "opaque-replay-only",
    }
    events = [
        {"type": "response.reasoning_summary_text.delta", "delta": "Compare "},
        {"type": "response.reasoning_text.delta", "delta": "private reasoning"},
        {"type": "response.reasoning_summary_text.delta", "delta": "the sections."},
        {"type": "response.reasoning_summary_text.delta", "delta": ""},
        {"type": "response.output_item.done", "item": reasoning},
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "name": "probe", "arguments": "{}", "call_id": "c1"},
        },
        {"type": "response.completed"},
    ]
    responses, _ = _run_turn(monkeypatch, events, stream=stream)
    assert [r.partial for r in responses] == ([True, True, True, False] if stream else [False])
    if stream:
        summary_parts = [r.content.parts[0] for r in responses[:2]]
        assert "".join(part.text for part in summary_parts) == "Compare the sections."
        assert all(part.thought and part.thought_signature is None for part in summary_parts)
        assert responses[2].content.parts[0].function_call.name == "probe"
    final = responses[-1].content
    assert final.parts[0].thought and final.parts[0].text == "Compare the sections."
    assert chatgpt.content_to_input_items(final) == [
        reasoning,
        {"type": "function_call", "call_id": "c1", "name": "probe", "arguments": "{}"},
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_summary_part_and_item_boundaries_match_final_text_and_keep_replay(monkeypatch, stream):
    first = {
        "type": "reasoning", "id": "rs_1", "encrypted_content": "opaque-one",
        "summary": [
            {"type": "summary_text", "text": "First heading"},
            {"type": "summary_text", "text": "Second heading"},
        ],
    }
    second = {
        "type": "reasoning", "id": "rs_2", "encrypted_content": "opaque-two",
        "summary": [{"type": "summary_text", "text": "Third heading"}],
    }

    def delta(item_id, summary_index, text):
        return {
            "type": "response.reasoning_summary_text.delta", "delta": text,
            "item_id": item_id, "summary_index": summary_index,
        }

    events = [
        delta("rs_1", 0, "First "), delta("rs_1", 0, "heading"),
        delta("rs_1", 1, ""), delta("rs_1", 1, "Second heading"),
        {"type": "response.output_item.done", "item": first},
        # A new reasoning item resets summary_index, but is still a new paragraph.
        delta("rs_2", 0, "Third "), delta("rs_2", 0, "heading"),
        {"type": "response.output_item.done", "item": second},
        {"type": "response.completed"},
    ]
    responses, _ = _run_turn(monkeypatch, events, stream=stream)
    final = responses[-1].content
    expected = "First heading\nSecond heading\nThird heading"
    assert "".join(part.text for part in final.parts) == expected
    assert chatgpt.content_to_input_items(final) == [first, second]
    if stream:
        partials = [part for response in responses[:-1] for part in response.content.parts]
        assert "".join(part.text for part in partials) == expected
        assert all(part.thought and part.thought_signature is None for part in partials)
    else:
        assert len(responses) == 1


def test_non_streaming_yields_one_aggregated_response_with_function_call(monkeypatch):
    events = [
        {"type": "response.output_text.delta", "delta": "checking"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "get_atlas_slice",
                "arguments": '{"position_mm": 4.5}',
                "call_id": "call_abc",
            },
        },
        {"type": "response.completed"},
    ]
    responses, _ = _run_turn(monkeypatch, events, stream=False)
    assert len(responses) == 1
    parts = responses[0].content.parts
    assert parts[0].text == "checking"
    call = parts[1].function_call
    assert (call.name, call.id, call.args) == ("get_atlas_slice", "call_abc", {"position_mm": 4.5})


def test_response_failed_raises(monkeypatch):
    events = [{"type": "response.failed", "response": {"error": {"message": "quota"}}}]
    with pytest.raises(RuntimeError, match="quota"):
        _run_turn(monkeypatch, events, stream=False)


def test_function_result_round_trip_body(monkeypatch):
    """A follow-up turn sends the tool result back as function_call_output."""
    monkeypatch.setattr(
        chatgpt, "stream_events", lambda body, **_kw: iter([{"type": "response.completed"}])
    )
    llm = chatgpt.ChatGptLlm(model="chatgpt/gpt-5.6-luna")
    request = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="where am i")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(id="c1", name="probe", args={"x": 1})
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="c1", name="probe", response={"ok": True}
                        )
                    )
                ],
            ),
        ]
    )
    body = llm.build_request_body(request)
    assert [item["type"] for item in body["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert body["input"][2]["call_id"] == "c1"


# --- wiring ------------------------------------------------------------------
def test_model_resolver_routes_chatgpt_prefix():
    from langslice.adk import model_resolver

    model = model_resolver.resolve_adk_model("chatgpt/gpt-5.6-luna")
    assert isinstance(model, chatgpt.ChatGptLlm)
    assert model.model == "gpt-5.6-luna"

    with pytest.raises(ValueError):
        model_resolver.resolve_adk_model("chatgpt/")


def test_registry_resolves_chatgpt_models():
    from google.adk.models.registry import LLMRegistry

    assert LLMRegistry.resolve("chatgpt/gpt-5.6-luna") is chatgpt.ChatGptLlm


def test_nonlinear_provider_uses_direct_images_edit(monkeypatch):
    """The registration path calls the direct edits endpoint: one GPT model
    (the pilot), one image model — no routing model rewriting the prompt."""
    from langslice.nonlinear import providers

    captured: dict[str, Any] = {}

    def fake_edit_image(prompt: str, references, **kwargs: Any) -> bytes:
        captured["prompt"] = prompt
        captured["references"] = list(references)
        captured.update(kwargs)
        return _png_bytes()

    monkeypatch.setattr(chatgpt, "edit_image", fake_edit_image)

    request = providers.SegmentationGenerationRequest(
        slice_image=Image.new("RGB", (30, 20)),
        reference_images=[Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))],
        prompt="warp it",
        provider="chatgpt",
    )
    generated = providers.generate_warped_segmentation_image(request)

    assert generated.provider == "openai-oauth"
    assert generated.route == "openai_oauth_images_edit"
    assert generated.model == "gpt-image-2"
    assert generated.revised_prompt is None  # nothing rewrites on this path
    assert captured["prompt"] == "warp it"  # delivered verbatim
    assert len(captured["references"]) == 3
    assert all(uri.startswith("data:image/png;base64,") for uri in captured["references"])
    assert captured["quality"] == "high"
    assert "size" not in captured  # the backend ignores it; we don't send it


def test_user_content_preserves_text_image_interleaving():
    from google.genai import types as gt

    from langslice.providers.openai_oauth import content_to_input_items

    png = b"\x89PNG\r\n\x1a\nfakebytes"
    content = gt.Content(
        role="user",
        parts=[
            gt.Part.from_text(text="0: a.tif"),
            gt.Part.from_bytes(mime_type="image/png", data=png),
            gt.Part.from_text(text="1: b.tif"),
            gt.Part.from_bytes(mime_type="image/png", data=png),
        ],
    )
    (item,) = content_to_input_items(content)
    kinds = [c["type"] for c in item["content"]]
    assert kinds == ["input_text", "input_image", "input_text", "input_image"]
    assert item["content"][0]["text"] == "0: a.tif"
    assert item["content"][2]["text"] == "1: b.tif"


def test_reasoning_items_are_kept_and_replayed_ahead_of_the_turn(monkeypatch):
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "compare 4.6 and 4.8"}],
        "encrypted_content": "opaque",
    }
    events = [
        {"type": "response.output_item.done", "item": reasoning},
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "name": "probe", "arguments": "{}", "call_id": "c1"},
        },
        {"type": "response.completed"},
    ]
    responses, _ = _run_turn(monkeypatch, events, stream=False)
    turn = responses[0].content
    assert turn.parts[0].thought and turn.parts[0].text == "compare 4.6 and 4.8"
    assert turn.parts[1].function_call.name == "probe"

    items = chatgpt.content_to_input_items(turn)
    assert [item["type"] for item in items] == ["reasoning", "function_call"]
    assert items[0] == reasoning


@pytest.mark.parametrize("detail,expected", [("original", "original"), ("invalid", "high")])
def test_coordinate_tool_images_preserve_requested_detail(detail, expected):
    response = types.FunctionResponse(
        id="coordinates", name="point_view", response={"image_detail": detail},
        parts=[types.FunctionResponsePart(inline_data=types.FunctionResponseBlob(
            data=_png_bytes(), mime_type="image/png"))],
    )
    output = chatgpt._function_call_output(response)["output"]
    image = next(item for item in output if item["type"] == "input_image")
    assert image["detail"] == expected
    assert image["image_url"].startswith("data:image/png;base64,")
