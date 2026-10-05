"""Usage telemetry is optional, content-free and observational."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from langslice.agent.session import run_agent_session
from langslice.agent.trace import SessionTrace
from langslice.providers import openai_oauth
from langslice.providers.usage import item_descriptor, request_descriptor, usage_diagnostics


def counters(inp=0, cached=0, writes=0, out=0):
    return dict(input_tokens=inp, cached_tokens=cached, cache_write_tokens=writes,
                output_tokens=out)


def test_attribution_reconciles_without_double_counting_children_or_guessing_ids():
    body = {"input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "secret text"},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,secret-image"},
    ]}, {"type": "reasoning", "id": "rs_old", "encrypted_content": "secret-reasoning"}],
        "instructions": "secret instructions", "tools": []}
    original = copy.deepcopy(body)
    raw = {"input_tokens": 20, "input_tokens_details": {"cached_tokens": 5,
           "cache_write_tokens": 0}, "output_tokens": 4, "attribution": {
        "items": {
            "msg_backend": {**counters(10, 5), "content": [counters(2), counters(8, 5)]},
            "rs_old": counters(3),
            "fc_new": counters(2, out=4),
        },
        "request_fields": {"instructions": counters(5)},
        "unknown": "secret extension",
    }}
    output = item_descriptor({"type": "function_call", "id": "fc_new",
                              "arguments": "secret arguments"})
    report = usage_diagnostics(raw, request_descriptor(body), [output])
    rows = report["attribution"]["items"]
    assert [row["mapping"] for row in rows] == [
        "unmapped", "exact_request_id", "exact_output_id",
    ]
    assert rows[0]["content_residual"] == counters()
    assert report["attribution_residual"] == counters()
    assert report["attribution_reconciled"]
    assert "secret" not in json.dumps(report)
    assert "msg_backend" not in json.dumps(report)
    assert body == original


def test_tool_images_are_described_from_output_and_hashes_detect_changed_content():
    item = {"type": "function_call_output", "call_id": "tool_1", "output": [
        {"type": "input_text", "text": "private"},
        {"type": "input_image", "image_url": "private image"},
    ]}
    first = item_descriptor(item)
    assert [part["type"] for part in first["content"]] == ["input_text", "input_image"]
    item["output"][1]["image_url"] = "different image"
    second = item_descriptor(item)
    assert first["content"][0] == second["content"][0]
    assert first["content"][1]["fingerprint"] != second["content"][1]["fingerprint"]


@pytest.mark.parametrize("bad", [None, -1, True, "0", 1.5])
def test_missing_or_invalid_counts_are_unknown_not_zero(bad):
    report = usage_diagnostics({"input_tokens_details": {"cache_write_tokens": bad}})
    assert report["totals"]["cache_write_tokens"] is None
    assert report["totals"]["cached_tokens"] is None
    assert "request" not in report


def test_incomplete_attribution_and_residuals_remain_visible():
    request = request_descriptor({"input": []})
    assert usage_diagnostics({}, request)["attribution_status"] == "unavailable"
    assert usage_diagnostics({"attribution": {}}, request)["attribution_status"] == "malformed"
    report = usage_diagnostics({"input_tokens": 5, "attribution": {
        "items": {}, "request_fields": {},
    }}, request)
    assert report["attribution_residual"]["input_tokens"] == 5
    assert report["attribution_residual"]["cached_tokens"] is None
    assert report["attribution_reconciled"] is False


@pytest.mark.parametrize("capture", [False, True])
def test_provider_diagnostics_do_not_change_request_or_model_content(monkeypatch, capture):
    bodies = []
    def fake_stream(body, **kwargs):
        bodies.append(copy.deepcopy(body))
        return iter([{"type": "response.completed", "response": {"usage": {
            "input_tokens": 10, "input_tokens_details": {"cached_tokens": 2,
            "cache_write_tokens": 0}, "output_tokens": 0,
        }}}])
    monkeypatch.setattr(openai_oauth, "stream_events", fake_stream)
    model = openai_oauth.OpenAIOAuthLlm(model="openai-oauth/test", capture_usage_details=capture)
    request = LlmRequest(contents=[types.Content(role="user", parts=[
        types.Part.from_text(text="private"),
    ])])
    before = model.build_request_body(request)
    async def collect():
        return [r async for r in model.generate_content_async(request)]
    response, = asyncio.run(collect())
    assert bodies == [before]
    assert not response.content.parts
    diagnostics = response.custom_metadata["usage_diagnostics"]
    assert diagnostics["totals"]["cache_write_tokens"] == 0
    assert ("request" in diagnostics) is capture
    assert "private" not in json.dumps(diagnostics)


@pytest.mark.parametrize("writes", [None, 0, 3])
def test_trace_preserves_usage_only_responses_and_optional_cache_writes(tmp_path, writes):
    agent = SimpleNamespace(name="test", model="fake", instruction="test")
    trace = SessionTrace(tmp_path, "usage", agent=agent)
    event = SimpleNamespace(content=None, usage_metadata=SimpleNamespace(
        prompt_token_count=10, cached_content_token_count=2, candidates_token_count=0,
    ), custom_metadata={"usage_diagnostics": usage_diagnostics({
        "input_tokens_details": {"cache_write_tokens": writes},
    })})
    trace.event(event, turn=1)
    row = json.loads(trace.path.read_text().splitlines()[-1])
    assert row["kind"] == "model" and row["step"] == 1
    assert row["usage_diagnostics"]["totals"]["cache_write_tokens"] == writes
    assert ("cache_write_tokens" in row["usage"]) is (writes is not None)


@pytest.mark.parametrize("traced", [False, True])
def test_session_enables_optional_diagnostics_only_when_traced(tmp_path, monkeypatch, traced):
    monkeypatch.delenv("LANGSLICE_TRACE_DIR", raising=False)
    if traced:
        monkeypatch.setenv("LANGSLICE_TRACE_DIR", str(tmp_path))
    model = openai_oauth.OpenAIOAuthLlm(model="openai-oauth/test")
    agent = LlmAgent(name="test", model=model)
    # Already done: test session setup without invoking any provider.
    asyncio.run(run_agent_session(
        agent=agent, run_label="usage_test", max_iterations=1, done=lambda: True,
        nudge_no_tool="continue", nudge_continue="continue",
        seed_message=types.Content(role="user", parts=[types.Part.from_text(text="test")]),
    ))
    assert model.capture_usage_details is traced


def test_diagnostics_survive_the_adk_session_to_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSLICE_TRACE_DIR", str(tmp_path))
    sent = []
    def fake_stream(body, **kwargs):
        sent.append(True)
        return iter([
            {"type": "response.output_text.delta", "delta": "done"},
            {"type": "response.completed", "response": {"usage": {
                "input_tokens": 10, "input_tokens_details": {"cached_tokens": 2,
                "cache_write_tokens": 0}, "output_tokens": 1,
            }}},
        ])
    monkeypatch.setattr(openai_oauth, "stream_events", fake_stream)
    agent = LlmAgent(name="test", model=openai_oauth.OpenAIOAuthLlm(model="openai-oauth/test"))
    asyncio.run(run_agent_session(
        agent=agent, run_label="usage_test", max_iterations=1, done=lambda: bool(sent),
        nudge_no_tool="continue", nudge_continue="continue",
        seed_message=types.Content(role="user", parts=[types.Part.from_text(text="test")]),
    ))
    path, = tmp_path.glob("*.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    model, = [row for row in rows if row["kind"] == "model"]
    assert model["usage"]["cache_write_tokens"] == 0
    assert model["usage_diagnostics"]["request"]["items"]
