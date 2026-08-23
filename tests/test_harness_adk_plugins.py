import asyncio

from google.adk.flows.llm_flows.functions import (
    _build_function_response_content,
    _extract_multimodal_parts,
)
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.adk.plugins import ModelCallPacingPlugin, RequestCapturePlugin


class _FakeTool:
    name = "fetch_atlas"
    response_scheduling = None


def test_langslice_tool_shape_reaches_adk_native_media_extraction():
    """Pin our tool return shape against ADK's real extraction.

    ADK only looks one container deep, so a flat list of Parts under a dict key
    is the deepest nesting that works. If a future ADK release changes that
    rule, this fails instead of silently sending the model image-free tool
    results.
    """
    image_parts = [
        types.Part.from_bytes(mime_type="image/jpeg", data=b"jpeg-one"),
        types.Part.from_bytes(mime_type="image/jpeg", data=b"jpeg-two"),
    ]
    tool_result = {
        "status": "ok",
        "positions_mm": [4.0, 6.0],
        "atlas_keys": ["atlas:4.00", "atlas:6.00"],
        "description": "Fetched 2 atlas sections",
        TOOL_MEDIA_PARTS_KEY: image_parts,
    }

    remaining, response_parts = _extract_multimodal_parts(tool_result)

    assert response_parts is not None
    assert [part.inline_data.data for part in response_parts if part.inline_data] == [
        b"jpeg-one",
        b"jpeg-two",
    ]
    # The media key is consumed entirely; the model reads clean JSON.
    assert remaining == {
        "status": "ok",
        "positions_mm": [4.0, 6.0],
        "atlas_keys": ["atlas:4.00", "atlas:6.00"],
        "description": "Fetched 2 atlas sections",
    }


def test_native_media_parts_ride_on_the_function_response_part():
    image_part = types.Part.from_bytes(mime_type="image/jpeg", data=b"jpeg-one")
    content = _build_function_response_content(
        _FakeTool(),  # type: ignore[arg-type]
        {"status": "ok", TOOL_MEDIA_PARTS_KEY: [image_part]},
        "call-1",
    )

    parts = content.parts or []
    # One function_response part carrying the image; no sibling image part, so
    # nothing double-injects it.
    assert len(parts) == 1
    function_response = parts[0].function_response
    assert function_response is not None
    assert function_response.response == {"status": "ok"}
    media = function_response.parts or []
    assert len(media) == 1
    assert media[0].inline_data is not None
    assert media[0].inline_data.data == b"jpeg-one"


def test_text_parts_would_leak_into_the_json_result():
    """Why tool text lives in JSON fields, not in Parts: ADK keeps text Parts."""
    remaining, response_parts = _extract_multimodal_parts(
        {"status": "ok", TOOL_MEDIA_PARTS_KEY: [types.Part.from_text(text="Atlas at 4.00 mm:")]}
    )

    assert response_parts is None
    assert isinstance(remaining, dict)
    assert TOOL_MEDIA_PARTS_KEY in remaining


def test_model_call_pacing_plugin_accepts_zero_delay():
    plugin = ModelCallPacingPlugin(0)
    request = LlmRequest(model="capture-model", contents=[])

    result = asyncio.run(
        plugin.before_model_callback(
            callback_context=None,  # type: ignore[arg-type]
            llm_request=request,
        )
    )

    assert result is None


def test_request_capture_plugin_redacts_inline_image_bytes(tmp_path):
    part = types.Part.from_bytes(mime_type="image/jpeg", data=b"not-a-real-jpeg")
    request = LlmRequest(
        model="capture-model",
        contents=[types.Content(role="user", parts=[part])],
    )
    plugin = RequestCapturePlugin(tmp_path, run_label="unit")

    asyncio.run(
        plugin.before_model_callback(callback_context=None, llm_request=request)  # type: ignore[arg-type]
    )

    captures = list(tmp_path.glob("unit_*.json"))
    assert len(captures) == 1
    text = captures[0].read_text(encoding="utf-8")
    assert "not-a-real-jpeg" not in text
    assert "inline_data" in text
    assert "byte_count" in text
    assert "image/jpeg" in text


def test_request_capture_plugin_counts_function_response_media(tmp_path):
    content = _build_function_response_content(
        _FakeTool(),  # type: ignore[arg-type]
        {
            "status": "ok",
            TOOL_MEDIA_PARTS_KEY: [
                types.Part.from_bytes(mime_type="image/jpeg", data=b"jpeg-one")
            ],
        },
        "call-1",
    )
    request = LlmRequest(model="capture-model", contents=[content])
    plugin = RequestCapturePlugin(tmp_path, run_label="fr")

    asyncio.run(
        plugin.before_model_callback(callback_context=None, llm_request=request)  # type: ignore[arg-type]
    )

    text = next(iter(tmp_path.glob("fr_*.json"))).read_text(encoding="utf-8")
    assert '"media_part_count": 1' in text
    assert "jpeg-one" not in text
