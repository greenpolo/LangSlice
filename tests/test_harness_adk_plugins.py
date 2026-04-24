import asyncio
from typing import Any, cast

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from langslice_harness.harness.estimation.adk_plugins import (
    _MULTIMODAL_TOOL_PARTS_BY_CALL_KEY,
    MULTIMODAL_PARTS_RESULT_KEY,
    ModelCallPacingPlugin,
    PersistentMultimodalToolResultsPlugin,
    RequestCapturePlugin,
    _coerce_part_history,
    _coerce_part_map,
)


def test_coerce_part_history_restores_serialized_parts():
    part = types.Part.from_bytes(mime_type="image/jpeg", data=b"fake-jpeg")
    serialized = part.model_dump(by_alias=True)

    coerced = _coerce_part_history([serialized])

    assert len(coerced) == 1
    assert isinstance(coerced[0], types.Part)
    assert coerced[0].inline_data is not None
    assert coerced[0].inline_data.data == b"fake-jpeg"


def test_coerce_part_map_restores_serialized_parts():
    part = types.Part.from_bytes(mime_type="image/jpeg", data=b"fake-jpeg")
    serialized = part.model_dump(by_alias=True)

    coerced = _coerce_part_map({"call-1": [serialized]})

    assert list(coerced) == ["call-1"]
    assert coerced["call-1"][0].inline_data is not None


def test_persistent_plugin_strips_private_parts_and_injects_after_function_response():
    image_part = types.Part.from_bytes(mime_type="image/jpeg", data=b"fake-jpeg")
    text_part = types.Part.from_text(text="Atlas at 5.00 mm:")
    plugin = PersistentMultimodalToolResultsPlugin()
    tool = cast(Any, type("Tool", (), {"name": "fetch_atlas"})())
    tool_context = cast(Any, type("ToolContext", (), {
        "state": {},
        "function_call_id": "call-1",
    })())

    public = asyncio.run(
        plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=tool_context,
            result={
                "status": "ok",
                "positions_mm": [5.0],
                MULTIMODAL_PARTS_RESULT_KEY: [text_part, image_part],
            },
        )
    )

    assert public == {"status": "ok", "positions_mm": [5.0]}
    assert _MULTIMODAL_TOOL_PARTS_BY_CALL_KEY in tool_context.state

    request = LlmRequest(
        model="capture-model",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="fetch_atlas",
                        response={"status": "ok", "positions_mm": [5.0]},
                    )
                ],
            )
        ],
    )
    parts = request.contents[0].parts
    assert parts is not None
    function_response = parts[0].function_response
    assert function_response is not None
    function_response.id = "call-1"
    callback_context = cast(Any, type("CallbackContext", (), {"state": tool_context.state})())

    asyncio.run(
        plugin.before_model_callback(
            callback_context=callback_context,  # type: ignore[arg-type]
            llm_request=request,
        )
    )

    injected = request.contents[0].parts or []
    assert [part.function_response is not None for part in injected] == [True, False, False]
    assert injected[1].text == "Atlas at 5.00 mm:"
    assert injected[2].inline_data is not None


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
