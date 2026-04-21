import asyncio

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from langslice.harness.estimation.adk_plugins import (
    RequestCapturePlugin,
    _coerce_part_history,
)


def test_coerce_part_history_restores_serialized_parts():
    part = types.Part.from_bytes(mime_type="image/jpeg", data=b"fake-jpeg")
    serialized = part.model_dump(by_alias=True)

    coerced = _coerce_part_history([serialized])

    assert len(coerced) == 1
    assert isinstance(coerced[0], types.Part)
    assert coerced[0].inline_data is not None
    assert coerced[0].inline_data.data == b"fake-jpeg"


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
