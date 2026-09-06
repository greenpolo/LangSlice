import asyncio

from google.adk.flows.llm_flows.functions import (
    _build_function_response_content,
    _extract_multimodal_parts,
)
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.adk.plugins import (
    DEFAULT_KEEP_LAST_TOOL_IMAGES,
    ModelCallPacingPlugin,
    RequestCapturePlugin,
    trim_stale_tool_images,
)


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


def _media_part(n: int) -> types.Part:
    frp = [
        types.FunctionResponsePart(
            inline_data=types.Blob(mime_type="image/jpeg", data=b"x" * 8)
        )
        for _ in range(n)
    ]
    return types.Part(
        function_response=types.FunctionResponse(
            name="fetch_atlas", response={"status": "ok"}, parts=frp
        )
    )


def _tool_turn(n_images: int) -> types.Content:
    return types.Content(role="user", parts=[_media_part(n_images)])


def test_trim_keeps_newest_tool_images_and_user_images():
    histology = types.Content(
        role="user",
        parts=[types.Part.from_bytes(mime_type="image/jpeg", data=b"slice")],
    )
    contents = [histology, _tool_turn(6), _tool_turn(6), _tool_turn(6)]
    out = trim_stale_tool_images(contents, keep_last=12)

    def n_media(c: types.Content) -> int:
        return sum(
            len(p.function_response.parts or [])
            for p in (c.parts or [])
            if p.function_response is not None
        )

    # Oldest tool call stripped, newest two kept, histology untouched.
    assert [n_media(c) for c in out[1:]] == [0, 6, 6]
    assert out[0].parts is not None and out[0].parts[0].inline_data is not None
    # JSON result survives on the stripped call, and says its images are gone.
    stripped = out[1].parts
    assert stripped is not None
    fr = stripped[0].function_response
    assert fr is not None and fr.response["status"] == "ok"
    assert "dropped from context" in fr.response["images"]
    assert contents[1].parts[0].function_response.response == {"status": "ok"}
    # Input list and its contents are not mutated.
    assert sum(n_media(c) for c in contents[1:]) == 18


def _kept(contents: list[types.Content]) -> list[int]:
    return [
        len(p.function_response.parts or [])
        for c in contents
        for p in (c.parts or [])
        if p.function_response is not None
    ]


def test_trim_noop_under_budget_and_always_keeps_newest():
    contents = [_tool_turn(5), _tool_turn(5)]
    assert trim_stale_tool_images(contents, keep_last=12) is contents
    # A single oversized newest call is still kept in full.
    big = [_tool_turn(6), _tool_turn(6), _tool_turn(20)]
    assert _kept(trim_stale_tool_images(big, keep_last=12)) == [0, 6, 20]


def test_trim_always_keeps_the_two_newest_calls_over_budget():
    """Two full sweeps must stay comparable however big they are.

    The 12-image budget plus a one-call floor left an agent with only its
    newest sweep in view: it could not compare candidate levels at all.
    """
    contents = [_tool_turn(8), _tool_turn(8), _tool_turn(8)]

    out = trim_stale_tool_images(contents, keep_last=4)

    assert _kept(out) == [0, 8, 8]


def test_trim_default_budget_holds_a_whole_positioning_run():
    """Two 8-image sweeps plus five 16-image set_positions batches: 96, kept.

    At 24 that run kept only its newest two calls, and GPT-6 Astra reported
    the older results as "attached but no image" in two debriefs.
    """
    assert DEFAULT_KEEP_LAST_TOOL_IMAGES == 128
    contents = [_tool_turn(8), _tool_turn(8), *(_tool_turn(16) for _ in range(5))]
    assert trim_stale_tool_images(contents) is contents
    # Three more batches push past 128: the oldest calls go, newest stay.
    more = [*contents, *(_tool_turn(16) for _ in range(3))]
    assert _kept(trim_stale_tool_images(more)) == [0, 0, 16, 16, 16, 16, 16, 16, 16, 16]


def test_trim_drop_is_monotone_no_holes():
    # Once an old call is dropped, even older small calls are dropped too.
    contents = [_tool_turn(2), _tool_turn(2), _tool_turn(5), _tool_turn(10)]
    assert _kept(trim_stale_tool_images(contents, keep_last=12)) == [0, 0, 5, 10]
