import asyncio

from google.adk.flows.llm_flows.functions import (
    _build_function_response_content,
    _extract_multimodal_parts,
)
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.adk.plugins import (
    DEFAULT_HIGH_WATER_IMAGES,
    DEFAULT_LOW_WATER_IMAGES,
    ModelCallPacingPlugin,
    RequestCapturePlugin,
    WorkingSetImages,
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


def _kept(contents: list[types.Content]) -> list[int]:
    return [
        len(p.function_response.parts or [])
        for c in contents
        for p in (c.parts or [])
        if p.function_response is not None
    ]


def _seed() -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="0: a.tif"),
            types.Part.from_bytes(mime_type="image/jpeg", data=b"slice"),
        ],
    )


def test_working_set_trims_in_batches_and_keeps_the_prefix_stable():
    """Above HIGH the set drops to LOW in one go; then nothing moves until
    the next overflow, so the request prefix stays byte-identical between
    trims (the upstream prompt cache pays only for a stable prefix)."""
    ws = WorkingSetImages(high=12, low=6)
    history = [_seed(), _tool_turn(4), _tool_turn(4), _tool_turn(4)]
    assert ws(history) is history  # 12 is not above 12
    history.append(_tool_turn(4))  # 16 > 12: cut to <= 6 (newest two calls: 8)
    assert _kept(ws(history)) == [0, 0, 4, 4]
    assert ws.trims == 1
    history.append(_tool_turn(1))  # 9 <= 12: the cut does not move
    assert _kept(ws(history)) == [0, 0, 4, 4, 1]
    assert ws.trims == 1
    history.append(_tool_turn(4))  # 13 > 12: cut again down to the newest two
    assert _kept(ws(history)) == [0, 0, 0, 0, 1, 4]
    assert ws.trims == 2


def test_working_set_dropped_results_say_so_and_do_not_mutate_the_input():
    ws = WorkingSetImages(high=4, low=2)
    contents = [_seed(), _tool_turn(3), _tool_turn(3), _tool_turn(3)]
    out = ws(contents)
    assert _kept(out) == [0, 3, 3]
    fr = out[1].parts[0].function_response
    assert fr is not None and fr.response["status"] == "ok"
    assert "dropped from context" in fr.response["images"]
    assert contents[1].parts[0].function_response.response == {"status": "ok"}
    assert _kept(contents) == [3, 3, 3]


def test_working_set_drops_the_seed_images_once_it_has_cut():
    ws = WorkingSetImages(high=4, low=2)
    contents = [_seed(), _tool_turn(3), _tool_turn(3)]
    same = ws(contents)
    assert same is contents  # under the bound: the seed strip is untouched
    contents.append(_tool_turn(3))
    out = ws(contents)
    parts = out[0].parts or []
    assert parts[0].text == "0: a.tif"  # labels stay
    assert parts[1].inline_data is None and "dropped from context" in (parts[1].text or "")
    assert contents[0].parts[1].inline_data is not None


def test_working_set_always_keeps_the_two_newest_calls():
    ws = WorkingSetImages(high=4, low=2)
    out = ws([_tool_turn(8), _tool_turn(8), _tool_turn(8)])
    assert _kept(out) == [0, 8, 8]


def test_working_set_defaults_fit_a_positioning_run():
    """Two 16-image batches (compare + write) fit; the third overflows and
    trims back to those two. ~48 images is ~24k tokens a call, against the
    170 images / 93k tokens run 4 carried with no working set."""
    assert (DEFAULT_HIGH_WATER_IMAGES, DEFAULT_LOW_WATER_IMAGES) == (48, 16)
    ws = WorkingSetImages()
    history = [_tool_turn(8), _tool_turn(8), _tool_turn(16), _tool_turn(16)]
    assert ws(history) is history
    history.append(_tool_turn(16))
    assert _kept(ws(history)) == [0, 0, 0, 16, 16]
