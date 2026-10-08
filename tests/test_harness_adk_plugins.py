import asyncio

from google.adk.agents import LlmAgent
from google.adk.flows.llm_flows.functions import (
    _build_function_response_content,
    _extract_multimodal_parts,
)
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import FunctionTool
from google.genai import types
from PIL import Image

from langslice.agent.plugins import (
    DEFAULT_KEEP_IMAGES,
    DEFAULT_MAX_IMAGES,
    ModelCallPacingPlugin,
    RequestCapturePlugin,
    RetiredToolsPlugin,
    StrictArgumentsPlugin,
    WorkingSetImages,
)
from langslice.agent.session import build_plugins, run_agent_session
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.ops.registry import RETIRED, retired_payload


class _FakeTool:
    name = "look"
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
            name="look", response={"status": "ok"}, parts=frp
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


def test_working_set_keeps_everything_until_the_cap_then_cuts_once():
    """Images that stay in the prefix are cached (~0.13x); a removal costs a
    cache break at that point on every later call. So nothing is dropped
    until the cap, then one batch cut to the keep level, oldest first, and
    the cut never moves back."""
    assert DEFAULT_KEEP_IMAGES < DEFAULT_MAX_IMAGES
    ws = WorkingSetImages(max_images=6, keep_images=3)
    history = [_seed(), _tool_turn(2), _tool_turn(2), _tool_turn(2)]
    assert ws(history) is history  # 6 images: at the cap, nothing cut
    history.append(_tool_turn(1))  # 7: over the cap
    assert _kept(ws(history)) == [0, 0, 2, 1]  # cut to <= 3, oldest first
    assert ws.cut == 2
    history.append(_tool_turn(1))  # 4 live: under the cap, cut stays put
    assert _kept(ws(history)) == [0, 0, 2, 1, 1]
    assert ws.cut == 2


def test_working_set_dropped_results_say_so_and_do_not_mutate_the_input():
    ws = WorkingSetImages(max_images=3, keep_images=3)
    contents = [_seed(), _tool_turn(3), _tool_turn(3)]
    out = ws(contents)
    assert _kept(out) == [0, 3]
    fr = out[1].parts[0].function_response
    assert fr is not None and fr.response["status"] == "ok"
    assert "dropped from context" in fr.response["images"]
    assert contents[1].parts[0].function_response.response == {"status": "ok"}
    assert _kept(contents) == [3, 3]


def test_transform_calls_do_not_cut_earlier_images():
    """No stage-boundary cut: channel strips and preprocessed-channel
    pictures seen before alignment stay in context."""
    ws = WorkingSetImages()

    def _turn(name: str, n: int) -> types.Content:
        part = _media_part(n)
        part.function_response.name = name  # type: ignore[union-attr]
        return types.Content(role="user", parts=[part])

    history = [_seed(), _turn("look", 4), _turn("set_preprocessed_channel_properties", 6),
               _turn("elastix_affine", 3), _turn("interactive_transform", 1)]
    assert ws(history) is history


def test_working_set_never_touches_the_seed_strip():
    """The seed strip heads the prefix: cached, and the one picture of every
    section the model always has."""
    ws = WorkingSetImages(max_images=1, keep_images=1)
    contents = [_seed(), _tool_turn(1), _tool_turn(1)]
    out = ws(contents)
    assert _kept(out) == [0, 1]
    assert out[0] is contents[0]
    assert (out[0].parts or [])[1].inline_data is not None


# --- retired tool names --------------------------------------------------------


class _Placeholder:
    """What ADK hands ``before_tool_callback`` for a name it does not know:
    a tool of that name with no function."""

    def __init__(self, name: str) -> None:
        self.name = name


def _retired_answer(tool) -> dict | None:
    return asyncio.run(RetiredToolsPlugin().before_tool_callback(
        tool=tool, tool_args={}, tool_context=None))


def test_a_retired_name_answers_with_the_tool_to_use():
    answer = _retired_answer(_Placeholder("fit_affine"))
    assert answer == retired_payload("fit_affine")
    assert answer is not None
    assert answer["error"] == "RETIRED_TOOL" and answer["use"] == "elastix_affine"
    for old, (new, _hint) in RETIRED.items():
        answer = _retired_answer(_Placeholder(old))
        assert answer is not None and answer["error"] == "RETIRED_TOOL", old
        assert answer.get("use") == (new or None), old
    # Nothing replaces search_position: no tool to use is named.
    answer = _retired_answer(_Placeholder("search_position"))
    assert answer is not None and "use" not in answer


def test_a_real_tool_and_an_unknown_name_are_left_to_adk():
    def look() -> dict:
        return {"status": "ok"}

    assert _retired_answer(FunctionTool(look)) is None
    assert _retired_answer(_Placeholder("no_such_tool")) is None
    # A real tool runs as itself, whatever its name.
    real = FunctionTool(look)
    real.name = "fit_affine"
    assert _retired_answer(real) is None


def test_the_session_answers_retired_names_before_the_argument_check():
    kinds = [type(plugin) for plugin in build_plugins("unit")]
    assert RetiredToolsPlugin in kinds and StrictArgumentsPlugin in kinds
    assert kinds.index(RetiredToolsPlugin) < kinds.index(StrictArgumentsPlugin)


# --- a scripted model through a whole session -----------------------------------------


class _ScriptLlm(BaseLlm):
    """Answers request N with ``script[N]`` (a list of parts), then text."""

    script: list = []
    requests: list = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        del stream
        self.requests.append(llm_request.model_copy(deep=True))
        step = len(self.requests) - 1
        parts = (self.script[step] if step < len(self.script)
                 else [types.Part.from_text(text="Nothing more.")])
        yield LlmResponse(content=types.Content(role="model", parts=parts), partial=False,
                          turn_complete=True)


def _session(script: list, *, background=None) -> tuple[_ScriptLlm, dict]:
    submitted: dict = {}

    def submit(summary: str) -> dict:
        """Hand in the job."""
        submitted["summary"] = summary
        return {"status": "ok"}

    model = _ScriptLlm(model="script", script=script, requests=[])
    agent = LlmAgent(name="unit", model=model, tools=[submit])
    asyncio.run(run_agent_session(
        agent=agent, seed_message=types.Content(
            role="user", parts=[types.Part.from_text(text="Register the stack.")]),
        done=lambda: bool(submitted), nudge_no_tool="NUDGE: call a tool",
        nudge_continue="NUDGE: continue", max_iterations=8, run_label="unit_session",
        background=background))
    return model, submitted


def _call(name: str, **args) -> types.Part:
    return types.Part.from_function_call(name=name, args=args)


def _responses(request: LlmRequest) -> list[types.FunctionResponse]:
    return [part.function_response for content in request.contents or []
            for part in content.parts or [] if part.function_response is not None]


def _texts(content: types.Content) -> list[str]:
    return [part.text for part in content.parts or [] if part.text]


def test_a_model_calling_a_retired_tool_gets_the_tool_to_use(monkeypatch):
    monkeypatch.delenv("LANGSLICE_TRACE_DIR", raising=False)
    model, submitted = _session([
        [_call("fit_affine", slices=["a.tif"])],
        [_call("submit", summary="done")],
    ])
    assert submitted == {"summary": "done"}
    answer, = _responses(model.requests[1])
    assert answer.name == "fit_affine"
    assert answer.response == retired_payload("fit_affine")
    assert answer.response["use"] == "elastix_affine"


def test_a_turn_ended_while_work_runs_gets_its_notice_as_the_next_message(monkeypatch):
    monkeypatch.delenv("LANGSLICE_TRACE_DIR", raising=False)
    picture = Image.new("RGB", (16, 12), "white")
    waits: list[int] = []

    def background():
        waits.append(1)
        if len(waits) == 1:
            return ["Work w1 (trace_borders, a.tif) finished: fitted.", picture]
        return None

    model, submitted = _session([
        [types.Part.from_text(text="Waiting for the trace.")],
        [types.Part.from_text(text="Still nothing to do.")],
        [_call("submit", summary="done")],
    ], background=background)
    assert submitted == {"summary": "done"}
    assert len(model.requests) == 3
    notice = model.requests[1].contents[-1]
    assert notice.role == "user"
    assert _texts(notice) == ["Work w1 (trace_borders, a.tif) finished: fitted."]
    images = [part for part in notice.parts or [] if part.inline_data is not None]
    assert len(images) == 1 and images[0].inline_data.mime_type == "image/jpeg"
    # Nothing running when the next turn ends: the nudge, as before.
    assert _texts(model.requests[2].contents[-1]) == ["NUDGE: call a tool"]
    assert len(waits) == 2
