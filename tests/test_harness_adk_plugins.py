import asyncio

from google.adk.agents import LlmAgent
from google.adk.flows.llm_flows.functions import (
    _build_function_response_content,
    _extract_multimodal_parts,
)
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image

from langslice.agent.plugins import (
    DEFAULT_KEEP_IMAGES,
    DEFAULT_MAX_IMAGES,
    RequestCapturePlugin,
    WorkingSetImages,
)
from langslice.agent.session import build_plugins, run_agent_session
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY


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

    # On the function response part itself: one part carrying the image, no
    # sibling image part, so nothing double-injects it.
    content = _build_function_response_content(
        _FakeTool(),  # type: ignore[arg-type]
        {"status": "ok", TOOL_MEDIA_PARTS_KEY: image_parts[:1]},
        "call-1",
    )
    parts = content.parts or []
    assert len(parts) == 1
    function_response = parts[0].function_response
    assert function_response is not None
    assert function_response.response == {"status": "ok"}
    media = function_response.parts or []
    assert len(media) == 1 and media[0].inline_data is not None
    assert media[0].inline_data.data == b"jpeg-one"


def test_request_capture_plugin_redacts_inline_bytes_and_counts_response_media(tmp_path):
    part = types.Part.from_bytes(mime_type="image/jpeg", data=b"not-a-real-jpeg")
    response = _build_function_response_content(
        _FakeTool(),  # type: ignore[arg-type]
        {"status": "ok", TOOL_MEDIA_PARTS_KEY: [
            types.Part.from_bytes(mime_type="image/jpeg", data=b"jpeg-one")]},
        "call-1",
    )
    request = LlmRequest(
        model="capture-model",
        contents=[types.Content(role="user", parts=[part]), response],
    )
    plugin = RequestCapturePlugin(tmp_path, run_label="unit")

    asyncio.run(
        plugin.before_model_callback(callback_context=None, llm_request=request)  # type: ignore[arg-type]
    )

    captures = list(tmp_path.glob("unit_*.json"))
    assert len(captures) == 1
    text = captures[0].read_text(encoding="utf-8")
    assert "not-a-real-jpeg" not in text and "jpeg-one" not in text
    assert "inline_data" in text and "byte_count" in text and "image/jpeg" in text
    assert '"media_part_count": 1' in text


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


def test_working_set_never_touches_the_seed_strip():
    """The seed strip heads the prefix: cached, and the one picture of every
    section the model always has."""
    ws = WorkingSetImages(max_images=1, keep_images=1)
    contents = [_seed(), _tool_turn(1), _tool_turn(1)]
    out = ws(contents)
    assert _kept(out) == [0, 1]
    assert out[0] is contents[0]
    assert (out[0].parts or [])[1].inline_data is not None


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
        nudge_continue="NUDGE: continue", run_label="unit_session",
        background=background))
    return model, submitted


def _call(name: str, **args) -> types.Part:
    return types.Part.from_function_call(name=name, args=args)


def _responses(request: LlmRequest) -> list[types.FunctionResponse]:
    return [part.function_response for content in request.contents or []
            for part in content.parts or [] if part.function_response is not None]


def _texts(content: types.Content) -> list[str]:
    return [part.text for part in content.parts or [] if part.text]


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


def test_session_uses_the_working_set_filter_first(monkeypatch):
    monkeypatch.delenv("LANGSLICE_ADK_CAPTURE_REQUESTS_DIR", raising=False)
    monkeypatch.delenv("LANGSLICE_ADK_MODEL_CALL_DELAY_S", raising=False)
    plugins = build_plugins("test")
    assert isinstance(plugins[0]._custom_filter, WorkingSetImages)
