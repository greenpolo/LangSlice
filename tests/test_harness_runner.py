import asyncio
import io
from collections.abc import AsyncGenerator

import pytest
from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from PIL import Image

from langslice_harness.harness.estimation._types import MultiSliceResult, PositionResult
from langslice_harness.harness.estimation.group import build_group_agent
from langslice_harness.harness.estimation.runner import (
    _prepare_target_payloads,
    run_group_session,
    run_single_slice_session,
)
from langslice_harness.harness.estimation.session import (
    ARTIFACT_TARGET,
    build_initial_state,
)
from langslice_harness.harness.estimation.single_slice import build_single_slice_agent
from langslice_harness.harness.estimation.validators import gate_submit_tool


def _count_inline_images(contents: list[types.Content]) -> int:
    count = 0
    for content in contents:
        for part in content.parts or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                Image.open(io.BytesIO(data)).load()
                count += 1
    return count


def _count_file_images(contents: list[types.Content]) -> int:
    count = 0
    for content in contents:
        for part in content.parts or []:
            file_data = getattr(part, "file_data", None)
            file_uri = getattr(file_data, "file_uri", None) if file_data else None
            if file_uri:
                count += 1
    return count


def _part_image_size(part: types.Part) -> tuple[int, int]:
    inline = part.inline_data
    assert inline is not None and inline.data is not None
    img = Image.open(io.BytesIO(inline.data))
    img.load()
    return img.size


class _FakeGeminiFile:
    def __init__(self, idx: int) -> None:
        self.name = f"files/target-{idx}"
        self.uri = f"https://generativelanguage.googleapis.com/v1beta/{self.name}"
        self.state = "ACTIVE"


class _FakeGeminiFiles:
    def __init__(self) -> None:
        self.uploaded: list[_FakeGeminiFile] = []
        self.deleted: list[str] = []

    def upload(self, *, file, config):  # noqa: ANN001 - mimics SDK surface
        file.read()
        uploaded = _FakeGeminiFile(len(self.uploaded) + 1)
        self.uploaded.append(uploaded)
        assert getattr(config, "mime_type", None) == "image/jpeg"
        return uploaded

    def get(self, *, name: str) -> _FakeGeminiFile:
        return next(f for f in self.uploaded if f.name == name)

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class _FakeGeminiClient:
    def __init__(self) -> None:
        self.files = _FakeGeminiFiles()


def test_initial_state_single_slice():
    state = build_initial_state(
        atlas_name="allen_mouse_25um",
        plane="coronal",
        pos_lo=0.0, pos_hi=13.2,
        n_slices=1, interval_mm=0.0, thickness_um=50,
        max_iterations=20,
    )
    assert state["atlas"] == "allen_mouse_25um"
    assert state["plane"] == "coronal"
    assert state["axis_label"] == "AP"
    assert state["n_slices"] == 1
    assert state["fetched_positions"] == []
    assert state["saw_broad_sweep"] is False
    assert state["saw_narrow_sweep"] is False
    assert state["submit_attempts"] == 0
    assert state["result"] is None


def test_artifact_target_constant():
    assert ARTIFACT_TARGET == "target"


def test_prepare_target_payloads_keeps_display_payload_only(monkeypatch):
    """Target preparation should only create the image shown to the model."""

    def _raise_if_called():
        raise AssertionError("inline target preparation should not create a Gemini client")

    monkeypatch.setattr("langslice_harness.vlm_config.get_client", _raise_if_called)

    payloads, uploaded_files = _prepare_target_payloads(
        [Image.new("RGB", (1200, 800), color=128)],
        atlas_long_edge=456,
        apply_clahe=False,
        model="local-gemma",
        target_transport="inline",
    )

    assert uploaded_files == []
    assert len(payloads) == 1
    payload = payloads[0]
    assert max(_part_image_size(payload.artifact_part)) == 456
    assert _part_image_size(payload.request_part) == _part_image_size(payload.artifact_part)


def test_build_single_slice_agent_registers_estimation_tools_only():
    agent = build_single_slice_agent(
        atlas_name="allen_mouse_25um", plane="coronal",
        species="mouse", pos_lo=0.0, pos_hi=13.2,
        model="gemini-3-flash-preview",
    )
    tool_names = {getattr(t, "__name__", None) or getattr(t, "name", None) for t in agent.tools}
    assert tool_names == {"fetch_atlas", "submit_estimate"}
    assert agent.instruction  # non-empty prompt


def test_build_single_slice_agent_sets_native_gemini_retry_options():
    agent = build_single_slice_agent(
        atlas_name="allen_mouse_25um", plane="coronal",
        species="mouse", pos_lo=0.0, pos_hi=13.2,
        model="gemini-3-flash-preview",
    )

    config = agent.generate_content_config
    assert config is not None
    assert config.http_options is not None
    assert config.http_options.retry_options is not None
    assert config.http_options.retry_options.initial_delay == 1
    assert config.http_options.retry_options.attempts == 5


def test_build_group_agent_registers_estimation_tools_only_and_callback():
    agent = build_group_agent(
        atlas_name="allen_mouse_25um", plane="coronal",
        species="mouse", pos_lo=0.0, pos_hi=13.2,
        n_slices=4, interval_mm=0.200, thickness_um=50,
        model="gemini-3-flash-preview",
    )
    assert agent.name == "group_position_estimator"
    assert len(agent.tools) == 2
    tool_names = {getattr(t, "__name__", None) or getattr(t, "name", None) for t in agent.tools}
    assert tool_names == {"fetch_atlas", "submit_group_estimate"}
    assert agent.before_tool_callback is gate_submit_tool
    assert agent.instruction  # non-empty prompt


def test_run_single_slice_session_happy_path(monkeypatch):
    """Runner completes when the model follows broad -> narrow -> submit."""
    try:
        from tests.fakes import install_fake_adk_model_scripted_submit
    except NotImplementedError as exc:
        pytest.skip(f"ADK fake-model seam unavailable: {exc}")
    try:
        install_fake_adk_model_scripted_submit(monkeypatch)
    except NotImplementedError as exc:
        pytest.skip(f"ADK fake-model seam unavailable: {exc}")

    img = Image.new("RGB", (456, 320), 128)
    result = asyncio.run(
        run_single_slice_session(
            image=img,
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=8,
            max_retries=1,
            target_transport="inline",
        )
    )
    assert result.position_mm is not None
    assert 0.0 <= result.position_mm <= 13.2


def test_run_group_session_drives_fake_to_completion(monkeypatch):
    """Runner completes when the group model follows broad -> narrow -> submit_group."""
    try:
        from tests.fakes import install_fake_adk_group_model_scripted_submit
    except NotImplementedError as exc:
        pytest.skip(f"ADK fake-model seam unavailable: {exc}")
    try:
        install_fake_adk_group_model_scripted_submit(
            monkeypatch, n_slices=4, interval_mm=0.2,
        )
    except NotImplementedError as exc:
        pytest.skip(f"ADK fake-model seam unavailable: {exc}")

    # Blank images keep CLAHE a no-op and avoid any OpenCV dependency in tests.
    images = [Image.new("L", (256, 256), color=128) for _ in range(4)]
    result = asyncio.run(
        run_group_session(
            images=images,
            atlas_name="allen_mouse_25um",
            interval_mm=0.2,
            thickness_um=50,
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=12,
            max_retries=1,
            apply_clahe=False,
            target_transport="inline",
        )
    )

    assert isinstance(result, MultiSliceResult)
    assert len(result.positions) == 4
    for pos in result.positions:
        assert isinstance(pos, PositionResult)
        assert 0.0 <= pos.position_mm <= 15.0  # loose sanity on atlas range
    # Monotonic, matching the scripted centre of 6.0 mm with 0.2 mm spacing.
    position_values = [p.position_mm for p in result.positions]
    assert position_values == sorted(position_values)
    assert all(abs(p - 6.0) <= 1.0 for p in position_values)
    # Reasoning text is the one propagated from the scripted fake — confirms
    # the runner reached submit rather than falling back to midpoint.
    lowered = result.group_reasoning.lower()
    assert "broad" in lowered and "narrow" in lowered
    assert "fell back" not in lowered


def test_group_runner_replays_all_multimodal_tool_images(monkeypatch):
    """Atlas images from earlier tool calls remain visible on later turns."""

    captured_image_counts: list[int] = []

    class CaptureMultimodalHistoryLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            n_function_responses = sum(
                1
                for content in contents
                for part in (content.parts or [])
                if getattr(part, "function_response", None) is not None
            )
            captured_image_counts.append(_count_inline_images(contents))

            if n_function_responses == 0:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif n_function_responses == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.6, 5.8, 6.0]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_group_estimate",
                    args={
                        "positions_mm": [5.7, 5.871],
                        "reasoning": "Both broad and narrow atlas sweeps are visible.",
                    },
                )

            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(
        LLMRegistry,
        "new_llm",
        staticmethod(lambda model: CaptureMultimodalHistoryLlm(model=model)),
    )

    images = [Image.new("L", (256, 256), color=128) for _ in range(2)]
    result = asyncio.run(
        run_group_session(
            images=images,
            atlas_name="allen_mouse_25um",
            interval_mm=0.171,
            thickness_um=50,
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=6,
            max_retries=1,
            apply_clahe=False,
            target_transport="inline",
        )
    )

    assert [p.position_mm for p in result.positions] == [5.7, 5.871]
    assert captured_image_counts == [2, 5, 8]


def test_group_runner_can_use_one_turn_multimodal_history(monkeypatch):
    """Old ADK image transport only carries the most recent tool images."""

    captured_image_counts: list[int] = []

    class CaptureOneTurnHistoryLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            n_function_responses = sum(
                1
                for content in contents
                for part in (content.parts or [])
                if getattr(part, "function_response", None) is not None
            )
            captured_image_counts.append(_count_inline_images(contents))

            if n_function_responses == 0:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif n_function_responses == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.6, 5.8, 6.0]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_group_estimate",
                    args={"positions_mm": [5.7, 5.871], "reasoning": "one-turn"},
                )

            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(
        LLMRegistry,
        "new_llm",
        staticmethod(lambda model: CaptureOneTurnHistoryLlm(model=model)),
    )

    images = [Image.new("L", (256, 256), color=128) for _ in range(2)]
    result = asyncio.run(
        run_group_session(
            images=images,
            atlas_name="allen_mouse_25um",
            interval_mm=0.171,
            thickness_um=50,
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=6,
            max_retries=1,
            apply_clahe=False,
            target_transport="inline",
            multimodal_history="one_turn",
        )
    )

    assert [p.position_mm for p in result.positions] == [5.7, 5.871]
    assert captured_image_counts == [2, 5, 5]


def test_single_runner_keeps_fetch_tool_response_structured(monkeypatch):
    """Successful fetch_atlas should not degrade into {'result': ...}."""

    captured_response_keys: list[list[str]] = []
    captured_image_counts: list[int] = []

    class CaptureStructuredToolResponseLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            keys: list[list[str]] = []
            for content in contents:
                for part in content.parts or []:
                    function_response = part.function_response
                    if function_response is None:
                        continue
                    response = function_response.response
                    assert isinstance(response, dict)
                    keys.append(sorted(response.keys()))
            captured_response_keys.extend(keys)
            captured_image_counts.append(_count_inline_images(contents))

            if not keys:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif len(keys) == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.8, 6.0, 6.2]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_estimate",
                    args={"position_mm": 6.0, "reasoning": "structured"},
                )

            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(
        LLMRegistry,
        "new_llm",
        staticmethod(lambda model: CaptureStructuredToolResponseLlm(model=model)),
    )

    result = asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=8,
            max_retries=1,
            apply_clahe=False,
            target_transport="inline",
        )
    )

    assert result.position_mm == 6.0
    assert ["description", "positions_mm", "status"] in captured_response_keys
    assert ["result"] not in captured_response_keys
    assert captured_image_counts == [1, 4, 7]


def test_single_runner_uses_file_api_target_for_native_gemini(monkeypatch):
    """Native Gemini should receive the target slice via File API, like main."""

    captured_first_request: list[types.Content] = []
    fake_client = _FakeGeminiClient()

    class CaptureInitialTransportLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            if not captured_first_request:
                captured_first_request.extend(contents)
            n_function_responses = sum(
                1
                for content in contents
                for part in (content.parts or [])
                if getattr(part, "function_response", None) is not None
            )
            if n_function_responses == 0:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif n_function_responses == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.8, 6.0, 6.2]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_estimate",
                    args={"position_mm": 6.0, "reasoning": "done"},
                )
            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(
        LLMRegistry,
        "new_llm",
        staticmethod(lambda model: CaptureInitialTransportLlm(model=model)),
    )
    monkeypatch.setattr("langslice_harness.vlm_config.supports_file_api", lambda: True)
    monkeypatch.setattr("langslice_harness.vlm_config.file_poll_timeout_s", lambda: 0.01)
    monkeypatch.setattr("langslice_harness.vlm_config.get_client", lambda: fake_client)

    result = asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=8,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert result.position_mm == 6.0
    assert len(fake_client.files.uploaded) == 1
    assert fake_client.files.deleted == ["files/target-1"]
    assert _count_file_images(captured_first_request) == 1
    assert _count_inline_images(captured_first_request) == 0


def test_single_runner_uses_file_api_target_for_native_gemma(monkeypatch):
    """Hosted Gemma API models should use the native Google File API path."""

    captured_first_request: list[types.Content] = []
    fake_client = _FakeGeminiClient()

    class CaptureInitialTransportLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            if not captured_first_request:
                captured_first_request.extend(contents)
            call = types.Part.from_function_call(
                name="fetch_atlas",
                args={"positions_mm": [4.0, 5.0, 6.0]},
            )
            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    monkeypatch.setattr(
        "langslice_harness.harness.estimation.model_resolver._load_adk_gemma_class",
        lambda: CaptureInitialTransportLlm,
    )
    monkeypatch.setattr("langslice_harness.vlm_config.supports_file_api", lambda: True)
    monkeypatch.setattr("langslice_harness.vlm_config.file_poll_timeout_s", lambda: 0.01)
    monkeypatch.setattr("langslice_harness.vlm_config.get_client", lambda: fake_client)

    result = asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="gemma-4-26b-a4b-it",
            max_iterations=1,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert "fell back" in result.reasoning
    assert len(fake_client.files.uploaded) == 1
    assert fake_client.files.deleted == ["files/target-1"]
    assert _count_file_images(captured_first_request) == 1
    assert _count_inline_images(captured_first_request) == 0


def test_single_runner_continues_same_session_with_nudge(monkeypatch):
    """A no-tool model turn should get a nudge in the same session, like main."""

    saw_nudge = False

    class NeedsNudgeLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            nonlocal saw_nudge
            del stream
            contents = list(llm_request.contents or [])
            n_function_responses = sum(
                1
                for content in contents
                for part in (content.parts or [])
                if getattr(part, "function_response", None) is not None
            )
            latest_text = " ".join(
                part.text or ""
                for part in (contents[-1].parts or [])
                if getattr(part, "text", None)
            )
            if n_function_responses == 0 and "Please continue" not in latest_text:
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="I need to inspect the atlas.")],
                    ),
                    partial=False,
                    turn_complete=True,
                )
                return

            if "Please continue" in latest_text:
                saw_nudge = True

            if n_function_responses == 0:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif n_function_responses == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.8, 6.0, 6.2]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_estimate",
                    args={"position_mm": 6.0, "reasoning": "nudge worked"},
                )

            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(
        LLMRegistry,
        "new_llm",
        staticmethod(lambda model: NeedsNudgeLlm(model=model)),
    )

    result = asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=8,
            max_retries=1,
            apply_clahe=False,
            target_transport="inline",
        )
    )

    assert saw_nudge is True
    assert result.position_mm == 6.0
    assert result.reasoning == "nudge worked"


def test_single_runner_long_no_tool_turn_gets_main_style_nudge(monkeypatch):
    """A long no-tool response should receive /main's explicit tool-call reminder."""

    requests_seen = 0
    saw_thought_nudge = False

    class ThoughtLeakThenSubmitLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            nonlocal requests_seen, saw_thought_nudge
            del stream
            requests_seen += 1
            contents = list(llm_request.contents or [])
            latest_text = " ".join(
                part.text or ""
                for part in (contents[-1].parts or [])
                if getattr(part, "text", None)
            )
            if "You wrote a long text response instead of calling a tool" in latest_text:
                saw_thought_nudge = True

            if requests_seen == 1:
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="analysis " * 1200)],
                    ),
                    partial=False,
                    turn_complete=True,
                    usage_metadata=types.GenerateContentResponseUsageMetadata(
                        candidates_token_count=1201,
                    ),
                )
                return

            n_function_responses = sum(
                1
                for content in contents
                for part in (content.parts or [])
                if getattr(part, "function_response", None) is not None
            )
            if n_function_responses == 0:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif n_function_responses == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.8, 6.0, 6.2]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_estimate",
                    args={"position_mm": 6.0, "reasoning": "thought nudge worked"},
                )

            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(
        LLMRegistry,
        "new_llm",
        staticmethod(lambda model: ThoughtLeakThenSubmitLlm(model=model)),
    )

    result = asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=8,
            max_retries=1,
            apply_clahe=False,
            target_transport="inline",
        )
    )

    assert saw_thought_nudge is True
    assert result.position_mm == 6.0
    assert result.reasoning == "thought nudge worked"


def test_group_runner_uses_file_api_targets_for_native_gemini(monkeypatch):
    """Native Gemini group sessions should upload each target once."""

    captured_first_request: list[types.Content] = []
    fake_client = _FakeGeminiClient()

    class CaptureInitialGroupTransportLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            if not captured_first_request:
                captured_first_request.extend(contents)
            n_function_responses = sum(
                1
                for content in contents
                for part in (content.parts or [])
                if getattr(part, "function_response", None) is not None
            )
            if n_function_responses == 0:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif n_function_responses == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.8, 6.0, 6.2]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_group_estimate",
                    args={"positions_mm": [5.9, 6.1], "reasoning": "done"},
                )
            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    from google.adk.models.registry import LLMRegistry

    monkeypatch.setattr(
        LLMRegistry,
        "new_llm",
        staticmethod(lambda model: CaptureInitialGroupTransportLlm(model=model)),
    )
    monkeypatch.setattr("langslice_harness.vlm_config.supports_file_api", lambda: True)
    monkeypatch.setattr("langslice_harness.vlm_config.file_poll_timeout_s", lambda: 0.01)
    monkeypatch.setattr("langslice_harness.vlm_config.get_client", lambda: fake_client)

    result = asyncio.run(
        run_group_session(
            images=[
                Image.new("RGB", (256, 256), color=128),
                Image.new("RGB", (256, 256), color=128),
            ],
            atlas_name="allen_mouse_25um",
            interval_mm=0.2,
            thickness_um=50,
            plane="coronal",
            model="gemini-3-flash-preview",
            max_iterations=8,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert [p.position_mm for p in result.positions] == [5.9, 6.1]
    assert len(fake_client.files.uploaded) == 2
    assert fake_client.files.deleted == ["files/target-1", "files/target-2"]
    assert _count_file_images(captured_first_request) == 2
    assert _count_inline_images(captured_first_request) == 0


def test_model_object_target_transport_stays_inline(monkeypatch):
    """Local/model-object paths must not touch Gemini File API."""

    captured_first_request: list[types.Content] = []

    class CaptureInlineTransportLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            if not captured_first_request:
                captured_first_request.extend(contents)
            call = types.Part.from_function_call(
                name="fetch_atlas",
                args={"positions_mm": [4.0, 5.0, 6.0]},
            )
            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    def _raise_if_called():
        raise AssertionError("local/model-object path should not create a Gemini client")

    monkeypatch.setattr("langslice_harness.vlm_config.get_client", _raise_if_called)

    result = asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model=CaptureInlineTransportLlm(model="local-gemma"),
            max_iterations=1,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert "fell back" in result.reasoning
    assert _count_inline_images(captured_first_request) == 1
    assert _count_file_images(captured_first_request) == 0


def test_litellm_proxy_model_string_resolves_to_inline_model_object(monkeypatch):
    """Open-model shorthand should resolve before LlmAgent construction."""

    captured_first_request: list[types.Content] = []

    class FakeLiteLlm(BaseLlm):
        def __init__(self, model: str, **kwargs) -> None:
            del kwargs
            super().__init__(model=model)

        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            if not captured_first_request:
                captured_first_request.extend(contents)
            n_function_responses = sum(
                1
                for content in contents
                for part in (content.parts or [])
                if getattr(part, "function_response", None) is not None
            )
            if n_function_responses == 0:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif n_function_responses == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.8, 6.0, 6.2]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_estimate",
                    args={"position_mm": 6.0, "reasoning": "resolved"},
                )
            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    monkeypatch.setattr(
        "langslice_harness.harness.estimation.model_resolver._load_litellm_class",
        lambda: FakeLiteLlm,
    )

    def _raise_if_called():
        raise AssertionError("LiteLLM proxy path should not create a Gemini client")

    monkeypatch.setattr("langslice_harness.vlm_config.get_client", _raise_if_called)

    result = asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="litellm-proxy:langslice-qwen36-plus",
            max_iterations=8,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert result.position_mm == 6.0
    assert result.reasoning == "resolved"
    assert _count_inline_images(captured_first_request) == 1
    assert _count_file_images(captured_first_request) == 0


def test_ollama_model_string_resolves_to_inline_model_object(monkeypatch):
    """Local Ollama shorthand should stay on inline transport."""

    captured_first_request: list[types.Content] = []

    class FakeLiteLlm(BaseLlm):
        def __init__(self, model: str, **kwargs) -> None:
            assert kwargs["api_base"] == "http://localhost:11434"
            super().__init__(model=model)

        async def generate_content_async(
            self, llm_request, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            contents = list(llm_request.contents or [])
            if not captured_first_request:
                captured_first_request.extend(contents)
            n_function_responses = sum(
                1
                for content in contents
                for part in (content.parts or [])
                if getattr(part, "function_response", None) is not None
            )
            if n_function_responses == 0:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [4.0, 5.0, 6.0]},
                )
            elif n_function_responses == 1:
                call = types.Part.from_function_call(
                    name="fetch_atlas",
                    args={"positions_mm": [5.8, 6.0, 6.2]},
                )
            else:
                call = types.Part.from_function_call(
                    name="submit_estimate",
                    args={"position_mm": 6.0, "reasoning": "ollama resolved"},
                )
            yield LlmResponse(
                content=types.Content(role="model", parts=[call]),
                partial=False,
                turn_complete=True,
            )

    monkeypatch.setattr(
        "langslice_harness.harness.estimation.model_resolver._load_litellm_class",
        lambda: FakeLiteLlm,
    )
    monkeypatch.setenv("LANGSLICE_OLLAMA_BASE", "http://localhost:11434")

    def _raise_if_called():
        raise AssertionError("Ollama path should not create a Gemini client")

    monkeypatch.setattr("langslice_harness.vlm_config.get_client", _raise_if_called)

    result = asyncio.run(
        run_single_slice_session(
            image=Image.new("RGB", (256, 256), color=128),
            atlas_name="allen_mouse_25um",
            plane="coronal",
            model="ollama:gemma4:26b",
            max_iterations=8,
            max_retries=1,
            apply_clahe=False,
        )
    )

    assert result.position_mm == 6.0
    assert result.reasoning == "ollama resolved"
    assert _count_inline_images(captured_first_request) == 1
    assert _count_file_images(captured_first_request) == 0
