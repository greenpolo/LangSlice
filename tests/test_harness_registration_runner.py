from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from PIL import Image

import langslice_harness.harness.registration.runner as registration_runner
import langslice_harness.harness.registration.tools as registration_tools
import langslice_harness.registration.runtime as registration_runtime
from langslice_harness.harness.registration.runner import run_registration_review_session
from langslice_harness.harness.registration.tools import (
    generate_registration_candidate,
    make_generate_registration_candidate_tool,
)
from langslice_harness.harness.registration.types import RegistrationCandidate
from langslice_harness.registration.runtime import RegistrationFailure
from langslice_harness.registration.types import RegistrationAnnotationSession


def _make_candidate(
    candidate_id: str,
    *,
    provider: str = "openai",
    model: str = "gpt-image-2",
    route: str = "openai_images",
    marker_count: int = 1,
) -> RegistrationCandidate:
    image = Image.new("RGB", (10, 8), color=(200, 100, 50))
    markers = [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(marker_count)]
    session = RegistrationAnnotationSession(
        workflow="image_gen_registration",
        target_count=0,
        metadata={
            "visualign_markers": markers,
            "n_markers": len(markers),
            "provider": provider,
            "model": model,
            "route": route,
            "candidate_id": candidate_id,
        },
    )
    return RegistrationCandidate(
        candidate_id=candidate_id,
        generated_segmentation=image,
        warped_atlas=image,
        warped_border_overlay=image,
        markers=markers,
        annotation_session=session,
        metadata={
            "generated": {
                "provider": provider,
                "model": model,
                "route": route,
                "revised_prompt": None,
            }
        },
    )


def _install_pipeline_fake(
    monkeypatch: Any,
    *,
    calls: list[dict[str, Any]],
    marker_count: int = 1,
    emit_callbacks: bool = False,
) -> None:
    def fake_pipeline(
        image: Image.Image,
        *,
        atlas_name: str,
        position_mm: float,
        plane: str = "coronal",
        provider: str = "google",
        image_model: str | None = None,
        prompt_revision: str | None = None,
        previous_candidate_id: str | None = None,
        candidate_id: str | None = None,
        debug_dir: str | None = None,
        on_progress=None,
        on_trace=None,
        openai_image_route: str = "images",
        review_model: str | None = None,
    ) -> RegistrationCandidate:
        del image
        calls.append(
            {
                "atlas_name": atlas_name,
                "position_mm": position_mm,
                "plane": plane,
                "provider": provider,
                "image_model": image_model,
                "prompt_revision": prompt_revision,
                "previous_candidate_id": previous_candidate_id,
                "candidate_id": candidate_id,
                "debug_dir": debug_dir,
                "openai_image_route": openai_image_route,
                "review_model": review_model,
                "on_progress": on_progress,
                "on_trace": on_trace,
            }
        )
        if emit_callbacks:
            if on_progress is not None:
                on_progress("candidate progress")
            if on_trace is not None:
                on_trace({"stage": "registration", "title": "candidate trace"})
        cid = candidate_id or f"candidate-{len(calls)}"
        candidate = _make_candidate(cid, marker_count=marker_count)
        candidate.metadata["generated"]["provider"] = provider
        candidate.metadata["generated"]["model"] = image_model
        candidate.metadata["generated"]["route"] = (
            "openai_images" if provider == "openai" else "google_genai"
        )
        candidate.metadata["generated"]["revised_prompt"] = prompt_revision
        candidate.metadata["previous_candidate_id"] = previous_candidate_id
        if prompt_revision is not None:
            candidate.metadata["prompt_revision"] = prompt_revision
        candidate.annotation_session.metadata["previous_candidate_id"] = previous_candidate_id
        if prompt_revision is not None:
            candidate.annotation_session.metadata["prompt_revision"] = prompt_revision
        candidate.annotation_session.metadata["visualign_markers"] = candidate.markers
        candidate.annotation_session.metadata["n_markers"] = len(candidate.markers)
        candidate.annotation_session.metadata["provider"] = provider
        candidate.annotation_session.metadata["model"] = image_model
        candidate.annotation_session.metadata["route"] = candidate.metadata["generated"]["route"]
        candidate.annotation_session.metadata["candidate_id"] = cid
        return candidate

    monkeypatch.setattr(
        registration_tools,
        "_generate_registration_candidate_pipeline",
        fake_pipeline,
    )


class _FakeToolContext:
    def __init__(
        self,
        state: dict[str, Any],
        saved_artifacts: list[str],
        *,
        artifact_service: Any | None = None,
        app_name: str = "",
        user_id: str = "",
        session_id: str = "",
    ) -> None:
        self.state = state
        self.saved_artifacts = saved_artifacts
        self.artifact_service = artifact_service
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id
        self.actions = SimpleNamespace(escalate=False)

    async def save_artifact(self, *, filename: str, artifact: Any) -> None:
        self.saved_artifacts.append(filename)
        self.state.setdefault("_saved", []).append((filename, artifact))
        if self.artifact_service is not None and self.app_name and self.user_id:
            await self.artifact_service.save_artifact(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id,
                filename=filename,
                artifact=artifact,
            )

    async def load_artifact(self, filename: str) -> Any:
        for saved_name, artifact in self.state.get("_saved", []):
            if saved_name == filename:
                return artifact
        if self.artifact_service is None or not self.app_name or not self.user_id:
            return None
        return await self.artifact_service.load_artifact(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
            filename=filename,
        )


class _FakeSession:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


class _FakeSessionService:
    def __init__(self) -> None:
        self.sessions: dict[tuple[str, str, str], _FakeSession] = {}

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        state: dict[str, Any],
    ) -> None:
        self.sessions[(app_name, user_id, session_id)] = _FakeSession(copy.deepcopy(state))

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> _FakeSession | None:
        return self.sessions.get((app_name, user_id, session_id))


class _FakeArtifactService:
    def __init__(self) -> None:
        self.saved: dict[tuple[str, str, str, str], Any] = {}

    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: Any,
    ) -> None:
        self.saved[(app_name, user_id, session_id, filename)] = artifact

    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
    ) -> Any:
        return self.saved.get((app_name, user_id, session_id, filename))


class _FakeApp:
    def __init__(self, *, name: str, root_agent: Any, plugins: list[Any]) -> None:
        self.name = name
        self.root_agent = root_agent
        self.plugins = plugins


class _FakeRunner:
    last_instance: ClassVar[_FakeRunner | None] = None

    def __init__(self, app: _FakeApp) -> None:
        self.app = app
        self.session_service = _FakeSessionService()
        self.artifact_service = _FakeArtifactService()
        _FakeRunner.last_instance = self

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: Any,
    ):
        del new_message
        session = await self.session_service.get_session(
            app_name=self.app.name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            raise RuntimeError("missing session")

        state = session.state
        script = list(getattr(self.app.root_agent, "flow_script", []))
        step_index = int(state.get("_flow_step", 0))
        if step_index >= len(script):
            yield SimpleNamespace(get_function_calls=lambda: [])
            return

        tool_name, tool_args = script[step_index]
        state["_flow_step"] = step_index + 1
        tool = next(
            tool for tool in self.app.root_agent.tools if getattr(tool, "__name__", "") == tool_name
        )
        context = _FakeToolContext(
            state=state,
            saved_artifacts=[],
            artifact_service=self.artifact_service,
            app_name=self.app.name,
            user_id=user_id,
            session_id=session_id,
        )
        await tool(**tool_args, tool_context=context)
        yield SimpleNamespace(
            get_function_calls=lambda: [SimpleNamespace(name=tool_name)],
        )


def _install_fake_runner(
    monkeypatch: Any,
    *,
    flow_script: list[tuple[str, dict[str, Any]]],
    captured_kwargs: dict[str, Any] | None = None,
) -> None:
    def fake_build_registration_review_agent(**kwargs: Any) -> Any:
        if captured_kwargs is not None:
            captured_kwargs.clear()
            captured_kwargs.update(kwargs)
        return SimpleNamespace(
            tools=[
                registration_tools.generate_registration_candidate,
                registration_tools.confirm_registration,
            ],
            flow_script=flow_script,
            kwargs=kwargs,
        )

    monkeypatch.setattr(registration_runner, "App", _FakeApp)
    monkeypatch.setattr(registration_runner, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(
        registration_runner,
        "build_registration_review_agent",
        fake_build_registration_review_agent,
    )


def test_generate_confirm_flow(monkeypatch: Any, tmp_path: Path) -> None:
    pipeline_calls: list[dict[str, Any]] = []
    captured_kwargs: dict[str, Any] = {}
    _install_pipeline_fake(monkeypatch, calls=pipeline_calls)
    _install_fake_runner(
        monkeypatch,
        flow_script=[
            ("generate_registration_candidate", {}),
            (
                "confirm_registration",
                {
                    "candidate_id": "candidate-1",
                    "reasoning": "The first preview best matches the slice.",
                },
            ),
        ],
        captured_kwargs=captured_kwargs,
    )

    candidate = asyncio.run(
        run_registration_review_session(
            image=Image.new("RGB", (12, 9), color=(64, 64, 64)),
            atlas_name="allen_mouse_25um",
            position_mm=1.25,
            image_provider="openai",
            image_model="gpt-image-2",
            model="gemini-3-flash-preview",
            pipeline_review_model="gpt-4.1",
            openai_image_route="images",
            debug_dir=str(tmp_path),
            max_candidates=3,
        )
    )

    assert candidate.candidate_id == "candidate-1"
    assert candidate.metadata["selection_reasoning"] == (
        "The first preview best matches the slice."
    )
    assert candidate.annotation_session.metadata["selection_reasoning"] == (
        "The first preview best matches the slice."
    )
    assert len(pipeline_calls) == 1
    assert pipeline_calls[0]["candidate_id"] == "candidate-1"
    assert pipeline_calls[0]["provider"] == "openai"
    assert pipeline_calls[0]["image_model"] == "gpt-image-2"
    assert pipeline_calls[0]["review_model"] == "gpt-4.1"
    assert pipeline_calls[0]["debug_dir"] == str(tmp_path)
    assert captured_kwargs["max_candidates"] == 3

    runner = _FakeRunner.last_instance
    assert runner is not None
    saved_names = {filename for (_, _, _, filename) in runner.artifact_service.saved.keys()}
    assert {"registration_source_slice.png"} <= saved_names
    assert any(name.startswith("candidate-1_") for name in saved_names)


def test_retry_generate_confirm_flow(monkeypatch: Any) -> None:
    pipeline_calls: list[dict[str, Any]] = []
    _install_pipeline_fake(monkeypatch, calls=pipeline_calls)
    _install_fake_runner(
        monkeypatch,
        flow_script=[
            ("generate_registration_candidate", {}),
            ("generate_registration_candidate", {"prompt_revision": "tighten ventricle edges"}),
            (
                "confirm_registration",
                {
                    "candidate_id": "candidate-2",
                    "reasoning": "The revised preview fits better.",
                },
            ),
        ],
    )

    candidate = asyncio.run(
        run_registration_review_session(
            image=Image.new("RGB", (12, 9), color=(64, 64, 64)),
            atlas_name="allen_mouse_25um",
            position_mm=1.25,
            image_provider="google",
            image_model="gemini-3-flash-preview",
            model="gemini-3-flash-preview",
            openai_image_route="images",
            max_candidates=3,
        )
    )

    assert candidate.candidate_id == "candidate-2"
    assert candidate.metadata["selection_reasoning"] == "The revised preview fits better."
    assert len(pipeline_calls) == 2
    assert pipeline_calls[0]["candidate_id"] == "candidate-1"
    assert pipeline_calls[1]["candidate_id"] == "candidate-2"
    assert pipeline_calls[1]["previous_candidate_id"] == "candidate-1"
    assert pipeline_calls[1]["prompt_revision"] == "tighten ventricle edges"


def test_run_registration_review_session_clamps_max_candidates_to_three(
    monkeypatch: Any,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    _install_pipeline_fake(monkeypatch, calls=[])
    _install_fake_runner(
        monkeypatch,
        flow_script=[
            ("generate_registration_candidate", {}),
            (
                "confirm_registration",
                {
                    "candidate_id": "candidate-1",
                    "reasoning": "The preview is fine.",
                },
            ),
        ],
        captured_kwargs=captured_kwargs,
    )

    candidate = asyncio.run(
        run_registration_review_session(
            image=Image.new("RGB", (12, 9), color=(64, 64, 64)),
            atlas_name="allen_mouse_25um",
            position_mm=1.25,
            image_provider="openai",
            image_model="gpt-image-2",
            model="gemini-3-flash-preview",
            pipeline_review_model="gpt-4.1",
            openai_image_route="images",
            max_candidates=4,
        )
    )

    assert candidate.candidate_id == "candidate-1"
    assert captured_kwargs["max_candidates"] == 3


def test_run_registration_review_session_resolves_adk_review_model(
    monkeypatch: Any,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    resolved_model = object()
    _install_pipeline_fake(monkeypatch, calls=[])
    _install_fake_runner(
        monkeypatch,
        flow_script=[
            ("generate_registration_candidate", {}),
            (
                "confirm_registration",
                {
                    "candidate_id": "candidate-1",
                    "reasoning": "The preview is fine.",
                },
            ),
        ],
        captured_kwargs=captured_kwargs,
    )
    monkeypatch.setattr(registration_runner, "resolve_adk_model", lambda model: resolved_model)

    candidate = asyncio.run(
        run_registration_review_session(
            image=Image.new("RGB", (12, 9), color=(64, 64, 64)),
            atlas_name="allen_mouse_25um",
            position_mm=1.25,
            image_provider="openai",
            image_model="gpt-image-2",
            model="gpt-4.1",
            pipeline_review_model="gpt-4.1",
            openai_image_route="images",
        )
    )

    assert candidate.candidate_id == "candidate-1"
    assert captured_kwargs["model"] is resolved_model


def test_run_registration_review_session_rejects_nonpositive_max_candidates() -> None:
    with pytest.raises(ValueError, match="max_candidates must be a positive integer"):
        asyncio.run(
            run_registration_review_session(
                image=Image.new("RGB", (12, 9), color=(64, 64, 64)),
                atlas_name="allen_mouse_25um",
                position_mm=1.25,
                max_candidates=0,
            )
        )


def test_run_registration_review_session_forwards_callbacks(
    monkeypatch: Any,
) -> None:
    pipeline_calls: list[dict[str, Any]] = []
    progress_messages: list[str] = []
    trace_events: list[dict[str, Any]] = []
    captured_kwargs: dict[str, Any] = {}
    _install_pipeline_fake(monkeypatch, calls=pipeline_calls, emit_callbacks=True)

    def fake_build_registration_review_agent(**kwargs: Any) -> Any:
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        generate_tool = make_generate_registration_candidate_tool(
            on_progress=kwargs.get("on_progress"),
            on_trace=kwargs.get("on_trace"),
        )
        return SimpleNamespace(
            tools=[
                generate_tool,
                registration_tools.confirm_registration,
            ],
            flow_script=[
                ("generate_registration_candidate", {}),
                (
                    "confirm_registration",
                    {
                        "candidate_id": "candidate-1",
                        "reasoning": "callback review",
                    },
                ),
            ],
            kwargs=kwargs,
        )

    monkeypatch.setattr(registration_runner, "App", _FakeApp)
    monkeypatch.setattr(registration_runner, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(
        registration_runner,
        "build_registration_review_agent",
        fake_build_registration_review_agent,
    )

    candidate = asyncio.run(
        run_registration_review_session(
            image=Image.new("RGB", (12, 9), color=(64, 64, 64)),
            atlas_name="allen_mouse_25um",
            position_mm=1.25,
            image_provider="openai",
            image_model="gpt-image-2",
            model="gemini-3-flash-preview",
            pipeline_review_model="gpt-4.1",
            openai_image_route="images",
            on_progress=progress_messages.append,
            on_trace=trace_events.append,
        )
    )

    assert candidate.candidate_id == "candidate-1"
    assert progress_messages
    assert any("candidate" in message.lower() for message in progress_messages)
    assert trace_events
    assert pipeline_calls[0]["on_progress"] is not None
    assert pipeline_calls[0]["on_trace"] is not None
    assert captured_kwargs["on_progress"] is not None
    assert captured_kwargs["on_trace"] is not None


def test_run_registration_review_session_requires_no_active_event_loop(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        registration_runtime.asyncio,
        "get_running_loop",
        lambda: object(),
    )

    async def fake_review_session(**kwargs: Any) -> RegistrationCandidate:
        del kwargs
        return _make_candidate("candidate-1")

    monkeypatch.setattr(
        registration_runtime,
        "run_registration_review_session",
        fake_review_session,
    )

    with pytest.raises(RegistrationFailure, match="active event loop"):
        registration_runtime._run_registration_review_session_sync(
            image=Image.new("RGB", (12, 9), color=(64, 64, 64)),
            atlas_name="allen_mouse_25um",
            position_mm=1.25,
            image_provider="google",
            image_model=None,
            review_model=None,
            pipeline_review_model=None,
            openai_image_route="images",
            max_candidates=3,
            debug_dir=None,
            on_progress=None,
            on_trace=None,
        )


def test_generate_registration_candidate_rejects_fourth_attempt() -> None:
    state = {
        "atlas_name": "allen_mouse_25um",
        "position_mm": 1.25,
        "image_artifact_name": "registration_source_slice.png",
        "image_provider": "openai",
        "image_model": "gpt-image-2",
        "openai_image_route": "images",
        "review_model": "gpt-4.1",
        "debug_dir": None,
        "max_candidates": 4,
        "registration_candidates": [
            {"candidate_id": "candidate-1"},
            {"candidate_id": "candidate-2"},
            {"candidate_id": "candidate-3"},
        ],
    }
    context = _FakeToolContext(
        state=state,
        saved_artifacts=[],
        artifact_service=None,
        app_name="",
        user_id="",
        session_id="",
    )

    result = asyncio.run(generate_registration_candidate(tool_context=context))

    assert result == {
        "status": "error",
        "error": "MAX_CANDIDATES_EXCEEDED",
        "max_candidates": 3,
        "candidate_count": 3,
    }
