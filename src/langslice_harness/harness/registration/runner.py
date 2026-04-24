"""Runner wrapper for the optional registration review agent."""

from __future__ import annotations

import copy
import io
from typing import Any

from google.adk.apps.app import App
from google.adk.runners import InMemoryRunner
from google.genai import types
from PIL import Image

from langslice_harness.harness.estimation.adk_plugins import PersistentMultimodalToolResultsPlugin
from langslice_harness.harness.estimation.model_resolver import resolve_adk_model
from langslice_harness.harness.registration.agent import build_registration_review_agent
from langslice_harness.harness.registration.tools import _HARDCAP_MAX_CANDIDATES
from langslice_harness.harness.registration.types import RegistrationCandidate
from langslice_harness.registration.types import RegistrationAnnotationSession

_APP_NAME = "langslice"
_USER_ID = "langslice-user"
_SOURCE_IMAGE_ARTIFACT = "registration_source_slice.png"
_DEFAULT_MAX_TURNS = 8


def _png_part(image: Image.Image) -> types.Part:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return types.Part.from_bytes(mime_type="image/png", data=buffer.getvalue())


def _image_from_part(part: Any) -> Image.Image:
    inline = getattr(part, "inline_data", None)
    data = getattr(inline, "data", None) if inline is not None else None
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise RuntimeError("Artifact did not contain inline image bytes")
    image = Image.open(io.BytesIO(bytes(data)))
    image.load()
    return image.convert("RGB")


def _coerce_annotation_session(data: dict[str, Any] | None) -> RegistrationAnnotationSession:
    session_data = dict(data or {})
    session_data.setdefault("workflow", "image_gen_registration")
    session_data.setdefault("metadata", {})
    session_data.setdefault("target_count", 0)
    session_data.setdefault("atlas_annotations", [])
    session_data.setdefault("slice_annotations", [])
    return RegistrationAnnotationSession(**session_data)


def _markers_from_record(record: dict[str, Any]) -> list[list[float]]:
    session = record.get("annotation_session")
    if isinstance(session, dict):
        metadata = session.get("metadata", {})
        if isinstance(metadata, dict):
            markers = metadata.get("visualign_markers", [])
            if isinstance(markers, list):
                return [list(map(float, marker)) for marker in markers if isinstance(marker, list)]
    return []


async def _load_candidate_artifact(
    *,
    runner: InMemoryRunner,
    session_id: str,
    filename: str,
) -> Image.Image:
    assert runner.artifact_service is not None
    part = await runner.artifact_service.load_artifact(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        session_id=session_id,
        filename=filename,
    )
    if part is None:
        raise RuntimeError(f"Candidate artifact {filename!r} was not found")
    return _image_from_part(part)


async def _candidate_from_record(
    *,
    runner: InMemoryRunner,
    session_id: str,
    record: dict[str, Any],
) -> RegistrationCandidate:
    artifact_names = dict(record.get("artifact_names", {}))
    segmentation = await _load_candidate_artifact(
        runner=runner,
        session_id=session_id,
        filename=str(artifact_names["generated_segmentation"]),
    )
    warped_atlas = await _load_candidate_artifact(
        runner=runner,
        session_id=session_id,
        filename=str(artifact_names["warped_atlas"]),
    )
    warped_border_overlay = await _load_candidate_artifact(
        runner=runner,
        session_id=session_id,
        filename=str(artifact_names["warped_border_overlay"]),
    )

    session = _coerce_annotation_session(record.get("annotation_session"))
    metadata = copy.deepcopy(record.get("candidate_metadata", {}))
    candidate = RegistrationCandidate(
        candidate_id=str(record["candidate_id"]),
        generated_segmentation=segmentation,
        warped_atlas=warped_atlas,
        warped_border_overlay=warped_border_overlay,
        markers=_markers_from_record(record),
        annotation_session=session,
        metadata=metadata,
    )
    if "reasoning" in record:
        candidate.metadata["selection_reasoning"] = str(record["reasoning"])
        candidate.annotation_session.metadata["selection_reasoning"] = str(record["reasoning"])
    return candidate


async def run_registration_review_session(
    *,
    image: Image.Image,
    atlas_name: str,
    position_mm: float,
    image_provider: str = "google",
    image_model: str | None = None,
    model: str | object | None = None,
    pipeline_review_model: str | None = None,
    openai_image_route: str = "images",
    max_candidates: int = 3,
    debug_dir: str | None = None,
    on_progress: Any = None,
    on_trace: Any = None,
    temperature: float = 0.2,
    media_resolution: str = "MEDIA_RESOLUTION_MEDIUM",
    thinking_config: object | None = None,
    max_turns: int = _DEFAULT_MAX_TURNS,
) -> RegistrationCandidate:
    """Run the registration review agent to completion and return the selected candidate."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")

    if model is None:
        from langslice_harness import vlm_config

        model = vlm_config.MODEL_NAME
    agent_model = resolve_adk_model(model)

    effective_max_candidates = min(int(max_candidates), _HARDCAP_MAX_CANDIDATES)

    agent = build_registration_review_agent(
        atlas_name=atlas_name,
        position_mm=position_mm,
        image_provider=image_provider,
        image_model=image_model,
        model=agent_model,
        openai_image_route=openai_image_route,
        max_candidates=effective_max_candidates,
        temperature=temperature,
        media_resolution=media_resolution,
        thinking_config=thinking_config,
        on_progress=on_progress,
        on_trace=on_trace,
    )
    app = App(
        name=_APP_NAME,
        root_agent=agent,
        plugins=[PersistentMultimodalToolResultsPlugin(persistent=True)],
    )
    runner = InMemoryRunner(app=app)
    assert runner.artifact_service is not None
    assert runner.session_service is not None

    session_id = "registration_review_session"
    state: dict[str, Any] = {
        "workflow": "image_gen_registration",
        "atlas_name": atlas_name,
        "position_mm": float(position_mm),
        "image_provider": image_provider,
        "provider": image_provider,
        "image_model": image_model,
        "review_model": pipeline_review_model,
        "openai_image_route": openai_image_route,
        "max_candidates": effective_max_candidates,
        "debug_dir": debug_dir,
        "image_artifact_name": _SOURCE_IMAGE_ARTIFACT,
        "registration_candidates": [],
        "candidate_count": 0,
        "result": None,
    }
    await runner.session_service.create_session(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        session_id=session_id,
        state=state,
    )

    source_part = _png_part(image)
    await runner.artifact_service.save_artifact(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        session_id=session_id,
        filename=_SOURCE_IMAGE_ARTIFACT,
        artifact=source_part,
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=(
                    "Review the image-gen registration preview. "
                    "Generate a candidate, inspect the three image outputs, "
                    "and confirm the best one when you are satisfied."
                )
            ),
            source_part,
        ],
    )

    final_session = None
    for _ in range(max_turns):
        async for _event in runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            new_message=message,
        ):
            pass

        final_session = await runner.session_service.get_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=session_id,
        )
        if final_session is not None and final_session.state.get("result") is not None:
            break
        message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "Continue the review. Generate another candidate if needed "
                        "or confirm the best existing candidate."
                    )
                )
            ],
        )

    if final_session is None or final_session.state.get("result") is None:
        raise RuntimeError("Registration review session did not confirm a candidate")

    result_record = dict(final_session.state["result"])
    candidate = await _candidate_from_record(
        runner=runner,
        session_id=session_id,
        record=result_record,
    )
    return candidate
