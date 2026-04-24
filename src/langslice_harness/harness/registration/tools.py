"""ADK tools for registration candidate review."""

from __future__ import annotations

import copy
import inspect
import io
from typing import Any

from google.genai import types
from PIL import Image

from langslice_harness.agent_trace import format_json
from langslice_harness.harness.estimation.adk_plugins import MULTIMODAL_PARTS_RESULT_KEY
from langslice_harness.harness.registration.image_gen_registration import (
    generate_registration_candidate as _generate_registration_candidate_pipeline,
)
from langslice_harness.harness.registration.types import RegistrationCandidate
from langslice_harness.registration.types import annotation_session_to_dict

_STATE_CANDIDATES_KEY = "registration_candidates"
_STATE_RESULT_KEY = "result"
_STATE_MAX_CANDIDATES_KEY = "max_candidates"
_STATE_IMAGE_ARTIFACT_KEY = "image_artifact_name"
_STATE_IMAGE_PROVIDER_KEY = "image_provider"
_STATE_IMAGE_MODEL_KEY = "image_model"
_STATE_OPENAI_IMAGE_ROUTE_KEY = "openai_image_route"
_STATE_REVIEW_MODEL_KEY = "review_model"
_STATE_DEBUG_DIR_KEY = "debug_dir"
_STATE_LAST_PROMPT_REVISION_KEY = "prompt_revision"
_HARDCAP_MAX_CANDIDATES = 3

_ARTIFACT_SEGMENTATION = "generated_segmentation"
_ARTIFACT_WARPED_ATLAS = "warped_atlas"
_ARTIFACT_BORDER_OVERLAY = "warped_border_overlay"


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _image_part(image: Image.Image, *, label: str) -> types.Part:
    del label
    return types.Part.from_bytes(mime_type="image/png", data=_png_bytes(image))


def _artifact_name(candidate_id: str, role: str) -> str:
    return f"{candidate_id}_{role}.png"


def _artifact_part_name_map(candidate_id: str) -> dict[str, str]:
    return {
        _ARTIFACT_SEGMENTATION: _artifact_name(candidate_id, _ARTIFACT_SEGMENTATION),
        _ARTIFACT_WARPED_ATLAS: _artifact_name(candidate_id, _ARTIFACT_WARPED_ATLAS),
        _ARTIFACT_BORDER_OVERLAY: _artifact_name(candidate_id, _ARTIFACT_BORDER_OVERLAY),
    }


def _candidate_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    records = state.get(_STATE_CANDIDATES_KEY, [])
    if isinstance(records, list):
        return [dict(record) for record in records if isinstance(record, dict)]
    return []


def _effective_max_candidates(state: dict[str, Any]) -> int:
    requested = int(state.get(_STATE_MAX_CANDIDATES_KEY, _HARDCAP_MAX_CANDIDATES))
    if requested <= 0:
        raise ValueError("max_candidates must be a positive integer")
    return min(requested, _HARDCAP_MAX_CANDIDATES)


async def _load_source_image(tool_context: Any) -> Image.Image:
    state = tool_context.state
    artifact_name = state.get(_STATE_IMAGE_ARTIFACT_KEY)
    if not isinstance(artifact_name, str) or not artifact_name:
        raise RuntimeError("Registration session is missing the source image artifact name")

    artifact = tool_context.load_artifact(artifact_name)
    if inspect.isawaitable(artifact):
        artifact = await artifact
    if artifact is None:
        raise RuntimeError(f"Source image artifact {artifact_name!r} was not found")

    inline_data = getattr(artifact, "inline_data", None)
    data = getattr(inline_data, "data", None) if inline_data is not None else None
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise RuntimeError(f"Source image artifact {artifact_name!r} did not contain image bytes")

    image = Image.open(io.BytesIO(bytes(data)))
    image.load()
    return image.convert("RGB")


def _candidate_record(
    candidate: RegistrationCandidate,
    *,
    artifact_names: dict[str, str],
    previous_candidate_id: str | None,
    prompt_revision: str | None,
) -> dict[str, Any]:
    session_dict = annotation_session_to_dict(candidate.annotation_session) or {}
    generated = candidate.metadata.get("generated", {})
    record: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "provider": generated.get("provider"),
        "model": generated.get("model"),
        "route": generated.get("route"),
        "markers_count": len(candidate.markers),
        "artifact_names": dict(artifact_names),
        "candidate_metadata": copy.deepcopy(candidate.metadata),
        "annotation_session": session_dict,
    }
    if previous_candidate_id is not None:
        record["previous_candidate_id"] = previous_candidate_id
    if prompt_revision is not None:
        record["prompt_revision"] = prompt_revision
    return record


async def _save_candidate_artifacts(
    *,
    candidate: RegistrationCandidate,
    candidate_id: str,
    tool_context: Any,
) -> dict[str, str]:
    artifact_names = _artifact_part_name_map(candidate_id)
    artifacts = {
        _ARTIFACT_SEGMENTATION: candidate.generated_segmentation,
        _ARTIFACT_WARPED_ATLAS: candidate.warped_atlas,
        _ARTIFACT_BORDER_OVERLAY: candidate.warped_border_overlay,
    }
    for role, image in artifacts.items():
        part = _image_part(image, label=role.replace("_", " ").title())
        await tool_context.save_artifact(filename=artifact_names[role], artifact=part)
    return artifact_names


def _with_multimodal_parts(result: dict[str, Any], parts: list[types.Part]) -> dict[str, Any]:
    out = dict(result)
    out[MULTIMODAL_PARTS_RESULT_KEY] = parts
    return out


async def _generate_registration_candidate_impl(
    *,
    prompt_revision: str | None = None,
    tool_context: Any = None,
    on_progress: Any = None,
    on_trace: Any = None,
) -> dict[str, Any]:
    """Generate one dense registration candidate and store it in session state."""
    if tool_context is None:
        raise RuntimeError("tool_context is required")

    state = tool_context.state
    candidates = _candidate_records(state)
    max_candidates = _effective_max_candidates(state)
    if len(candidates) >= max_candidates:
        return {
            "status": "error",
            "error": "MAX_CANDIDATES_EXCEEDED",
            "max_candidates": max_candidates,
            "candidate_count": len(candidates),
        }

    source_image = await _load_source_image(tool_context)
    previous_candidate_id = candidates[-1]["candidate_id"] if candidates else None
    candidate_index = len(candidates) + 1
    candidate_id = f"candidate-{candidate_index}"
    active_prompt_revision = prompt_revision

    generated = _generate_registration_candidate_pipeline(
        source_image,
        atlas_name=str(state["atlas_name"]),
        position_mm=float(state["position_mm"]),
        provider=str(state.get(_STATE_IMAGE_PROVIDER_KEY, state.get("provider", "google"))),
        image_model=state.get(_STATE_IMAGE_MODEL_KEY),
        prompt_revision=active_prompt_revision,
        previous_candidate_id=previous_candidate_id,
        candidate_id=candidate_id,
        debug_dir=state.get(_STATE_DEBUG_DIR_KEY),
        on_progress=on_progress,
        on_trace=on_trace,
        openai_image_route=str(state.get(_STATE_OPENAI_IMAGE_ROUTE_KEY, "images")),
        review_model=state.get(_STATE_REVIEW_MODEL_KEY),
    )

    artifact_names = await _save_candidate_artifacts(
        candidate=generated,
        candidate_id=candidate_id,
        tool_context=tool_context,
    )
    record = _candidate_record(
        generated,
        artifact_names=artifact_names,
        previous_candidate_id=previous_candidate_id,
        prompt_revision=active_prompt_revision,
    )
    candidates.append(record)
    state[_STATE_CANDIDATES_KEY] = candidates
    state["candidate_count"] = len(candidates)
    state["last_candidate_id"] = candidate_id
    if active_prompt_revision is not None:
        state[_STATE_LAST_PROMPT_REVISION_KEY] = active_prompt_revision

    generated_parts = [
        types.Part.from_text(
            text=(
                f"Candidate {candidate_id}: {record['markers_count']} markers, "
                f"provider={record['provider']}, model={record['model']}, route={record['route']}"
            )
        ),
        _image_part(generated.generated_segmentation, label="Model-generated segmentation"),
        _image_part(generated.warped_atlas, label="Elastix-warped atlas"),
        _image_part(
            generated.warped_border_overlay,
            label="Warped atlas borders over histology slice",
        ),
        types.Part.from_text(text=format_json(record)),
    ]

    return _with_multimodal_parts(
        {
            "status": "ok",
            "candidate_id": candidate_id,
            "markers_count": record["markers_count"],
            "provider": record["provider"],
            "model": record["model"],
            "route": record["route"],
            "artifact_names": artifact_names,
            "previous_candidate_id": previous_candidate_id,
            "prompt_revision": active_prompt_revision,
            "candidate_metadata": copy.deepcopy(record["candidate_metadata"]),
            "annotation_session": copy.deepcopy(record["annotation_session"]),
        },
        generated_parts,
    )


def make_generate_registration_candidate_tool(
    *,
    on_progress: Any = None,
    on_trace: Any = None,
) -> Any:
    """Create the ADK tool wrapper with optional callback plumbing."""

    async def generate_registration_candidate(
        prompt_revision: str | None = None,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        return await _generate_registration_candidate_impl(
            prompt_revision=prompt_revision,
            tool_context=tool_context,
            on_progress=on_progress,
            on_trace=on_trace,
        )

    return generate_registration_candidate


async def generate_registration_candidate(
    prompt_revision: str | None = None,
    tool_context: Any = None,
) -> dict[str, Any]:
    return await _generate_registration_candidate_impl(
        prompt_revision=prompt_revision,
        tool_context=tool_context,
    )


async def confirm_registration(
    candidate_id: str,
    reasoning: str,
    tool_context: Any,
) -> dict[str, Any]:
    """Confirm one stored candidate and escalate back to the agent."""
    state = tool_context.state
    candidates = _candidate_records(state)
    selected = next(
        (record for record in candidates if record.get("candidate_id") == candidate_id), None
    )
    if selected is None:
        return {
            "status": "error",
            "error": "UNKNOWN_CANDIDATE",
            "candidate_id": candidate_id,
        }

    result = copy.deepcopy(selected)
    result["reasoning"] = str(reasoning)
    state[_STATE_RESULT_KEY] = result
    state["selected_candidate_id"] = candidate_id
    state["selected_reasoning"] = str(reasoning)
    tool_context.actions.escalate = True
    return {
        "status": "ok",
        "candidate_id": candidate_id,
        "reasoning": str(reasoning),
    }
