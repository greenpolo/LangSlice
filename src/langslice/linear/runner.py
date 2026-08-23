"""Runner wrapper: drives nudge-on-no-tool-call, retry-with-fresh-session, and max-iteration cap."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google.adk.apps.app import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.genai import types
from PIL import Image

from langslice.adk.model_resolver import (
    _env,
    _env_float,
    resolve_adk_model,
)
from langslice.adk.plugins import (
    ModelCallPacingPlugin,
    RequestCapturePlugin,
)
from langslice.atlas.core import (
    get_in_plane_long_edge,
    get_position_range_mm,
    load_atlas,
)
from langslice.image_prep import (
    adaptive_preprocess,
    normalize_image,
    prepare_image_for_vlm,
)
from langslice.linear._types import MultiSliceResult, PositionResult
from langslice.linear.group import build_group_agent
from langslice.linear.session import (
    ARTIFACT_TARGET,
    build_initial_state,
    target_key,
)
from langslice.linear.single_slice import build_single_slice_agent
from langslice.space import Plane

logger = logging.getLogger(__name__)

_APP_NAME = "langslice"
_USER_ID = "langslice-user"

_DEFAULT_MAX_ITERATIONS_SINGLE = 20
_DEFAULT_MAX_ITERATIONS_GROUP = 25
_DEFAULT_MAX_RETRIES = 2
_THOUGHT_LEAK_CANDIDATE_TOKEN_THRESHOLD = 1000

_MEDIA_RES_MAP: dict[str, str] = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
}

_TARGET_TRANSPORTS = {"auto", "inline", "file_api"}

_NUDGE_BROAD = (
    "Please continue. Call `fetch_atlas` with widely spaced positions "
    "(e.g., [2, 4, 6, 8, 10]) to find the correct neighborhood."
)
_NUDGE_NARROW = (
    "Please narrow down. Call `fetch_atlas` with tightly spaced positions "
    "around your best candidate (e.g., [4.0, 4.2, 4.4, 4.6, 4.8])."
)
_NUDGE_VERIFY = (
    "Please continue. Verify your candidate by checking nearby positions "
    "with `fetch_atlas`, or call `submit_estimate` if confident."
)


def _pick_nudge(state: dict[str, Any]) -> str:
    if not state.get("saw_broad_sweep"):
        return _NUDGE_BROAD
    if not state.get("saw_narrow_sweep"):
        return _NUDGE_NARROW
    return _NUDGE_VERIFY


def _thought_leak_nudge(submit_tool: str) -> str:
    return (
        "You wrote a long text response instead of calling a tool. "
        "Do NOT repeat this. Briefly state your reasoning, then call a tool: "
        f"`fetch_atlas` or `{submit_tool}`."
    )


def _event_candidate_token_count(event: Any) -> int:
    usage = getattr(event, "usage_metadata", None)
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get("candidates_token_count", 0)
    else:
        value = getattr(usage, "candidates_token_count", 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_thought_leak_event(event: Any, function_calls: list[Any]) -> bool:
    if function_calls:
        return False
    return _event_candidate_token_count(event) > _THOUGHT_LEAK_CANDIDATE_TOKEN_THRESHOLD


def _normalize_media_resolution(media_resolution: str | None) -> str:
    if media_resolution is None:
        return "MEDIA_RESOLUTION_MEDIUM"
    normalized = media_resolution.strip()
    if normalized.startswith("MEDIA_RESOLUTION_"):
        return normalized
    return _MEDIA_RES_MAP.get(normalized.lower(), "MEDIA_RESOLUTION_MEDIUM")


@dataclass(frozen=True)
class _TargetPayload:
    artifact_part: types.Part
    request_part: types.Part


def _build_plugins(
    run_label: str, trace_recorder: BasePlugin | None = None
) -> list[BasePlugin]:
    plugins: list[BasePlugin] = []
    if trace_recorder is not None:
        plugins.append(trace_recorder)
    model_call_delay_s = _env_float("LANGSLICE_ADK_MODEL_CALL_DELAY_S")
    if model_call_delay_s is not None and model_call_delay_s > 0:
        plugins.append(ModelCallPacingPlugin(model_call_delay_s))
    capture_dir = _env("LANGSLICE_ADK_CAPTURE_REQUESTS_DIR")
    if capture_dir is not None:
        plugins.append(RequestCapturePlugin(capture_dir, run_label=run_label))
    return plugins


def _encode_target_jpeg_bytes(
    image: Image.Image, atlas_long_edge: int, *, apply_clahe: bool = True,
) -> bytes:
    """Normalize, downscale, optionally CLAHE, and encode as JPEG bytes.

    Order matches whole-brain estimation:
    normalize → downscale-to-atlas-long-edge → CLAHE. CLAHE is on by default.
    """
    normalized = normalize_image(image)
    prepped = prepare_image_for_vlm(normalized, max_long_edge=atlas_long_edge).image
    if apply_clahe:
        prepped = adaptive_preprocess(prepped)
    buf = io.BytesIO()
    prepped.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _image_bytes_to_part(jpeg_bytes: bytes) -> types.Part:
    return types.Part.from_bytes(mime_type="image/jpeg", data=jpeg_bytes)


def _is_native_google_genai_model(model: str | object) -> bool:
    if not isinstance(model, str):
        return False
    normalized = model.strip().lower()
    return normalized.startswith(("gemini-", "gemma-")) or normalized.startswith((
        "models/gemini-",
        "models/gemma-",
    ))


def _resolve_target_transport(model: str | object, target_transport: str) -> str:
    normalized = target_transport.strip().lower()
    if normalized not in _TARGET_TRANSPORTS:
        allowed = ", ".join(sorted(_TARGET_TRANSPORTS))
        raise ValueError(f"target_transport must be one of: {allowed}")
    if normalized == "inline":
        return "inline"

    from langslice.providers import vlm_config

    can_use_file_api = _is_native_google_genai_model(model) and vlm_config.supports_file_api()
    if normalized == "file_api":
        if not can_use_file_api:
            raise ValueError(
                "target_transport='file_api' requires a native Gemini API model "
                "and a backend with File API support"
            )
        return "file_api"
    return "file_api" if can_use_file_api else "inline"


def _upload_target_image(
    *,
    jpeg_bytes: bytes,
    display_name: str,
) -> tuple[Any, Any, types.Part]:
    from langslice.adk.sdk_helpers import _wait_for_uploaded_file
    from langslice.providers import vlm_config

    client = vlm_config.get_client()
    uploaded = client.files.upload(
        file=io.BytesIO(jpeg_bytes),
        config=types.UploadFileConfig(
            mime_type="image/jpeg",
            display_name=display_name,
        ),
    )
    file_name = getattr(uploaded, "name", None)
    if not isinstance(file_name, str) or not file_name:
        raise RuntimeError("Gemini File API upload for target image returned no file name")

    active = _wait_for_uploaded_file(
        client,
        file_name=file_name,
        timeout_s=vlm_config.file_poll_timeout_s(),
    )
    file_uri = getattr(active, "uri", None) or getattr(uploaded, "uri", None)
    if not isinstance(file_uri, str) or not file_uri:
        raise RuntimeError("Gemini File API upload for target image returned no URI")

    return client, uploaded, types.Part.from_uri(
        file_uri=file_uri,
        mime_type="image/jpeg",
    )


def _prepare_target_payloads(
    images: list[Image.Image],
    *,
    atlas_long_edge: int,
    apply_clahe: bool,
    model: str | object,
    target_transport: str,
) -> tuple[list[_TargetPayload], list[tuple[Any, Any]]]:
    resolved_transport = _resolve_target_transport(model, target_transport)
    payloads: list[_TargetPayload] = []
    uploaded_files: list[tuple[Any, Any]] = []
    for idx, image in enumerate(images, start=1):
        jpeg_bytes = _encode_target_jpeg_bytes(
            image,
            atlas_long_edge,
            apply_clahe=apply_clahe,
        )
        artifact_part = _image_bytes_to_part(jpeg_bytes)
        request_part = artifact_part
        if resolved_transport == "file_api":
            client, uploaded, request_part = _upload_target_image(
                jpeg_bytes=jpeg_bytes,
                display_name=f"target_slice_{idx}",
            )
            uploaded_files.append((client, uploaded))
        payloads.append(_TargetPayload(
            artifact_part=artifact_part,
            request_part=request_part,
        ))
    return payloads, uploaded_files


def _delete_uploaded_files(uploaded_files: list[tuple[Any, Any]]) -> None:
    for client, uploaded in uploaded_files:
        file_name = getattr(uploaded, "name", None)
        if not isinstance(file_name, str) or not file_name:
            continue
        try:
            client.files.delete(name=file_name)
        except Exception as exc:
            logger.warning("Failed to delete Gemini file %s: %s", file_name, exc)


async def _run_estimation_session(
    *,
    images: list[Image.Image],
    atlas_name: str,
    plane: Plane,
    model: str | object,
    species: str | None,
    max_iterations: int,
    max_retries: int,
    thinking_level: str,
    apply_clahe: bool,
    target_transport: str,
    trace_recorder: BasePlugin | None,
    include_thought_summaries: bool,
    run_label: str,
    session_id_prefix: str,
    submit_tool_name: str,
    n_slices: int,
    interval_mm: float,
    thickness_um: int,
    build_agent: Callable[[str, float, float, object, object | None], Any],
    build_new_message: Callable[[list[_TargetPayload]], types.Content],
    build_result: Callable[[dict[str, Any]], Any],
    build_fallback: Callable[[float, float], Any],
) -> Any:
    """Shared attempt/retry/nudge loop behind both public session runners.

    Single- and group-slice sessions differ only in agent construction, the
    initial user message, and how a submitted/fallback result is shaped; the
    atlas setup, plugin wiring, retry loop, tool-call counting, thought-leak
    detection, and nudge-on-stall logic are identical and live here.
    """
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=plane)
    if plane == "sagittal":
        # Canonical hemisphere range: the two hemispheres are mirror images
        # so the agent only needs to estimate within one. Halving pos_hi
        # constrains both the prompt's reported range AND fetch_atlas's
        # position clamp (state['pos_hi']) to the canonical hemisphere.
        pos_hi = pos_hi / 2.0
    atlas_long_edge = get_in_plane_long_edge(atlas, plane=plane)

    # MEDIUM thinking is the validated sweet spot for Flash (0.14mm MAE on M01).
    # Only wire a thinking_config for string-named Gemini models; LiteLlm
    # wrappers drive their own thinking knobs.
    thinking_cfg: object | None = None
    if isinstance(model, str) and _is_native_google_genai_model(model):
        from langslice.providers import vlm_config
        thinking_cfg = vlm_config.build_thinking_config(
            model,
            thinking_level,
            include_thoughts=include_thought_summaries,
        )

    agent_model = resolve_adk_model(model)
    species_val = species or str(atlas.metadata.get("species", "mouse"))
    agent = build_agent(species_val, pos_lo, pos_hi, agent_model, thinking_cfg)

    # Atlas images returned by `fetch_atlas` ride along on the function-response
    # event (ADK 2.7+ native media tool results), so session history replays
    # every earlier sweep to the model without any plugin.
    app = App(
        name=_APP_NAME,
        root_agent=agent,
        plugins=_build_plugins(run_label, trace_recorder),
    )
    runner = InMemoryRunner(app=app)
    # InMemoryRunner always wires in-memory services, but the base class types
    # them Optional; assert for the type checker.
    assert runner.artifact_service is not None
    assert runner.session_service is not None

    initial_state_template = build_initial_state(
        atlas_name=atlas_name,
        plane=plane,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        n_slices=n_slices,
        interval_mm=interval_mm,
        thickness_um=thickness_um,
        max_iterations=max_iterations,
    )

    # Prepare inline artifact copies for tools, and optionally Gemini File API
    # request parts for native Gemini parity with the legacy implementation.
    target_payloads, uploaded_files = _prepare_target_payloads(
        images,
        atlas_long_edge=atlas_long_edge,
        apply_clahe=apply_clahe,
        model=model,
        target_transport=target_transport,
    )
    # Single-slice sessions seed one artifact under the shared ARTIFACT_TARGET
    # key; group sessions seed one per slice under 'target:<N>'.
    artifact_specs: list[tuple[str, _TargetPayload]] = (
        [(ARTIFACT_TARGET, target_payloads[0])]
        if run_label == "single_slice"
        else [(target_key(i + 1), payload) for i, payload in enumerate(target_payloads)]
    )

    try:
        for attempt in range(max_retries):
            session_id = f"{session_id_prefix}{attempt}"
            await runner.session_service.create_session(
                app_name=_APP_NAME,
                user_id=_USER_ID,
                session_id=session_id,
                state=dict(initial_state_template),
            )

            # Seed the target image(s) as artifacts so the session retains a
            # debuggable copy of exactly what the model was shown.
            for filename, payload in artifact_specs:
                await runner.artifact_service.save_artifact(
                    app_name=_APP_NAME,
                    user_id=_USER_ID,
                    session_id=session_id,
                    filename=filename,
                    artifact=payload.artifact_part,
                )

            new_message = build_new_message(target_payloads)

            tool_call_count = 0
            capped = False
            turn_count = 0
            final = None
            saw_thought_leak = False
            while turn_count < max_iterations and not capped:
                turn_count += 1
                try:
                    async for event in runner.run_async(
                        user_id=_USER_ID,
                        session_id=session_id,
                        new_message=new_message,
                    ):
                        fcs = event.get_function_calls() or []
                        if _is_thought_leak_event(event, fcs):
                            candidate_tokens = _event_candidate_token_count(event)
                            saw_thought_leak = True
                            logger.warning(
                                "Thought leak detected (%d candidate tokens) on "
                                "%s attempt %d.",
                                candidate_tokens,
                                run_label,
                                attempt + 1,
                            )
                        tool_call_count += len(fcs)
                        if fcs:
                            names = [getattr(fc, "name", "?") for fc in fcs]
                            logger.info(
                                "%s attempt %d: tool call #%d -> %s (total=%d)",
                                run_label,
                                attempt + 1,
                                tool_call_count,
                                names,
                                tool_call_count,
                            )
                        if tool_call_count > max_iterations:
                            logger.warning(
                                "Hit max_iterations=%d on attempt %d; forcing end.",
                                max_iterations,
                                attempt + 1,
                            )
                            capped = True
                            break

                        # Re-read session state after each event (mutated by tools).
                        current = await runner.session_service.get_session(
                            app_name=_APP_NAME,
                            user_id=_USER_ID,
                            session_id=session_id,
                        )
                        if current is not None and current.state.get("result") is not None:
                            break
                except Exception:
                    logger.exception(
                        "%s attempt %d raised inside runner.run_async", run_label, attempt + 1
                    )
                    raise

                final = await runner.session_service.get_session(
                    app_name=_APP_NAME,
                    user_id=_USER_ID,
                    session_id=session_id,
                )
                if final is not None and final.state.get("result") is not None:
                    break

                # Match the legacy loop: if the model stopped without a
                # result, continue the same conversation with a nudge.
                final_state = final.state if final is not None else initial_state_template
                nudge = (
                    _thought_leak_nudge(submit_tool_name)
                    if saw_thought_leak
                    else _pick_nudge(final_state)
                )
                saw_thought_leak = False
                logger.info(
                    "%s attempt %d stopped without a result; continuing same "
                    "session with nudge=%r.",
                    run_label,
                    attempt + 1,
                    nudge,
                )
                new_message = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=nudge)],
                )

            if final is not None and final.state.get("result") is not None:
                return build_result(final.state["result"])

            final_state = final.state if final is not None else initial_state_template
            nudge = _pick_nudge(final_state)
            logger.info(
                "%s attempt %d did not submit (capped=%s); nudge=%r; retrying with "
                "fresh session.",
                run_label,
                attempt + 1,
                capped,
                nudge,
            )
    finally:
        _delete_uploaded_files(uploaded_files)

    return build_fallback(pos_lo, pos_hi)


async def run_single_slice_session(
    *,
    image: Image.Image,
    atlas_name: str,
    plane: Plane = "coronal",
    model: str | object = "gemini-3-flash-preview",
    species: str | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS_SINGLE,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    temperature: float = 1.0,
    thinking_level: str = "MEDIUM",
    media_resolution: str | None = "medium",
    apply_clahe: bool = True,
    target_transport: str = "auto",
    trace_recorder: BasePlugin | None = None,
    include_thought_summaries: bool = False,
) -> PositionResult:
    """Drive a single-slice position-estimation session to completion.

    Seeds the target image as an artifact, runs the ADK single-slice agent,
    counts function-call events against ``max_iterations``, and retries with
    a fresh session up to ``max_retries`` times if the agent does not submit.
    Falls back to the atlas midpoint if all retries are exhausted.
    """
    axis_label = {"coronal": "AP", "sagittal": "ML", "horizontal": "DV"}[plane]

    def _build_agent(
        species_val: str, pos_lo: float, pos_hi: float, agent_model: object,
        thinking_cfg: object | None,
    ) -> Any:
        return build_single_slice_agent(
            atlas_name=atlas_name,
            plane=plane,
            species=species_val,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            model=agent_model,
            temperature=temperature,
            media_resolution=_normalize_media_resolution(media_resolution),
            thinking_config=thinking_cfg,
        )

    def _build_new_message(target_payloads: list[_TargetPayload]) -> types.Content:
        target_payload = target_payloads[0]
        return types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"Target slice (artifact key: '{ARTIFACT_TARGET}'):"
                ),
                target_payload.request_part,
                types.Part.from_text(
                    text=(
                        f"Determine this {plane} slice's {axis_label} "
                        f"position in the {atlas_name} atlas."
                    )
                ),
            ],
        )

    def _build_result(result: dict[str, Any]) -> PositionResult:
        return PositionResult(
            position_mm=float(result["position_mm"]),
            reasoning=str(result["reasoning"]),
        )

    def _build_fallback(pos_lo: float, pos_hi: float) -> PositionResult:
        # Fallback: atlas midpoint. Keep this phrase aligned with group
        # fallback handling and eval fallback detection.
        mid = (pos_lo + pos_hi) / 2.0
        logger.warning(
            "All %d retries exhausted; falling back to %.2f mm midpoint.",
            max_retries,
            mid,
        )
        return PositionResult(
            position_mm=mid,
            reasoning=(
                "Model did not submit within iteration+retry budget; "
                "fell back to atlas midpoint."
            ),
        )

    return await _run_estimation_session(
        images=[image],
        atlas_name=atlas_name,
        plane=plane,
        model=model,
        species=species,
        max_iterations=max_iterations,
        max_retries=max_retries,
        thinking_level=thinking_level,
        apply_clahe=apply_clahe,
        target_transport=target_transport,
        trace_recorder=trace_recorder,
        include_thought_summaries=include_thought_summaries,
        run_label="single_slice",
        session_id_prefix="single_slice_attempt_",
        submit_tool_name="submit_estimate",
        n_slices=1,
        interval_mm=0.0,
        thickness_um=50,
        build_agent=_build_agent,
        build_new_message=_build_new_message,
        build_result=_build_result,
        build_fallback=_build_fallback,
    )


async def run_group_session(
    *,
    images: list[Image.Image],
    atlas_name: str,
    interval_mm: float,
    thickness_um: int = 50,
    plane: Plane = "coronal",
    model: str | object = "gemini-3-flash-preview",
    species: str | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS_GROUP,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    temperature: float = 1.0,
    thinking_level: str = "MEDIUM",
    media_resolution: str | None = "medium",
    apply_clahe: bool = True,
    target_transport: str = "auto",
    trace_recorder: BasePlugin | None = None,
    include_thought_summaries: bool = False,
) -> MultiSliceResult:
    """Drive a multi-slice group position-estimation session to completion.

    Seeds each target image as a distinct ``target:<N>`` artifact, runs the
    ADK group agent, counts function-call events against ``max_iterations``,
    and retries with a fresh session up to ``max_retries`` times if the agent
    does not submit. Falls back to ``n_slices`` positions centered on the
    atlas midpoint with the requested ``interval_mm`` spacing if all retries
    are exhausted.
    """
    n_slices = len(images)
    if not 2 <= n_slices <= 8:
        raise ValueError(f"Expected 2-8 slices, got {n_slices}")

    def _build_agent(
        species_val: str, pos_lo: float, pos_hi: float, agent_model: object,
        thinking_cfg: object | None,
    ) -> Any:
        return build_group_agent(
            atlas_name=atlas_name,
            plane=plane,
            species=species_val,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            n_slices=n_slices,
            interval_mm=interval_mm,
            thickness_um=thickness_um,
            model=agent_model,
            temperature=temperature,
            media_resolution=_normalize_media_resolution(media_resolution),
            thinking_config=thinking_cfg,
        )

    def _build_new_message(target_payloads: list[_TargetPayload]) -> types.Content:
        parts: list[types.Part] = []
        for i, payload in enumerate(target_payloads):
            parts.append(
                types.Part.from_text(
                    text=f"Slice {i + 1} (artifact: '{target_key(i + 1)}'):"
                )
            )
            parts.append(payload.request_part)
        parts.append(types.Part.from_text(text="Determine the position of each slice."))
        return types.Content(role="user", parts=parts)

    def _build_result(result: dict[str, Any]) -> MultiSliceResult:
        reasoning = str(result["reasoning"])
        positions = [
            PositionResult(position_mm=float(p), reasoning=reasoning)
            for p in result["positions_mm"]
        ]
        return MultiSliceResult(positions=positions, group_reasoning=reasoning)

    def _build_fallback(pos_lo: float, pos_hi: float) -> MultiSliceResult:
        # Fallback: center the group around the atlas midpoint with requested
        # interval spacing, clamped to the atlas range. Keep the reasoning
        # phrase aligned with run_single_slice_session for consistent traces.
        mid = (pos_lo + pos_hi) / 2.0
        span = (n_slices - 1) * interval_mm
        start = mid - span / 2.0
        fallback_positions = [
            max(pos_lo, min(pos_hi, start + i * interval_mm)) for i in range(n_slices)
        ]
        logger.warning(
            "All %d retries exhausted; falling back to %d positions centered at %.2f mm.",
            max_retries,
            n_slices,
            mid,
        )
        fallback_reasoning = (
            "Model did not submit within iteration+retry budget; "
            "fell back to atlas midpoint."
        )
        return MultiSliceResult(
            positions=[
                PositionResult(position_mm=p, reasoning=fallback_reasoning)
                for p in fallback_positions
            ],
            group_reasoning=fallback_reasoning,
        )

    return await _run_estimation_session(
        images=images,
        atlas_name=atlas_name,
        plane=plane,
        model=model,
        species=species,
        max_iterations=max_iterations,
        max_retries=max_retries,
        thinking_level=thinking_level,
        apply_clahe=apply_clahe,
        target_transport=target_transport,
        trace_recorder=trace_recorder,
        include_thought_summaries=include_thought_summaries,
        run_label="group",
        session_id_prefix="group_attempt_",
        submit_tool_name="submit_group_estimate",
        n_slices=n_slices,
        interval_mm=interval_mm,
        thickness_um=thickness_um,
        build_agent=_build_agent,
        build_new_message=_build_new_message,
        build_result=_build_result,
        build_fallback=_build_fallback,
    )
