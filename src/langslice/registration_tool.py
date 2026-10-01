"""Agent image tool at the boundary between linear placement and border correction.

The first image reply is retained automatically. This tool produces annotation
artifacts, not a fitted deformation or a replacement linear transform.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from langslice.atlas.render import annotation_slice
from langslice.nonlinear.border_refinement import (
    border_overlay,
    extract_thinned_lines,
    smooth_border_overlay,
)
from langslice.nonlinear.border_registration import pixel_center_map
from langslice.nonlinear.image_gen_helpers import (
    _merge_classified,
)
from langslice.nonlinear.image_gen_registration import prepare_canvas
from langslice.nonlinear.prompts import (
    border_correction_tool_prompt,
    supplied_prompt_is_gpt_twin,
)
from langslice.nonlinear.providers import (
    SegmentationGenerationRequest,
    generate_warped_segmentation_image,
)
from langslice.providers.openai_oauth import DEFAULT_IMAGE_MODEL
from langslice.providers.registry import canonical_provider
from langslice.registration_handoff import prepare_linear_registration
from langslice.space import Plane

if TYPE_CHECKING:
    from langslice.linear.engine import EngineContext
    from langslice.linear.state import StackState


#: Width of the placed boundaries in Image 1, in canvas pixels at a 1536-px long edge.
BORDER_WIDTH_PX = 2.0

#: Changes whenever the model's inputs change for the same geometry (prompt text,
#: attachment order, border rendering), so a saved reply is reused only for the
#: exact inputs that produced it.
INPUT_VERSION = "2026-09-29 smooth borders; route-supplied GPT twin"

#: Attachment roles in request order, per prompt twin.
_GPT_ATTACHMENTS = (("Image 1: clean photograph", "original"),
                    ("Image 2: placed borders", "rough_overlay"))
_GEMINI_ATTACHMENTS = (("Image 1: placed borders", "rough_overlay"),
                       ("Image 2: clean photograph", "original"))


def correction_instructions(plane: Plane, provider: str | None = None) -> str:
    """Expose the tool's base image prompt to the agent, which may lightly edit it."""
    return border_correction_tool_prompt(plane, provider=provider)


def prompt_diff(base: str, edited: str) -> str:
    """Word-level diff of an edited prompt: removed words [-so-], added {+so+}."""
    old, new = base.split(" "), edited.split(" ")
    parts: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == "equal":
            parts.append(" ".join(old[i1:i2]))
            continue
        if i2 > i1:
            parts.append("[-" + " ".join(old[i1:i2]) + "-]")
        if j2 > j1:
            parts.append("{+" + " ".join(new[j1:j2]) + "+}")
    return " ".join(parts)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def correction_fingerprint(state: StackState, ctx: EngineContext, section_id: str) -> str:
    """Identify the source image, placement and rendering settings for a correction."""
    record = state.by_id(section_id)
    if record is None:
        raise ValueError(f"Unknown section: {section_id}")
    source = Path(ctx.image_path(record.id)).resolve()
    stat = source.stat()
    transform = record.transform or {}
    nonlinear = dict(ctx.spec.to_dict().get("nonlinear") or {})
    # The deformable-fit engine never reaches the image call; leaving it in
    # made every trace saved before the field existed (or under another
    # engine choice) stale at an unchanged placement.
    nonlinear.pop("engine", None)
    return _digest({
        "source": str(source), "source_size": stat.st_size, "source_mtime": stat.st_mtime_ns,
        "section_id": record.id, "atlas": state.atlas, "plane": state.plane,
        "position_mm": record.position_mm, "angles": state.cutting_angles_deg,
        "flip": record.flip, "rotation_deg": record.rotation_deg,
        "transform": {key: transform.get(key) for key in (
            "params", "calibration", "orientation", "stale", "spline",
        )},
        "preprocess": ctx.spec.preprocess,
        "pixel_size_um": ctx.spec.inputs.get("pixel_size_um"),
        "nonlinear": nonlinear,
    })


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


#: Image-model requests one stack session keeps in flight at once. The agent
#: never reads a reply, so calls run at the agent's pace rather than one by one.
MAX_CONCURRENT_IMAGE_CALLS = 8


def correct_slice(
    state: StackState,
    ctx: EngineContext,
    section_id: str,
    *,
    prompt: str = "",
    out: Path,
    provider: str = "openai-oauth",
    image_model: str | None = None,
) -> dict[str, Any]:
    """Make one image edit and wait for its result.

    The synchronous form of :func:`start_correction`: prepare, then run the
    image call in this thread.
    """
    record, job = start_correction(
        state, ctx, section_id, prompt=prompt, out=out,
        provider=provider, image_model=image_model,
    )
    return job() if job is not None else record


def start_correction(
    state: StackState,
    ctx: EngineContext,
    section_id: str,
    *,
    prompt: str = "",
    out: Path,
    provider: str = "openai-oauth",
    image_model: str | None = None,
) -> tuple[dict[str, Any], Callable[[], dict[str, Any]] | None]:
    """Prepare one image edit from an existing calibrated linear placement.

    Returns ``(record, job)``. Everything that reads the stack state happens
    here, so ``job`` (the image call and its artifacts) may run in another
    thread while the agent keeps working. ``job`` is ``None`` when there is
    nothing to run: a saved reply at this geometry (returned with
    ``cached``), or an attempt already in progress or interrupted.

    *prompt* is the agent's edited copy of the base prompt (blank sends the
    base); the base, the prompt sent and their word diff are saved with the
    attempt. There is no
    atlas search, model veto, or selection step. Repeated calls at the same
    geometry return the first image reply. Failed transports that returned no
    image may be retried; no reply is discarded. Different edits cannot
    regenerate or replace an existing reply.
    The caller checkpoints the returned record; the source linear state is untouched.
    """
    provider = canonical_provider(provider)
    if provider == "none":
        raise ValueError("The image correction tool requires an image-model provider")
    model = image_model or (DEFAULT_IMAGE_MODEL if provider == "openai-oauth" else None)
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    # Validate prerequisites before spending a call or marking an attempt.
    prepared = prepare_linear_registration(state, ctx, section_id)
    fingerprint = correction_fingerprint(state, ctx, section_id)
    call_key = _digest({
        "geometry": fingerprint, "provider": provider, "model": model, "inputs": INPUT_VERSION,
    })
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(section_id).name)
    call_directory = Path(out).resolve() / name / call_key[:24]
    result_path = call_directory / "result.json"
    previous: dict[str, Any] | None = None
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if not isinstance(saved, dict):
            raise ValueError("Saved image correction result must be an object")
        previous = saved
        if previous.get("status") == "ok" or previous.get("raw_received"):
            return {**previous, "cached": True}, None

    canvas, unpadded, ox, oy, _ = prepare_canvas(
        prepared.image, provider=provider, image_model=model, canvas_pad=0,
        native_canvas=True,
    )
    labels = np.asarray(annotation_slice(
        ctx.atlas, prepared.position_mm, plane=prepared.plane,
        pitch_deg=prepared.pitch_deg, yaw_deg=prepared.yaw_deg,
    ))
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Atlas annotation must be a two-dimensional integer label map")
    placement = pixel_center_map(prepared.image.size, unpadded, (ox, oy)) @ prepared.atlas_to_slice
    # float64 preserves large atlas IDs across nearest-neighbour projection.
    placed = cv2.warpAffine(
        labels.astype(np.float64), placement[:2], canvas.size,
        flags=cv2.INTER_NEAREST, borderValue=0,
    ).astype(labels.dtype)
    if not placed.any():
        raise ValueError("The supplied linear placement contains no atlas regions on the image")
    rough = smooth_border_overlay(
        canvas, _merge_classified(labels, ctx.atlas), placement,
        width_px=BORDER_WIDTH_PX * max(canvas.size) / 1536,
    )
    gpt_twin = supplied_prompt_is_gpt_twin(provider)
    base_prompt = border_correction_tool_prompt(prepared.plane, provider=provider)
    sent = prompt.strip() or base_prompt

    # Exclusive creation detects interrupted requests with no known outcome.
    # A recorded transport failure may start another attempt under the same key.
    in_progress = call_directory / ".in_progress"
    try:
        call_directory.mkdir(parents=True, exist_ok=previous is not None)
        with in_progress.open("x") as handle:
            handle.write("Request in progress; no completed outcome recorded yet.\n")
    except FileExistsError:
        return {
            "status": "error", "error": "IMAGE_CALL_IN_PROGRESS_OR_INTERRUPTED",
            "id": section_id, "geometry_fingerprint": fingerprint,
            "artifact_dir": str(call_directory),
        }, None
    attempt = int((previous or {}).get("attempt", 0)) + 1
    directory = call_directory / f"attempt-{attempt:02d}"
    directory.mkdir()
    paths = {key: str(directory / filename) for key, filename in {
        "original": "input_slice.png", "rough_overlay": "rough_border_overlay.png",
        "raw_reply": "raw_reply.png", "lines_on_original": "lines_on_original.png",
        "prompt": "prompt.txt", "base_prompt": "base_prompt.txt", "prompt_diff": "prompt_diff.txt",
    }.items()}
    canvas.save(paths["original"])
    rough.save(paths["rough_overlay"])
    Path(paths["prompt"]).write_text(sent)
    Path(paths["base_prompt"]).write_text(base_prompt)
    Path(paths["prompt_diff"]).write_text(prompt_diff(base_prompt, sent) + "\n")
    request = {
        "id": section_id, "geometry_fingerprint": fingerprint,
        "prompt_edited": sent != base_prompt,
        "provider": provider, "model": model,
        "linear_handoff": prepared.metadata, "atlas_to_canvas": placement.tolist(),
        "attachments": [
            {"role": role, "path": paths[key],
             "sha256": hashlib.sha256(Path(paths[key]).read_bytes()).hexdigest()}
            for role, key in (_GPT_ATTACHMENTS if gpt_twin else _GEMINI_ATTACHMENTS)
        ],
    }
    _write_json(directory / "request.json", request)
    result: dict[str, Any] = {
        "id": section_id, "geometry_fingerprint": fingerprint,
        "prompt_edited": sent != base_prompt,
        "provider": provider, "model": model,
        "output_kind": "border_annotation", "fit_performed": False,
        "artifact_dir": str(directory), "artifact_paths": paths, "cached": False,
        "attempt": attempt, "raw_received": False,
    }
    submitted = {**result, "status": "running"}

    def job() -> dict[str, Any]:
        try:
            first, second = (canvas, rough) if gpt_twin else (rough, canvas)
            generated = generate_warped_segmentation_image(SegmentationGenerationRequest(
                slice_image=first, reference_images=[second], prompt=sent,
                provider=provider, model=model, openai_image_route="images", mode="edit",
            ))
            result["raw_received"] = True
            raw = generated.image.convert("RGB")
            raw.save(paths["raw_reply"])
            mask = extract_thinned_lines(raw, canvas.size)
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / "extracted_lines.png")
            border_overlay(canvas, mask).save(paths["lines_on_original"])
            # Even an empty drawing is retained; anatomy is not a submit gate.
            result.update(
                status="ok", route=generated.route, raw_size=list(raw.size),
                model_border_pixels=int(mask.sum()),
            )
        except Exception as exc:
            result.update(status="error", error=type(exc).__name__, message=str(exc))
        _write_json(directory / "result.json", result)
        _write_json(result_path, result)
        in_progress.unlink()
        return dict(result)

    return submitted, job
