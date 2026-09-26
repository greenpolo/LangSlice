"""Agent image tool at the boundary between linear placement and border correction.

The first image reply is retained automatically. This tool produces annotation
artifacts, not a fitted deformation or a replacement linear transform.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from langslice.atlas.render import annotation_slice
from langslice.nonlinear.border_refinement import border_overlay, extract_thinned_lines
from langslice.nonlinear.border_registration import pixel_center_map
from langslice.nonlinear.image_gen_helpers import (
    _extract_borders_from_classified,
    _merge_classified,
    line_width_px,
)
from langslice.nonlinear.image_gen_registration import prepare_canvas
from langslice.nonlinear.prompts import border_correction_tool_prompt
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


def correction_instructions(plane: Plane) -> str:
    """Expose the tool's fixed image task to the agent composing supplementary notes."""
    return border_correction_tool_prompt(plane)


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
        "nonlinear": ctx.spec.to_dict().get("nonlinear"),
    })


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def correct_slice(
    state: StackState,
    ctx: EngineContext,
    section_id: str,
    *,
    additional_notes: str = "",
    out: Path,
    provider: str = "openai-oauth",
    image_model: str | None = None,
) -> dict[str, Any]:
    """Make one fixed-prompt image edit from an existing calibrated linear placement.

    There is no atlas search, prompt override, model veto, or selection step.
    Repeated calls at the same geometry return the first image reply. Failed
    transports that returned no image may be retried; no reply is discarded.
    Different notes cannot regenerate or replace an existing reply.
    The caller checkpoints the returned record; the source linear state is untouched.
    """
    if not isinstance(additional_notes, str):
        raise ValueError("additional_notes must be text")
    provider = canonical_provider(provider)
    if provider == "none":
        raise ValueError("The image correction tool requires an image-model provider")
    model = image_model or (DEFAULT_IMAGE_MODEL if provider == "openai-oauth" else None)
    # Validate prerequisites before spending a call or marking an attempt.
    prepared = prepare_linear_registration(state, ctx, section_id)
    fingerprint = correction_fingerprint(state, ctx, section_id)
    call_key = _digest({"geometry": fingerprint, "provider": provider, "model": model})
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
            return {**previous, "cached": True}

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
    boundaries = _extract_borders_from_classified(_merge_classified(placed, ctx.atlas)) > 0
    rough = border_overlay(canvas, boundaries, line_width_px(max(canvas.size)))
    prompt = border_correction_tool_prompt(prepared.plane, additional_notes)

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
        }
    attempt = int((previous or {}).get("attempt", 0)) + 1
    directory = call_directory / f"attempt-{attempt:02d}"
    directory.mkdir()
    paths = {key: str(directory / filename) for key, filename in {
        "original": "input_slice.png", "rough_overlay": "rough_border_overlay.png",
        "raw_reply": "raw_reply.png", "lines_on_original": "lines_on_original.png",
        "prompt": "prompt.txt",
    }.items()}
    canvas.save(paths["original"])
    rough.save(paths["rough_overlay"])
    Path(paths["prompt"]).write_text(prompt)
    request = {
        "id": section_id, "geometry_fingerprint": fingerprint,
        "additional_notes": additional_notes.strip(), "provider": provider, "model": model,
        "linear_handoff": prepared.metadata, "atlas_to_canvas": placement.tolist(),
        "attachments": [
            {"role": role, "path": paths[key],
             "sha256": hashlib.sha256(Path(paths[key]).read_bytes()).hexdigest()}
            for role, key in (("Image 1: placed borders", "rough_overlay"),
                              ("Image 2: clean photograph", "original"))
        ],
    }
    _write_json(directory / "request.json", request)
    result: dict[str, Any] = {
        "id": section_id, "geometry_fingerprint": fingerprint,
        "additional_notes": additional_notes.strip(), "provider": provider, "model": model,
        "output_kind": "border_annotation", "fit_performed": False,
        "artifact_dir": str(directory), "artifact_paths": paths, "cached": False,
        "attempt": attempt, "raw_received": False,
    }
    try:
        generated = generate_warped_segmentation_image(SegmentationGenerationRequest(
            slice_image=rough, reference_images=[canvas], prompt=prompt,
            provider=provider, model=model, openai_image_route="images", mode="edit",
        ))
        result["raw_received"] = True
        raw = generated.image.convert("RGB")
        raw.save(paths["raw_reply"])
        mask = extract_thinned_lines(raw, canvas.size)
        Image.fromarray(mask.astype(np.uint8) * 255).save(directory / "extracted_lines.png")
        border_overlay(canvas, mask).save(paths["lines_on_original"])
        # Even an empty drawing is retained and shown; anatomy is not a submit gate.
        result.update(
            status="ok", route=generated.route, raw_size=list(raw.size),
            model_border_pixels=int(mask.sum()),
        )
    except Exception as exc:
        result.update(status="error", error=type(exc).__name__, message=str(exc))
    _write_json(directory / "result.json", result)
    _write_json(result_path, result)
    in_progress.unlink()
    return result
