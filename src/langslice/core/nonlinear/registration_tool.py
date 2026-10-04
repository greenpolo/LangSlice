"""Agent image tool at the boundary between linear placement and border correction.

The first image reply is retained automatically. This tool produces annotation
artifacts, not a fitted deformation or a replacement linear transform.

The image model is an argument (:class:`langslice.providers.registry.ImageModel`,
resolved by a door), so this module imports no provider and is core-clean
(``tests/test_core_imports.py``); the geometry it starts from is the core's
(:mod:`langslice.core.handoff`).
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

from langslice.core.affine import pixel_center_map
from langslice.core.atlas.render import annotation_slice
from langslice.core.handoff import correction_fingerprint, digest, prepare_linear_registration
from langslice.core.nonlinear.border_refinement import (
    border_overlay,
    extract_thinned_lines,
    smooth_border_overlay,
)
from langslice.core.nonlinear.image_gen_helpers import (
    _merge_classified,
)
from langslice.core.nonlinear.image_gen_registration import prepare_canvas
from langslice.core.nonlinear.prompts import (
    border_correction_tool_prompt,
    supplied_prompt_is_gpt_twin,
)
from langslice.core.nonlinear.types import SegmentationGenerationRequest
from langslice.core.space import Plane
from langslice.core.workspace import Workspace

if TYPE_CHECKING:
    from langslice.core.state import StackState
    from langslice.providers.registry import ImageModel

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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


#: Image-model requests one stack session keeps in flight at once. The agent
#: never reads a reply, so calls run at the agent's pace rather than one by one.
MAX_CONCURRENT_IMAGE_CALLS = 8


def correct_slice(
    state: StackState,
    ctx: Workspace,
    section_id: str,
    *,
    prompt: str = "",
    out: Path,
    image_model: ImageModel,
) -> dict[str, Any]:
    """Make one image edit and wait for its result.

    The synchronous form of :func:`start_correction`: prepare, then run the
    image call in this thread.
    """
    record, job = start_correction(
        state, ctx, section_id, prompt=prompt, out=out, image_model=image_model,
    )
    return job() if job is not None else record


def start_correction(
    state: StackState,
    ctx: Workspace,
    section_id: str,
    *,
    prompt: str = "",
    out: Path | None = None,
    image_model: ImageModel,
    calls_dir: Path | None = None,
) -> tuple[dict[str, Any], Callable[[], dict[str, Any]] | None]:
    """Prepare one image edit from an existing calibrated linear placement.

    *image_model* is the model to call, resolved by the caller
    (:func:`langslice.providers.registry.resolve_image_model`; a door binds
    the run's provider): this module never chooses or imports a provider.

    Returns ``(record, job)``. Everything that reads the stack state happens
    here, so ``job`` (the image call and its artifacts) may run in another
    thread while the agent keeps working. ``job`` is ``None`` when there is
    nothing to run: a saved reply at this geometry (returned with
    ``cached``), or an attempt already in progress or interrupted.

    *prompt* is the agent's edited copy of the base prompt (blank sends the
    base); the base, the prompt sent and their word diff are saved with the
    attempt. The calls are saved in *calls_dir* (the section's own folder,
    one subfolder per call key; the job's ``sections/<name>/image_correction``)
    or, without it, in ``<out>/<section filename>``. There is no
    atlas search, model veto, or selection step. Repeated calls at the same
    geometry return the first image reply. Failed transports that returned no
    image may be retried; no reply is discarded. Different edits cannot
    regenerate or replace an existing reply.
    The caller checkpoints the returned record; the source linear state is untouched.
    """
    provider, model = image_model.provider, image_model.model
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    # Validate prerequisites before spending a call or marking an attempt.
    prepared = prepare_linear_registration(state, ctx, section_id)
    fingerprint = correction_fingerprint(state, ctx, section_id)
    call_key = digest({
        "geometry": fingerprint, "provider": provider, "model": model, "inputs": INPUT_VERSION,
    })
    if calls_dir is None:
        if out is None:
            raise ValueError("start_correction needs calls_dir or an out folder")
        calls_dir = Path(out) / re.sub(r"[^a-zA-Z0-9._-]", "_", Path(section_id).name)
    call_directory = Path(calls_dir).resolve() / call_key[:24]
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
            generated = image_model.call(SegmentationGenerationRequest(
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
