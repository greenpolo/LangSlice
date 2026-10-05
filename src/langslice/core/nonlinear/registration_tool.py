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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.affine import pixel_center_map
from langslice.core.atlas.render import annotation_slice
from langslice.core.handoff import (
    LinearRegistrationInput,
    correction_fingerprint,
    digest,
    prepare_linear_registration,
)
from langslice.core.nonlinear.border_refinement import (
    border_overlay,
    extract_thinned_lines,
    smooth_border_overlay,
)
from langslice.core.nonlinear.image_gen_helpers import _merge_classified, line_width_px
from langslice.core.nonlinear.image_gen_registration import (
    outlined_atlas_template,
    prepare_canvas,
)
from langslice.core.nonlinear.prompts import (
    border_correction_tool_prompt,
    pass1_atlas_prompt,
    pass2_atlas_prompt,
    supplied_prompt_is_gpt_twin,
)
from langslice.core.nonlinear.types import GeneratedSegmentation, SegmentationGenerationRequest
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
INPUT_VERSION = "supplied-1"

#: Attachment roles in request order, per prompt twin.
_GPT_ATTACHMENTS = (("Image 1: clean photograph", "original"),
                    ("Image 2: placed borders", "rough_overlay"))
_GEMINI_ATTACHMENTS = (("Image 1: placed borders", "rough_overlay"),
                       ("Image 2: clean photograph", "original"))


def correction_instructions(plane: Plane, provider: str | None = None) -> str:
    """Expose the tool's base image prompt to the agent, which may lightly edit it."""
    return border_correction_tool_prompt(plane, provider=provider)


#: Where a profile's own prompt names the section plane.
PLANE_PLACEHOLDER = "{plane}"


def profile_prompt(image_model: ImageModel, plane: Plane) -> tuple[str, bool]:
    """The base prompt *image_model* is sent and whether the clean photograph
    is its Image 1 (else the placed borders are).

    A model profile's own ``prompt`` (``{plane}`` replaced by the plane) in
    the order it names (``photograph_first``; None: the provider's); without
    one, LangSlice's prompt for the provider
    (:func:`~langslice.core.nonlinear.prompts.border_correction_tool_prompt`).
    """
    provider = image_model.provider
    own = getattr(image_model, "prompt", None)
    order = getattr(image_model, "photograph_first", None)
    photograph_first = supplied_prompt_is_gpt_twin(provider) if order is None else bool(order)
    if own is None:
        return border_correction_tool_prompt(plane, provider=provider), photograph_first
    return str(own).replace(PLANE_PLACEHOLDER, plane), photograph_first


def profile_marks(image_model: ImageModel) -> dict[str, Any]:
    """What a trace record says of an untested profile (``profile``, ``untested``
    True); nothing for a provider's own model and prompt."""
    if getattr(image_model, "tested", True):
        return {}
    return {"profile": getattr(image_model, "profile", None) or "custom", "untested": True}


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


def _saved_result(result_path: Path) -> dict[str, Any] | None:
    """The result saved under one call key, if any."""
    if not result_path.exists():
        return None
    saved = json.loads(result_path.read_text())
    if not isinstance(saved, dict):
        raise ValueError("Saved image correction result must be an object")
    return saved


def _open_attempt(
    call_directory: Path, previous: dict[str, Any] | None, section_id: str, fingerprint: str,
) -> tuple[Path, Path, int] | dict[str, Any]:
    """``(in_progress, attempt directory, attempt)`` for a new attempt under a
    call key, or the error record when one is in progress or was interrupted.

    Exclusive creation detects interrupted requests with no known outcome. A
    recorded transport failure may start another attempt under the same key.
    """
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
    return in_progress, directory, attempt


def _finish_attempt(
    directory: Path, result_path: Path, in_progress: Path, result: dict[str, Any],
) -> dict[str, Any]:
    """Save an attempt's outcome (its own and the call key's) and release the key."""
    _write_json(directory / "result.json", result)
    _write_json(result_path, result)
    in_progress.unlink()
    return dict(result)


def _calls_folder(calls_dir: Path | None, out: Path | None, section_id: str) -> Path:
    if calls_dir is not None:
        return Path(calls_dir)
    if out is None:
        raise ValueError("start_correction needs calls_dir or an out folder")
    return Path(out) / re.sub(r"[^a-zA-Z0-9._-]", "_", Path(section_id).name)


def _canvas_and_placement(
    prepared: LinearRegistrationInput, ctx: Workspace, provider: str, model: str | None,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """``(canvas, labels, atlas_to_canvas)``: the model-facing canvas of the
    section render, the native atlas plane at its position and cutting angles,
    and the section's linear placement of that plane on the canvas (native
    atlas pixel centres to canvas pixel centres; what ``fit_deformable``
    maps the traced lines back through, ``core.deformation.traced_lines``)."""
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
    return canvas, labels, placement


#: Image-model requests one stack session keeps in flight at once. The agent
#: never reads a reply, so calls run at the agent's pace rather than one by one.
MAX_CONCURRENT_IMAGE_CALLS = 8


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
    attempt. The base is *image_model*'s profile prompt when it has one
    (:func:`profile_prompt`), and a profile's own prompt or attachment order
    is part of the call key; an untested profile's request and result say
    so (``profile``, ``untested`` True: :func:`profile_marks`). The calls
    are saved in *calls_dir* (the section's own folder, one subfolder per
    call key; the job's ``sections/<name>/image_correction``)
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
    key: dict[str, Any] = {
        "geometry": fingerprint, "provider": provider, "model": model, "inputs": INPUT_VERSION,
    }
    own_prompt = getattr(image_model, "prompt", None)
    own_order = getattr(image_model, "photograph_first", None)
    if own_prompt is not None or own_order is not None:  # a profile's own inputs
        key["profile_inputs"] = digest({"prompt": own_prompt, "photograph_first": own_order})
    call_key = digest(key)
    call_directory = _calls_folder(calls_dir, out, section_id).resolve() / call_key[:24]
    result_path = call_directory / "result.json"
    previous = _saved_result(result_path)
    if previous is not None and (previous.get("status") == "ok"
                                 or previous.get("raw_received")):
        return {**previous, "cached": True}, None

    canvas, labels, placement = _canvas_and_placement(prepared, ctx, provider, model)
    rough = smooth_border_overlay(
        canvas, _merge_classified(labels, ctx.atlas), placement,
        width_px=BORDER_WIDTH_PX * max(canvas.size) / 1536,
    )
    base_prompt, gpt_twin = profile_prompt(image_model, prepared.plane)
    sent = prompt.strip() or base_prompt
    marks = profile_marks(image_model)

    opened = _open_attempt(call_directory, previous, section_id, fingerprint)
    if isinstance(opened, dict):
        return opened, None
    in_progress, directory, attempt = opened
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
        "provider": provider, "model": model, **marks,
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
        "provider": provider, "model": model, **marks,
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
                provider=provider, model=model,
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
        return _finish_attempt(directory, result_path, in_progress, result)

    return submitted, job


@dataclass(frozen=True)
class AtlasDrawing:
    """Route "atlas"'s model calls on one canvas (:func:`draw_from_atlas`)."""

    #: The final reply (pass 2's when there was one): the lines to extract.
    image: Image.Image
    #: Model calls made.
    model_calls: int
    #: Each call's transport route, in call order.
    transports: list[str | None]
    #: Pass 2's inputs by file name, with two passes: pass 1's lines on the tissue.
    artifacts: dict[str, Image.Image]


def draw_from_atlas(
    canvas: Image.Image,
    outlined_atlas: Image.Image,
    *,
    plane: Plane,
    provider: str,
    model: str | None,
    passes: int,
    edit: Callable[[SegmentationGenerationRequest], GeneratedSegmentation],
    on_reply: Callable[[int, Image.Image], None] | None = None,
) -> AtlasDrawing:
    """Route "atlas"'s model calls: boundaries drawn from nothing on the clean tissue.

    Pass 1 sends *canvas* (Image 1, the clean tissue) and *outlined_atlas*
    (Image 2, :func:`~langslice.core.nonlinear.image_gen_registration.outlined_atlas_template`)
    with :func:`~langslice.core.nonlinear.prompts.pass1_atlas_prompt`. With
    *passes* 2 a second call sends the clean tissue, pass 1's extracted
    lines redrawn on it, and the outlined atlas, with
    :func:`~langslice.core.nonlinear.prompts.pass2_atlas_prompt`. No
    placement is shown to the model. *edit* is the image model's call;
    *on_reply* receives each reply as it arrives (pass number, image), so a
    caller keeps pass 1's reply even when pass 2 is refused (pass 1 drew no
    lines) or fails.
    """
    if passes not in (1, 2):
        raise ValueError("passes must be 1 or 2")
    reply = edit(SegmentationGenerationRequest(
        slice_image=canvas.convert("RGB"), reference_images=[outlined_atlas],
        prompt=pass1_atlas_prompt(plane, provider), provider=provider, model=model,
        metadata={"route": "atlas", "pass": 1},
    ))
    if on_reply is not None:
        on_reply(1, reply.image)
    transports: list[str | None] = [reply.route]
    if passes == 1:
        return AtlasDrawing(reply.image, 1, transports, {})
    pass1_lines = extract_thinned_lines(reply.image, canvas.size)
    if not pass1_lines.any():
        raise ValueError(
            "Pass 1 returned no usable yellow anatomical boundaries; pass 2 has nothing to correct"
        )
    lines_on_tissue = border_overlay(canvas, pass1_lines, line_width_px(max(canvas.size)))
    reply = edit(SegmentationGenerationRequest(
        slice_image=canvas.convert("RGB"), reference_images=[lines_on_tissue, outlined_atlas],
        prompt=pass2_atlas_prompt(plane, provider), provider=provider, model=model,
        metadata={"route": "atlas", "pass": 2},
    ))
    if on_reply is not None:
        on_reply(2, reply.image)
    transports.append(reply.route)
    return AtlasDrawing(reply.image, 2, transports,
                        {"pass1_lines_on_tissue.png": lines_on_tissue})


#: Changes whenever route "atlas"'s model inputs change for the same geometry
#: (prompt text, attachments, the outlined atlas), like :data:`INPUT_VERSION`.
ATLAS_INPUT_VERSION = "atlas-1"


def start_atlas_correction(
    state: StackState,
    ctx: Workspace,
    section_id: str,
    *,
    passes: int = 1,
    image_model: ImageModel,
    calls_dir: Path,
) -> tuple[dict[str, Any], Callable[[], dict[str, Any]] | None]:
    """Prepare one section's placement-free trace: route "atlas" on the job.

    The model is shown the clean section and the outlined grayscale atlas
    plane at the section's position and cutting angles
    (:func:`draw_from_atlas`: pass 1, and with *passes* 2 a corrective
    pass 2), never the section's
    placement. The result is recorded exactly as :func:`start_correction`'s
    (same keys, same call-key folders and attempts under *calls_dir*, the
    extracted lines and ``request.json``'s ``atlas_to_canvas``), so
    ``fit_deformable``'s traced fit sections, the submit gate and the maps
    read it unchanged: the section's written linear placement is where the
    fit of the lines starts (``atlas_to_canvas``), as for a
    ``trace_borders`` reply. Requires a position and a written transform
    (:func:`langslice.core.handoff.prepare_linear_registration`). Returns
    ``(record, job)`` like :func:`start_correction`; the first reply at a
    geometry and pass count is reused (``cached``). With two passes, pass
    1's reply is saved as it arrives (``pass1_raw_correction.png``) and
    counts as a received reply, so a failed or refused pass 2 is not paid
    for again.
    """
    if passes not in (1, 2):
        raise ValueError("passes must be 1 or 2")
    provider, model = image_model.provider, image_model.model
    prepared = prepare_linear_registration(state, ctx, section_id)
    fingerprint = correction_fingerprint(state, ctx, section_id)
    call_key = digest({
        "geometry": fingerprint, "provider": provider, "model": model,
        "inputs": ATLAS_INPUT_VERSION, "route": "atlas", "passes": passes,
    })
    call_directory = Path(calls_dir).resolve() / call_key[:24]
    result_path = call_directory / "result.json"
    previous = _saved_result(result_path)
    if previous is not None and (previous.get("status") == "ok"
                                 or previous.get("raw_received")):
        return {**previous, "cached": True}, None

    canvas, labels, placement = _canvas_and_placement(prepared, ctx, provider, model)
    outlined = outlined_atlas_template(
        ctx.atlas, prepared.position_mm, prepared.plane, pitch_deg=prepared.pitch_deg,
        yaw_deg=prepared.yaw_deg, section_aspect=canvas.width / canvas.height,
        native_labels=labels,
    )
    opened = _open_attempt(call_directory, previous, section_id, fingerprint)
    if isinstance(opened, dict):
        return opened, None
    in_progress, directory, attempt = opened
    paths = {key: str(directory / filename) for key, filename in {
        "original": "input_slice.png", "outlined_atlas": "outlined_atlas.png",
        "raw_reply": "raw_reply.png", "lines_on_original": "lines_on_original.png",
        "prompt": "prompt.txt", **({"pass2_prompt": "pass2_prompt.txt"} if passes == 2 else {}),
    }.items()}
    canvas.save(paths["original"])
    outlined.save(paths["outlined_atlas"])
    # The exact text each pass sends.
    Path(paths["prompt"]).write_text(pass1_atlas_prompt(prepared.plane, provider))
    if passes == 2:
        Path(paths["pass2_prompt"]).write_text(pass2_atlas_prompt(prepared.plane, provider))
    request = {
        "id": section_id, "geometry_fingerprint": fingerprint, "trace_route": "atlas",
        "passes": passes, "provider": provider, "model": model,
        "linear_handoff": prepared.metadata, "atlas_to_canvas": placement.tolist(),
        "attachments": [
            {"role": role, "path": paths[key],
             "sha256": hashlib.sha256(Path(paths[key]).read_bytes()).hexdigest()}
            for role, key in (("Image 1: clean photograph", "original"),
                              ("Image 2: outlined atlas", "outlined_atlas"))
        ],
    }
    _write_json(directory / "request.json", request)
    result: dict[str, Any] = {
        "id": section_id, "geometry_fingerprint": fingerprint, "prompt_edited": False,
        "provider": provider, "model": model, "trace_route": "atlas", "passes": passes,
        "output_kind": "border_annotation", "fit_performed": False,
        "artifact_dir": str(directory), "artifact_paths": paths, "cached": False,
        "attempt": attempt, "raw_received": False,
    }
    submitted = {**result, "artifact_paths": dict(paths), "status": "running"}

    def received(pass_number: int, image: Image.Image) -> None:
        result["raw_received"] = True
        if passes == 2 and pass_number == 1:
            paths["pass1_raw_correction"] = str(directory / "pass1_raw_correction.png")
            image.save(paths["pass1_raw_correction"])

    def job() -> dict[str, Any]:
        try:
            drawing = draw_from_atlas(
                canvas, outlined, plane=prepared.plane, provider=provider, model=model,
                passes=passes, edit=image_model.call, on_reply=received,
            )
            for filename, image in drawing.artifacts.items():
                paths[Path(filename).stem] = str(directory / filename)
                image.save(directory / filename)
            raw = drawing.image.convert("RGB")
            raw.save(paths["raw_reply"])
            mask = extract_thinned_lines(raw, canvas.size)
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / "extracted_lines.png")
            border_overlay(canvas, mask).save(paths["lines_on_original"])
            # Even an empty drawing is retained; anatomy is not a submit gate.
            result.update(
                status="ok", route=drawing.transports, model_calls=drawing.model_calls,
                raw_size=list(raw.size), model_border_pixels=int(mask.sum()),
            )
        except Exception as exc:
            result.update(status="error", error=type(exc).__name__, message=str(exc))
        return _finish_attempt(directory, result_path, in_progress, result)

    return submitted, job
