"""Model-facing canvas geometry, atlas templates, and the registration entrypoint.

Exactly two routes, both border-based (see ``border_registration.py`` for the
route selection and orchestration): this module holds the geometry every
route shares — the working canvas (:func:`prepare_canvas`), the aspect-ratio
lineup helpers, the grayscale atlas template, and the outlined-atlas render
route "atlas" shows the model instead of a placement.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.nonlinear.image_gen_helpers import (
    _annotation_slice,
    _extract_borders_from_classified,
    _merge_classified,
    line_width_px,
)
from langslice.core.nonlinear.model_prompts import aspect_ratio_limits, native_output_size
from langslice.core.nonlinear.types import Deformation, RegistrationCandidate
from langslice.core.space import Plane

if TYPE_CHECKING:
    from langslice.providers.registry import ImageCall

#: Long edge every model-facing atlas render is NEAREST/LANCZOS-upscaled to at
#: least. The atlas is coarse (a 25um coronal plate is ~456px across); below
#: this the thin bands and small nuclei the model has to place stop being
#: legible.
MODEL_MAP_MIN_LONG_EDGE = 1024

#: Relative aspect-ratio difference above which a returned image is treated as
#: letterboxed inside a different frame and cropped back. Lanes with fixed
#: output frames answer at the nearest legal aspect, a percent or two off.
_ASPECT_TOLERANCE = 0.005

_MAX_LONG_EDGE = 2048


def letterbox_to_aspect(
    image: Image.Image, aspect: float, fill: tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Center *image* on a *fill* canvas of the given width/height ratio.

    Grows one axis, never scales or crops: the atlas renders keep their own
    geometry and still arrive in the section's frame, which is what lets the
    prompt tell the model to answer in that one shared frame.
    """
    width, height = image.size
    if width / height < aspect:
        canvas_size = (round(height * aspect), height)
    else:
        canvas_size = (width, round(width / aspect))
    canvas = Image.new("RGB", canvas_size, fill)
    canvas.paste(
        image.convert("RGB"),
        ((canvas_size[0] - width) // 2, (canvas_size[1] - height) // 2),
    )
    return canvas


def upscale_to_min_long_edge(
    image: Image.Image,
    resample: Image.Resampling,
    min_long_edge: int = MODEL_MAP_MIN_LONG_EDGE,
) -> Image.Image:
    """Enlarge *image* until its long edge reaches *min_long_edge*; never shrink."""
    factor = max(1.0, min_long_edge / max(image.size))
    if factor == 1.0:
        return image
    return image.resize(
        (round(image.width * factor), round(image.height * factor)), resample
    )


def crop_to_aspect(
    image: Image.Image, aspect: float, tolerance: float = _ASPECT_TOLERANCE
) -> Image.Image:
    """Center-crop *image* to the given width/height ratio, never stretch.

    The model is asked for the frame it was given; a lane with fixed output
    frames returns the nearest legal one instead, with our frame letterboxed
    inside it. Cropping the excess back off puts every painted boundary where
    the model put it — resampling the whole thing to the frame would slide
    them all off the tissue.
    """
    width, height = image.size
    if abs((width / height) / aspect - 1.0) <= tolerance:
        return image
    if width / height > aspect:
        new_width = round(height * aspect)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    new_height = round(width / aspect)
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    """Median color of the image's border ring — the slide background.

    Padding in this color (instead of black) keeps the margin visually
    continuous with the image's own background, for both the image model
    and the downstream foreground mask.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    ring = max(2, int(round(0.02 * max(rgb.shape[:2]))))
    edges = np.concatenate(
        [
            rgb[:ring].reshape(-1, 3),
            rgb[-ring:].reshape(-1, 3),
            rgb[:, :ring].reshape(-1, 3),
            rgb[:, -ring:].reshape(-1, 3),
        ]
    )
    r, g, b = (int(v) for v in np.median(edges, axis=0))
    return (r, g, b)


def prepare_canvas(
    image: Image.Image,
    *,
    canvas_pad: float = 0.0,
    image_model: str | None = None,
    provider: str | None = None,
    native_canvas: bool = True,
    canvas_long_edge: int | None = None,
    quality: str | None = None,
) -> tuple[Image.Image, tuple[int, int], float, float, int]:
    """Downsample, pad, and aspect-snap the slice into the working canvas.

    Returns ``(slice_image, unpadded_size, origin_x, origin_y, pad_px)``.
    Padding uses the slice's own background color, and the aspect-ratio
    clamp pads the short axis (pad only, never crop): in edit mode the
    model paints on ITS canvas, and resampling a mismatched ratio back
    onto the slice would silently undo the pixel alignment.

    ``native_canvas`` (the default) sizes the canvas to the frame the image
    path returns (:func:`model_prompts.native_output_size`): the layout is
    worked out at the long-edge rule, scaled to fit that frame, and padded
    out to it exactly, so the model edits on the output's own pixel grid.
    Nash 2026-09-11: every input the model must rescale to its output is a
    pixel the output cannot carry, paid for twice (input tokens in, a
    resample out). The slice is resampled ONCE, straight from the original
    to its final size. ``canvas_long_edge`` instead pins the long edge (an
    experiment knob: the model is shown a smaller canvas and its output is
    resampled DOWN onto it).
    """
    fill = _background_color(image)
    limits = aspect_ratio_limits(image_model, provider)

    def layout(scale: float) -> tuple[tuple[int, int], int, tuple[int, int], tuple[int, int]]:
        """(slice size, pad px, snapped canvas size, slice offset) at *scale*."""
        tw, th = _target_size_for_slice(image, canvas_long_edge)
        tw, th = max(1, round(tw * scale)), max(1, round(th * scale))
        pad = int(round(float(canvas_pad) * max(tw, th))) if canvas_pad else 0
        cw, ch = tw + 2 * pad, th + 2 * pad
        if limits:
            if cw / ch > limits[1]:
                ch = math.ceil(cw / limits[1])
            elif cw / ch < limits[0]:
                cw = math.ceil(ch * limits[0])
        offset = ((cw - tw) // 2, (ch - th) // 2)
        return (tw, th), pad, (cw, ch), offset

    scale = 1.0
    frame: tuple[int, int] | None = None
    if native_canvas and canvas_long_edge is None:
        _, _, canvas0, _ = layout(1.0)
        frame = native_output_size(image_model, provider, canvas0, quality)
        if frame is not None:
            scale = min(frame[0] / canvas0[0], frame[1] / canvas0[1])
    target_size, pad_px, canvas_size, offset = layout(scale)
    if frame is not None:
        # Pad out to the frame exactly (never crop): the request IS the frame.
        extra = ((frame[0] - canvas_size[0]) // 2, (frame[1] - canvas_size[1]) // 2)
        canvas_size = frame
        offset = (offset[0] + extra[0], offset[1] + extra[1])

    slice_image = _resize_if_needed(image, target_size)
    if canvas_size != target_size:
        canvas = Image.new("RGB", canvas_size, fill)
        canvas.paste(slice_image, offset)
        slice_image = canvas
    return slice_image, target_size, float(offset[0]), float(offset[1]), pad_px


def _target_size_for_slice(
    image: Image.Image, max_long_edge: int | None = None
) -> tuple[int, int]:
    width, height = image.size
    long_edge = max(width, height)
    limit = max_long_edge or _MAX_LONG_EDGE
    if long_edge <= limit:
        return width, height

    scale = limit / float(long_edge)
    return int(width * scale), int(height * scale)


def _resize_if_needed(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image.convert("RGB")
    return image.convert("RGB").resize(size, resample=Image.Resampling.LANCZOS)


def _orient_pil(
    image: Image.Image, atlas: Any, plane: Plane, image_axes: str | None,
    atlas_mirror_lr: bool = False,
) -> Image.Image:
    """Rotate/flip an atlas render into the user's image frame (no-op if unset)."""
    if image_axes:
        from langslice.core.space import atlas_space_context, orient_slice_to_axes

        arr = orient_slice_to_axes(np.asarray(image), atlas_space_context(atlas), plane, image_axes)
        image = Image.fromarray(arr)
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if atlas_mirror_lr else image


def _model_facing_template(
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> Image.Image:
    """The grayscale atlas template of one plane, contrast-normalized for the model.

    ``get_reference_slice`` normalizes by the volume's brightest voxel, which
    leaves a typical plate dim; the model reads structure off this image, so
    it is restretched on the 99.5th percentile of the plate's own tissue
    (a linear rescale, exactly what the April benchmark sent).
    """
    from langslice.core.atlas import get_reference_slice

    gray = np.asarray(
        get_reference_slice(
            atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
        ),
        dtype=np.float32,
    )
    tissue = gray[gray > 0]
    peak = float(np.percentile(tissue, 99.5)) if tissue.size else 0.0
    scaled = np.clip(255.0 * gray / max(peak, 1.0), 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def _overlay_borders(
    base_image: Image.Image, borders: np.ndarray, line_px: int = 1
) -> Image.Image:
    """Draw atlas-region borders over a base image.

    Yellow core on a black rim: readable on violet Nissl, white brightfield,
    and dark fluorescence alike (plain cyan vanished on cyan-tinted Nissl).
    ``line_px`` grows the yellow core before the rim is drawn, so a render at
    a different working resolution than the historical 2048px canvas still
    gets a legible (not hairline-thin, not bloated) line — see
    :func:`~langslice.core.nonlinear.image_gen_helpers.line_width_px`.
    """
    overlay_rgb = np.asarray(base_image.convert("RGB"), dtype=np.uint8).copy()
    core = np.asarray(borders) > 0
    kernel = np.ones((3, 3), np.uint8)
    if line_px > 1:
        core = cv2.dilate(core.astype(np.uint8), kernel, iterations=line_px - 1) > 0
    rim = cv2.dilate(core.astype(np.uint8), kernel) > 0
    overlay_rgb[rim] = (0, 0, 0)
    overlay_rgb[core] = (255, 255, 0)
    return Image.fromarray(overlay_rgb, mode="RGB")


def outlined_atlas_template(
    atlas: Any,
    position_mm: float,
    plane: Plane,
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    image_axes: str | None = None,
    atlas_mirror_lr: bool = False,
    section_aspect: float | None = None,
    native_labels: np.ndarray | None = None,
) -> Image.Image:
    """The grayscale atlas plate with thin yellow family borders — route "atlas"'s
    only atlas-facing input.

    Oriented by ``image_axes`` then explicit ``atlas_mirror_lr``, never inferred
    from a silhouette. ``native_labels`` is the already-oriented annotation
    plane when the caller has it (the oblique resample is the expensive step);
    otherwise it is sampled and oriented here. Labels are NEAREST-upscaled to
    the frame the plate is LANCZOS-upscaled to, then letterboxed to the
    section's aspect.
    """
    template = _orient_pil(
        _model_facing_template(atlas, position_mm, plane, pitch_deg, yaw_deg),
        atlas, plane, image_axes, atlas_mirror_lr,
    )
    if native_labels is None:
        native_labels = _annotation_slice(
            atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            blackout=False,
        )
        if image_axes:
            from langslice.core.space import atlas_space_context, orient_slice_to_axes

            native_labels = orient_slice_to_axes(
                native_labels, atlas_space_context(atlas), plane, image_axes
            )
        if atlas_mirror_lr:
            native_labels = np.fliplr(native_labels)
    if native_labels.shape != (template.height, template.width):
        raise ValueError("Native atlas labels and grayscale template must share a pixel frame")

    grown = upscale_to_min_long_edge(template, Image.Resampling.LANCZOS)
    if grown.size != template.size:
        labels_up = np.asarray(
            Image.fromarray(native_labels.astype(np.int32), mode="I").resize(
                grown.size, Image.Resampling.NEAREST
            ),
            dtype=np.int64,
        )
    else:
        labels_up = native_labels
    borders = _extract_borders_from_classified(_merge_classified(labels_up, atlas))
    outlined = _overlay_borders(grown, borders, line_width_px(max(grown.size)))
    if section_aspect is not None:
        outlined = letterbox_to_aspect(outlined, section_aspect)
    return outlined


def generate_registration_candidate(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    plane: Plane = "coronal",
    provider: str = "google",
    image_model: str | None = None,
    image_prompt: str | None = None,
    generated_image: Image.Image | None = None,
    image_axes: str | None = None,
    canvas_pad: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    deformation: Deformation = "deformable",
    passes: int = 1,
    previous_candidate_id: str | None = None,
    candidate_id: str | None = None,
    debug_dir: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    openai_image_route: str = "images",
    review_model: str | None = None,
    thinking_level: str | None = None,
    native_canvas: bool = True,
    canvas_long_edge: int | None = None,
    initial_atlas_to_slice: Sequence[Sequence[float]] | np.ndarray | None = None,
    initial_alignment_source: str = "supplied",
    atlas_mirror_lr: bool = False,
    image_call: ImageCall | None = None,
) -> RegistrationCandidate:
    """Generate one dense border-based registration candidate.

    Exactly two routes, chosen by whether a placement is supplied:

    - ``initial_atlas_to_slice`` given -> route "supplied": one model call
      (:func:`~langslice.core.nonlinear.border_refinement.border_refinement_prompt`)
      moves a rough placement's drawn boundaries onto the visible tissue.
    - Not given -> route "atlas": no placement to correct. One model call
      (:func:`~langslice.core.nonlinear.prompts.pass1_atlas_prompt`) draws
      boundaries on the clean tissue against the outlined atlas template
      (:func:`outlined_atlas_template`); an optional second call
      (``passes=2``) corrects them. The same silhouette-moments placement
      (:mod:`langslice.core.nonlinear.prior`) and residual border fit as
      "supplied" then produce the candidate.

    ``provider="none"`` calls no model on either route: it retains a supplied
    placement, or fits the silhouette placement with zero residual. This is a
    thin dispatcher; the routing and both routes' orchestration live in
    :mod:`langslice.core.nonlinear.border_registration` (avoids a circular import
    at module load time).

    *image_call* is the image model's edit, resolved by the caller
    (:class:`langslice.providers.registry.ImageModel`'s ``call``); a model
    call without it is refused (``provider="none"`` and a replayed
    *generated_image* need none).
    """
    from langslice.core.nonlinear.border_registration import generate_border_registration_candidate

    return generate_border_registration_candidate(
        image,
        atlas_name=atlas_name, position_mm=position_mm, plane=plane,
        provider=provider, image_model=image_model, image_prompt=image_prompt,
        generated_image=generated_image, image_axes=image_axes,
        atlas_mirror_lr=atlas_mirror_lr,
        initial_atlas_to_slice=initial_atlas_to_slice,
        initial_alignment_source=initial_alignment_source,
        canvas_pad=canvas_pad, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
        deformation=deformation, passes=passes,
        previous_candidate_id=previous_candidate_id, candidate_id=candidate_id,
        debug_dir=debug_dir, on_progress=on_progress, on_trace=on_trace,
        openai_image_route=openai_image_route, review_model=review_model,
        thinking_level=thinking_level, native_canvas=native_canvas,
        canvas_long_edge=canvas_long_edge, image_call=image_call,
    )
