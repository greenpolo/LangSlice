"""Correct placed atlas boundaries while keeping the original photograph authoritative.

The image-model half of route "supplied": the request (rough borders drawn on
the photograph, the clean photograph beside it), the yellow-line extraction
and the review overlays. The fit after it is the deformable package's
(:mod:`langslice.core.nonlinear.border_fit`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.atlas.render import placed_border_coverage
from langslice.core.nonlinear.image_gen_helpers import (
    _extract_borders_from_classified,
    _merge_classified,
    line_width_px,
)
from langslice.core.nonlinear.prompts import border_refinement_prompt
from langslice.core.nonlinear.types import SegmentationGenerationRequest
from langslice.core.provider_names import canonical_provider
from langslice.core.space import Plane

if TYPE_CHECKING:
    from langslice.providers.registry import ImageCall

__all__ = [
    "BorderRefinementResult",
    "border_overlay",
    "border_refinement_prompt",
    "extract_thinned_lines",
    "refine_borders",
    "thin",
    "yellow_mask",
]


def yellow_mask(rgb: np.ndarray) -> np.ndarray:
    """Extract saturated yellow, allowing the model's orange/green color drift."""
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hue, sat, val = (hsv[..., i] for i in range(3))
    return (hue >= 18) & (hue <= 42) & (sat >= 90) & (val >= 110)


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen skeletonization, retaining connected one-pixel boundaries."""
    img = np.asarray(mask, dtype=bool).astype(np.uint8)
    while True:
        removed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = np.zeros_like(img)
            for i in range(8):
                transitions += ((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8)
            if step == 0:
                corner = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                corner = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            drop = (img == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
            drop &= corner
            if drop.any():
                img[drop] = 0
                removed = True
        if not removed:
            return img.astype(bool)


def smooth_border_overlay(
    image: Image.Image,
    labels: np.ndarray,
    atlas_to_image: np.ndarray,
    *,
    width_px: float = 2.0,
    smoothing_px: float = 0.8,
    supersample: int = 3,
) -> Image.Image:
    """Placed atlas region boundaries as smooth, single, antialiased yellow lines.

    *labels* is an atlas-resolution region map and *atlas_to_image* the 3x3
    (or 2x3) map from its pixel centres to *image*'s. A nearest-neighbour warp
    of the label map magnifies the atlas grid into a staircase; instead each
    region's indicator is blurred by *smoothing_px* atlas pixels, warped
    bilinearly onto a *supersample*-times finer grid, and every fine pixel
    takes the region that covers it most. Boundaries between neighbouring
    fine pixels are one shared line (tracing each region separately draws a
    shared edge twice, one atlas pixel apart), widened to *width_px* image
    pixels and area-averaged back down, so the edges are antialiased. The
    coverage itself is :func:`langslice.core.atlas.render.placed_border_coverage`,
    shared with the deformable fit's border images.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    coverage = placed_border_coverage(
        labels, atlas_to_image, image.size, width_px=width_px,
        smoothing_px=smoothing_px, supersample=supersample,
    )
    alpha = coverage[..., None]
    blended = base * (1.0 - alpha) + np.array([255.0, 255.0, 0.0]) * alpha
    return Image.fromarray(np.rint(blended).astype(np.uint8))


def border_overlay(image: Image.Image, mask: np.ndarray, width: int = 1) -> Image.Image:
    """Replace only boundary pixels on the original photograph."""
    pixels = np.array(image.convert("RGB"))
    drawn = np.asarray(mask, dtype=bool)
    if width > 1:
        drawn = cv2.dilate(drawn.astype(np.uint8), np.ones((width, width), np.uint8)) > 0
    pixels[drawn] = (255, 255, 0)
    return Image.fromarray(pixels)


def extract_thinned_lines(raw: Image.Image, canvas_size: tuple[int, int]) -> np.ndarray:
    """Boolean thinned yellow-line mask from a raw model reply, on *canvas_size*.

    Shared by the line extraction in :func:`refine_borders` and
    route "atlas"'s pass-2 input construction (:mod:`border_registration`),
    which needs pass 1's lines redrawn on the clean tissue at the same canvas
    before the second call. Extracts before any interpolation, so tissue
    colors cannot blend into yellow; crops back to the canvas aspect first —
    a lane with a fixed output frame answers at its own aspect, ours
    letterboxed inside it, so crop back rather than stretch.
    """
    from langslice.core.nonlinear.image_gen_registration import crop_to_aspect

    raw_mask = Image.fromarray(yellow_mask(np.asarray(raw.convert("RGB"))))
    framed = crop_to_aspect(raw_mask, canvas_size[0] / canvas_size[1])
    return thin(np.asarray(framed.resize(canvas_size, Image.Resampling.NEAREST)))


@dataclass
class BorderRefinementResult:
    """The model's correction of the rough borders, before any fit.

    The fit after it is :func:`langslice.core.nonlinear.border_fit.fit_border_lines`.
    """

    rough_border_overlay: Image.Image
    raw_model_image: Image.Image
    model_border_mask: np.ndarray
    model_border_overlay: Image.Image
    metadata: dict[str, Any]


def refine_borders(
    image: Image.Image,
    rough_labels: np.ndarray,
    atlas: Any,
    *,
    provider: str = "google",
    model: str | None = None,
    plane: Plane = "coronal",
    review_model: str | None = None,
    openai_image_route: str = "images",
    thinking_level: str | None = None,
    generated_image: Image.Image | None = None,
    image_prompt: str | None = None,
    image_call: ImageCall | None = None,
) -> BorderRefinementResult:
    """Ask the model to move the rough borders onto the tissue; extract its lines.

    Image 1 is the photograph with the rough placement's family borders drawn
    on it, Image 2 the clean photograph; the reply's yellow lines are
    extracted on the original canvas. ``provider="none"`` without a
    *generated_image* calls no model: the rough borders stand as the lines
    (``model_free``). A *generated_image* replays a reply with no call.
    Nothing is fitted here (the caller fits the lines,
    :func:`langslice.core.nonlinear.border_fit.fit_border_lines`), and the raw
    reply is untouched.
    """
    labels = np.asarray(rough_labels)
    shape = (image.height, image.width)
    if labels.shape != shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Rough atlas labels must be an integer map on the image canvas")
    if not np.any(labels):
        raise ValueError("Rough atlas placement contains no regions")
    original = image.convert("RGB")
    moving = _extract_borders_from_classified(_merge_classified(labels, atlas))
    line_width = line_width_px(max(image.size))
    rough = border_overlay(original, moving > 0, line_width)
    prompt = image_prompt or border_refinement_prompt(plane)
    metadata: dict[str, Any] = {
        "workflow": "border_refinement", "prompt": prompt,
        "provider": canonical_provider(provider), "model": model,
        "line_width_px": line_width, "input_size": list(image.size),
    }
    if generated_image is None and canonical_provider(provider) == "none":
        return BorderRefinementResult(
            rough, rough.copy(), moving > 0, rough.copy(),
            {**metadata, "model_called": False, "model_free": True},
        )
    if generated_image is None:
        if image_call is None:
            raise ValueError(
                "no image_call: pass the resolved image model's call"
                " (providers.registry.resolve_image_model(provider).call)"
            )
        generated = image_call(SegmentationGenerationRequest(
            slice_image=rough, reference_images=[original], prompt=prompt,
            provider=provider, model=model, review_model=review_model,
            openai_image_route=openai_image_route, thinking_level=thinking_level,
            metadata=dict(metadata),
        ))
        raw = generated.image.copy()
        metadata.update({"model_called": True, "route": generated.route})
    else:
        raw = generated_image.copy()
        metadata.update({"model_called": False, "replayed": True})

    mask = extract_thinned_lines(raw, image.size)
    if not mask.any():
        raise ValueError("Image model returned no usable yellow anatomical boundaries")
    metadata.update({
        "raw_output_size": list(raw.size), "model_border_pixels": int(mask.sum()),
        "rough_border_pixels": int((moving > 0).sum()),
    })
    return BorderRefinementResult(rough, raw, mask, border_overlay(original, mask), metadata)
