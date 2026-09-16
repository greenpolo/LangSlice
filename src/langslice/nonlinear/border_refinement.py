"""Correct placed atlas boundaries while keeping the original photograph authoritative."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.nonlinear.image_gen_helpers import (
    _compute_deformation_field,
    _extract_borders_from_classified,
    _merge_classified,
    _register_channel_stacks,
    _run_elastix_april_borders,
    _warp_classified_labels,
)
from langslice.nonlinear.providers import (
    SegmentationGenerationRequest,
    generate_warped_segmentation_image,
)
from langslice.nonlinear.types import Deformation
from langslice.providers.registry import canonical_provider
from langslice.space import Plane


def border_refinement_prompt(plane: Plane = "coronal") -> str:
    """The rough-plus-raw experiment wording, without a mouse-only assumption.

    Sentence audit: the first two sentences identify actual attachments and
    their order; the next four specify moving existing lines against visible
    anatomy, retaining correct lines and resolving indistinct boundaries. The
    last three pin the photograph, boundary identities, style and single-image
    output. 'Mouse' is removed and the section plane is parameterized. 'Automatic'
    is removed because supplied placement may come from an agent or a person;
    this leaves the instruction to correct existing rough lines unchanged.
    """
    return (
        f"Image 1 is a photograph of a brain {plane} section with thin yellow "
        "anatomical region boundaries placed by a rough alignment. "
        "Image 2 is the same photograph in exactly the same frame, without the lines, "
        "so the underlying tissue edges are visible.\n\n"
        "Edit Image 1 so that every yellow line lies on the edge of the region it "
        "encloses, as that edge appears in Image 2. "
        "Move a line by sliding or bending it to follow this specimen's anatomy. "
        "A line that already sits on its edge stays as it is. "
        "Where an internal boundary is indistinct, use the neighboring visible structures "
        "and the supplied region arrangement to place it.\n\n"
        "The photograph in the output is Image 2 unchanged beneath the corrected lines: "
        "the same tissue and background, brain size and position, and frame. "
        "The set of boundaries stays the same, each enclosing the corresponding region "
        "in the same thin bright yellow. "
        "The output is one image: the original photograph with the corrected yellow "
        "boundaries replacing the supplied yellow boundaries."
    )


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


def border_overlay(image: Image.Image, mask: np.ndarray, width: int = 1) -> Image.Image:
    """Replace only boundary pixels on the original photograph."""
    pixels = np.array(image.convert("RGB"))
    drawn = np.asarray(mask, dtype=bool)
    if width > 1:
        drawn = cv2.dilate(drawn.astype(np.uint8), np.ones((width, width), np.uint8)) > 0
    pixels[drawn] = (255, 255, 0)
    return Image.fromarray(pixels)


@dataclass
class BorderRefinementResult:
    rough_border_overlay: Image.Image
    raw_model_image: Image.Image
    model_border_mask: np.ndarray
    model_border_overlay: Image.Image
    fitted_labels: np.ndarray
    fitted_border_overlay: Image.Image
    result_transform: Any | None
    deformation_field: np.ndarray
    elapsed: float
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
    deformation: Deformation = "bspline",
    image_prompt: str | None = None,
) -> BorderRefinementResult:
    """Fit label-preserving residual deformation to a model's corrected lines.

    All arrays use the original input canvas. The returned displacement maps
    output coordinates back into that rough canvas; the caller composes it
    with initial placement for atlas-space exports. Raw replies are untouched.
    """
    if deformation not in {"affine", "bspline"}:
        raise ValueError(f"Unsupported border deformation: {deformation}")
    labels = np.asarray(rough_labels)
    shape = (image.height, image.width)
    if labels.shape != shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Rough atlas labels must be an integer map on the image canvas")
    if not np.any(labels):
        raise ValueError("Rough atlas placement contains no regions")
    original = image.convert("RGB")
    moving = _extract_borders_from_classified(_merge_classified(labels, atlas))
    line_width = max(1, round(2 * max(image.size) / 2048))
    rough = border_overlay(original, moving > 0, line_width)
    prompt = image_prompt or border_refinement_prompt(plane)
    metadata: dict[str, Any] = {
        "workflow": "border_refinement", "prompt": prompt,
        "provider": canonical_provider(provider), "model": model,
        "deformation": deformation, "line_width_px": line_width,
        "input_size": list(image.size),
    }
    if generated_image is None and canonical_provider(provider) == "none":
        return BorderRefinementResult(
            rough, rough.copy(), moving > 0, rough.copy(), labels.copy(), rough.copy(),
            None, np.zeros((*shape, 2), dtype=np.float64), 0.0,
            {**metadata, "model_called": False, "model_free": True},
        )
    if generated_image is None:
        generated = generate_warped_segmentation_image(SegmentationGenerationRequest(
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

    # Import lazily: the candidate orchestrator may itself call this module.
    from langslice.nonlinear.image_gen_registration import crop_to_aspect

    # Extract before interpolation so tissue colors cannot blend into yellow.
    raw_mask = Image.fromarray(yellow_mask(np.asarray(raw.convert("RGB"))))
    framed = crop_to_aspect(raw_mask, image.width / image.height)
    mask = thin(np.asarray(framed.resize(image.size, Image.Resampling.NEAREST)))
    if not mask.any():
        raise ValueError("Image model returned no usable yellow anatomical boundaries")
    fixed = mask.astype(np.uint8) * 255
    if deformation == "bspline":
        transform, elapsed = _run_elastix_april_borders(fixed, moving)
    else:
        transform, elapsed = _register_channel_stacks(
            [fixed], [moving], deformation="affine"
        )
    fitted = _warp_classified_labels(labels, transform)
    field = _compute_deformation_field(transform, moving)
    if field is None or field.shape != (*shape, 2) or not np.isfinite(field).all():
        raise RuntimeError("Border registration did not produce a finite canvas deformation field")
    fitted_borders = _extract_borders_from_classified(_merge_classified(fitted, atlas)) > 0
    metadata.update({
        "raw_output_size": list(raw.size), "model_border_pixels": int(mask.sum()),
        "rough_border_pixels": int((moving > 0).sum()),
    })
    return BorderRefinementResult(
        rough, raw, mask, border_overlay(original, mask), fitted,
        border_overlay(original, fitted_borders), transform, field, elapsed, metadata,
    )
