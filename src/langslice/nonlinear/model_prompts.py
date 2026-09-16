"""Prompt and image-path facts for the registration task.

The registration prompt is the APRIL LINEUP text: the model is handed a
colored atlas region map as Image 1 and asked to deform that map onto the
tissue in Image 3, using the grayscale template in Image 2 to read the
anatomy. There are two wordings of the same task, one per model family —
:data:`V14` for Gemini, :data:`V14_EDIT_MOUSE` for the GPT image lanes —
each written to its vendor's own image-prompting guide. Both are
byte-identical to the benchmark's copies in
``_local/nonlinear_eval/prompts.py`` (``v14`` and ``v14edit_mouse``), which
is how they were measured; ``tests/test_nonlinear_prompts.py`` compares them
whenever that local file is present.

Tune the text against real runs, not per-model prose.
"""

from __future__ import annotations

import math

from langslice.space import Plane


def image_model_family(image_model: str | None) -> str:
    """Coarse family of an image model, for image-path capability lookups."""
    model = (image_model or "").lower()
    if "nano-banana" in model or "gemini" in model or "imagen" in model:
        return "nano-banana"
    return "gpt-image"


def aspect_ratio_limits(
    image_model: str | None, provider: str | None = None
) -> tuple[float, float] | None:
    """(min, max) output aspect ratio (w/h) the image path supports, or None.

    Outside the range the working canvas is padded (never cropped) to the
    nearest bound: in edit mode the model paints on ITS canvas, and
    resampling a mismatched ratio back onto the slice would undo the pixel
    alignment.

    Verified 2026-08-25: gpt-image-2 via the OpenAI API accepts arbitrary
    WIDTHxHEIGHT within 1:3..3:1; via openai-oauth (Codex backend) the size
    parameter is ignored and the output matches the input image's aspect
    exactly (probed up to 2.35:1), so the same range is a safe envelope.
    """
    if image_model_family(image_model) == "gpt-image":
        return (1.0 / 3.0, 3.0)
    return None


# v14 (2026-09-12): the pinned April base rewritten to Google's shared image-prompting
# guide: edit template up front, each input's role stated, positive framing everywhere
# (no literal negatives). Shared by Gemini Pro and Flash; tested on Flash.
V14 = (
    "Edit Image 1 so that its colored regions lie on the brain tissue in Image 3. "
    "Change only the shapes and positions of the colored regions; keep everything else "
    "exactly the same.\n\n"
    "IMAGE 1: A colored region map of a mouse brain coronal section. Each anatomical "
    "region is one flat, solid color. This is the image to edit.\n"
    "IMAGE 2: A grayscale reference showing the same anatomy as Image 1, for identifying "
    "which structure each colored region labels.\n"
    "IMAGE 3: A real histology photograph of a mouse brain coronal section. This is the "
    "target: the regions are moved onto this tissue.\n\n"
    "Warp each colored region in Image 1 so it sits on the corresponding structure visible "
    "in Image 3. The map is symmetric and idealized; the real tissue is asymmetric, "
    "stretched, and may have tears. Use Image 2 to match the map's structures to the "
    "features in the histology, and follow the natural left-right asymmetry of this "
    "particular section.\n\n"
    "Every region keeps its exact color from Image 1, and the output contains exactly the "
    "regions of Image 1, each painted flat and opaque. Where tissue is damaged or missing "
    "in Image 3, paint the expected regions in their expected place.\n\n"
    "Images 1, 2 and 3 share one frame. The output fills that same frame with the map at "
    "the size, scale and position of the tissue in Image 3: the map's outer edge lies on "
    "the tissue's outer edge, and every painted pixel sits on the tissue it labels, so the "
    "brain stays exactly where, and as large as, it is in Image 3.\n\n"
    "The output is the deformed colored regions on a plain black background, and nothing "
    "else."
)

# Codex twin of V14: same text, OpenAI gpt-image-2.5 guide wording (index + describe each
# input, invariants listed), and the species/atlas anchor the Codex lineage keeps (T1).
V14_EDIT_MOUSE = (
    "Edit Image 1 (the first image, the colored region map) so that its colored regions "
    "lie on the mouse brain tissue in Image 3 (the third image). Change only the shapes "
    "and positions of the colored regions; keep everything else exactly the same.\n\n"
    "IMAGE 1: A colored region map from the Allen Mouse Brain Atlas, a coronal plate. Each "
    "anatomical region is one flat, solid color. This is the image to edit.\n"
    "IMAGE 2: The grayscale Allen Mouse Brain Atlas template of the same plate, for "
    "identifying which structure each colored region labels.\n"
    "IMAGE 3: A real histology photograph of a mouse brain coronal section. This is the "
    "target: the regions are moved onto this tissue.\n\n"
    "Warp each colored region in Image 1 so it sits on the corresponding structure visible "
    "in Image 3. The atlas plate is symmetric and idealized; the real tissue is asymmetric, "
    "stretched, and may have tears. Use Image 2 to match the plate's structures to the "
    "features in the histology, and follow the natural left-right asymmetry of this "
    "particular section.\n\n"
    "Invariants: every region keeps its exact color from Image 1; the output contains "
    "exactly the regions of Image 1, each painted flat and opaque; the background stays "
    "black; this is a mouse brain and the regions are those of Image 1. Where tissue is "
    "damaged or missing in Image 3, paint the expected regions in their expected place.\n\n"
    "Images 1, 2 and 3 share one frame. The output fills that same frame with the map at "
    "the size, scale and position of the tissue in Image 3: the map's outer edge lies on "
    "the tissue's outer edge, and every painted pixel sits on the tissue it labels, so the "
    "brain stays exactly where, and as large as, it is in Image 3.\n\n"
    "The output is the deformed colored regions on a plain black background, and nothing "
    "else: no text, no labels, no extra elements."
)


def base_segmentation_prompt(plane: Plane = "coronal", provider: str | None = None) -> str:
    """The April-lineup registration prompt for *provider*'s model family.

    ``gemini-api`` gets :data:`V14`; every GPT image lane (``openai-api``,
    ``openai-oauth``) gets :data:`V14_EDIT_MOUSE`. Both texts are written
    for a coronal section — the only plane they were measured on — and name
    the plane twice; a sagittal or horizontal job gets the same text with
    that word swapped, which leaves the coronal prompt byte-identical to the
    benchmark's.
    """
    from langslice.providers.registry import canonical_provider

    text = V14 if canonical_provider(provider or "gemini-api") == "gemini-api" else V14_EDIT_MOUSE
    if plane != "coronal":
        text = text.replace("coronal", plane)
    return text


#: Codex-lane (openai-oauth) output: a fixed ~1.573 Mpx budget at the INPUT
#: aspect, size/quality ignored. Read off 19 outputs on 2026-09-11: width =
#: floor(sqrt(budget * aspect)), height = round(budget / width), within 1 px.
_CODEX_OUTPUT_BUDGET_PX = 1024 * 1536

#: Gemini image models return one fixed frame per (aspect ratio, size tier);
#: measured live on gemini-3.1-flash-lite-image / flash-image, 2026-09-11.
#: Portrait entries are the transposed landscape ones (assumed, not measured).
_GEMINI_FRAMES: dict[str, dict[str, tuple[int, int]]] = {
    "1K": {"1:1": (1024, 1024), "4:3": (1200, 896), "3:2": (1264, 848), "5:4": (1152, 928)},
    "512": {"3:2": (624, 416)},
}
_GEMINI_TIER_ALIASES = {"1K": "1K", "512": "512", "512P": "512", "512PX": "512", "0.5K": "512"}


def _nearest_gemini_frame(aspect: float, tier: str) -> tuple[str, tuple[int, int]]:
    frames = dict(_GEMINI_FRAMES[tier])
    for key, (w, h) in _GEMINI_FRAMES[tier].items():
        a, b = key.split(":")
        if a != b:
            frames[f"{b}:{a}"] = (h, w)
    return min(frames.items(), key=lambda kv: abs(math.log((kv[1][0] / kv[1][1]) / aspect)))


def gemini_aspect_for(aspect: float, quality: str | None = None) -> str:
    """The Gemini ``aspect_ratio`` string nearest a canvas aspect (w/h)."""
    tier = _GEMINI_TIER_ALIASES.get((quality or "1K").upper(), "1K")
    return _nearest_gemini_frame(aspect, tier if tier in _GEMINI_FRAMES else "1K")[0]


def native_output_size(
    image_model: str | None,
    provider: str | None,
    canvas_size: tuple[int, int],
    quality: str | None = None,
) -> tuple[int, int] | None:
    """The frame the image path will hand back for a canvas of this size.

    The working canvas is built AT this size so the edited image and the
    output share one pixel grid: no resample into the model, none out of it,
    and no input tokens spent on pixels the output cannot carry. None when
    the lane's frame is not known (the canvas then keeps the long-edge rule).

    - openai-oauth (Codex): the fixed budget at the canvas aspect.
    - openai-api: the same budget on the endpoint's 16-px grid (the request
      size then equals the canvas; ``_api_edit_size`` sends it verbatim).
    - gemini-api: the model's fixed frame for the nearest legal aspect at the
      requested tier (``quality`` = 1K | 512; 2K/4K frames not measured).
    """
    from langslice.providers.registry import canonical_provider

    w, h = canvas_size
    aspect = w / h
    canon = canonical_provider(provider) if provider else None
    family = image_model_family(image_model)
    if canon == "openai-oauth" or (canon is None and family == "gpt-image"):
        out_w = math.floor(math.sqrt(_CODEX_OUTPUT_BUDGET_PX * aspect))
        return out_w, round(_CODEX_OUTPUT_BUDGET_PX / out_w)
    if canon == "openai-api":
        # Same budget as the Codex lane (lanes stay comparable, same output
        # tokens) on the endpoint's 16-px grid; the canvas is padded out to
        # it, so the request IS the canvas and nothing is stretched.
        out_w = max(16, round(math.sqrt(_CODEX_OUTPUT_BUDGET_PX * aspect) / 16) * 16)
        return out_w, max(16, round(_CODEX_OUTPUT_BUDGET_PX / out_w / 16) * 16)
    if canon == "gemini-api" or family == "nano-banana":
        tier = _GEMINI_TIER_ALIASES.get((quality or "1K").upper(), "1K")
        if tier not in _GEMINI_FRAMES:
            return None
        return _nearest_gemini_frame(aspect, tier)[1]
    return None
