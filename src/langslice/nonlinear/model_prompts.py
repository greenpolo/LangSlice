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

from langslice.space import Plane

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

    Outside the range the section canvas is padded (never cropped) to the
    nearest bound. Every model-facing image shares the section's frame, so a
    ratio the image path cannot return would come back letterboxed and have
    to be cropped back.

    Verified 2026-08-25: gpt-image-2 via the OpenAI API accepts arbitrary
    WIDTHxHEIGHT within 1:3..3:1; via openai-oauth (Codex backend) the size
    parameter is ignored and the output matches the input image's aspect
    exactly (probed up to 2.35:1), so the same range is a safe envelope.
    """
    if image_model_family(image_model) == "gpt-image":
        return (1.0 / 3.0, 3.0)
    return None


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
