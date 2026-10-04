"""Image-path facts for the registration task: model families and output frames.

Registration prompt TEXT lives in :mod:`langslice.core.nonlinear.prompts` now (the
colormap-lineup prompts this module used to hold — ``V14``,
``V14_EDIT_MOUSE``, ``base_segmentation_prompt`` — were deleted with the
colormap workflow). What stays here is provider-network fact, not task
wording: which model family a name belongs to, the aspect-ratio envelope an
image path accepts, and the fixed output frame each lane hands back for a
given canvas, so :func:`~langslice.core.nonlinear.image_gen_registration.prepare_canvas`
can build the working canvas AT that frame instead of resampling into or out
of the model.
"""

from __future__ import annotations

import math


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
    from langslice.core.provider_names import canonical_provider

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
