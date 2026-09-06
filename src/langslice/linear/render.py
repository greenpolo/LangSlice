"""Rendering the stack: sections as the run sees them, plus the status table.

Every path that shows or measures a section goes through :func:`render_slice`,
so the pixels the agent judges are the pixels a fit is computed on. Corrections
are applied in one order everywhere: ROTATE first, then FLIP left-right.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from google.genai import types
from PIL import Image, ImageDraw

from langslice.image_prep import (
    adaptive_preprocess,
    crop_to_tissue,
    normalize_image,
    prepare_image_for_vlm,
)
from langslice.linear.state import SliceState, StackState

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox that renders
    from langslice.linear.engine import EngineContext

#: Long edge for ``view_slices`` images: enough to zoom past the seed-message
#: stack images without paying full-resolution tokens.
VIEW_LONG_EDGE = 1024
#: Long edge for the per-section images in the seed message. Small on purpose:
#: the whole stack (40-odd sections) rides in one user message and stays in
#: context for the whole session.
SEED_IMAGE_LONG_EDGE = 512
#: Working frame for transform fits and their preview panels.
PREVIEW_LONG_EDGE = 512

_ROTATE_OPS = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def image_to_part(img: Image.Image, *, quality: int = 85) -> types.Part:
    """One PIL image as a JPEG ``types.Part``."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return types.Part.from_bytes(mime_type="image/jpeg", data=buf.getvalue())


def render_slice(
    ctx: EngineContext,
    record: SliceState,
    *,
    long_edge: int = VIEW_LONG_EDGE,
    frame: bool = False,
) -> Image.Image:
    """One section as the run sees it: normalized, framed, enhanced, corrected.

    With ``spec.preprocess == "auto"`` (the default) the section is run through
    :func:`~langslice.image_prep.adaptive_preprocess` — per-channel CLAHE plus a
    DAPI-weighted grayscale blend — so dim fluorescence reads like the atlas
    instead of like a black field. Display only: the user's file is never
    touched.

    *frame* crops to the tissue plus a small margin before the resize, so the
    section fills its frame about as much as a cropped atlas render does. It is
    off by default because it changes the image's coordinate frame: only the
    paths that SHOW a section to a model set it, never a fit, whose parameters
    are normalized against the render they were computed on.

    Renders are cached on *ctx*, so the returned image is shared: read it,
    never mutate it in place.
    """
    key = (record.id, record.flip, record.rotation_deg, long_edge, ctx.spec.preprocess, frame)
    cached = ctx.render_cache.get(key)
    if cached is not None:
        return cached

    with Image.open(ctx.image_path(record.id)) as handle:
        # Detach from the file handle: prepare_image_for_vlm can hand back the
        # very object it was given when no resize is needed.
        source = normalize_image(handle.copy())
    if frame:
        source = crop_to_tissue(source)
    prepped = prepare_image_for_vlm(source, max_long_edge=long_edge).image
    if ctx.spec.preprocess == "auto":
        prepped = adaptive_preprocess(prepped)
    rotate = _ROTATE_OPS.get(int(record.rotation_deg) % 360)
    if rotate is not None:
        prepped = prepped.transpose(rotate)
    if record.flip:
        prepped = prepped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    ctx.render_cache[key] = prepped
    return prepped


# --- the status table ----------------------------------------------------


def status_rows(state: StackState) -> list[dict[str, Any]]:
    """One row per section in corrected order. The ``ls`` of the environment.

    ``spacing_to_next_mm`` is the distance to the next section in corrected
    order that carries a position, and null when this section has none or no
    placed section follows it. Data only: no comparison against the nominal
    interval, no verdict.
    """
    ordered = state.in_order()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(ordered):
        here = record.position_mm
        spacing: float | None = None
        if here is not None:
            following = next(
                (
                    value
                    for r in ordered[index + 1 :]
                    if (value := r.position_mm) is not None
                ),
                None,
            )
            if following is not None:
                spacing = round(following - here, 3)
        transform = record.transform or {}
        rows.append(
            {
                "index": record.index_corrected,
                "id": record.id,
                "position_mm": round(here, 3) if here is not None else None,
                "spacing_to_next_mm": spacing,
                "flip": record.flip,
                "rotation_deg": record.rotation_deg,
                "damaged": record.damaged,
                "damage_note": record.damage_note,
                "transform": transform.get("kind"),
                "confidence": record.confidence,
                "caveats": list(record.caveats),
            }
        )
    return rows


def status_text(state: StackState) -> str:
    """The status rows as one line per section, for a text message."""
    lines = [
        "index  id  position_mm  spacing_to_next_mm  flags  transform  confidence"
    ]
    for row in status_rows(state):
        flags: list[str] = []
        if row["flip"]:
            flags.append("flipped")
        if row["rotation_deg"]:
            flags.append(f"rotated {row['rotation_deg']}")
        if row["damaged"]:
            note = row["damage_note"]
            flags.append(f"damaged: {note}" if note else "damaged")
        flags.extend(row["caveats"])
        position = (
            "unplaced" if row["position_mm"] is None else f"{row['position_mm']:.3f} mm"
        )
        spacing = (
            "-"
            if row["spacing_to_next_mm"] is None
            else f"{row['spacing_to_next_mm']:.3f}"
        )
        lines.append(
            f"{row['index']:>3}  {row['id']}  {position}  {spacing}"
            + (f"  [{'; '.join(flags)}]" if flags else "")
            + (f"  transform={row['transform']}" if row["transform"] else "")
            + (f"  confidence={row['confidence']}" if row["confidence"] else "")
        )
    return "\n".join(lines)


def slice_flags(record: SliceState) -> list[str]:
    """The section's current corrections, as short human-readable flags."""
    flags: list[str] = []
    if record.rotation_deg:
        flags.append(f"rotated {record.rotation_deg}")
    if record.flip:
        flags.append("flipped")
    if record.damaged:
        flags.append(
            f"damaged: {record.damage_note}" if record.damage_note else "damaged"
        )
    return flags


def stack_image_parts(
    state: StackState, ctx: EngineContext, *, long_edge: int = SEED_IMAGE_LONG_EDGE
) -> list[types.Part]:
    """The whole stack as labelled text+image pairs, in corrected order.

    Each section gets a one-line label — ``"<corrected index>: <filename>"``
    plus any flags — immediately followed by its own image, rendered through
    :func:`render_slice` so preprocessing and corrections are already applied.

    One image per section rather than one thumbnail grid: a grid splits a fixed
    vision-encoder patch budget across every section at once and lets
    neighbouring sections share patch boundaries. A labelled sequence at a
    modest resolution reads better, and the label is what binds each set of
    pixels to a filename the model can quote back.
    """
    parts: list[types.Part] = [
        types.Part.from_text(
            text=(
                f"The {len(state.slices)} sections of the stack follow, in "
                "their current corrected order, one image each. Every image is "
                "preceded by its label '<index>: <filename>' and is rendered "
                "with any rotation and flip already applied."
            )
        )
    ]
    for record in state.in_order():
        flags = slice_flags(record)
        label = f"{record.index_corrected}: {record.id}"
        if flags:
            label += f"  [{'; '.join(flags)}]"
        parts.append(types.Part.from_text(text=label))
        parts.append(
            image_to_part(render_slice(ctx, record, long_edge=long_edge, frame=True))
        )
    return parts


def preview_panel(
    section: Image.Image, atlas_section: Image.Image, matrix: np.ndarray
) -> Image.Image:
    """Transformed section | atlas section | overlay, side by side, labelled.

    The overlay puts the transformed section in magenta and the atlas section
    in green: where the two agree the pixels go white, and every mismatch shows
    up as a coloured fringe on the side that is out of place.
    """
    width, height = section.size
    section_gray = np.asarray(section.convert("L"), dtype=np.uint8)
    warped = cv2.warpAffine(
        section_gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0
    )
    atlas_resized = atlas_section.convert("RGB").resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    atlas_gray = np.asarray(atlas_resized.convert("L"), dtype=np.uint8)
    overlay = np.dstack([warped, atlas_gray, warped])

    cells = [
        (Image.fromarray(warped, mode="L").convert("RGB"), "transformed section"),
        (atlas_resized, "atlas section"),
        (Image.fromarray(overlay, mode="RGB"), "overlay: magenta=section, green=atlas"),
    ]
    label_px = 14
    panel = Image.new("RGB", (width * len(cells), height + label_px), (16, 16, 16))
    draw = ImageDraw.Draw(panel)
    for index, (image, label) in enumerate(cells):
        panel.paste(image, (index * width, 0))
        draw.text((index * width + 3, height + 2), label, fill=(235, 235, 235))
    return panel
