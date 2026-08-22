"""Atlas grid-image builder used by the image-gen registration tool.

Historically this module also hosted the legacy pre-ADK tool-loop (schemas,
nudge/sweep heuristics, submit validators). That code had no callers after
the ADK migration and was deleted; `_build_atlas_grid` remains because
`harness/estimation/image_gen.py` still imports it from here.
"""

from __future__ import annotations

from typing import Any, cast

from PIL import Image

from langslice_harness.image_prep import normalize_image


def _build_atlas_grid(
    atlas: object,
    positions: list[float],
    *,
    target_image: Image.Image | None = None,
    grid_width: int = 2048,
    cell_width: int | None = None,
    show_borders: bool = False,
    max_positions: int = 8,
) -> Image.Image:
    """Build a comparison image: target slice (left) + 2x2 atlas grid (right)."""
    from PIL import ImageDraw, ImageFont

    atlas_obj = cast(Any, atlas)
    n = min(len(positions), max_positions)

    if n <= 2:
        cols = n
    elif n <= 4:
        cols = 2
    elif n <= 6:
        cols = 3
    else:
        cols = 4

    if cell_width is not None:
        grid_width = cell_width * cols

    slices: list[tuple[Image.Image, float]] = []
    for pos in positions[:max_positions]:
        try:
            if show_borders:
                from langslice_harness.atlas.core import get_composite_slice

                ref_img = get_composite_slice(atlas_obj, pos)
            else:
                from langslice_harness.atlas.core import get_reference_slice

                ref_img = get_reference_slice(atlas_obj, pos)
            slices.append((normalize_image(ref_img), pos))
        except (ValueError, IndexError):
            pass

    if not slices:
        return Image.new("RGB", (grid_width, grid_width // 2), (0, 0, 0))

    label_height = 100
    gap = 12

    try:
        font_large = ImageFont.truetype("arial.ttf", 70)
        font_small = ImageFont.truetype("arial.ttf", 50)
    except OSError:
        font_large = ImageFont.load_default()
        font_small = font_large

    cell_width = grid_width // cols
    sample_w, sample_h = slices[0][0].size
    aspect = sample_h / sample_w
    cell_img_height = int(cell_width * aspect)
    cell_height = cell_img_height + label_height
    rows = (len(slices) + cols - 1) // cols
    grid_height = rows * cell_height

    target_section_width = 0
    target_resized = None
    if target_image is not None:
        tw, th = target_image.size
        target_img_height = grid_height - label_height
        target_scale = target_img_height / th
        target_section_width = int(tw * target_scale) + gap
        target_resized = target_image.resize(
            (int(tw * target_scale), target_img_height),
            Image.Resampling.LANCZOS,
        )

    total_width = target_section_width + grid_width
    canvas = Image.new("RGB", (total_width, grid_height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    if target_resized is not None:
        canvas.paste(target_resized.convert("RGB"), (0, 0))
        label = "TARGET SLICE"
        bbox = draw.textbbox((0, 0), label, font=font_large)
        text_w = bbox[2] - bbox[0]
        text_x = (target_section_width - gap - text_w) // 2
        text_y = grid_height - label_height + 10
        draw.text((text_x, text_y), label, fill=(255, 200, 0), font=font_large)

    for idx, (img, pos) in enumerate(slices):
        row = idx // cols
        col = idx % cols
        x = target_section_width + col * cell_width
        y = row * cell_height

        upscaled = img.resize(
            (cell_width - gap, cell_img_height), Image.Resampling.LANCZOS
        )
        canvas.paste(upscaled.convert("RGB"), (x, y))

        label = f"[{idx + 1}] {pos:.2f} mm"
        bbox = draw.textbbox((0, 0), label, font=font_small)
        text_w = bbox[2] - bbox[0]
        text_x = x + (cell_width - gap - text_w) // 2
        text_y = y + cell_img_height + 8
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font_small)

    return canvas
