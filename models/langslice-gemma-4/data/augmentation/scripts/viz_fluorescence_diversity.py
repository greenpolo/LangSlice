"""Fluorescence diversity panel — 9 variants showing varied mode sampling.

Renders 9 full-section images using different seeds. Mode is sampled
per-image from FLUORESCENCE_MODES weights, so the grid should naturally
show ≥4 visually distinct modes across 9 panels.

To ensure mode diversity is visible, seeds are chosen to sample across the
weight distribution. The mode name is printed as a label on each panel so
the viewer can see which modes appear.

Output: tmp/outputs/fluorescence/diversity.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from augmentation.fluorescence_pipeline import render_fluorescence_section
from augmentation.modes import FLUORESCENCE_MODES, sample_mode
from PIL import Image, ImageDraw, ImageFont

from langslice_harness.atlas.core import (
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_MM = 5.335
TARGET_PX_UM = 5.0
# 9 seeds chosen to produce diverse mode coverage across the weight table
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99]


def upsampled_inputs(atlas: object, ap_mm: float):
    pil = get_reference_slice(atlas, ap_mm).convert("L")
    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)  # type: ignore[arg-type]
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)  # type: ignore[union-attr]

    src_um = atlas.resolution[0]  # type: ignore[attr-defined]
    scale = src_um / TARGET_PX_UM
    h0, w0 = ann.shape
    h, w = int(round(h0 * scale)), int(round(w0 * scale))
    ref_arr = np.asarray(pil.resize((w, h), Image.LANCZOS), dtype=np.uint8)
    ann_up = np.asarray(
        Image.fromarray(ann.astype(np.int32), "I").resize((w, h), Image.NEAREST),
        dtype=np.int32,
    )
    return ref_arr, ann_up


def _try_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label_panel(arr: np.ndarray, mode_name: str) -> np.ndarray:
    """Add a small mode-name label in the top-left corner."""
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    font = _try_font(11)
    label = mode_name
    # dark background strip for readability
    try:
        tw = draw.textlength(label, font=font)
    except AttributeError:
        tw = len(label) * 7
    draw.rectangle([(0, 0), (min(int(tw) + 8, arr.shape[1]), 18)], fill=(10, 10, 10))
    draw.text((4, 2), label, fill=(220, 220, 120), font=font)
    return np.asarray(img)


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")
    ref, ann = upsampled_inputs(atlas, AP_MM)
    cells: list[np.ndarray] = []
    h0 = w0 = 0
    mode_names_seen: list[str] = []

    for s in SEEDS:
        # Peek at which mode this seed will pick (same rng state as pipeline start)
        # The pipeline consumes rng for gamma and floor before mode sampling,
        # so we re-derive the mode name by simulating the first rng draws.
        rng_peek = np.random.default_rng(s)
        # consume gamma and floor draws to match pipeline state
        _ = rng_peek.uniform(0.9, 1.7)  # gamma
        _ = rng_peek.uniform(0.10, 0.20)  # floor
        chosen_mode = sample_mode(rng_peek, FLUORESCENCE_MODES)
        mode_name = chosen_mode.name

        out = render_fluorescence_section(
            ref,
            ann,
            atlas,
            seed=s,
            pixel_size_um=TARGET_PX_UM,
        )
        arr = np.clip(out * 255, 0, 255).astype(np.uint8)
        arr = _label_panel(arr, mode_name)
        cells.append(arr)
        h0, w0 = arr.shape[:2]
        mode_names_seen.append(mode_name)
        print(f"  seed={s}  mode={mode_name}", flush=True)

    unique_modes = set(mode_names_seen)
    print(f"\n  Unique modes in grid ({len(unique_modes)}): {sorted(unique_modes)}")

    cols = 3
    rows = (len(SEEDS) + cols - 1) // cols
    grid = np.zeros((rows * h0, cols * w0, 3), dtype=np.uint8)
    for idx, img in enumerate(cells):
        r, c = divmod(idx, cols)
        grid[r * h0 : (r + 1) * h0, c * w0 : (c + 1) * w0] = img

    out_path = Path("tmp/outputs/fluorescence/diversity.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
