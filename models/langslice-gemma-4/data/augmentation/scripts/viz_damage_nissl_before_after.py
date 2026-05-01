"""Nissl before/after damage comparison — 2-panel side-by-side at higher resolution.

Output: tmp/outputs/damage_nissl_before_after.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.nissl_pipeline import render_nissl_section
from langslice_harness.atlas.core import (
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_MM = 5.335
TARGET_PX_UM = 5.0
SEED = 42
CREAM = (0.85, 0.78, 0.65)
OUT_PATH = Path("tmp/outputs/damage_nissl_before_after.png")

PANEL_H = 480
PANEL_W = 640
LABEL_H = 36


def _try_font(size: int = 16):
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> int:
    print("Loading atlas...", flush=True)
    atlas = load_atlas("allen_mouse_25um")

    pil = get_reference_slice(atlas, AP_MM).convert("L")
    ctx_space = atlas_space_context(atlas)
    axis = slice_axis_index(ctx_space, "coronal")
    idx = position_mm_to_index(atlas, AP_MM)
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)

    src_um = atlas.resolution[0]
    scale = src_um / TARGET_PX_UM
    h0, w0 = ann.shape
    h, w = int(round(h0 * scale)), int(round(w0 * scale))
    ref = np.asarray(pil.resize((w, h), Image.LANCZOS), dtype=np.uint8)
    ann_up = np.asarray(
        Image.fromarray(ann.astype(np.int32), "I").resize((w, h), Image.NEAREST),
        dtype=np.int32,
    )

    print("Rendering clean...", flush=True)
    clean = render_nissl_section(
        ref, ann_up, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM,
        cream_base=CREAM, apply_damage=False,
    )

    print("Rendering damaged...", flush=True)
    damaged = render_nissl_section(
        ref, ann_up, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM,
        cream_base=CREAM, apply_damage=True, damage_intensity="medium",
    )

    def _to_panel(arr: np.ndarray, label: str) -> np.ndarray:
        img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
        img = img.resize((PANEL_W, PANEL_H), Image.LANCZOS)
        arr_p = np.asarray(img)

        bar = np.full((LABEL_H, PANEL_W, 3), 25, dtype=np.uint8)
        img_bar = Image.fromarray(bar)
        draw = ImageDraw.Draw(img_bar)
        font = _try_font(16)
        draw.text((8, 8), label, fill=(230, 230, 230), font=font)
        return np.concatenate([np.asarray(img_bar), arr_p], axis=0)

    panel_clean = _to_panel(clean, "Nissl — clean (apply_damage=False)")
    panel_dam = _to_panel(damaged, "Nissl — medium damage (apply_damage=True)")

    # 2-pixel divider
    divider = np.full((LABEL_H + PANEL_H, 2, 3), 80, dtype=np.uint8)
    final = np.concatenate([panel_clean, divider, panel_dam], axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(OUT_PATH)
    print(f"Saved -> {OUT_PATH}  ({final.shape[1]}x{final.shape[0]} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
