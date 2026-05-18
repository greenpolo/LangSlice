"""Brightfield DAB+hematoxylin compare: real vs procedural.

Two columns:
    Left:  real reference — NF IHC + hematoxylin (DAB brown + blue-violet nuclei)
           from PMC10658376 Fig 9 panel J, CC BY 4.0
    Right: procedural brightfield section at mode=pan_neuronal, counterstain=hematoxylin
           (cortex crop shown at matched scale so cellular detail is visible)

Output: tmp/outputs/brightfield/compare.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.brightfield_pipeline import render_brightfield_section
from langslice_harness.atlas.core import (
    get_reference_slice, load_atlas, position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_MM = 5.335
TARGET_PX_UM = 5.0
SEED = 42


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


def _add_label(img: Image.Image, text: str, sub: str = "") -> Image.Image:
    """Add a black label bar at the top with white main text."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
        small = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        small = font
    bar_h = 52
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(10, 10, 10))
    draw.text((8, 4), text, fill=(255, 255, 255), font=font)
    if sub:
        draw.text((8, 32), sub, fill=(180, 200, 255), font=small)
    return img


def _separator(h: int, w: int = 8) -> np.ndarray:
    """Thin white separator column."""
    return np.full((h, w, 3), 220, dtype=np.uint8)


def main() -> int:
    real_path = Path("tmp/outputs/brightfield/real_dab_hematoxylin_section.png")
    if not real_path.exists():
        print(f"ERROR: Real reference not found at {real_path}")
        return 1

    # --- Load real reference ---
    real_img = Image.open(real_path).convert("RGB")
    # Crop away the label strip at the bottom (legend / scale bar text)
    real_arr = np.asarray(real_img)
    # Find where the image has the staining content (not white legend strip)
    # The legend strip at bottom is mostly white — crop to just the tissue
    row_means = real_arr.mean(axis=(1, 2))
    # Keep rows up to last row that's darker than 240 (non-white)
    dark_rows = np.where(row_means < 240)[0]
    if len(dark_rows) > 0:
        real_arr = real_arr[:dark_rows[-1] + 1, :, :]
    real_img = Image.fromarray(real_arr)
    print(f"Real reference (cropped): {real_img.size}")

    # --- Render procedural section and crop to cortex ---
    atlas = load_atlas("allen_mouse_25um")
    ref, ann = upsampled_inputs(atlas, AP_MM)
    out = render_brightfield_section(
        ref, ann, atlas,
        seed=SEED,
        pixel_size_um=TARGET_PX_UM,
        mode="pan_neuronal",
        counterstain="hematoxylin",
    )
    h_full, w_full = out.shape[:2]
    print(f"Full procedural section: {w_full}x{h_full}")

    # Crop to isocortex region (upper-centre strip: roughly top 1/4 height,
    # centred horizontally).  This matches the magnification of the real
    # reference image which is a ~400-µm wide cortex crop at ~50x.
    crop_h = h_full // 4
    crop_w = min(w_full // 3, crop_h * 2)  # keep landscape aspect
    y0 = h_full // 8  # slightly below the very top edge (sub-dural tissue)
    x0 = (w_full - crop_w) // 2
    proc_crop = out[y0:y0 + crop_h, x0:x0 + crop_w, :]
    proc_arr = np.clip(proc_crop * 255, 0, 255).astype(np.uint8)
    proc_img = Image.fromarray(proc_arr)
    print(f"Procedural crop: {proc_img.size}")

    # --- Match heights ---
    target_h = max(real_img.height, proc_img.height)
    rw = int(round(real_img.width * target_h / real_img.height))
    real_img = real_img.resize((rw, target_h), Image.LANCZOS)
    pw = int(round(proc_img.width * target_h / proc_img.height))
    proc_img = proc_img.resize((pw, target_h), Image.LANCZOS)

    # --- Add labels ---
    real_img = _add_label(
        real_img,
        "REAL — NF IHC + Hematoxylin",
        "Mouse brain; PMC10658376 Fig.9J, CC BY 4.0",
    )
    proc_img = _add_label(
        proc_img,
        "PROCEDURAL — pan_neuronal + hematoxylin",
        f"AP={AP_MM:.3f}mm cortex crop; seed={SEED}",
    )

    # --- Composite ---
    sep = _separator(target_h)
    grid = np.concatenate(
        [np.asarray(real_img), sep, np.asarray(proc_img)],
        axis=1,
    )
    out_path = Path("tmp/outputs/brightfield/compare.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
