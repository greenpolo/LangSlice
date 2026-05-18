"""ISH diversity panel — 9 sections with mode-sampled counterstain + signal.

Each cell is a different seed, so both the expression pattern AND the ISH
mode (allen_style, nbt_nfr, dab_hematoxylin, fish, nissl_nbt) are randomly
sampled.  Over 9 panels you should see roughly the expected mode distribution:
~3 allen/nbt_nfr, ~2 dab_hematoxylin, ~1-2 fish, ~0-1 nissl_nbt.

The mode actually chosen for each panel is printed to stdout.

Output: tmp/outputs/ish/diversity.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.ish_pipeline import render_ish_section
from augmentation.modes import ISH_MODES, sample_mode
from langslice_harness.atlas.core import (
    get_reference_slice, load_atlas, position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_MM = 5.335
TARGET_PX_UM = 5.0
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99]


def _try_font(size: int = 11) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label_panel(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    font = _try_font(11)
    draw.rectangle([(0, 0), (arr.shape[1], 20)], fill=(0, 0, 0, 180))
    draw.text((3, 3), text, fill=(240, 240, 240), font=font)
    return np.asarray(img)


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


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")
    ref, ann = upsampled_inputs(atlas, AP_MM)
    cells: list[np.ndarray] = []
    h0 = w0 = 0
    mode_counts: dict[str, int] = {}

    for s in SEEDS:
        # Determine which mode this seed will pick (replicate the rng state
        # that render_ish_section uses before calling sample_mode).
        # render_ish_section: rng → masks (no rng calls) → gamma (1) → floor (1)
        # → sample_mode (1 draw).  Replicate those 3 draws to peek at mode.
        rng_peek = np.random.default_rng(s)
        rng_peek.uniform(0.9, 1.7)   # gamma
        rng_peek.uniform(0.10, 0.20) # floor
        chosen = sample_mode(rng_peek, ISH_MODES)

        out = render_ish_section(ref, ann, atlas, seed=s, pixel_size_um=TARGET_PX_UM)
        arr = np.clip(out * 255, 0, 255).astype(np.uint8)
        arr = _label_panel(arr, f"s={s} {chosen.name}")
        cells.append(arr)
        h0, w0 = arr.shape[:2]
        mode_counts[chosen.name] = mode_counts.get(chosen.name, 0) + 1
        print(f"  seed={s:3d}  mode={chosen.name}", flush=True)

    print("\nMode counts:", mode_counts)

    cols = 3
    rows = (len(SEEDS) + cols - 1) // cols
    grid = np.zeros((rows * h0, cols * w0, 3), dtype=np.uint8)
    for idx, img in enumerate(cells):
        r, c = divmod(idx, cols)
        grid[r * h0:(r + 1) * h0, c * w0:(c + 1) * w0] = img

    out_path = Path("tmp/outputs/ish/diversity.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
