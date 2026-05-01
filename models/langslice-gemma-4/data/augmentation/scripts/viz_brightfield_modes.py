"""Brightfield mode + counterstain comparison grid.

Six columns (3 modes × 2 counterstains) × three rows (seeds) — shows how the
same atlas slice renders across pan-neuronal / sparse-interneuron / myelin
staining patterns and none / hematoxylin counterstain variants.

Grid layout (columns left to right):
    pan_neuronal+none | pan_neuronal+hematoxylin
    sparse_interneuron+none | sparse_interneuron+hematoxylin
    myelin+none | myelin+hematoxylin
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.brightfield_pipeline import (
    BRIGHTFIELD_MODES, render_brightfield_section,
)
from langslice_harness.atlas.core import (
    get_reference_slice, load_atlas, position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_MM = 5.335
TARGET_PX_UM = 5.0
SEEDS = [11, 22, 33]
COUNTERSTAINS = ["none", "hematoxylin"]


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


def _label(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    # Wrap long labels
    lines = text.split(" + ")
    label_h = 56
    draw.rectangle([(0, 0), (img.width, label_h)], fill=(0, 0, 0))
    if len(lines) == 2:
        draw.text((8, 2), lines[0], fill=(255, 255, 255), font=font)
        try:
            small_font = ImageFont.truetype("arial.ttf", 32)
        except OSError:
            small_font = font
        draw.text((8, 36), f"+ {lines[1]}", fill=(180, 180, 255), font=small_font)
    else:
        draw.text((8, 10), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")
    ref, ann = upsampled_inputs(atlas, AP_MM)

    # Build column list: mode × counterstain pairs
    columns: list[tuple[str, str]] = [
        (mode, cs)
        for mode in BRIGHTFIELD_MODES
        for cs in COUNTERSTAINS
    ]
    n_cols = len(columns)  # 6
    n_rows = len(SEEDS)    # 3

    cells: list[list[np.ndarray]] = []
    for s in SEEDS:
        row: list[np.ndarray] = []
        for mode, cs in columns:
            out = render_brightfield_section(
                ref, ann, atlas,
                seed=s,
                pixel_size_um=TARGET_PX_UM,
                mode=mode,
                counterstain=cs,
            )
            arr = np.clip(out * 255, 0, 255).astype(np.uint8)
            label = f"{mode} + {cs}"
            arr = _label(arr, label)
            row.append(arr)
            print(f"  seed={s}  mode={mode}  counterstain={cs}  done", flush=True)
        cells.append(row)

    h0, w0 = cells[0][0].shape[:2]
    grid = np.zeros((n_rows * h0, n_cols * w0, 3), dtype=np.uint8)
    for r, row in enumerate(cells):
        for c, img in enumerate(row):
            grid[r * h0:(r + 1) * h0, c * w0:(c + 1) * w0] = img

    out_path = Path("tmp/outputs/brightfield/modes.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
