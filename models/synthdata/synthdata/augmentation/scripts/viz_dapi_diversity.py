"""End-to-end DAPI diversification panel.

Single AP slice rendered with 9 different seeds via render_dapi_section.
Each cell uses its own random gamma, density, sigma, aspect ratio, intensity,
tone shift, and brightness/contrast — the full step-4 diversification stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image

from augmentation.dapi_pipeline import render_dapi_section
from langslice_harness.atlas.core import (
    get_reference_slice, load_atlas, position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_MM = 5.335
TARGET_PX_UM = 5.0
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


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")
    ref, ann = upsampled_inputs(atlas, AP_MM)
    print(f"shape={ref.shape}  rendering {len(SEEDS)} variants...", flush=True)

    cells: list[np.ndarray] = []
    h0 = w0 = 0
    for s in SEEDS:
        out = render_dapi_section(ref, ann, atlas, seed=s, pixel_size_um=TARGET_PX_UM)
        arr = np.clip(out * 255, 0, 255).astype(np.uint8)
        cells.append(arr)
        h0, w0 = arr.shape[:2]
        print(f"  seed={s}  done", flush=True)

    cols = 3
    rows = (len(SEEDS) + cols - 1) // cols
    grid = np.zeros((rows * h0, cols * w0, 3), dtype=np.uint8)
    for idx, img in enumerate(cells):
        r, c = divmod(idx, cols)
        grid[r * h0:(r + 1) * h0, c * w0:(c + 1) * w0] = img

    out_path = Path("tmp/outputs/dapi/diversity.png")
    Image.fromarray(grid).save(out_path)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
