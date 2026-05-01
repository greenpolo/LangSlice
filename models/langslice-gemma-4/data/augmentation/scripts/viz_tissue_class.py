"""Render tissue-class masks colored over the atlas reference template.

Saves a single PNG with three rows (AP=2.0 / 5.335 / 8.0 mm) and four columns
(reference / gray / white / ventricle).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image

from augmentation.transforms.tissue_class import classify_tissue
from langslice_harness.atlas.core import (
    get_reference_slice, load_atlas, position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_VALUES_MM = (2.0, 5.335, 8.0)
COLORS_RGB = {
    "gray_matter":  (255, 220,  80),  # warm yellow
    "white_matter": ( 70, 200, 255),  # cyan
    "ventricle":    (255,  80, 200),  # magenta
}


def annotation_slice(atlas: object, ap_mm: float) -> np.ndarray:
    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)  # type: ignore[arg-type]
    return np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)  # type: ignore[union-attr]


def reference_slice_rgb(atlas: object, ap_mm: float) -> np.ndarray:
    pil = get_reference_slice(atlas, ap_mm).convert("L")
    arr = np.asarray(pil, dtype=np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")

    rows: list[list[np.ndarray]] = []
    max_h = max_w = 0
    cells: list[tuple[int, int, np.ndarray]] = []

    for r, ap in enumerate(AP_VALUES_MM):
        ref = reference_slice_rgb(atlas, ap)
        ann = annotation_slice(atlas, ap)
        masks = classify_tissue(ann, atlas)
        cells.append((r, 0, ref))
        cells.append((r, 1, overlay(ref, masks["gray_matter"], COLORS_RGB["gray_matter"])))
        cells.append((r, 2, overlay(ref, masks["white_matter"], COLORS_RGB["white_matter"])))
        cells.append((r, 3, overlay(ref, masks["ventricle"], COLORS_RGB["ventricle"])))
        max_h = max(max_h, ref.shape[0])
        max_w = max(max_w, ref.shape[1])

        # quick stats
        total = masks["tissue"].sum()
        gm_pct = 100 * masks["gray_matter"].sum() / max(total, 1)
        wm_pct = 100 * masks["white_matter"].sum() / max(total, 1)
        vs_pct = 100 * masks["ventricle"].sum() / max(total, 1)
        print(f"AP={ap:5.3f}  tissue={total}px  gm={gm_pct:5.1f}%  wm={wm_pct:5.1f}%  vs={vs_pct:5.1f}%")

    grid = np.zeros((3 * max_h, 4 * max_w, 3), dtype=np.uint8)
    for r, c, img in cells:
        h, w = img.shape[:2]
        y0, x0 = r * max_h, c * max_w
        grid[y0:y0 + h, x0:x0 + w] = img

    out = Path("tmp/outputs/tissue_class/grid.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, "RGB").save(out)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
