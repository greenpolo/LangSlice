"""Render the DAPI gray/white texture pair on real atlas slices.

Three atlas APs (2.0 / 5.335 / 8.0 mm) × four columns:
    1. atlas reference template
    2. gray-matter texture only
    3. white-matter texture only
    4. combined (gray + white) on a near-black background

Renders at 5 µm/px so individual nuclei are resolvable. The atlas slice is
upsampled by lanczos before texture synthesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from augmentation.density import atlas_grayscale_density_map
from augmentation.transforms.base import TransformContext
from augmentation.transforms.texture import DAPIGrayMatterNuclei, DAPIWhiteMatterNuclei
from augmentation.transforms.tissue_class import classify_tissue
from PIL import Image

from langslice_harness.atlas.core import (
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_VALUES_MM = (2.0, 5.335, 8.0)
TARGET_PX_UM = 5.0


def slice_inputs(atlas: object, ap_mm: float):
    ref_pil = get_reference_slice(atlas, ap_mm).convert("L")
    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)  # type: ignore[arg-type]
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)  # type: ignore[union-attr]

    src_um_per_px: float = atlas.resolution[0]  # type: ignore[attr-defined]
    scale = src_um_per_px / TARGET_PX_UM
    h0, w0 = ann.shape
    h, w = int(round(h0 * scale)), int(round(w0 * scale))

    ref_up = ref_pil.resize((w, h), Image.LANCZOS)
    ref_arr = np.asarray(ref_up, dtype=np.uint8)

    # Annotation must be upsampled with NEAREST to preserve region IDs.
    ann_up = np.asarray(
        Image.fromarray(ann.astype(np.int32), "I").resize((w, h), Image.NEAREST),
        dtype=np.int32,
    )
    return ref_arr, ann_up


def to_rgb(gray_arr: np.ndarray) -> np.ndarray:
    return np.stack([gray_arr] * 3, axis=-1)


def render_on(atlas: object, ap_mm: float, *, gamma: float = 1.2, seed: int = 42):
    ref_arr, ann = slice_inputs(atlas, ap_mm)
    masks = classify_tissue(ann, atlas)
    h, w = ann.shape

    base = np.zeros((h, w, 3), dtype=np.float32)
    density = atlas_grayscale_density_map(ref_arr, masks["tissue"], gamma=gamma, floor=0.15)
    ctx = TransformContext(
        modality="dapi",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=TARGET_PX_UM,
        tissue_class_masks=masks,
    )
    rng = np.random.default_rng(seed)

    gm_only = DAPIGrayMatterNuclei(p=1.0)(base, rng=rng, ctx=ctx)
    wm_only = DAPIWhiteMatterNuclei(p=1.0)(base, rng=rng, ctx=ctx)

    rng2 = np.random.default_rng(seed)
    combined = DAPIGrayMatterNuclei(p=1.0)(base, rng=rng2, ctx=ctx)
    combined = DAPIWhiteMatterNuclei(p=1.0)(combined, rng=rng2, ctx=ctx)

    return to_rgb(ref_arr), gm_only, wm_only, combined


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")
    cells: list[tuple[int, int, np.ndarray]] = []
    max_h = max_w = 0

    for r, ap in enumerate(AP_VALUES_MM):
        ref, gm, wm, combined = render_on(atlas, ap)
        cells.append((r, 0, ref))
        for c, arr in enumerate([gm, wm, combined], start=1):
            cells.append((r, c, np.clip(arr * 255, 0, 255).astype(np.uint8)))
        h, w = ref.shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        print(f"AP={ap:5.3f}  shape={ref.shape}", flush=True)

    grid = np.zeros((3 * max_h, 4 * max_w, 3), dtype=np.uint8)
    for r, c, img in cells:
        h, w = img.shape[:2]
        y0, x0 = r * max_h, c * max_w
        grid[y0 : y0 + h, x0 : x0 + w] = img

    out = Path("tmp/outputs/dapi/textures_grid.png")
    Image.fromarray(grid, "RGB").save(out)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
