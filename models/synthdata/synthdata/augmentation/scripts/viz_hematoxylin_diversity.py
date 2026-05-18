"""End-to-end hematoxylin counterstain diversification panel — 9 variants.

Each panel renders a full atlas slice with the hematoxylin counterstain
renderer using a different seed and substrate-tone preset. Demonstrates:
- Substrate tone variation across the 4 presets (pink-cream, warm, cool, violet)
- Density variation (low / mid / high staining)
- Brightness/contrast and tone-shift variation

This script is self-contained: it calls render_hematoxylin_counterstain
directly + lightweight post-processing rather than a full pipeline module.
Track B (B2 brightfield+hematoxylin refactor) integrates the counterstain
into the full pipeline; this script is for standalone visual proof.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image

from augmentation.counterstain import (
    _HEMATOXYLIN_SUBSTRATE_PRESETS,
    render_hematoxylin_counterstain,
)
from augmentation.density import atlas_grayscale_density_map
from augmentation.transforms.base import TransformContext
from augmentation.transforms.tissue_class import classify_tissue
from langslice_harness.atlas.core import (
    get_reference_slice, load_atlas, position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_MM = 5.335
TARGET_PX_UM = 5.0
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99]


def upsampled_inputs(atlas: object, ap_mm: float):
    pil = get_reference_slice(atlas, ap_mm).convert("L")
    ctx_space = atlas_space_context(atlas)
    axis = slice_axis_index(ctx_space, "coronal")
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


def _apply_tone_shift(canvas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Warm / cool drift in substrate tone."""
    warm = rng.uniform(-0.04, 0.08)
    cool = rng.uniform(-0.03, 0.04)
    out = canvas.copy()
    out[..., 0] += warm * 0.4
    out[..., 1] += warm * 0.2
    out[..., 2] += cool * 0.4
    return np.clip(out, 0.0, 1.0)


def _apply_brightness_contrast(
    canvas: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    brightness = rng.uniform(0.88, 1.08)
    contrast = rng.uniform(0.88, 1.12)
    mean = canvas.mean(axis=(0, 1), keepdims=True)
    out = (canvas - mean) * contrast + mean
    return np.clip(out * brightness, 0.0, 1.0)


def render_hematoxylin_section(
    reference_slice: np.ndarray,
    annotation_slice: np.ndarray,
    atlas: object,
    *,
    seed: int,
    pixel_size_um: float,
    substrate_base: tuple[float, float, float] | None = None,
    gamma_range: tuple[float, float] = (0.8, 1.6),
) -> np.ndarray:
    """Standalone hematoxylin section render (no full pipeline file needed).

    Combines classify_tissue + atlas_grayscale_density_map + counterstain
    renderer + tone-shift + brightness/contrast. Track B will integrate
    render_hematoxylin_counterstain into the full pipeline; this function
    exists only for the diversity script.
    """
    rng = np.random.default_rng(seed)

    # Pick substrate preset per image if not fixed
    if substrate_base is None:
        substrate_base = _HEMATOXYLIN_SUBSTRATE_PRESETS[
            int(rng.integers(0, len(_HEMATOXYLIN_SUBSTRATE_PRESETS)))
        ]

    masks = classify_tissue(annotation_slice, atlas)
    gamma = float(rng.uniform(*gamma_range))
    floor = float(rng.uniform(0.10, 0.20))
    density = atlas_grayscale_density_map(
        reference_slice, masks["tissue"], gamma=gamma, floor=floor,
    )

    ctx = TransformContext(
        modality="hematoxylin",
        annotation_slice=annotation_slice,
        density_map=density,
        tissue_mask=masks["tissue"],
        pixel_size_um=pixel_size_um,
        tissue_class_masks=masks,
    )

    canvas = render_hematoxylin_counterstain(
        reference_slice,
        annotation_slice,
        atlas,
        masks=masks,
        density_map=density,
        ctx=ctx,
        rng=rng,
        pixel_size_um=pixel_size_um,
        substrate_base=substrate_base,
        gamma_range=(gamma, gamma),  # already drawn above
    )
    canvas = _apply_tone_shift(canvas, rng)
    canvas = _apply_brightness_contrast(canvas, rng)
    return canvas


def main() -> int:
    atlas = load_atlas("allen_mouse_25um")
    ref, ann = upsampled_inputs(atlas, AP_MM)
    print(f"shape={ref.shape}  rendering {len(SEEDS)} variants...", flush=True)

    cells: list[np.ndarray] = []
    h0 = w0 = 0
    for s in SEEDS:
        out = render_hematoxylin_section(ref, ann, atlas, seed=s, pixel_size_um=TARGET_PX_UM)
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

    out_path = Path("tmp/outputs/hematoxylin/diversity.png")
    Image.fromarray(grid).save(out_path)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
