"""Side-by-side Nissl comparison: real Mouse10 crops vs procedural rendering.

Top row:
    real GM crop | procedural GM patch
Bottom row:
    real WM crop | procedural WM patch

Tunes the procedural pipeline against a known real sample. The synthetic
patches use a uniform GM-only or WM-only mask + uniform high density so the
comparison isolates color/sigma/density choices, not anatomy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image

from augmentation.transforms.base import TransformContext
from augmentation.transforms.texture import (
    NisslGrayMatterCellBodies, NisslWhiteMatterCellBodies,
)


REAL_GM = Path("tmp/outputs/nissl/real_gm_crop.png")
REAL_WM = Path("tmp/outputs/nissl/real_wm_crop.png")
OUT = Path("tmp/outputs/nissl/compare.png")


def _ctx_uniform_class(h: int, w: int, *, kind: str) -> TransformContext:
    if kind not in ("gray_matter", "white_matter"):
        raise ValueError(kind)
    mask = np.ones((h, w), dtype=bool)
    other = np.zeros((h, w), dtype=bool)
    return TransformContext(
        modality="nissl",
        annotation_slice=None,
        density_map=np.full((h, w), 0.85, dtype=np.float32),
        tissue_mask=mask,
        pixel_size_um=5.0,
        tissue_class_masks={
            "gray_matter": mask if kind == "gray_matter" else other,
            "white_matter": mask if kind == "white_matter" else other,
            "ventricle": np.zeros((h, w), dtype=bool),
            "tissue": mask,
            "background": np.zeros((h, w), dtype=bool),
        },
    )


def _make_canvas(h: int, w: int, cream: tuple[float, float, float], rng) -> np.ndarray:
    canvas = np.empty((h, w, 3), dtype=np.float32)
    cream_arr = np.array(cream, dtype=np.float32)
    canvas[:] = cream_arr
    # subtle per-pixel cream noise
    canvas += rng.uniform(-0.02, 0.02, canvas.shape).astype(np.float32)
    return np.clip(canvas, 0.0, 1.0)


def render_gm_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cream = (0.74, 0.82, 0.95)  # blue-cream Mouse10 substrate
    canvas = _make_canvas(h, w, cream, rng)
    transform = NisslGrayMatterCellBodies(
        p=1.0,
        density_range_per_mm2=(2200.0, 3200.0),
        sigma_range_scale_small=(0.5, 0.9),
        sigma_range_scale_large=(1.4, 2.2),
        large_cell_fraction=0.08,
        intensity_range=(0.25, 0.55),
        cream=cream,
        cytoplasm_blur_radius_scale=0.0,
    )
    return transform(canvas, rng=rng, ctx=_ctx_uniform_class(h, w, kind="gray_matter"))


def render_wm_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cream = (0.78, 0.84, 0.95)  # slightly lighter / bluer cream for WM
    canvas = _make_canvas(h, w, cream, rng)
    transform = NisslWhiteMatterCellBodies(
        p=1.0,
        density_range_per_mm2=(150.0, 350.0),
        sigma_range_scale=(0.4, 0.8),
        intensity_range=(0.20, 0.40),
        cream=cream,
    )
    return transform(canvas, rng=rng, ctx=_ctx_uniform_class(h, w, kind="white_matter"))


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _label(arr: np.ndarray, text: str) -> np.ndarray:
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    tw = draw.textlength(text, font=font)
    draw.rectangle([(0, 0), (tw + 20, 36)], fill=(0, 0, 0))
    draw.text((10, 6), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main() -> int:
    real_gm = np.asarray(Image.open(REAL_GM).convert("RGB"))
    real_wm = np.asarray(Image.open(REAL_WM).convert("RGB"))

    proc_gm = _to_uint8(render_gm_patch(real_gm.shape[0], real_gm.shape[1], seed=0))
    proc_wm = _to_uint8(render_wm_patch(real_wm.shape[0], real_wm.shape[1], seed=1))

    real_gm = _label(real_gm, "REAL GM (cortex)")
    proc_gm = _label(proc_gm, "PROCEDURAL GM")
    real_wm = _label(real_wm, "REAL WM (corpus callosum)")
    proc_wm = _label(proc_wm, "PROCEDURAL WM")

    gm_row = np.concatenate([real_gm, proc_gm], axis=1)
    wm_row = np.concatenate([real_wm, proc_wm], axis=1)

    # Pad widths to the wider row
    target_w = max(gm_row.shape[1], wm_row.shape[1])
    def pad_w(a):
        if a.shape[1] == target_w:
            return a
        pad = np.full((a.shape[0], target_w - a.shape[1], 3), 32, dtype=np.uint8)
        return np.concatenate([a, pad], axis=1)

    gm_row, wm_row = pad_w(gm_row), pad_w(wm_row)
    grid = np.concatenate([gm_row, wm_row], axis=0)
    Image.fromarray(grid).save(OUT)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
