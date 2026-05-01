"""Side-by-side ISH comparison: real Allen ISH crops vs procedural rendering.

Three rows:
    real GM cortex          | procedural GM patch
    real WM corpus callosum | procedural WM patch
    real expressed hippocampus | procedural expressed patch
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.transforms.base import TransformContext
from augmentation.transforms.texture import ISHExpressionPuncta


REAL_GM = Path("tmp/outputs/ish/real_gm_crop.png")
REAL_WM = Path("tmp/outputs/ish/real_wm_crop.png")
REAL_HP = Path("tmp/outputs/ish/real_hippocampus_crop.png")
OUT = Path("tmp/outputs/ish/compare.png")


def _ctx_uniform_gm(h: int, w: int, density_value: float = 0.85) -> TransformContext:
    gm = np.ones((h, w), dtype=bool)
    return TransformContext(
        modality="ish",
        annotation_slice=None,
        density_map=np.full((h, w), density_value, dtype=np.float32),
        tissue_mask=gm,
        pixel_size_um=10.0,
        tissue_class_masks={
            "gray_matter": gm,
            "white_matter": np.zeros((h, w), dtype=bool),
            "ventricle": np.zeros((h, w), dtype=bool),
            "tissue": gm,
            "background": np.zeros((h, w), dtype=bool),
        },
    )


def _make_canvas(h: int, w: int, cream: tuple[float, float, float], rng) -> np.ndarray:
    canvas = np.empty((h, w, 3), dtype=np.float32)
    cream_arr = np.array(cream, dtype=np.float32)
    canvas[:] = cream_arr
    canvas += rng.uniform(-0.02, 0.02, canvas.shape).astype(np.float32)
    return np.clip(canvas, 0.0, 1.0)


def render_gm_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    """Light cortical-region rendering — minimal expression."""
    rng = np.random.default_rng(seed)
    cream = (0.86, 0.82, 0.92)  # lavender-cream
    canvas = _make_canvas(h, w, cream, rng)
    transform = ISHExpressionPuncta(
        p=1.0,
        density_range_per_mm2=(200.0, 500.0),
        sigma_range_scale=(0.4, 0.8),
        intensity_range=(0.45, 0.80),
        wash_strength_range=(0.02, 0.05),
        cream=cream,
    )
    return transform(canvas, rng=rng, ctx=_ctx_uniform_gm(h, w, density_value=0.45))


def render_wm_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    """Corpus callosum: very low density expression — pale substrate, sparse specks."""
    rng = np.random.default_rng(seed)
    cream = (0.88, 0.84, 0.93)
    canvas = _make_canvas(h, w, cream, rng)
    transform = ISHExpressionPuncta(
        p=1.0,
        density_range_per_mm2=(80.0, 180.0),
        sigma_range_scale=(0.4, 0.7),
        intensity_range=(0.30, 0.55),
        wash_strength_range=(0.01, 0.03),
        cream=cream,
    )
    return transform(canvas, rng=rng, ctx=_ctx_uniform_gm(h, w, density_value=0.20))


def render_expressed_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    """Densely-expressed region (hippocampus pyramidal-layer-like)."""
    rng = np.random.default_rng(seed)
    cream = (0.86, 0.82, 0.92)
    canvas = _make_canvas(h, w, cream, rng)
    transform = ISHExpressionPuncta(
        p=1.0,
        density_range_per_mm2=(2500.0, 4000.0),
        sigma_range_scale=(0.5, 0.9),
        intensity_range=(0.65, 0.95),
        wash_strength_range=(0.06, 0.14),
        cream=cream,
    )
    return transform(canvas, rng=rng, ctx=_ctx_uniform_gm(h, w, density_value=1.0))


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _label(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    tw = draw.textlength(text, font=font)
    draw.rectangle([(0, 0), (tw + 12, 22)], fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main() -> int:
    real_gm = np.asarray(Image.open(REAL_GM).convert("RGB"))
    real_wm = np.asarray(Image.open(REAL_WM).convert("RGB"))
    real_hp = np.asarray(Image.open(REAL_HP).convert("RGB"))

    proc_gm = _to_uint8(render_gm_patch(real_gm.shape[0], real_gm.shape[1], seed=0))
    proc_wm = _to_uint8(render_wm_patch(real_wm.shape[0], real_wm.shape[1], seed=1))
    proc_hp = _to_uint8(render_expressed_patch(real_hp.shape[0], real_hp.shape[1], seed=2))

    real_gm = _label(real_gm, "REAL GM cortex")
    proc_gm = _label(proc_gm, "PROC GM")
    real_wm = _label(real_wm, "REAL WM corpus callosum")
    proc_wm = _label(proc_wm, "PROC WM")
    real_hp = _label(real_hp, "REAL hippocampus (expressed)")
    proc_hp = _label(proc_hp, "PROC expressed")

    rows = []
    for r, p in [(real_gm, proc_gm), (real_wm, proc_wm), (real_hp, proc_hp)]:
        rows.append(np.concatenate([r, p], axis=1))

    target_w = max(r.shape[1] for r in rows)
    def pad(a):
        if a.shape[1] == target_w:
            return a
        pad_arr = np.full((a.shape[0], target_w - a.shape[1], 3), 32, dtype=np.uint8)
        return np.concatenate([a, pad_arr], axis=1)

    grid = np.concatenate([pad(r) for r in rows], axis=0)
    Image.fromarray(grid).save(OUT)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
