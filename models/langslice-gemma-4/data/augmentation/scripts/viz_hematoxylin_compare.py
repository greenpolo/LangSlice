"""Side-by-side hematoxylin comparison: real H+LFB mouse-brain crops vs procedural.

Top row:
    real GM crop (cortex) | procedural GM patch
Bottom row:
    real WM crop (corpus callosum) | procedural WM patch

The synthetic patches use a uniform GM-only or WM-only mask + uniform high
density to isolate color / sigma / density choices from anatomy.

Real reference: Bmouzon (CC BY-SA 4.0), Wikimedia Commons.
Hematoxylin & LFB stained mouse brain coronal section.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from augmentation.transforms.base import TransformContext
from augmentation.transforms.texture import (
    HematoxylinGrayMatterNuclei,
    HematoxylinWhiteMatterNuclei,
)
from PIL import Image

REAL_GM = Path("tmp/outputs/hematoxylin/real_gm_crop.png")
REAL_WM = Path("tmp/outputs/hematoxylin/real_wm_crop.png")
OUT = Path("tmp/outputs/hematoxylin/compare.png")


def _ctx_uniform_class(h: int, w: int, *, kind: str) -> TransformContext:
    if kind not in ("gray_matter", "white_matter"):
        raise ValueError(kind)
    mask = np.ones((h, w), dtype=bool)
    other = np.zeros((h, w), dtype=bool)
    return TransformContext(
        modality="hematoxylin",
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


def _make_canvas(
    h: int,
    w: int,
    substrate: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Pale-cream substrate with subtle per-pixel noise."""
    canvas = np.empty((h, w, 3), dtype=np.float32)
    arr = np.array(substrate, dtype=np.float32)
    canvas[:] = arr
    canvas += rng.uniform(-0.015, 0.015, canvas.shape).astype(np.float32)
    return np.clip(canvas, 0.0, 1.0)


def render_gm_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    substrate = (0.92, 0.88, 0.90)  # classic pale pink-cream
    canvas = _make_canvas(h, w, substrate, rng)
    transform = HematoxylinGrayMatterNuclei(
        p=1.0,
        density_range_per_mm2=(3200.0, 4200.0),
        sigma_range_scale=(0.5, 0.9),
        intensity_range=(0.55, 0.90),
    )
    return transform(canvas, rng=rng, ctx=_ctx_uniform_class(h, w, kind="gray_matter"))


def render_wm_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    substrate = (0.93, 0.89, 0.91)  # slightly lighter substrate for WM
    canvas = _make_canvas(h, w, substrate, rng)
    transform = HematoxylinWhiteMatterNuclei(
        p=1.0,
        density_range_per_mm2=(250.0, 400.0),
        sigma_range_scale=(0.4, 0.7),
        intensity_range=(0.35, 0.65),
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
    draw.rectangle([(0, 0), (int(tw) + 20, 36)], fill=(0, 0, 0))
    draw.text((10, 6), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize uint8 HWC array to (h, w)."""
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.LANCZOS))


def main() -> int:
    real_gm_raw = np.asarray(Image.open(REAL_GM).convert("RGB"))
    real_wm_raw = np.asarray(Image.open(REAL_WM).convert("RGB"))

    # Normalise both pairs to the same (h, w) = GM crop dims for a clean grid
    panel_h, panel_w = 200, 300

    real_gm = _resize_to(real_gm_raw, panel_h, panel_w)
    real_wm = _resize_to(real_wm_raw, panel_h, panel_w)

    proc_gm = _to_uint8(render_gm_patch(panel_h, panel_w, seed=0))
    proc_wm = _to_uint8(render_wm_patch(panel_h, panel_w, seed=1))

    real_gm = _label(real_gm, "REAL GM (cortex, H+LFB)")
    proc_gm = _label(proc_gm, "PROCEDURAL GM")
    real_wm = _label(real_wm, "REAL WM (corpus callosum)")
    proc_wm = _label(proc_wm, "PROCEDURAL WM")

    gm_row = np.concatenate([real_gm, proc_gm], axis=1)
    wm_row = np.concatenate([real_wm, proc_wm], axis=1)

    grid = np.concatenate([gm_row, wm_row], axis=0)
    Image.fromarray(grid).save(OUT)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
