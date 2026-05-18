"""Side-by-side Nuclear Fast Red comparison: synthetic real crops vs procedural.

Top row:
    real GM crop | procedural GM patch
Bottom row:
    real WM crop | procedural WM patch

The procedural patches use a uniform GM-only or WM-only mask + uniform high
density to isolate color / sigma / density choices from anatomy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from augmentation.counterstain import render_nuclear_fast_red_counterstain
from augmentation.transforms.base import TransformContext
from PIL import Image

REAL_GM = Path("tmp/outputs/nfr/real_gm_crop.png")
REAL_WM = Path("tmp/outputs/nfr/real_wm_crop.png")
OUT = Path("tmp/outputs/nfr/compare.png")


def _ctx_uniform_class(h: int, w: int, *, kind: str) -> tuple[
    TransformContext, dict[str, np.ndarray], np.ndarray, np.ndarray,
]:
    if kind not in ("gray_matter", "white_matter"):
        raise ValueError(kind)
    mask = np.ones((h, w), dtype=bool)
    other = np.zeros((h, w), dtype=bool)
    masks = {
        "gray_matter": mask if kind == "gray_matter" else other,
        "white_matter": mask if kind == "white_matter" else other,
        "ventricle": np.zeros((h, w), dtype=bool),
        "tissue": mask,
        "background": np.zeros((h, w), dtype=bool),
    }
    density = np.full((h, w), 0.85, dtype=np.float32)
    ann = np.zeros((h, w), dtype=np.int32)
    ctx = TransformContext(
        modality="nfr",
        annotation_slice=ann,
        density_map=density,
        tissue_mask=mask,
        pixel_size_um=5.0,
        tissue_class_masks=masks,
    )
    return ctx, masks, density, ann


def render_gm_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ctx, masks, density, ann = _ctx_uniform_class(h, w, kind="gray_matter")
    ref = np.full((h, w), 220, dtype=np.uint8)
    return render_nuclear_fast_red_counterstain(
        ref, ann, object(),
        masks=masks, density_map=density, ctx=ctx, rng=rng, pixel_size_um=5.0,
        substrate_base=(0.93, 0.85, 0.85),
        density_range_per_mm2_gm=(3200.0, 4200.0),
    )


def render_wm_patch(h: int, w: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ctx, masks, density, ann = _ctx_uniform_class(h, w, kind="white_matter")
    ref = np.full((h, w), 220, dtype=np.uint8)
    return render_nuclear_fast_red_counterstain(
        ref, ann, object(),
        masks=masks, density_map=density, ctx=ctx, rng=rng, pixel_size_um=5.0,
        substrate_base=(0.94, 0.86, 0.86),
        density_range_per_mm2_wm=(250.0, 400.0),
    )


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

    panel_h, panel_w = 200, 300

    real_gm = _resize_to(real_gm_raw, panel_h, panel_w)
    real_wm = _resize_to(real_wm_raw, panel_h, panel_w)

    proc_gm = _to_uint8(render_gm_patch(panel_h, panel_w, seed=0))
    proc_wm = _to_uint8(render_wm_patch(panel_h, panel_w, seed=1))

    real_gm = _label(real_gm, "REAL GM (NFR)")
    proc_gm = _label(proc_gm, "PROCEDURAL GM")
    real_wm = _label(real_wm, "REAL WM (NFR)")
    proc_wm = _label(proc_wm, "PROCEDURAL WM")

    gm_row = np.concatenate([real_gm, proc_gm], axis=1)
    wm_row = np.concatenate([real_wm, proc_wm], axis=1)

    grid = np.concatenate([gm_row, wm_row], axis=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(OUT)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
