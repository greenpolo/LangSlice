"""Posterior-wing + hemibrain damage preview grid.

Renders a clean Nissl section, then forces each PosteriorWingDamage and
HemibrainPreparation mode on top of identical baselines so the cuts and
fraying can be visually compared side-by-side.

Output: tmp/outputs/wing_and_hemibrain_damage.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import copy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.nissl_pipeline import render_nissl_section
from augmentation.transforms.base import TransformContext
from augmentation.transforms.damage import HemibrainPreparation, PosteriorWingDamage
from augmentation.transforms.tissue_class import classify_tissue
from langslice_harness.atlas.core import (
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

ATLAS_NAME = "allen_mouse_25um"
POSTERIOR_AP_MM = 9.8
ANTERIOR_AP_MM = 5.5
TARGET_PX_UM = 5.0
SEED = 42
CREAM = (0.85, 0.78, 0.65)
OUT_PATH = Path("tmp/outputs/wing_and_hemibrain_damage.png")
SOLO_DIR = Path("tmp/outputs/wing_damage_solo")

PANEL_W = 720
PANEL_H = 520
LABEL_H = 30


def _try_font(size: int = 13):
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _load_slice(atlas, ap_mm: float):
    pil = get_reference_slice(atlas, ap_mm).convert("L")
    space_ctx = atlas_space_context(atlas)
    axis = slice_axis_index(space_ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)

    src_um = atlas.resolution[0]
    scale = src_um / TARGET_PX_UM
    h0, w0 = ann.shape
    h, w = int(round(h0 * scale)), int(round(w0 * scale))
    ref = np.asarray(pil.resize((w, h), Image.LANCZOS), dtype=np.uint8)
    ann_up = np.asarray(
        Image.fromarray(ann.astype(np.int32), "I").resize((w, h), Image.NEAREST),
        dtype=np.int32,
    )
    return ref, ann_up


def _build_ctx(atlas, ann: np.ndarray, *, plane: str, position_mm: float) -> TransformContext:
    masks = classify_tissue(ann, atlas)
    return TransformContext(
        modality="nissl",
        annotation_slice=ann.copy(),
        density_map=None,
        tissue_mask=masks["tissue"].copy(),
        pixel_size_um=TARGET_PX_UM,
        tissue_class_masks={k: v.copy() for k, v in masks.items()},
        plane=plane,
        position_mm=position_mm,
    )


def _to_panel(arr: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    img = img.resize((PANEL_W, PANEL_H), Image.LANCZOS)
    arr_p = np.asarray(img)

    bar = np.full((LABEL_H, PANEL_W, 3), 25, dtype=np.uint8)
    img_bar = Image.fromarray(bar)
    draw = ImageDraw.Draw(img_bar)
    draw.text((6, 7), label, fill=(230, 230, 230), font=_try_font(13))
    return np.concatenate([np.asarray(img_bar), arr_p], axis=0)


def _row(panels: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(panels, axis=1)


def _pad_to_width(row: np.ndarray, target_w: int) -> np.ndarray:
    if row.shape[1] >= target_w:
        return row
    pad = np.full((row.shape[0], target_w - row.shape[1], 3), 18, dtype=np.uint8)
    return np.concatenate([row, pad], axis=1)


def _force(transform, image: np.ndarray, ctx: TransformContext, seed: int) -> np.ndarray:
    """Apply a damage transform deterministically with a fresh ctx copy.

    The transform mutates ctx in place; we deep-copy so each invocation starts
    from the same baseline ctx.
    """
    return transform(image.copy(), rng=np.random.default_rng(seed), ctx=copy.deepcopy(ctx))


def _save_solo(arr: np.ndarray, name: str) -> None:
    SOLO_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).save(SOLO_DIR / f"{name}.png")


def render_posterior_wing_row(atlas) -> np.ndarray:
    ref, ann = _load_slice(atlas, POSTERIOR_AP_MM)
    print(f"Posterior slice shape: ref={ref.shape}", flush=True)

    baseline = render_nissl_section(
        ref,
        ann,
        atlas,
        seed=SEED,
        pixel_size_um=TARGET_PX_UM,
        cream_base=CREAM,
        apply_damage=False,
        plane="coronal",
        position_mm=POSTERIOR_AP_MM,
    )
    ctx = _build_ctx(atlas, ann, plane="coronal", position_mm=POSTERIOR_AP_MM)
    _save_solo(baseline, "posterior_baseline")

    modes = [
        ("baseline", None),
        ("left_missing", "left_missing"),
        ("right_missing", "right_missing"),
        ("both_missing", "both_missing"),
        ("both_detached", "both_detached"),
        ("left_missing + right_detached", "left_missing_right_detached"),
    ]
    panels = []
    for label, mode in modes:
        if mode is None:
            panels.append(_to_panel(baseline, f"posterior  AP={POSTERIOR_AP_MM}  {label}"))
            continue
        transform = PosteriorWingDamage(
            p=1.0,
            mode=mode,
            posterior_min_position_mm=0.0,
        )
        damaged = _force(transform, baseline, ctx, seed=SEED + 1)
        _save_solo(damaged, f"posterior_{mode}")
        panels.append(_to_panel(damaged, f"posterior  {label}"))
    return _row(panels)


def render_hemibrain_row(atlas) -> np.ndarray:
    ref, ann = _load_slice(atlas, ANTERIOR_AP_MM)
    print(f"Anterior slice shape: ref={ref.shape}", flush=True)

    baseline = render_nissl_section(
        ref,
        ann,
        atlas,
        seed=SEED,
        pixel_size_um=TARGET_PX_UM,
        cream_base=CREAM,
        apply_damage=False,
        plane="coronal",
        position_mm=ANTERIOR_AP_MM,
    )
    ctx = _build_ctx(atlas, ann, plane="coronal", position_mm=ANTERIOR_AP_MM)

    _save_solo(baseline, "hemibrain_baseline")
    panels = [
        _to_panel(baseline, f"hemibrain  AP={ANTERIOR_AP_MM}  baseline"),
    ]
    for keep_side in ("left", "right"):
        transform = HemibrainPreparation(p=1.0, keep_side=keep_side)
        damaged = _force(transform, baseline, ctx, seed=SEED + 2)
        _save_solo(damaged, f"hemibrain_keep_{keep_side}")
        panels.append(_to_panel(damaged, f"hemibrain  keep={keep_side}"))
    return _row(panels)


def main() -> int:
    print("Loading atlas...", flush=True)
    atlas = load_atlas(ATLAS_NAME)

    print("Rendering posterior-wing row...", flush=True)
    posterior = render_posterior_wing_row(atlas)
    print("Rendering hemibrain row...", flush=True)
    hemibrain = render_hemibrain_row(atlas)

    target_w = max(posterior.shape[1], hemibrain.shape[1])
    posterior = _pad_to_width(posterior, target_w)
    hemibrain = _pad_to_width(hemibrain, target_w)
    grid = np.concatenate([posterior, hemibrain], axis=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(OUT_PATH)
    print(f"Saved -> {OUT_PATH}  ({grid.shape[1]}x{grid.shape[0]} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
