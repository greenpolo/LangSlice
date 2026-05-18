"""Damage demo grid — 5 modalities × 3 intensity levels.

Grid layout:
    5 rows  (DAPI / Nissl / Brightfield / Fluorescence / ISH)
    3 cols  (clean / medium damage / heavy damage)

All rows use the same atlas slice (AP=5.335) and the same seed per row so
the variation between columns is purely from the damage layer.

Output: tmp/outputs/damage.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "models/langslice-gemma-4/data")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from augmentation.brightfield_pipeline import render_brightfield_section
from augmentation.dapi_pipeline import render_dapi_section
from augmentation.fluorescence_pipeline import render_fluorescence_section
from augmentation.ish_pipeline import render_ish_section
from augmentation.modes import FLUORESCENCE_MODES, ISH_MODES
from augmentation.nissl_pipeline import render_nissl_section
from langslice_harness.atlas.core import (
    get_reference_slice,
    load_atlas,
    position_mm_to_index,
)
from langslice_harness.atlas.space import atlas_space_context, slice_axis_index

AP_MM = 5.335
TARGET_PX_UM = 5.0
SEED = 42
OUT_PATH = Path("tmp/outputs/damage.png")

THUMB_H = 220
THUMB_W = 300
LABEL_H = 28


# ---------------------------------------------------------------------------
# Atlas slice loading
# ---------------------------------------------------------------------------


def load_atlas_slice(atlas, ap_mm: float, target_px_um: float):
    pil = get_reference_slice(atlas, ap_mm).convert("L")
    ctx = atlas_space_context(atlas)
    axis = slice_axis_index(ctx, "coronal")
    idx = position_mm_to_index(atlas, ap_mm)
    ann = np.take(np.asarray(atlas.annotation), idx, axis=axis).astype(np.int32)

    src_um = atlas.resolution[0]
    scale = src_um / target_px_um
    h0, w0 = ann.shape
    h, w = int(round(h0 * scale)), int(round(w0 * scale))
    ref_arr = np.asarray(pil.resize((w, h), Image.LANCZOS), dtype=np.uint8)
    ann_up = np.asarray(
        Image.fromarray(ann.astype(np.int32), "I").resize((w, h), Image.NEAREST),
        dtype=np.int32,
    )
    return ref_arr, ann_up


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _try_font(size: int = 13):
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _to_thumb(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    return np.asarray(img)


def _label_panel(thumb: np.ndarray, label: str) -> np.ndarray:
    bar = np.full((LABEL_H, THUMB_W, 3), 30, dtype=np.uint8)
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    draw.text((4, 5), label, fill=(220, 220, 220), font=_try_font(12))
    return np.concatenate([np.asarray(img), thumb], axis=0)


def _row_label_strip(label: str, cell_h: int) -> np.ndarray:
    strip = np.full((cell_h, 55, 3), 20, dtype=np.uint8)
    txt_img = Image.new("RGB", (cell_h, 32), (20, 20, 20))
    draw = ImageDraw.Draw(txt_img)
    draw.text((4, 8), label, fill=(200, 200, 200), font=_try_font(13))
    txt_rot = txt_img.rotate(90, expand=True)
    img_v = Image.fromarray(strip)
    x_off = max(0, (55 - txt_rot.width) // 2)
    y_off = max(0, (cell_h - txt_rot.height) // 2)
    img_v.paste(txt_rot, (x_off, y_off))
    return np.asarray(img_v)


# ---------------------------------------------------------------------------
# Row renderers
# ---------------------------------------------------------------------------


def render_row_dapi(ref, ann, atlas) -> list[np.ndarray]:
    clean = render_dapi_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, apply_damage=False)
    medium = render_dapi_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, apply_damage=True, damage_intensity="medium")
    heavy = render_dapi_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, apply_damage=True, damage_intensity="heavy")
    return [
        _label_panel(_to_thumb(clean), "DAPI clean"),
        _label_panel(_to_thumb(medium), "DAPI medium damage"),
        _label_panel(_to_thumb(heavy), "DAPI heavy damage"),
    ]


def render_row_nissl(ref, ann, atlas) -> list[np.ndarray]:
    cream = (0.85, 0.78, 0.65)
    clean = render_nissl_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, cream_base=cream, apply_damage=False)
    medium = render_nissl_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, cream_base=cream, apply_damage=True, damage_intensity="medium")
    heavy = render_nissl_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, cream_base=cream, apply_damage=True, damage_intensity="heavy")
    return [
        _label_panel(_to_thumb(clean), "Nissl clean"),
        _label_panel(_to_thumb(medium), "Nissl medium damage"),
        _label_panel(_to_thumb(heavy), "Nissl heavy damage"),
    ]


def render_row_brightfield(ref, ann, atlas) -> list[np.ndarray]:
    clean = render_brightfield_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode="pan_neuronal", counterstain="none", apply_damage=False)
    medium = render_brightfield_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode="pan_neuronal", counterstain="none", apply_damage=True, damage_intensity="medium")
    heavy = render_brightfield_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode="pan_neuronal", counterstain="none", apply_damage=True, damage_intensity="heavy")
    return [
        _label_panel(_to_thumb(clean), "Brightfield clean"),
        _label_panel(_to_thumb(medium), "Brightfield medium damage"),
        _label_panel(_to_thumb(heavy), "Brightfield heavy damage"),
    ]


def render_row_fluorescence(ref, ann, atlas) -> list[np.ndarray]:
    mode_map = {m.name: m for m in FLUORESCENCE_MODES}
    fm = mode_map["dapi_gfp"]
    clean = render_fluorescence_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode=fm, apply_damage=False)
    medium = render_fluorescence_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode=fm, apply_damage=True, damage_intensity="medium")
    heavy = render_fluorescence_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode=fm, apply_damage=True, damage_intensity="heavy")
    return [
        _label_panel(_to_thumb(clean), "Fluor dapi_gfp clean"),
        _label_panel(_to_thumb(medium), "Fluor dapi_gfp medium"),
        _label_panel(_to_thumb(heavy), "Fluor dapi_gfp heavy"),
    ]


def render_row_ish(ref, ann, atlas) -> list[np.ndarray]:
    mode_map = {m.name: m for m in ISH_MODES}
    im = mode_map["allen_style"]
    clean = render_ish_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode=im, apply_damage=False)
    medium = render_ish_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode=im, apply_damage=True, damage_intensity="medium")
    heavy = render_ish_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, mode=im, apply_damage=True, damage_intensity="heavy")
    return [
        _label_panel(_to_thumb(clean), "ISH allen_style clean"),
        _label_panel(_to_thumb(medium), "ISH allen_style medium"),
        _label_panel(_to_thumb(heavy), "ISH allen_style heavy"),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("Loading atlas...", flush=True)
    atlas = load_atlas("allen_mouse_25um")
    ref, ann = load_atlas_slice(atlas, AP_MM, TARGET_PX_UM)
    print(f"Slice shape: ref={ref.shape}  ann={ann.shape}", flush=True)

    row_specs = [
        ("DAPI",        render_row_dapi),
        ("Nissl",       render_row_nissl),
        ("Brightfield", render_row_brightfield),
        ("Fluorescence", render_row_fluorescence),
        ("ISH",         render_row_ish),
    ]

    rows_rendered: list[np.ndarray] = []
    cell_h = LABEL_H + THUMB_H

    for row_label, row_fn in row_specs:
        print(f"Rendering row: {row_label}...", flush=True)
        panels = row_fn(ref, ann, atlas)
        row_img = np.concatenate(panels, axis=1)
        sidebar = _row_label_strip(row_label, cell_h)
        row_with_sidebar = np.concatenate([sidebar, row_img], axis=1)
        rows_rendered.append(row_with_sidebar)
        print(f"  row shape: {row_with_sidebar.shape}", flush=True)

    max_w = max(r.shape[1] for r in rows_rendered)
    padded = []
    for r in rows_rendered:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 20, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)

    grid = np.concatenate(padded, axis=0)

    # Column header
    col_labels = ["Clean (no damage)", "Medium damage", "Heavy damage"]
    col_header_h = 28
    header_bar = np.full((col_header_h, max_w, 3), 15, dtype=np.uint8)
    img_h = Image.fromarray(header_bar)
    draw_h = ImageDraw.Draw(img_h)
    font_h = _try_font(13)
    sidebar_w = 55
    col_w = THUMB_W
    for ci, col_lbl in enumerate(col_labels):
        x = sidebar_w + ci * col_w + col_w // 2 - 40
        draw_h.text((x, 7), col_lbl, fill=(200, 200, 200), font=font_h)
    header_arr = np.asarray(img_h)

    final = np.concatenate([header_arr, grid], axis=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}  ({final.shape[1]}x{final.shape[0]} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
