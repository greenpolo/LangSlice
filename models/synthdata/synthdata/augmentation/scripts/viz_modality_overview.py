"""Unified modality overview grid — one panel per (modality × mode/counterstain) cell.

Grid layout (5 rows × 3 columns, same AP slice, same seed=42 for all):

    Row 0 — DAPI (no mode parameter):
        col 0: default render
        col 1: (n/a — padded blank)
        col 2: (n/a — padded blank)

    Row 1 — Nissl (3 cream-base presets):
        col 0: warm-cream (0.85, 0.78, 0.65)  — thionine-like
        col 1: neutral-cream (0.78, 0.78, 0.85)
        col 2: blue-cream (0.74, 0.82, 0.95)  — cresyl-violet (Mouse10-like)

    Row 2 — Brightfield (mode × counterstain combos):
        col 0: pan_neuronal + none
        col 1: pan_neuronal + hematoxylin
        col 2: myelin + none

    Row 3 — Fluorescence (3 representative FLUORESCENCE_MODES):
        col 0: dapi_only
        col 1: dapi_gfp
        col 2: dapi_pv_som_vip

    Row 4 — ISH (3 representative ISH_MODES):
        col 0: allen_style   (none + nbt_bcip)
        col 1: nbt_nfr       (nfr  + nbt_bcip)
        col 2: dab_hematoxylin (hematoxylin + dab)

Optional 6th row (tract tracing):
    col 0: tract_aav_gfp_neurotrace
    col 1–2: blank

Output: tmp/outputs/overview.png
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
from augmentation.modes import (
    FLUORESCENCE_MODES,
    ISH_MODES,
)
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
OUT_PATH = Path("tmp/outputs/overview.png")

# ---- Thumbnail size per panel (pixels) ----
# Full render then thumbnail down so labels are readable at the final composite size.
THUMB_H = 220
THUMB_W = 300
LABEL_H = 28  # height of label bar above each panel


# ---------------------------------------------------------------------------
# Atlas slice loading
# ---------------------------------------------------------------------------


def load_atlas_slice(atlas, ap_mm: float, target_px_um: float):
    """Return (ref_arr uint8, ann_arr int32) upsampled to target_px_um."""
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
# Rendering helpers
# ---------------------------------------------------------------------------


def _to_thumb(arr: np.ndarray) -> np.ndarray:
    """Float32 HWC [0,1] -> uint8 (THUMB_H, THUMB_W, 3)."""
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    return np.asarray(img)


def _blank_panel() -> np.ndarray:
    """Dark-gray blank placeholder for 'n/a' cells."""
    arr = np.full((THUMB_H, THUMB_W, 3), 40, dtype=np.uint8)
    # Add "n/a" text in the center
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    font = _try_font(22)
    draw.text((THUMB_W // 2 - 20, THUMB_H // 2 - 14), "n/a", fill=(130, 130, 130), font=font)
    return np.asarray(img)


def _try_font(size: int = 13) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    for name in ("arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label_panel(thumb: np.ndarray, label: str) -> np.ndarray:
    """Return (LABEL_H + THUMB_H) × THUMB_W panel with a labelled header bar."""
    bar = np.zeros((LABEL_H, THUMB_W, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)  # near-black bar
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    font = _try_font(12)
    draw.text((4, 5), label, fill=(220, 220, 220), font=font)
    bar = np.asarray(img)
    return np.concatenate([bar, thumb], axis=0)


# ---------------------------------------------------------------------------
# Per-row renderers
# ---------------------------------------------------------------------------


def render_row_dapi(ref, ann, atlas) -> list[np.ndarray]:
    """Row 0: DAPI — clean | damaged | blank.

    Column 0: apply_damage=False (clean render for reference).
    Column 1: apply_damage=True  (default; realistic damaged output).
    Column 2: n/a.
    """
    clean = render_dapi_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, apply_damage=False)
    damaged = render_dapi_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, apply_damage=True)
    panels = [
        _label_panel(_to_thumb(clean), "DAPI clean"),
        _label_panel(_to_thumb(damaged), "DAPI + damage"),
        _label_panel(_blank_panel(), "DAPI (n/a)"),
    ]
    return panels


def render_row_nissl(ref, ann, atlas) -> list[np.ndarray]:
    """Row 1: Nissl — warm-cream clean | warm-cream damaged | blue-cream damaged."""
    cream_warm = (0.85, 0.78, 0.65)
    cream_blue = (0.74, 0.82, 0.95)
    clean = render_nissl_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, cream_base=cream_warm, apply_damage=False)
    damaged_warm = render_nissl_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, cream_base=cream_warm, apply_damage=True)
    damaged_blue = render_nissl_section(ref, ann, atlas, seed=SEED, pixel_size_um=TARGET_PX_UM, cream_base=cream_blue, apply_damage=True)
    panels = [
        _label_panel(_to_thumb(clean), "Nissl warm-cream clean"),
        _label_panel(_to_thumb(damaged_warm), "Nissl warm-cream + damage"),
        _label_panel(_to_thumb(damaged_blue), "Nissl blue-cream + damage"),
    ]
    return panels


def render_row_brightfield(ref, ann, atlas) -> list[np.ndarray]:
    """Row 2: Brightfield — 3 mode × counterstain combos (with default damage)."""
    combos = [
        ("pan_neuronal", "none",        "BF pan_neuronal+none"),
        ("pan_neuronal", "hematoxylin", "BF pan_neuronal+hematox"),
        ("myelin",       "none",        "BF myelin+none"),
    ]
    panels = []
    for mode, cs, label in combos:
        out = render_brightfield_section(
            ref, ann, atlas,
            seed=SEED, pixel_size_um=TARGET_PX_UM,
            mode=mode, counterstain=cs,
            apply_damage=True,
        )
        panels.append(_label_panel(_to_thumb(out), label))
    return panels


def render_row_fluorescence(ref, ann, atlas) -> list[np.ndarray]:
    """Row 3: Fluorescence — 3 representative FLUORESCENCE_MODES (with default damage)."""
    target_names = ("dapi_only", "dapi_gfp", "dapi_pv_som_vip")
    mode_map = {m.name: m for m in FLUORESCENCE_MODES}
    panels = []
    for name in target_names:
        fm = mode_map[name]
        out = render_fluorescence_section(
            ref, ann, atlas,
            seed=SEED, pixel_size_um=TARGET_PX_UM,
            mode=fm,
            apply_damage=True,
        )
        panels.append(_label_panel(_to_thumb(out), f"Fluor {name}"))
    return panels


def render_row_ish(ref, ann, atlas) -> list[np.ndarray]:
    """Row 4: ISH — 3 representative ISH_MODES (with default damage)."""
    target_names = ("allen_style", "nbt_nfr", "dab_hematoxylin")
    mode_map = {m.name: m for m in ISH_MODES}
    panels = []
    for name in target_names:
        im = mode_map[name]
        out = render_ish_section(
            ref, ann, atlas,
            seed=SEED, pixel_size_um=TARGET_PX_UM,
            mode=im,
            apply_damage=True,
        )
        panels.append(_label_panel(_to_thumb(out), f"ISH {name}"))
    return panels


def render_row_tract(ref, ann, atlas) -> list[np.ndarray]:
    """Row 5 (optional): Tract tracing — 1 panel + 2 blanks."""
    mode_map = {m.name: m for m in FLUORESCENCE_MODES}
    fm = mode_map["tract_aav_gfp_neurotrace"]
    out = render_fluorescence_section(
        ref, ann, atlas,
        seed=SEED, pixel_size_um=TARGET_PX_UM,
        mode=fm,
        apply_damage=True,
    )
    panels = [
        _label_panel(_to_thumb(out), "Tract aav_gfp+NeuroTrace"),
        _label_panel(_blank_panel(), "Tract (n/a)"),
        _label_panel(_blank_panel(), "Tract (n/a)"),
    ]
    return panels


# ---------------------------------------------------------------------------
# Row-label sidebar
# ---------------------------------------------------------------------------


def _row_label_strip(label: str, cell_h: int) -> np.ndarray:
    """Vertical sidebar strip (cell_h × 55 px) with rotated row label."""
    strip = np.full((cell_h, 55, 3), 20, dtype=np.uint8)
    img_v = Image.fromarray(strip)
    # Draw text on a horizontal image, then rotate 90°.
    txt_img = Image.new("RGB", (cell_h, 32), (20, 20, 20))
    draw = ImageDraw.Draw(txt_img)
    font = _try_font(13)
    draw.text((4, 8), label, fill=(200, 200, 200), font=font)
    txt_rot = txt_img.rotate(90, expand=True)
    # Paste into the strip, centred horizontally.
    x_off = max(0, (55 - txt_rot.width) // 2)
    y_off = max(0, (cell_h - txt_rot.height) // 2)
    img_v.paste(txt_rot, (x_off, y_off))
    return np.asarray(img_v)


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
        ("Tract",       render_row_tract),
    ]

    rows_rendered: list[np.ndarray] = []
    cell_h = LABEL_H + THUMB_H  # height of one panel cell

    for row_label, row_fn in row_specs:
        print(f"Rendering row: {row_label}...", flush=True)
        panels = row_fn(ref, ann, atlas)
        # Concatenate 3 panels side by side.
        row_img = np.concatenate(panels, axis=1)  # (cell_h, 3*THUMB_W, 3)
        # Prepend a row-label sidebar.
        sidebar = _row_label_strip(row_label, cell_h)
        row_with_sidebar = np.concatenate([sidebar, row_img], axis=1)
        rows_rendered.append(row_with_sidebar)
        print(f"  row shape: {row_with_sidebar.shape}", flush=True)

    # Pad all rows to the same width (they should match already, but be safe).
    max_w = max(r.shape[1] for r in rows_rendered)
    padded = []
    for r in rows_rendered:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 20, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)

    grid = np.concatenate(padded, axis=0)

    # Add column-header row at the very top.
    col_labels = ["Mode A (clean/default)", "Mode B (damaged)", "Mode C (damaged)"]
    col_header_h = 28
    header_bar = np.full((col_header_h, max_w, 3), 15, dtype=np.uint8)
    img_h = Image.fromarray(header_bar)
    draw_h = ImageDraw.Draw(img_h)
    font_h = _try_font(13)
    sidebar_w = 55
    col_w = THUMB_W
    for ci, col_lbl in enumerate(col_labels):
        x = sidebar_w + ci * col_w + col_w // 2 - 25
        draw_h.text((x, 7), col_lbl, fill=(200, 200, 200), font=font_h)
    header_arr = np.asarray(img_h)

    final = np.concatenate([header_arr, grid], axis=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}  ({final.shape[1]}x{final.shape[0]} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
