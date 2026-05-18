"""Build the vertical microscopy pipeline figure for the LangSlice submission.

Stages stacked top-to-bottom:
  1. Brain Slicing      - vibratome + agarose + section strip
  2. Processing         - wellplate with 3 aligned labels
  3. Microscope Imaging - slides on a microscope
  4. Registration       - registration icon + LangSlice logo

Uniform GAP=60 between every visual element. Each source is auto-cropped on
its alpha bbox before fitting so images fill their cells instead of floating
in a sea of transparent padding.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path("C:/LabSoftware/LangSlice")
VIDEO_ASSETS = Path("C:/LabSoftware/langslice-video/assets")
OUT_DIR = REPO / "assets" / "demo_registrations" / "promo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGO_ICON = REPO / "assets" / "langslice_icon_thicker.png"

BG = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (57, 162, 250)
DIM = (165, 175, 190)

FONT_BOLD = "C:/Windows/Fonts/consolab.ttf"
FONT_REG = "C:/Windows/Fonts/consola.ttf"


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


WIDTH = 1200
MARGIN_X = 80
INNER_W = WIDTH - 2 * MARGIN_X
GAP = 60
LABEL_H = 110
ARROW_H = 100

LABEL_FONT = font(60, bold=True)


@lru_cache(maxsize=64)
def load_cropped(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    bbox = img.split()[3].getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    return img


def aspect(path: Path) -> float:
    w, h = load_cropped(path).size
    return w / h


def fit_height(path: Path, target_w: int) -> int:
    return int(target_w / aspect(path))


def fit_width(path: Path, target_h: int) -> int:
    return int(target_h * aspect(path))


def paste_fit(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    iw, ih = img.size
    scale = min(bw / iw, bh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    px = x0 + (bw - nw) // 2
    py = y0 + (bh - nh) // 2
    canvas.paste(resized, (px, py), resized if resized.mode == "RGBA" else None)


def draw_label_band(canvas: Image.Image, y: int, number: int, title: str) -> None:
    draw = ImageDraw.Draw(canvas)
    segs = [("{ ", BLUE), (f"{number}. ", BLUE), (title, WHITE), (" }", BLUE)]
    total = sum(LABEL_FONT.getlength(s) for s, _ in segs)
    x = (WIDTH - total) // 2
    bbox = LABEL_FONT.getbbox("Mg")
    text_h = bbox[3] - bbox[1]
    y_text = y + (LABEL_H - text_h) // 2 - bbox[1]
    for s, col in segs:
        draw.text((x, y_text), s, font=LABEL_FONT, fill=col)
        x += LABEL_FONT.getlength(s)


def draw_down_arrow(canvas: Image.Image, y: int) -> None:
    draw = ImageDraw.Draw(canvas)
    cx = WIDTH // 2
    shaft_top = y + 8
    shaft_bot = y + ARROW_H - 30
    draw.rectangle([cx - 5, shaft_top, cx + 5, shaft_bot], fill=BLUE)
    head_h, head_w = 26, 38
    draw.polygon(
        [(cx - head_w, shaft_bot - 2), (cx + head_w, shaft_bot - 2), (cx, shaft_bot + head_h)],
        fill=BLUE,
    )


# ----- precomputed per-stage layouts (so build() can size the canvas first) -----

def layout_stage1() -> dict:
    cell_w = (INNER_W - GAP) // 2
    h_vib = fit_height(VIDEO_ASSETS / "Vibratome.png", cell_w)
    h_agar = fit_height(VIDEO_ASSETS / "AgaroseMouseBrain.png", cell_w)
    top_h = max(h_vib, h_agar)

    n = 7
    strip_inner_gap = 14
    sw = (INNER_W - strip_inner_gap * (n - 1)) // n
    strip_h = max(fit_height(VIDEO_ASSETS / f"slice{i + 1}bio.png", sw) for i in range(n))
    return {"top_h": top_h, "strip_h": strip_h, "cell_w": cell_w,
            "sw": sw, "strip_inner_gap": strip_inner_gap, "n": n,
            "content_h": top_h + GAP + strip_h}


def layout_stage2() -> dict:
    label_row_h = 100  # label row above (caps at 32pt label + 26pt sub + 8 padding)
    plate_h = fit_height(VIDEO_ASSETS / "Slice_Processing_Steps_NOlabels.png", INNER_W)
    return {"label_row_h": label_row_h, "plate_h": plate_h,
            "content_h": label_row_h + GAP + plate_h}


def layout_stage3() -> dict:
    img_h = fit_height(VIDEO_ASSETS / "Slides+Microscope.png", INNER_W)
    return {"img_h": img_h, "content_h": img_h}


def layout_stage4() -> dict:
    cell_w = (INNER_W - GAP) // 2
    # right cell: brain icon stacked above wordmark
    word_band_h = 110
    icon_h_right = cell_w  # logo is square once cropped (or near-square)
    right_total = icon_h_right + GAP + word_band_h
    # left cell: registration icon fills available content height
    left_h = right_total
    return {"cell_w": cell_w, "icon_h_right": icon_h_right,
            "word_band_h": word_band_h, "content_h": right_total,
            "left_h": left_h}


# ----- stage renderers (consume the precomputed layout) -----

def render_stage1(canvas: Image.Image, y: int, L: dict) -> None:
    paste_fit(canvas, load_cropped(VIDEO_ASSETS / "Vibratome.png"),
              (MARGIN_X, y, MARGIN_X + L["cell_w"], y + L["top_h"]))
    paste_fit(canvas, load_cropped(VIDEO_ASSETS / "AgaroseMouseBrain.png"),
              (MARGIN_X + L["cell_w"] + GAP, y, MARGIN_X + 2 * L["cell_w"] + GAP, y + L["top_h"]))
    strip_y = y + L["top_h"] + GAP
    for i in range(L["n"]):
        slc = load_cropped(VIDEO_ASSETS / f"slice{i + 1}bio.png")
        x0 = MARGIN_X + i * (L["sw"] + L["strip_inner_gap"])
        paste_fit(canvas, slc, (x0, strip_y, x0 + L["sw"], strip_y + L["strip_h"]))


def render_stage2(canvas: Image.Image, y: int, L: dict) -> None:
    src = load_cropped(VIDEO_ASSETS / "Slice_Processing_Steps_NOlabels.png")
    new_w = INNER_W
    new_h = L["plate_h"]
    plate = src.resize((new_w, new_h), Image.LANCZOS)
    plate_x = MARGIN_X
    plate_y = y + L["label_row_h"] + GAP
    canvas.paste(plate, (plate_x, plate_y), plate)

    centers_frac = [0.148, 0.502, 0.851]
    labels = ["Permeabilize", "Primary antibody", "Secondary + DAPI"]
    sublabels = ["& block", "(target protein)", "(fluorophore)"]
    big = font(34, bold=True)
    small = font(26, bold=False)
    draw = ImageDraw.Draw(canvas)
    for frac, lab, sub in zip(centers_frac, labels, sublabels):
        cx = plate_x + int(frac * new_w)
        w1 = big.getlength(lab)
        draw.text((cx - w1 // 2, y + 4), lab, font=big, fill=WHITE)
        w2 = small.getlength(sub)
        draw.text((cx - w2 // 2, y + 4 + 44), sub, font=small, fill=DIM)


def render_stage3(canvas: Image.Image, y: int, L: dict) -> None:
    img = load_cropped(VIDEO_ASSETS / "Slides+Microscope.png")
    resized = img.resize((INNER_W, L["img_h"]), Image.LANCZOS)
    canvas.paste(resized, (MARGIN_X, y), resized)


def render_stage4(canvas: Image.Image, y: int, L: dict) -> None:
    cell_w = L["cell_w"]
    # left: registration icon centered in its cell
    paste_fit(canvas, load_cropped(VIDEO_ASSETS / "Slice_Registration.png"),
              (MARGIN_X, y, MARGIN_X + cell_w, y + L["left_h"]))

    right_x0 = MARGIN_X + cell_w + GAP
    # right top: brain icon
    paste_fit(canvas, load_cropped(LOGO_ICON),
              (right_x0, y, right_x0 + cell_w, y + L["icon_h_right"]))
    # right bottom: wordmark in a band with uniform GAP above
    word_y = y + L["icon_h_right"] + GAP
    word_font = font(80, bold=True)
    draw = ImageDraw.Draw(canvas)
    parts = [("{ ", BLUE), ("Lang", WHITE), ("Slice", BLUE), (" }", BLUE)]
    total = sum(word_font.getlength(s) for s, _ in parts)
    cx = right_x0 + cell_w // 2
    x = cx - total // 2
    bbox = word_font.getbbox("Mg")
    text_h = bbox[3] - bbox[1]
    y_text = word_y + (L["word_band_h"] - text_h) // 2 - bbox[1]
    for s, col in parts:
        draw.text((x, y_text), s, font=word_font, fill=col)
        x += word_font.getlength(s)


def build(out_path: Path) -> None:
    layouts = [
        ("Brain Slicing", layout_stage1(), render_stage1),
        ("Processing", layout_stage2(), render_stage2),
        ("Microscope Imaging", layout_stage3(), render_stage3),
        ("Registration", layout_stage4(), render_stage4),
    ]
    n = len(layouts)

    total_h = GAP
    for i, (_, L, _) in enumerate(layouts):
        total_h += LABEL_H + GAP + L["content_h"] + GAP
        if i < n - 1:
            total_h += ARROW_H + GAP

    canvas = Image.new("RGB", (WIDTH, total_h), BG)

    y = GAP
    for i, (title, L, render) in enumerate(layouts):
        draw_label_band(canvas, y, i + 1, title)
        y += LABEL_H + GAP
        render(canvas, y, L)
        y += L["content_h"] + GAP
        if i < n - 1:
            draw_down_arrow(canvas, y)
            y += ARROW_H + GAP

    canvas.save(out_path)
    print(f"wrote {out_path} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    build(OUT_DIR / "pipeline_v7.png")
