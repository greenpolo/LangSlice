"""Build a square 2x2 version of the microscopy pipeline figure.

U-flow:

  [ 1. Brain Slicing ]                  [ 4. Registration / LangSlice ]
         |                                          ^
         v                                          |
  [ 2. Processing ] -------------> [ 3. Microscope Imaging ]

Same aesthetic as make_pipeline.py: black BG, BLUE accents, WHITE text,
Consolas Bold, { N. Title } label bands, blue arrows. The arrows match the
original vertical pipeline (no text labels — these stages are pure
narrative transitions).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path("C:/LabSoftware/LangSlice")
VIDEO_ASSETS = Path("C:/LabSoftware/langslice-video/assets")
OUT_DIR = REPO / "assets" / "demo_registrations" / "promo"

LOGO_ICON = REPO / "assets" / "langslice_icon_thicker.png"

BG = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (57, 162, 250)
DIM = (165, 175, 190)

FONT_BOLD = "C:/Windows/Fonts/consolab.ttf"
FONT_REG = "C:/Windows/Fonts/consola.ttf"


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


# ---- layout ----
WIDTH = 2200
MARGIN = 70
ARROW_GUTTER_H = 200
ARROW_GUTTER_W = 240
CELL_W = (WIDTH - 2 * MARGIN - ARROW_GUTTER_W) // 2
CELL_H = 720

LABEL_BAND_H = 100
LABEL_FONT = font(54, bold=True)

INNER_PAD = 30  # padding inside each cell for content


@lru_cache(maxsize=64)
def load_cropped(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    bbox = img.split()[3].getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    return img


def aspect(img: Image.Image) -> float:
    return img.size[0] / img.size[1]


def fit_w(img: Image.Image, w: int) -> Image.Image:
    h = int(round(w / aspect(img)))
    return img.resize((w, h), Image.LANCZOS)


def fit_h(img: Image.Image, h: int) -> Image.Image:
    w = int(round(h * aspect(img)))
    return img.resize((w, h), Image.LANCZOS)


def paste_rgba(canvas: Image.Image, img: Image.Image, xy: tuple[int, int]) -> None:
    canvas.paste(img, xy, img if img.mode == "RGBA" else None)


def draw_bracketed_label_at(canvas: Image.Image, cx: int, cy: int,
                            segments: list,
                            font_obj: ImageFont.FreeTypeFont) -> None:
    draw = ImageDraw.Draw(canvas)
    total = sum(font_obj.getlength(s) for s, _ in segments)
    bbox = font_obj.getbbox("Mg")
    text_h = bbox[3] - bbox[1]
    x = cx - total // 2
    y_text = cy - text_h // 2 - bbox[1]
    for text, color in segments:
        draw.text((x, y_text), text, font=font_obj, fill=color)
        x += font_obj.getlength(text)


def draw_panel_title(canvas: Image.Image, cell_x: int, cell_y: int,
                     num: int, title: str) -> None:
    cx = cell_x + CELL_W // 2
    cy = cell_y + LABEL_BAND_H // 2
    segs = [("{ ", BLUE), (f"{num}. ", BLUE), (title, WHITE), (" }", BLUE)]
    draw_bracketed_label_at(canvas, cx, cy, segs, LABEL_FONT)


# ---- arrows ----
SHAFT = 8
HEAD_LEN = 36
HEAD_HW = 22


def draw_down_arrow(canvas: Image.Image, cx: int, y_top: int, y_bot: int) -> None:
    draw = ImageDraw.Draw(canvas)
    shaft_bot = y_bot - HEAD_LEN
    draw.rectangle([cx - SHAFT, y_top, cx + SHAFT, shaft_bot], fill=BLUE)
    draw.polygon([(cx - HEAD_HW, shaft_bot),
                  (cx + HEAD_HW, shaft_bot),
                  (cx, y_bot)], fill=BLUE)


def draw_up_arrow(canvas: Image.Image, cx: int, y_top: int, y_bot: int) -> None:
    draw = ImageDraw.Draw(canvas)
    shaft_top = y_top + HEAD_LEN
    draw.rectangle([cx - SHAFT, shaft_top, cx + SHAFT, y_bot], fill=BLUE)
    draw.polygon([(cx - HEAD_HW, shaft_top),
                  (cx + HEAD_HW, shaft_top),
                  (cx, y_top)], fill=BLUE)


def draw_right_arrow(canvas: Image.Image, x_left: int, x_right: int, cy: int) -> None:
    draw = ImageDraw.Draw(canvas)
    shaft_right = x_right - HEAD_LEN
    draw.rectangle([x_left, cy - SHAFT, shaft_right, cy + SHAFT], fill=BLUE)
    draw.polygon([(shaft_right, cy - HEAD_HW),
                  (shaft_right, cy + HEAD_HW),
                  (x_right, cy)], fill=BLUE)


# ---- cell renderers ----
# Each render_stage_N takes the main canvas and (cell_x, cell_y) and renders
# the stage's content centered vertically in the cell's content region
# (below the title band). All content fits within CELL_W and we cap heights
# so that all 4 stages look roughly balanced.

def cell_content_box(cell_x: int, cell_y: int) -> tuple[int, int, int, int]:
    """Inner content rectangle (excluding title band) for a cell."""
    x0 = cell_x + INNER_PAD
    x1 = cell_x + CELL_W - INNER_PAD
    y0 = cell_y + LABEL_BAND_H + 10
    y1 = cell_y + CELL_H - INNER_PAD
    return x0, y0, x1, y1


def render_stage1(canvas: Image.Image, cell_x: int, cell_y: int) -> None:
    """Brain Slicing: vibratome + agarose brain on top, slice strip below."""
    draw_panel_title(canvas, cell_x, cell_y, 1, "Brain Slicing")
    x0, y0, x1, y1 = cell_content_box(cell_x, cell_y)
    inner_w = x1 - x0
    inner_h = y1 - y0

    top_gap = 24
    strip_gap = 32

    # Top row: two side-by-side images at equal width
    top_cell_w = (inner_w - top_gap) // 2

    # Strip: 7 slice icons across the full inner width
    n = 7
    sg = 10
    strip_cell_w = (inner_w - sg * (n - 1)) // n
    strip_imgs_raw = [load_cropped(VIDEO_ASSETS / f"slice{i+1}bio.png") for i in range(n)]
    strip_imgs = [fit_w(im, strip_cell_w) for im in strip_imgs_raw]
    strip_h = max(im.height for im in strip_imgs)

    # Top images fit to whatever vertical space remains, capped by their aspect-derived height at top_cell_w
    avail_for_top = inner_h - strip_gap - strip_h
    vib = load_cropped(VIDEO_ASSETS / "Vibratome.png")
    agar = load_cropped(VIDEO_ASSETS / "AgaroseMouseBrain.png")
    vib_h_at_w = int(top_cell_w / aspect(vib))
    agar_h_at_w = int(top_cell_w / aspect(agar))
    natural_top_h = max(vib_h_at_w, agar_h_at_w)
    top_h = min(natural_top_h, avail_for_top)
    # Refit if capped by height
    vib_fit = fit_h(vib, top_h)
    agar_fit = fit_h(agar, top_h)

    total_content_h = top_h + strip_gap + strip_h
    block_y0 = y0 + (inner_h - total_content_h) // 2

    # paste top row
    vib_x = x0 + (top_cell_w - vib_fit.width) // 2
    agar_x = x0 + top_cell_w + top_gap + (top_cell_w - agar_fit.width) // 2
    paste_rgba(canvas, vib_fit, (vib_x, block_y0))
    paste_rgba(canvas, agar_fit, (agar_x, block_y0))

    # paste strip
    strip_y = block_y0 + top_h + strip_gap
    sx = x0
    for im in strip_imgs:
        paste_rgba(canvas, im, (sx, strip_y + (strip_h - im.height) // 2))
        sx += strip_cell_w + sg


def render_stage2(canvas: Image.Image, cell_x: int, cell_y: int) -> None:
    """Processing: labels above + plate image."""
    draw_panel_title(canvas, cell_x, cell_y, 2, "Processing")
    x0, y0, x1, y1 = cell_content_box(cell_x, cell_y)
    inner_w = x1 - x0
    inner_h = y1 - y0

    plate_raw = load_cropped(VIDEO_ASSETS / "Slice_Processing_Steps_NOlabels.png")
    plate = fit_w(plate_raw, inner_w)
    # plate is very wide-short (~5:1); add label row above

    label_row_h = 78
    gap = 24
    total_h = label_row_h + gap + plate.height

    block_y0 = y0 + (inner_h - total_h) // 2

    # Plate first (we need its position to align labels)
    plate_x = x0 + (inner_w - plate.width) // 2
    plate_y = block_y0 + label_row_h + gap
    paste_rgba(canvas, plate, (plate_x, plate_y))

    # Labels above each of the 3 wellplate groups
    centers_frac = [0.148, 0.502, 0.851]
    labels = ["Permeabilize", "Primary antibody", "Secondary + DAPI"]
    sublabels = ["& block", "(target protein)", "(fluorophore)"]
    big = font(28, bold=True)
    small = font(22, bold=False)
    draw = ImageDraw.Draw(canvas)
    label_top = block_y0
    for frac, lab, sub in zip(centers_frac, labels, sublabels):
        cx = plate_x + int(frac * plate.width)
        w1 = big.getlength(lab)
        draw.text((cx - w1 // 2, label_top), lab, font=big, fill=WHITE)
        w2 = small.getlength(sub)
        draw.text((cx - w2 // 2, label_top + 38), sub, font=small, fill=DIM)


def render_stage3(canvas: Image.Image, cell_x: int, cell_y: int) -> None:
    """Microscope Imaging: slides + microscope image."""
    draw_panel_title(canvas, cell_x, cell_y, 3, "Microscope Imaging")
    x0, y0, x1, y1 = cell_content_box(cell_x, cell_y)
    inner_w = x1 - x0
    inner_h = y1 - y0

    img = fit_w(load_cropped(VIDEO_ASSETS / "Slides+Microscope.png"), inner_w)
    if img.height > inner_h:
        img = fit_h(img, inner_h)
    px = x0 + (inner_w - img.width) // 2
    py = y0 + (inner_h - img.height) // 2
    paste_rgba(canvas, img, (px, py))


def render_stage4(canvas: Image.Image, cell_x: int, cell_y: int) -> None:
    """Registration / LangSlice: registration icon + brain icon + wordmark."""
    draw_panel_title(canvas, cell_x, cell_y, 4, "Registration")
    x0, y0, x1, y1 = cell_content_box(cell_x, cell_y)
    inner_w = x1 - x0
    inner_h = y1 - y0

    cell_gap = 30
    left_w = (inner_w - cell_gap) // 2
    right_w = left_w

    reg = load_cropped(VIDEO_ASSETS / "Slice_Registration.png")
    brain = load_cropped(LOGO_ICON)
    wordmark_h = 90
    inner_gap = 24

    # Right side stacks brain icon over wordmark
    brain_max_h = inner_h - wordmark_h - inner_gap
    brain_fit_by_w = int(right_w / aspect(brain))
    brain_h = min(brain_fit_by_w, brain_max_h)
    brain_fit = fit_h(brain, brain_h)
    right_block_h = brain_fit.height + inner_gap + wordmark_h

    # Left side: registration icon scaled to match right block height
    reg_h_max = min(inner_h, right_block_h * 1.05)
    reg_w_at_h = int(reg_h_max * aspect(reg))
    if reg_w_at_h > left_w:
        reg_fit = fit_w(reg, left_w)
    else:
        reg_fit = fit_h(reg, int(reg_h_max))

    # Vertical centering
    left_block_y0 = y0 + (inner_h - reg_fit.height) // 2
    right_block_y0 = y0 + (inner_h - right_block_h) // 2

    # Place left
    left_x0 = x0
    paste_rgba(canvas, reg_fit, (left_x0 + (left_w - reg_fit.width) // 2,
                                  left_block_y0))

    # Place right: brain + wordmark
    right_x0 = x0 + left_w + cell_gap
    bx = right_x0 + (right_w - brain_fit.width) // 2
    paste_rgba(canvas, brain_fit, (bx, right_block_y0))

    word_y = right_block_y0 + brain_fit.height + inner_gap
    word_font = font(72, bold=True)
    parts = [("{ ", BLUE), ("Lang", WHITE), ("Slice", BLUE), (" }", BLUE)]
    draw_bracketed_label_at(canvas, right_x0 + right_w // 2,
                             word_y + wordmark_h // 2, parts, word_font)


# ---- build ----

def build(out_path: Path) -> None:
    total_h = 2 * CELL_H + ARROW_GUTTER_H + 2 * MARGIN
    canvas = Image.new("RGB", (WIDTH, total_h), BG)

    left_x = MARGIN
    right_x = MARGIN + CELL_W + ARROW_GUTTER_W
    top_y = MARGIN
    bot_y = MARGIN + CELL_H + ARROW_GUTTER_H

    # Render the 4 cells in U-flow order
    render_stage1(canvas, left_x, top_y)         # TL
    render_stage2(canvas, left_x, bot_y)         # BL
    render_stage3(canvas, right_x, bot_y)        # BR
    render_stage4(canvas, right_x, top_y)        # TR

    # Arrow column centers
    left_col_cx = left_x + CELL_W // 2
    right_col_cx = right_x + CELL_W // 2
    bot_row_cy = bot_y + CELL_H // 2

    gutter_h_y0 = MARGIN + CELL_H
    gutter_h_y1 = bot_y
    gutter_v_x0 = MARGIN + CELL_W
    gutter_v_x1 = right_x

    # A: down arrow stage1 -> stage2 (left side)
    draw_down_arrow(canvas, left_col_cx, gutter_h_y0 + 30, gutter_h_y1 - 30)

    # C: up arrow stage3 -> stage4 (right side)
    draw_up_arrow(canvas, right_col_cx, gutter_h_y0 + 30, gutter_h_y1 - 30)

    # B: right arrow stage2 -> stage3 (bottom)
    draw_right_arrow(canvas, gutter_v_x0 + 30, gutter_v_x1 - 30, bot_row_cy)

    canvas.save(out_path)
    print(f"wrote {out_path} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    build(OUT_DIR / "pipeline_square_v2.png")
