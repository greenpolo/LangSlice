"""Build the square-layout registration figure for the LangSlice submission.

2x2 grid with U-shaped flow:

  [ 1. Inputs ]                          [ 4. Fully registered slice ]
       |                                                ^
       v  Nano Banana                                   |  Apply deformation
                                                        |
  [ 2. Atlas warped onto slice ] -- Elastix --> [ 3. Extract deformation ]

Same aesthetic as the vertical version (black BG, BLUE accents, WHITE text,
Consolas Bold, { N. Title } label bands, blue arrows).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (57, 162, 250)
DIM = (165, 175, 190)

FONT_BOLD = "C:/Windows/Fonts/consolab.ttf"

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent.parent
CAND = ROOT.parent / "slice_19_clahe" / "registration" / "candidate-ae9899978a28"

SLICE_PATH = REPO / "web-demo" / "public" / "demo_brain" / "slice_19.png"
ATLAS_COLORED_PATH = CAND / "input_colored_regions.png"
GENERATED_PATH = CAND / "generated_segmentation.png"
DEFORMATION_VIZ_PATH = ROOT / "deformation_viz_v4.png"
FINAL_OVERLAY_PATH = ROOT / "slice_19_warped_on_color.png"


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD, size=size)


# ---- layout constants ----
WIDTH = 2200
MARGIN = 70
ARROW_GUTTER_H = 200   # vertical gutter between rows
ARROW_GUTTER_W = 240   # horizontal gutter between cols
CELL_W = (WIDTH - 2 * MARGIN - ARROW_GUTTER_W) // 2
CELL_H = 700           # uniform cell height
LABEL_BAND_H = 90
SUBCAP_H = 45

LABEL_FONT = font(50, bold=True)
SUBCAP_FONT = font(30, bold=True)
ARROW_LABEL_FONT = font(36, bold=True)


@lru_cache(maxsize=32)
def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@lru_cache(maxsize=32)
def load_cropped(path: Path, threshold: int = 12, margin: int = 6) -> Image.Image:
    """Trim near-black padding around the actual content."""
    img = load(path)
    gray = img.convert("L")
    # PIL's getbbox treats >0 as content; threshold by point first.
    mask = gray.point(lambda v: 255 if v > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(img.size[0], x1 + margin)
    y1 = min(img.size[1], y1 + margin)
    return img.crop((x0, y0, x1, y1))


def aspect_cropped(path: Path) -> float:
    im = load_cropped(path)
    return im.size[0] / im.size[1]


def aspect(path: Path) -> float:
    im = load(path)
    return im.size[0] / im.size[1]


def fit_to_box(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    iw, ih = img.size
    scale = min(max_w / iw, max_h / ih)
    return img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)


def paste_centered(canvas: Image.Image, img: Image.Image,
                   box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    fit = fit_to_box(img, x1 - x0, y1 - y0)
    px = x0 + ((x1 - x0) - fit.width) // 2
    py = y0 + ((y1 - y0) - fit.height) // 2
    canvas.paste(fit, (px, py))


def draw_bracketed_label_at(canvas: Image.Image, cx: int, cy: int,
                            segments: list,
                            font_obj: ImageFont.FreeTypeFont) -> None:
    """Draw `{ ... }` style label centered at (cx, cy)."""
    draw = ImageDraw.Draw(canvas)
    total = sum(font_obj.getlength(s) for s, _ in segments)
    bbox = font_obj.getbbox("Mg")
    text_h = bbox[3] - bbox[1]
    x = cx - total // 2
    y_text = cy - text_h // 2 - bbox[1]
    for text, color in segments:
        draw.text((x, y_text), text, font=font_obj, fill=color)
        x += font_obj.getlength(text)


def draw_multi_line_label_at(canvas: Image.Image, cx: int, cy: int,
                             lines: list[list],
                             font_obj: ImageFont.FreeTypeFont,
                             line_gap: int = 6) -> None:
    """Draw a multi-line bracketed label centered vertically at cy."""
    bbox = font_obj.getbbox("Mg")
    line_h = bbox[3] - bbox[1] + line_gap
    total_h = line_h * len(lines)
    y0 = cy - total_h // 2
    for i, segs in enumerate(lines):
        line_cy = y0 + i * line_h + line_h // 2
        draw_bracketed_label_at(canvas, cx, line_cy, segs, font_obj)


def draw_panel_title(canvas: Image.Image, cell_x: int, cell_y: int,
                     num: int, title: str) -> None:
    """Numbered { N. Title } band at the top of a cell."""
    cx = cell_x + CELL_W // 2
    cy = cell_y + LABEL_BAND_H // 2
    segs = [("{ ", BLUE), (f"{num}. ", BLUE), (title, WHITE), (" }", BLUE)]
    draw_bracketed_label_at(canvas, cx, cy, segs, LABEL_FONT)


def draw_subcap(canvas: Image.Image, cx: int, cy: int, text: str) -> None:
    draw_bracketed_label_at(canvas, cx, cy, [(text, WHITE)], SUBCAP_FONT)


# ---- arrow drawing ----

SHAFT = 8
HEAD_LEN = 36
HEAD_HW = 22


def draw_down_arrow(canvas: Image.Image, cx: int, y_top: int, y_bot: int) -> None:
    draw = ImageDraw.Draw(canvas)
    shaft_top = y_top
    shaft_bot = y_bot - HEAD_LEN
    draw.rectangle([cx - SHAFT, shaft_top, cx + SHAFT, shaft_bot], fill=BLUE)
    draw.polygon([(cx - HEAD_HW, shaft_bot),
                  (cx + HEAD_HW, shaft_bot),
                  (cx, y_bot)], fill=BLUE)


def draw_up_arrow(canvas: Image.Image, cx: int, y_top: int, y_bot: int) -> None:
    draw = ImageDraw.Draw(canvas)
    shaft_top = y_top + HEAD_LEN
    shaft_bot = y_bot
    draw.rectangle([cx - SHAFT, shaft_top, cx + SHAFT, shaft_bot], fill=BLUE)
    draw.polygon([(cx - HEAD_HW, shaft_top),
                  (cx + HEAD_HW, shaft_top),
                  (cx, y_top)], fill=BLUE)


def draw_right_arrow(canvas: Image.Image, x_left: int, x_right: int, cy: int) -> None:
    draw = ImageDraw.Draw(canvas)
    shaft_left = x_left
    shaft_right = x_right - HEAD_LEN
    draw.rectangle([shaft_left, cy - SHAFT, shaft_right, cy + SHAFT], fill=BLUE)
    draw.polygon([(shaft_right, cy - HEAD_HW),
                  (shaft_right, cy + HEAD_HW),
                  (x_right, cy)], fill=BLUE)


# ---- cell renderers ----

def render_stage1(canvas: Image.Image, cell_x: int, cell_y: int) -> None:
    """Two sub-images side by side with captions."""
    draw_panel_title(canvas, cell_x, cell_y, 1, "Inputs")

    inner_gap = 30
    sub_w = (CELL_W - inner_gap) // 2

    # Content region below title
    content_y0 = cell_y + LABEL_BAND_H + 20
    content_h = CELL_H - LABEL_BAND_H - 40

    sub_cap_y = content_y0 + SUBCAP_H // 2
    img_y0 = content_y0 + SUBCAP_H + 20
    img_h = content_h - SUBCAP_H - 30

    # Captions
    left_cx = cell_x + sub_w // 2
    right_cx = cell_x + sub_w + inner_gap + sub_w // 2
    draw_subcap(canvas, left_cx, sub_cap_y, "Target slice")
    draw_subcap(canvas, right_cx, sub_cap_y, "Atlas colored regions")

    # Images (auto-crop near-black padding)
    paste_centered(canvas, load_cropped(SLICE_PATH),
                   (cell_x, img_y0, cell_x + sub_w, img_y0 + img_h))
    paste_centered(canvas, load_cropped(ATLAS_COLORED_PATH),
                   (cell_x + sub_w + inner_gap, img_y0,
                    cell_x + sub_w + inner_gap + sub_w, img_y0 + img_h))


def render_single(canvas: Image.Image, cell_x: int, cell_y: int,
                  num: int, title: str, path: Path, crop: bool = True) -> None:
    draw_panel_title(canvas, cell_x, cell_y, num, title)
    content_y0 = cell_y + LABEL_BAND_H + 20
    content_h = CELL_H - LABEL_BAND_H - 40
    img = load_cropped(path) if crop else load(path)
    paste_centered(canvas, img,
                   (cell_x, content_y0,
                    cell_x + CELL_W, content_y0 + content_h))


# ---- build ----

def build(out_path: Path) -> None:
    total_h = 2 * CELL_H + ARROW_GUTTER_H + 2 * MARGIN

    canvas = Image.new("RGB", (WIDTH, total_h), BG)

    # Cell positions
    left_x = MARGIN
    right_x = MARGIN + CELL_W + ARROW_GUTTER_W
    top_y = MARGIN
    bot_y = MARGIN + CELL_H + ARROW_GUTTER_H

    # Render the four cells
    render_stage1(canvas, left_x, top_y)
    render_single(canvas, left_x, bot_y, 2, "Atlas warped onto slice", GENERATED_PATH)
    render_single(canvas, right_x, bot_y, 3, "Extract deformation calculation",
                  DEFORMATION_VIZ_PATH)
    render_single(canvas, right_x, top_y, 4, "Fully registered slice", FINAL_OVERLAY_PATH)

    # ---- arrows ----
    left_col_cx = left_x + CELL_W // 2
    right_col_cx = right_x + CELL_W // 2
    bot_row_cy = bot_y + CELL_H // 2
    gutter_h_y0 = MARGIN + CELL_H
    gutter_h_y1 = bot_y

    # A: down arrow on the left side (stage 1 -> stage 2), label above shaft
    label_h_zone = 70
    a_label_y = gutter_h_y0 + label_h_zone // 2 + 6
    a_shaft_top = gutter_h_y0 + label_h_zone + 10
    a_shaft_bot = gutter_h_y1 - 8
    draw_bracketed_label_at(canvas, left_col_cx, a_label_y,
                            [("{ ", BLUE), ("Nano Banana", WHITE), (" }", BLUE)],
                            ARROW_LABEL_FONT)
    draw_down_arrow(canvas, left_col_cx, a_shaft_top, a_shaft_bot)

    # C: up arrow on the right side (stage 3 -> stage 4). Label is two lines.
    c_label_zone = 110
    c_label_y = gutter_h_y1 - c_label_zone // 2 - 6
    c_shaft_top = gutter_h_y0 + 8
    c_shaft_bot = gutter_h_y1 - c_label_zone - 10
    draw_up_arrow(canvas, right_col_cx, c_shaft_top, c_shaft_bot)
    draw_multi_line_label_at(canvas, right_col_cx, c_label_y, [
        [("{ ", BLUE), ("Apply deformation", WHITE)],
        [("to atlas coordinates", WHITE), (" }", BLUE)],
    ], ARROW_LABEL_FONT)

    # B: right arrow between stage 2 and stage 3, label above shaft
    gutter_v_x0 = MARGIN + CELL_W
    gutter_v_x1 = right_x
    b_label_y = bot_row_cy - 38
    b_shaft_y = bot_row_cy + 24
    b_shaft_x0 = gutter_v_x0 + 8
    b_shaft_x1 = gutter_v_x1 - 8
    draw_bracketed_label_at(canvas, (gutter_v_x0 + gutter_v_x1) // 2, b_label_y,
                            [("{ ", BLUE), ("Elastix", WHITE), (" }", BLUE)],
                            ARROW_LABEL_FONT)
    draw_right_arrow(canvas, b_shaft_x0, b_shaft_x1, b_shaft_y)

    canvas.save(out_path)
    print(f"wrote {out_path} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    build(ROOT / "registration_square_v2.png")
