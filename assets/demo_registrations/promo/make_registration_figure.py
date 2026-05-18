"""Build the registration-pipeline figure for the LangSlice submission.

Vertical 4-stage flow showing how a slice gets registered:

  1. Inputs: target slice + atlas colored regions
        |  Nano Banana
        v
  2. Atlas warped onto slice (Nano Banana output)
        |  Elastix
        v
  3. Extract deformation calculation (B-spline control-point field)
        |  Apply deformation to atlas coordinates
        v
  4. Fully registered slice (atlas borders overlaid on slice)

Matches the pipeline/promo aesthetic: black BG, BLUE accents, WHITE text,
Consolas Bold font, { N. Title } label bands, blue arrows.
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
# Match the combined-promo aesthetic: borders on the raw colored slice
# (not the CLAHE'd grayscale one).
FINAL_OVERLAY_PATH = ROOT / "slice_19_warped_on_color.png"


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD, size=size)


# ---- layout constants ----
WIDTH = 1400
MARGIN_X = 80
INNER_W = WIDTH - 2 * MARGIN_X
GAP = 40               # vertical gap between elements within a stage
STAGE_GAP = 36         # extra gap above/below each stage
ARROW_H = 150          # space the arrow + label occupy between stages
LABEL_BAND_H = 100     # numbered title band
SUBCAP_H = 50          # per-image caption row (stage 1 only)

LABEL_FONT = font(54, bold=True)
SUBCAP_FONT = font(34, bold=True)
ARROW_LABEL_FONT = font(40, bold=True)


@lru_cache(maxsize=32)
def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


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
    bw, bh = x1 - x0, y1 - y0
    fit = fit_to_box(img, bw, bh)
    px = x0 + (bw - fit.width) // 2
    py = y0 + (bh - fit.height) // 2
    canvas.paste(fit, (px, py))


def draw_bracketed_label(canvas: Image.Image, y: int, segments: list,
                         font_obj: ImageFont.FreeTypeFont,
                         band_h: int) -> None:
    """Draw `{ ... }` style label centered horizontally at vertical band [y, y+band_h]."""
    draw = ImageDraw.Draw(canvas)
    total = sum(font_obj.getlength(s) for s, _ in segments)
    x = (WIDTH - total) // 2
    bbox = font_obj.getbbox("Mg")
    text_h = bbox[3] - bbox[1]
    y_text = y + (band_h - text_h) // 2 - bbox[1]
    for text, color in segments:
        draw.text((x, y_text), text, font=font_obj, fill=color)
        x += font_obj.getlength(text)


def draw_numbered_band(canvas: Image.Image, y: int, num: int, title: str) -> None:
    segs = [("{ ", BLUE), (f"{num}. ", BLUE), (title, WHITE), (" }", BLUE)]
    draw_bracketed_label(canvas, y, segs, LABEL_FONT, LABEL_BAND_H)


def draw_subcap(canvas: Image.Image, cx: int, y: int, text: str,
                color=WHITE) -> None:
    """Centered single-line caption for sub-images in stage 1."""
    draw = ImageDraw.Draw(canvas)
    w = SUBCAP_FONT.getlength(text)
    bbox = SUBCAP_FONT.getbbox("Mg")
    h = bbox[3] - bbox[1]
    y_text = y + (SUBCAP_H - h) // 2 - bbox[1]
    draw.text((cx - w // 2, y_text), text, font=SUBCAP_FONT, fill=color)


def draw_arrow_block(canvas: Image.Image, y: int, label: str) -> None:
    """Centered blue down-arrow with bracketed text label above it."""
    draw = ImageDraw.Draw(canvas)
    cx = WIDTH // 2

    label_h = 60
    arrow_zone_h = ARROW_H - label_h
    segs = [("{ ", BLUE), (label, WHITE), (" }", BLUE)]
    draw_bracketed_label(canvas, y, segs, ARROW_LABEL_FONT, label_h)

    arrow_y0 = y + label_h
    shaft_top = arrow_y0 + 10
    shaft_bot = arrow_y0 + arrow_zone_h - 36
    draw.rectangle([cx - 7, shaft_top, cx + 7, shaft_bot], fill=BLUE)
    head_h, head_w = 34, 48
    draw.polygon(
        [(cx - head_w, shaft_bot - 2),
         (cx + head_w, shaft_bot - 2),
         (cx, shaft_bot + head_h)],
        fill=BLUE,
    )


# ---- stage layouts ----

def layout_stage1() -> dict:
    """Two images side by side with per-image caption above each."""
    cell_w = (INNER_W - GAP) // 2
    max_h = 500
    h1 = min(max_h, int(cell_w / aspect(SLICE_PATH)))
    h2 = min(max_h, int(cell_w / aspect(ATLAS_COLORED_PATH)))
    img_h = max(h1, h2)
    return {
        "cell_w": cell_w,
        "img_h": img_h,
        "content_h": SUBCAP_H + GAP // 2 + img_h,
    }


def layout_single_image(path: Path, max_h: int = 620) -> dict:
    h = min(max_h, int(INNER_W / aspect(path)))
    w = int(h * aspect(path))
    return {"img_w": w, "img_h": h, "content_h": h}


def render_stage1(canvas: Image.Image, y: int, L: dict) -> None:
    cell_w = L["cell_w"]
    img_h = L["img_h"]
    sub_y = y
    img_y = y + SUBCAP_H + GAP // 2

    left_x0 = MARGIN_X
    right_x0 = MARGIN_X + cell_w + GAP

    draw_subcap(canvas, left_x0 + cell_w // 2, sub_y, "Target slice")
    draw_subcap(canvas, right_x0 + cell_w // 2, sub_y, "Atlas colored regions")

    paste_centered(canvas, load(SLICE_PATH),
                   (left_x0, img_y, left_x0 + cell_w, img_y + img_h))
    paste_centered(canvas, load(ATLAS_COLORED_PATH),
                   (right_x0, img_y, right_x0 + cell_w, img_y + img_h))


def render_single_image(canvas: Image.Image, y: int, L: dict, path: Path) -> None:
    img = load(path)
    paste_centered(canvas, img,
                   (MARGIN_X, y, MARGIN_X + INNER_W, y + L["content_h"]))


# ---- build ----

def build(out_path: Path) -> None:
    stages = [
        ("Inputs", layout_stage1(), render_stage1, None),
        ("Atlas warped onto slice", layout_single_image(GENERATED_PATH),
         render_single_image, GENERATED_PATH),
        ("Extract deformation calculation",
         layout_single_image(DEFORMATION_VIZ_PATH, max_h=640),
         render_single_image, DEFORMATION_VIZ_PATH),
        ("Fully registered slice",
         layout_single_image(FINAL_OVERLAY_PATH, max_h=680),
         render_single_image, FINAL_OVERLAY_PATH),
    ]
    arrow_labels = [
        "Nano Banana",
        "Elastix",
        "Apply deformation to atlas coordinates",
    ]

    # Pre-compute canvas height
    total_h = STAGE_GAP
    for i, (_, L, _, _) in enumerate(stages):
        total_h += LABEL_BAND_H + GAP + L["content_h"] + STAGE_GAP
        if i < len(stages) - 1:
            total_h += ARROW_H + STAGE_GAP

    canvas = Image.new("RGB", (WIDTH, total_h), BG)

    y = STAGE_GAP
    for i, (title, L, render, path) in enumerate(stages):
        draw_numbered_band(canvas, y, i + 1, title)
        y += LABEL_BAND_H + GAP
        if path is None:
            render(canvas, y, L)
        else:
            render(canvas, y, L, path)
        y += L["content_h"] + STAGE_GAP
        if i < len(stages) - 1:
            draw_arrow_block(canvas, y, arrow_labels[i])
            y += ARROW_H + STAGE_GAP

    canvas.save(out_path)
    print(f"wrote {out_path} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    build(ROOT / "registration_figure_v4.png")
