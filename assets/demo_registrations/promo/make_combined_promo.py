"""Combined registration promo: 3 inputs (left) → 1 arrow → 3 registered (right).

Matches the single-slice promo aesthetic (black bg, blue arrow + plus signs,
white model names in Consolas Bold), but stacks all three Walsh-Lab demo
slices into one figure with a single horizontal arrow in the middle.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (0, 0, 0)
BLUE = (57, 162, 250)
WHITE = (255, 255, 255)
FONT_PATH = "C:/Windows/Fonts/consolab.ttf"

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent.parent.parent / "web-demo" / "public" / "demo_brain"

TRIPLES = [
    ("slice_06.png", "slice_06_warped_on_color.png"),
    ("slice_11.png", "slice_11_generated_on_color.png"),
    ("slice_19.png", "slice_19_warped_on_color.png"),
]


def fit_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    new_w = int(round(w * target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def main() -> None:
    slice_h = 460
    inter_slice_gap = 36

    lefts = [fit_height(Image.open(SRC / l).convert("RGB"), slice_h) for l, _ in TRIPLES]
    rights = [fit_height(Image.open(ROOT / r).convert("RGB"), slice_h) for _, r in TRIPLES]
    left_w = max(im.width for im in lefts)
    right_w = max(im.width for im in rights)

    name_font = ImageFont.truetype(FONT_PATH, 64)
    plus_font = ImageFont.truetype(FONT_PATH, 48)

    names = ["Gemma 4", "Nano Banana", "Elastix"]
    line_h = 80
    plus_h = 60
    stack_h = line_h * 3 + plus_h * 2

    tmp = Image.new("RGB", (10, 10), BG)
    tdraw = ImageDraw.Draw(tmp)
    max_text_w = max(tdraw.textbbox((0, 0), n, font=name_font)[2] for n in names)

    arrow_len = 360
    head_w = 70
    side_pad = 90
    column_w = max(max_text_w, arrow_len) + side_pad * 2

    margin = 80
    col_gap = 40  # slim gutter between the slice columns and the center column

    canvas_w = left_w + right_w + column_w + col_gap * 2 + margin * 2
    column_h = slice_h * 3 + inter_slice_gap * 2
    canvas_h = column_h + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    left_x = margin
    column_x0 = margin + left_w + col_gap
    right_x = column_x0 + column_w + col_gap

    # paste 3 left and 3 right slices, each centered in its row + horiz-centered
    # within its column (in case widths differ slightly across slices)
    for i, (l, r) in enumerate(zip(lefts, rights)):
        y = margin + i * (slice_h + inter_slice_gap)
        canvas.paste(l, (left_x + (left_w - l.width) // 2, y))
        canvas.paste(r, (right_x + (right_w - r.width) // 2, y))

    # vertically center the label-stack + arrow in the middle column
    arrow_top_gap = 56
    block_h = stack_h + arrow_top_gap + head_w
    block_top = margin + (column_h - block_h) // 2
    column_cx = column_x0 + column_w // 2

    y = block_top
    rows = []
    for i, n in enumerate(names):
        rows.append((n, WHITE, name_font, line_h))
        if i < len(names) - 1:
            rows.append(("+", BLUE, plus_font, plus_h))
    for text, color, font, row_h in rows:
        draw.text((column_cx, y + row_h // 2), text,
                  font=font, fill=color, anchor="mm")
        y += row_h

    # single horizontal arrow below the stack
    arrow_y = block_top + stack_h + arrow_top_gap + head_w // 2
    arrow_x0 = column_cx - arrow_len // 2
    arrow_x1 = column_cx + arrow_len // 2
    stroke = 14
    draw.line([(arrow_x0, arrow_y), (arrow_x1 - head_w + 4, arrow_y)],
              fill=BLUE, width=stroke)
    draw.polygon([
        (arrow_x1, arrow_y),
        (arrow_x1 - head_w, arrow_y - head_w * 0.85),
        (arrow_x1 - head_w, arrow_y + head_w * 0.85),
    ], fill=BLUE)

    out = ROOT / "promo_registration_combined.png"
    canvas.save(out)
    print(f"wrote {out} ({canvas_w}x{canvas_h})")

    # also copy into shipping assets dir
    ship = ROOT.parent.parent / "promo_registration_combined.png"
    canvas.save(ship)
    print(f"wrote {ship}")


if __name__ == "__main__":
    main()
