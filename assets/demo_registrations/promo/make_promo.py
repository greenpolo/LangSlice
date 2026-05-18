"""Build promo composites: input slice -> arrow with label -> registered slice."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BG = (0, 0, 0)
BLUE = (57, 162, 250)
WHITE = (255, 255, 255)
FONT_PATH = "C:/Windows/Fonts/consolab.ttf"  # Consolas Bold

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent.parent.parent / "web-demo" / "public" / "demo_brain"

TRIPLES = [
    ("slice_06", "slice_06_warped_on_color.png", "3.90 mm AP"),
    ("slice_11", "slice_11_generated_on_color.png", "5.65 mm AP"),
    ("slice_19", "slice_19_warped_on_color.png", "7.03 mm AP"),
]


def fit_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    new_w = int(round(w * target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def measure_segments(
    draw: ImageDraw.ImageDraw,
    segments: list[tuple[str, tuple[int, int, int], ImageFont.FreeTypeFont]],
) -> tuple[int, int]:
    """Total pixel width + max height for a sequence of (text, color, font) segments."""
    width = 0
    height = 0
    for text, _, font in segments:
        bbox = draw.textbbox((0, 0), text, font=font)
        width += bbox[2] - bbox[0]
        height = max(height, bbox[3] - bbox[1])
    return width, height


def draw_segments(
    draw: ImageDraw.ImageDraw,
    segments: list[tuple[str, tuple[int, int, int], ImageFont.FreeTypeFont]],
    center_x: int,
    baseline_y: int,
) -> None:
    """Render mixed-font multi-color label centered horizontally at (center_x, baseline_y)."""
    total_w, _ = measure_segments(draw, segments)
    x = center_x - total_w // 2
    for text, color, font in segments:
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((x, baseline_y), text, font=font, fill=color, anchor="lm")
        x += bbox[2] - bbox[0]


def build_one(left_path: Path, right_path: Path, out_path: Path) -> None:
    slice_h = 700
    left = fit_height(Image.open(left_path).convert("RGB"), slice_h)
    right = fit_height(Image.open(right_path).convert("RGB"), slice_h)

    name_font = ImageFont.truetype(FONT_PATH, 50)
    plus_font = ImageFont.truetype(FONT_PATH, 38)

    names = ["Gemma 4", "Nano Banana", "Elastix"]
    line_h = 64
    plus_h = 50
    # Stack: name | + | name | + | name (5 rows)
    stack_h = line_h * 3 + plus_h * 2

    # Find widest text row to size the center column.
    tmp = Image.new("RGB", (10, 10), BG)
    tdraw = ImageDraw.Draw(tmp)
    max_text_w = max(
        tdraw.textbbox((0, 0), name, font=name_font)[2] for name in names
    )

    arrow_len = 280  # shorter arrow
    head_w = 60
    side_pad = 80  # padding around the stack inside the center column
    column_w = max(max_text_w, arrow_len) + side_pad * 2

    margin = 60
    canvas_w = left.width + right.width + column_w + margin * 2
    canvas_h = slice_h + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    canvas.paste(left, (margin, margin))
    canvas.paste(right, (margin + left.width + column_w, margin))

    draw = ImageDraw.Draw(canvas)

    column_cx = margin + left.width + column_w // 2
    column_cy = margin + slice_h // 2

    # Lay out the stack vertically centered around column_cy, with arrow below.
    arrow_top_gap = 50  # space between stack bottom and arrow
    block_h = stack_h + arrow_top_gap + head_w  # rough block height
    stack_top = column_cy - block_h // 2

    y = stack_top
    rows: list[tuple[str, tuple[int, int, int], ImageFont.FreeTypeFont, int]] = []
    for i, name in enumerate(names):
        rows.append((name, WHITE, name_font, line_h))
        if i < len(names) - 1:
            rows.append(("+", BLUE, plus_font, plus_h))

    for text, color, font, row_h in rows:
        draw.text((column_cx, y + row_h // 2), text,
                  font=font, fill=color, anchor="mm")
        y += row_h

    # Short arrow below the stack
    arrow_y = stack_top + stack_h + arrow_top_gap + head_w // 2
    arrow_x0 = column_cx - arrow_len // 2
    arrow_x1 = column_cx + arrow_len // 2
    stroke = 12
    draw.line([(arrow_x0, arrow_y), (arrow_x1 - head_w + 4, arrow_y)],
              fill=BLUE, width=stroke)
    draw.polygon([
        (arrow_x1, arrow_y),
        (arrow_x1 - head_w, arrow_y - head_w * 0.8),
        (arrow_x1 - head_w, arrow_y + head_w * 0.8),
    ], fill=BLUE)

    canvas.save(out_path)
    print(f"  {out_path.name}: {canvas.size[0]}x{canvas.size[1]}  "
          f"(stack {max_text_w}x{stack_h}, column {column_w})")


def main() -> None:
    out_dir = ROOT
    for slice_name, registered_filename, _ in TRIPLES:
        left = SRC / f"{slice_name}.png"
        right = ROOT / registered_filename
        out = out_dir / f"{slice_name}_promo_v6.png"
        build_one(left, right, out)


if __name__ == "__main__":
    main()
