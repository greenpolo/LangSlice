"""Build the LangSlice agent-loop figure: Inspect -> Explore -> Submit."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BG = (0, 0, 0)
BLUE = (57, 162, 250)
WHITE = (255, 255, 255)
DIM = (140, 150, 170)
FONT_PATH = "C:/Windows/Fonts/consolab.ttf"

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent.parent
SLICE_PATH = REPO / "web-demo" / "public" / "demo_brain" / "slice_06.png"
ATLAS_DIR = REPO / "web-demo" / "public" / "atlas" / "allen_mouse_25um" / "sections"
GEMMA_PATH = REPO / "assets" / "gemma_logo_synth.png"


def fit_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    new_w = int(round(w * target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def fit_width(img: Image.Image, target_w: int) -> Image.Image:
    w, h = img.size
    new_h = int(round(h * target_w / w))
    return img.resize((target_w, new_h), Image.LANCZOS)


def composite_rgba_on(rgba: Image.Image, bg_rgb: tuple[int, int, int]) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, bg_rgb + (255,))
    out = Image.alpha_composite(bg, rgba.convert("RGBA"))
    return out.convert("RGB")


def draw_arrow(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int) -> None:
    head_w = 60
    stroke = 12
    draw.line([(x0, y), (x1 - head_w + 4, y)], fill=BLUE, width=stroke)
    draw.polygon([
        (x1, y),
        (x1 - head_w, y - head_w * 0.8),
        (x1 - head_w, y + head_w * 0.8),
    ], fill=BLUE)


GEM_SIZE = 240
GEM_TOP = 40
TOOL_FONT_PT = 30
CONTENT_TOP = GEM_TOP + GEM_SIZE + 40  # vertical anchor for content


def place_gemma(panel: Image.Image, gemma: Image.Image) -> None:
    gem = gemma.copy().resize((GEM_SIZE, GEM_SIZE), Image.LANCZOS)
    panel.paste(gem, ((panel.width - GEM_SIZE) // 2, GEM_TOP))


def make_inspect_panel(panel_w: int, panel_h: int, gemma: Image.Image) -> Image.Image:
    panel = Image.new("RGB", (panel_w, panel_h), BG)
    place_gemma(panel, gemma)

    raw = Image.open(SLICE_PATH).convert("RGB")
    avail_h = panel_h - CONTENT_TOP - 40
    avail_w = panel_w - 60
    slice_img = fit_height(raw, avail_h)
    if slice_img.width > avail_w:
        slice_img = fit_width(slice_img, avail_w)
    slice_x = (panel_w - slice_img.width) // 2
    slice_y = CONTENT_TOP + (avail_h - slice_img.height) // 2
    panel.paste(slice_img, (slice_x, slice_y))
    return panel


def make_explore_panel(panel_w: int, panel_h: int, gemma: Image.Image) -> Image.Image:
    panel = Image.new("RGB", (panel_w, panel_h), BG)
    pdraw = ImageDraw.Draw(panel)
    place_gemma(panel, gemma)

    # 3 atlas sections at varied AP. Section index 156 ~= AP 3.9 mm (slice_06 target).
    samples = [(80, "AP 2.0"), (156, "AP 3.9"), (240, "AP 6.0")]
    atlas_imgs = [
        composite_rgba_on(Image.open(ATLAS_DIR / f"{i:03d}.png"), BG)
        for i, _ in samples
    ]

    # Tool-name caption sits between gemma and thumbs.
    cap_font = ImageFont.truetype(FONT_PATH, 38)
    cap_y = CONTENT_TOP - 10
    pdraw.text((panel_w // 2, cap_y), "fetch_atlas(...)",
               font=cap_font, fill=DIM, anchor="mm")

    thumbs_top = CONTENT_TOP + 50
    avail_h = panel_h - thumbs_top - 80  # leave room for AP labels
    avail_w = panel_w - 80
    gap = 36
    thumb_h = avail_h
    thumbs = [fit_height(im, thumb_h) for im in atlas_imgs]
    total_w = sum(t.width for t in thumbs) + gap * (len(thumbs) - 1)
    if total_w > avail_w:
        scale = avail_w / total_w
        thumb_h = int(thumb_h * scale)
        thumbs = [fit_height(im, thumb_h) for im in atlas_imgs]
        total_w = sum(t.width for t in thumbs) + gap * (len(thumbs) - 1)
    x = (panel_w - total_w) // 2
    y = thumbs_top + (avail_h - thumb_h) // 2
    ap_font = ImageFont.truetype(FONT_PATH, 34)
    for i, (thumb, (_, ap_label)) in enumerate(zip(thumbs, samples)):
        is_chosen = i == len(thumbs) // 2
        panel.paste(thumb, (x, y))
        if is_chosen:
            pad = 10
            pdraw.rectangle(
                [x - pad, y - pad, x + thumb.width + pad - 1, y + thumb.height + pad - 1],
                outline=BLUE, width=5,
            )
        cap_color = BLUE if is_chosen else DIM
        pdraw.text((x + thumb.width // 2, y + thumb.height + 34),
                   ap_label, font=ap_font, fill=cap_color, anchor="mm")
        x += thumb.width + gap

    return panel


def make_submit_panel(panel_w: int, panel_h: int, gemma: Image.Image) -> Image.Image:
    panel = Image.new("RGB", (panel_w, panel_h), BG)
    pdraw = ImageDraw.Draw(panel)
    place_gemma(panel, gemma)

    big_font = ImageFont.truetype(FONT_PATH, 88)
    bracket_font = ImageFont.truetype(FONT_PATH, 110)

    # Vertically center { AP: 3.90 mm } in the content region.
    cy = CONTENT_TOP + (panel_h - CONTENT_TOP - 40) // 2

    label_parts = [
        ("{ ", BLUE, bracket_font),
        ("AP: 3.90 mm", WHITE, big_font),
        (" }", BLUE, bracket_font),
    ]
    total_w = 0
    for text, _, f in label_parts:
        b = pdraw.textbbox((0, 0), text, font=f)
        total_w += b[2] - b[0]
    x = (panel_w - total_w) // 2
    for text, color, f in label_parts:
        pdraw.text((x, cy), text, font=f, fill=color, anchor="lm")
        b = pdraw.textbbox((0, 0), text, font=f)
        x += b[2] - b[0]

    return panel


def build() -> Path:
    gemma = Image.open(GEMMA_PATH).convert("RGB")

    inspect_w = 820
    explore_w = 1340
    submit_w = 820
    panel_h = 860
    arrow_w = 200
    margin = 60
    label_band_h = 90

    panel_widths = [inspect_w, explore_w, submit_w]
    canvas_w = sum(panel_widths) + arrow_w * 2 + margin * 2
    canvas_h = panel_h + label_band_h + margin * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)

    panels = [
        ("Inspect", make_inspect_panel(inspect_w, panel_h, gemma)),
        ("Explore", make_explore_panel(explore_w, panel_h, gemma)),
        ("Submit", make_submit_panel(submit_w, panel_h, gemma)),
    ]

    x = margin
    label_font = ImageFont.truetype(FONT_PATH, 56)
    bracket_label_font = ImageFont.truetype(FONT_PATH, 80)
    cdraw = ImageDraw.Draw(canvas)

    for i, ((name, panel), pw) in enumerate(zip(panels, panel_widths)):
        canvas.paste(panel, (x, margin))
        center_x = x + pw // 2
        label_y = margin + panel_h + label_band_h // 2

        # "{ N. Name }" with chunky bracket like the slice promo.
        text_inner = f"{i + 1}. {name}"
        segments = [
            ("{  ", BLUE, bracket_label_font),
            (text_inner, WHITE, label_font),
            ("  }", BLUE, bracket_label_font),
        ]
        total_w = 0
        for text, _, f in segments:
            b = cdraw.textbbox((0, 0), text, font=f)
            total_w += b[2] - b[0]
        sx = center_x - total_w // 2
        for text, color, f in segments:
            cdraw.text((sx, label_y), text, font=f, fill=color, anchor="lm")
            b = cdraw.textbbox((0, 0), text, font=f)
            sx += b[2] - b[0]

        x += pw

        if i < len(panels) - 1:
            arrow_y = margin + panel_h // 2
            arrow_x0 = x + 30
            arrow_x1 = x + arrow_w - 30
            draw_arrow(cdraw, arrow_x0, arrow_x1, arrow_y)
            x += arrow_w

    out = ROOT / "agent_loop_v4.png"
    canvas.save(out)
    print(f"  {out.name}: {canvas.size[0]}x{canvas.size[1]}")
    return out


if __name__ == "__main__":
    build()
