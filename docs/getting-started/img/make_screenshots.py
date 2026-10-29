"""Crop, blur and mark the app screenshots on the Connect pages.

Source: screenshots taken on Linux on 2026-10-29 (Codex in the ChatGPT
desktop app, Claude Desktop's Code tab, Antigravity), named
"Screenshot From 2026-10-29 HH-MM-SS.png". Each picture is cropped to the
part its step needs; any personal text left inside the crop (names, chat
and project titles, installed plugins) is blurred, and the control the step
names is outlined. Usage: python make_screenshots.py SCREENSHOT_DIR OUT_DIR
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

src, out = Path(sys.argv[1]), Path(sys.argv[2])
MARK = (255, 122, 26)

# name: (source time, crop box, boxes to blur, boxes to outline), in source pixels
PICTURES = {
    "codex-new-chat.png": (
        "14-09-34", (0, 44, 913, 768),
        [(60, 128, 288, 768), (8, 722, 46, 762)],
        [(60, 96, 284, 126)],
    ),
    "codex-plugins-add-menu.png": (
        "14-10-02", (0, 44, 913, 310),
        [(60, 205, 288, 310)],
        [(8, 184, 44, 220), (808, 134, 869, 166), (685, 204, 864, 232)],
    ),
    "codex-add-marketplace-form.png": (
        "14-10-11", (157, 177, 757, 591), [], [(177, 287, 737, 322)],
    ),
    "claude-code-tab.png": (
        "14-07-52", (16, 10, 1087, 842),
        [(24, 238, 302, 842), (316, 55, 620, 92), (326, 130, 1066, 278)],
        [(260, 14, 296, 42), (316, 720, 536, 744)],
    ),
    "claude-plugins-add-menu.png": (
        "14-08-20", (16, 10, 1087, 410),
        [(24, 238, 302, 410)],
        [(24, 156, 293, 188), (497, 158, 569, 186), (963, 158, 1045, 186), (844, 198, 1039, 226)],
    ),
    "claude-add-marketplace.png": (
        "14-08-30", (192, 294, 911, 558), [], [(209, 428, 894, 497)],
    ),
    "claude-marketplace-url.png": (
        "14-08-52", (192, 276, 911, 577), [], [(209, 448, 894, 476)],
    ),
    "antigravity-plugin-skill.png": (
        "14-22-23", (452, 374, 1238, 568), [], [(466, 387, 1224, 410)],
    ),
    "antigravity-install-message.png": (
        "14-23-34", (318, 312, 1104, 472), [], [],
    ),
}

for name, (stamp, crop, blurs, marks) in PICTURES.items():
    img = Image.open(src / f"Screenshot From 2026-10-29 {stamp}.png").convert("RGB")
    for box in blurs:
        region = img.crop(box)
        for _ in range(3):
            region = region.filter(ImageFilter.GaussianBlur(14))
        img.paste(region, box[:2])
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in marks:
        draw.rounded_rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), radius=8, outline=MARK, width=3)
    img = img.crop(crop)
    img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        out / name, optimize=True
    )
    print(name, img.size, (out / name).stat().st_size // 1024, "KB")
