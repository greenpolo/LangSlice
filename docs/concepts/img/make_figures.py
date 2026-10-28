"""Compose the Concepts figures from LangSlice's own saved pictures.

Source: section F1_PHAL_s097 of brain F1 (Kondo et al., EBRAINS,
doi:10.25493/2MX9-3XF, CC BY 4.0), registered by LangSlice's agent; pictures
drawn with `langslice-job JOB look` (overlay with warp none, overlay, section,
positioning). Usage: python compose.py JOB_FOLDER OUT_DIR
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

job, out = Path(sys.argv[1]), Path(sys.argv[2])
v = job / "sections/F1_PHAL_s097/views"
linear = Image.open(v / "000244_look_overlay/view.jpg").convert("RGB")
warped = Image.open(v / "000248_look_overlay/view.jpg").convert("RGB")
positioning = Image.open(job / "views/000249_look_positioning/view.jpg").convert("RGB")

FONT = Path("/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf")
def font(size):
    return ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()


def label(img, text, size=54):
    band = round(size * 1.6)
    canvas = Image.new("RGB", (img.width, img.height + band), "white")
    canvas.paste(img, (0, band))
    ImageDraw.Draw(canvas).text((12, 10), text, fill="black", font=font(size))
    return canvas


def row(images, gap=24, height=560):
    scaled = [im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
              for im in images]
    w = sum(i.width for i in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (w, height), "white")
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


# Overlays: 1400x1232 with a caption band at the bottom; crop to the brain.
box = (90, 70, 1330, 1070)
row([
    label(linear.crop(box), "After in-plane alignment"),
    label(warped.crop(box), "After deformation"),
]).save(out / "stages_overlays.jpg", quality=88)

# Positioning: drop the caption line at the bottom.
positioning.crop((0, 0, positioning.width, positioning.height - 30)).save(
    out / "stages_positioning.jpg", quality=88)

# Ventricle close-up, linear vs deformed (the same pixels in both overlays).
vbox = (470, 300, 900, 640)
row([label(linear.crop(vbox), "Before deformation", 28),
     label(warped.crop(vbox), "After deformation", 28)], height=460).save(
    out / "check_ventricles.jpg", quality=90)
print("ok")
