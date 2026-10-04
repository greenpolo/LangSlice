"""Checks for VLM image preparation."""

import cv2
import numpy as np
from PIL import Image

from langslice.core.image_prep import (
    FRAME_MARGIN,
    crop_to_mask,
    crop_to_tissue,
    prepare_image_for_vlm,
)


def _fill(image: Image.Image, *, background: int) -> float:
    """Fraction of pixels that differ from *background*."""
    arr = np.asarray(image.convert("L"), dtype=np.int16)
    return float((np.abs(arr - background) > 20).mean())


def test_prepare_image_for_vlm_downsamples_and_tracks_pixel_size() -> None:
    big_image = Image.new("RGB", (8000, 4000), (10, 20, 30))
    prep = prepare_image_for_vlm(big_image, pixel_size_um=4.0)
    assert prep.output_size == (4096, 2048)
    assert prep.downsampled
    assert prep.effective_pixel_size_um is not None
    assert abs(prep.effective_pixel_size_um - (4.0 * 8000 / 4096)) < 1e-6

    square = Image.new("RGB", (5000, 5000), (1, 2, 3))
    square_prep = prepare_image_for_vlm(square)
    assert square_prep.output_size[0] * square_prep.output_size[1] <= 12_000_000
    assert max(square_prep.output_size) <= 4096

    small = Image.new("RGB", (1200, 800), (5, 5, 5))
    small_prep = prepare_image_for_vlm(small, pixel_size_um=2.5)
    assert small_prep.output_size == (1200, 800)
    assert not small_prep.downsampled
    assert small_prep.effective_pixel_size_um == 2.5


# --- framing -------------------------------------------------------------


def test_crop_to_tissue_lifts_a_small_section_to_a_full_frame() -> None:
    """Dark-on-light: a section adrift in a big scan comes back filling it."""
    canvas = np.full((400, 400, 3), 240, dtype=np.uint8)
    cv2.ellipse(canvas, (150, 220), (60, 40), 0, 0, 360, (30, 30, 30), -1)
    section = Image.fromarray(canvas, mode="RGB")

    framed = crop_to_tissue(section)

    assert _fill(section, background=240) < 0.06
    assert _fill(framed, background=240) > 0.45
    assert framed.size < section.size


def test_crop_to_tissue_keeps_a_dim_interior(monkeypatch) -> None:
    """Light-on-dark fluorescence: a dim core is tissue, not background."""
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.ellipse(canvas, (200, 150), (80, 55), 0, 0, 360, (200, 200, 200), -1)
    cv2.ellipse(canvas, (200, 150), (40, 25), 0, 0, 360, (60, 60, 60), -1)

    framed = crop_to_tissue(Image.fromarray(canvas, mode="RGB"))

    # The whole ellipse survives: 2 * 80 wide plus the margin, not the ring.
    assert framed.width >= 160
    assert framed.height >= 110


def test_crop_to_tissue_ignores_debris_beside_the_section() -> None:
    """A neighbouring fragment must not drag the frame open around both."""
    canvas = np.full((400, 400, 3), 240, dtype=np.uint8)
    cv2.ellipse(canvas, (120, 120), (60, 45), 0, 0, 360, (30, 30, 30), -1)
    cv2.circle(canvas, (360, 370), 12, (30, 30, 30), -1)  # a far-away speck
    slide = Image.fromarray(canvas, mode="RGB")

    framed = crop_to_tissue(slide)

    # The box is the section's, not the section-plus-speck bounding box.
    assert framed.width < 200 and framed.height < 200
    assert _fill(framed, background=240) > 0.45


def test_crop_to_tissue_keeps_both_pieces_of_a_parted_section() -> None:
    """Two bulbs cut apart are one section: the frame holds both."""
    canvas = np.full((400, 400, 3), 240, dtype=np.uint8)
    cv2.ellipse(canvas, (110, 200), (50, 70), 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(canvas, (290, 200), (45, 65), 0, 0, 360, (30, 30, 30), -1)
    slide = Image.fromarray(canvas, mode="RGB")

    framed = crop_to_tissue(slide)

    # Both bulbs span x 60..335: the box is wider than either bulb alone.
    assert framed.width > 250


def test_crop_to_tissue_leaves_a_frame_it_cannot_read() -> None:
    uniform = Image.new("RGB", (100, 80), (120, 120, 120))
    assert crop_to_tissue(uniform).size == uniform.size

    noise = Image.fromarray(
        np.random.default_rng(0).integers(60, 200, (80, 120, 3)).astype(np.uint8),
        mode="RGB",
    )
    assert crop_to_tissue(noise).size == noise.size


def test_crop_to_mask_scales_the_mask_and_adds_the_margin() -> None:
    image = Image.new("RGB", (200, 200), (10, 10, 10))
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True  # a quarter-scale mask: 40..120 px in the image

    framed = crop_to_mask(image, mask)

    expected = 80 + 2 * round(FRAME_MARGIN * 80)
    assert abs(framed.width - expected) <= 2
    assert abs(framed.height - expected) <= 2
    assert crop_to_mask(image, np.zeros((50, 50), dtype=bool)).size == image.size


def test_read_working_image_takes_the_smallest_pyramid_level_large_enough(tmp_path) -> None:
    import tifffile

    from langslice.core.image_prep import read_working_image

    path = tmp_path / "scan.tif"
    base = np.random.default_rng(0).integers(0, 255, (4000, 6000, 3), dtype=np.uint8)
    with tifffile.TiffWriter(path) as tif:
        tif.write(base, subifds=2, tile=(256, 256))
        tif.write(base[::2, ::2], subfiletype=1, tile=(256, 256))  # 3000 px
        tif.write(base[::4, ::4], subfiletype=1, tile=(256, 256))  # 1500 px
    image, file_px_per_px = read_working_image(path, min_edge=1536)
    assert image.mode == "RGB" and image.size == (3000, 2000)
    assert file_px_per_px == 2.0


def test_read_working_image_downsamples_a_plain_file_once(tmp_path) -> None:
    from langslice.core.image_prep import read_working_image

    path = tmp_path / "plain.png"
    Image.fromarray(np.zeros((1000, 5000), dtype=np.uint8)).save(path)
    image, file_px_per_px = read_working_image(path, max_edge=2500)
    assert image.size == (2500, 500) and image.mode == "RGB"
    assert file_px_per_px == 2.0

    small = tmp_path / "small.png"
    Image.fromarray(np.zeros((30, 40, 3), dtype=np.uint8)).save(small)
    assert read_working_image(small)[0].size == (40, 30)
