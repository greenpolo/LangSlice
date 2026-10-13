"""Checks for VLM image preparation."""

import numpy as np
from PIL import Image

from langslice.core.image_prep import (
    prepare_image_for_vlm,
)


def test_prepare_image_for_vlm_downsamples_keeping_the_aspect() -> None:
    big_image = Image.new("RGB", (8000, 4000), (10, 20, 30))
    assert prepare_image_for_vlm(big_image).size == (4096, 2048)

    square = Image.new("RGB", (5000, 5000), (1, 2, 3))
    square_size = prepare_image_for_vlm(square).size
    assert square_size[0] * square_size[1] <= 12_000_000
    assert max(square_size) <= 4096

    small = Image.new("RGB", (1200, 800), (5, 5, 5))
    assert prepare_image_for_vlm(small) is small


# --- framing -------------------------------------------------------------


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
