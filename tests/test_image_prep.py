"""Checks for VLM image preparation."""

from PIL import Image

from langslice_harness.image_prep import prepare_image_for_vlm


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
