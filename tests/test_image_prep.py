"""Checks for image ingest metadata and VLM preparation."""

from pathlib import Path
from typing import cast

from PIL import Image

from langslice.image_prep import (
    detect_pixel_size_um,
    load_image_state,
    prepare_image_for_vlm,
    render_slice_agent_image,
)


def test_image_prep_pipeline(tmp_path: Path) -> None:
    tiff_path = tmp_path / "metadata.tif"
    Image.new("RGB", (128, 64), (255, 255, 255)).save(tiff_path, dpi=(5080, 5080))
    detection = detect_pixel_size_um(str(tiff_path))
    assert detection is not None
    assert detection.source == "tiff_resolution"
    assert abs(detection.pixel_size_um - 5.0) < 0.05

    png_path = tmp_path / "fallback.png"
    Image.new("RGB", (256, 128), (0, 0, 0)).save(png_path)
    state = load_image_state(str(png_path), fallback_pixel_size_um=4.0)
    assert state.pixel_size_source == "manual_existing"
    assert state.metadata_pixel_size_um is None
    assert state.pixel_size_um == 4.0
    assert state.original_size == (256, 128)
    assert state.vlm_size == (256, 128)
    assert state.channel_labels == ("Red", "Green", "Blue")

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


def test_render_slice_agent_image_applies_channel_and_tone_controls() -> None:
    image = Image.new("RGB", (1, 1), (100, 150, 200))

    blue_only = render_slice_agent_image(image, channel_enabled=(False, False, True))
    assert blue_only.getpixel((0, 0)) == (0, 0, 200)

    brighter = render_slice_agent_image(image, exposure_ev=1.0)
    assert brighter.getpixel((0, 0)) == (200, 255, 255)

    dimmer = render_slice_agent_image(image, brightness=-0.2, contrast=0.5)
    dimmer_pixel = cast(tuple[int, int, int], dimmer.getpixel((0, 0)))
    assert dimmer_pixel[0] < 100
