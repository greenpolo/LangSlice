"""A compressed TIFF opens through a job: tifffile decodes LZW (and the other
TIFF codecs) only with ``imagecodecs``, a core dependency for that reason.
Without it the job was made and its first picture failed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from tests.cli_child import install
from tests.synthetic_stack import ID0, PIXEL_SIZE_UM, atlas_loader, write_sections


def test_an_lzw_tiff_opens_through_a_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import langslice

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    install(atlas_loader(), monkeypatch.setattr)
    drawn = tmp_path / "drawn"
    write_sections(drawn)
    pixels = np.asarray(Image.open(drawn / ID0).convert("L"))
    images = tmp_path / "stack"
    images.mkdir()
    path = images / "s0.tif"
    tifffile.imwrite(path, pixels, compression="lzw", photometric="minisblack")
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].compression == tifffile.COMPRESSION.LZW

    with langslice.create_job(images, positions={"s0.tif": 0.15},
                              pixel_size_um=PIXEL_SIZE_UM) as job:
        reply = job.look(mode="section", sections=["s0.tif"])
        assert reply["status"] == "ok", reply
        picture = np.asarray(reply.images[0].convert("L"))
        assert picture.std() > 1  # the section, not a blank tile
