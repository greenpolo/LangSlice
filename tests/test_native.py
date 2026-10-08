"""Zooms read the section from its image file at the file's own resolution
(:mod:`langslice.core.native`): the windowed reader, the orientation
inverses, the look carried by the local fit, and the zoom pictures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from langslice.core.look import LookRequest, look, redraw
from langslice.core.native import (
    fit_look,
    read_region,
    unorient,
    unoriented_fractions,
    unoriented_points,
)
from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace
from tests.deformable_synthetic import SyntheticAtlas

_TURNS = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
          270: Image.Transpose.ROTATE_270}


def _oriented(image: Image.Image, rotation: int, flip: bool) -> Image.Image:
    if rotation:
        image = image.transpose(_TURNS[rotation])
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else image


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("flip", [False, True])
def test_the_orientation_inverses_undo_the_render_turn_and_flip(rotation: int, flip: bool):
    rng = np.random.default_rng(rotation + flip)
    base = (rng.random((30, 50, 3)) * 255).astype(np.uint8)
    turned = np.asarray(_oriented(Image.fromarray(base), rotation, flip))
    rows, cols = turned.shape[:2]
    yy, xx = np.indices((rows, cols))
    ux, uy = unoriented_points(xx, yy, (cols, rows), rotation, flip)
    assert np.array_equal(base[uy.astype(int), ux.astype(int)], turned)
    assert np.array_equal(np.asarray(unorient(Image.fromarray(turned), rotation, flip)), base)
    fx, fy = unoriented_fractions((xx + 0.5) / cols, (yy + 0.5) / rows, rotation, flip)
    width, height = base.shape[1], base.shape[0]
    assert np.array_equal(base[(fy * height).astype(int), (fx * width).astype(int)], turned)


def test_read_region_reads_a_window_of_any_tiff_layout(tmp_path: Path):
    rng = np.random.default_rng(0)
    rgb = (rng.random((900, 1200, 3)) * 255).astype(np.uint8)
    tifffile.imwrite(tmp_path / "tiled.tif", rgb, tile=(256, 256))
    with tifffile.TiffWriter(tmp_path / "pyramid.tif") as writer:
        writer.write(rgb, tile=(256, 256), subifds=1)
        writer.write(rgb[::2, ::2], tile=(256, 256), subfiletype=1)
    pages = (rng.random((3, 400, 500)) * 4000).astype(np.uint16)
    tifffile.imwrite(tmp_path / "pages.tif", pages, photometric="minisblack")
    Image.fromarray(rgb).save(tmp_path / "plain.png")

    box = (100, 200, 400, 360)
    for name in ("tiled.tif", "pyramid.tif", "plain.png"):
        region = read_region(tmp_path / name, box)
        assert region.file_size == (1200, 900) and region.stride == 1, name
        assert np.array_equal(region.pages[0], rgb[200:360, 100:400]), name
    coarse = read_region(tmp_path / "pyramid.tif", box, step=2)
    assert coarse.downsample == 2.0  # the pyramid level, not a stride over the file
    assert np.array_equal(coarse.pages[0], rgb[::2, ::2][100:180, 50:200])
    rx, ry = coarse.to_read(np.array([100.5]), np.array([200.5]))
    assert (float(rx[0]), float(ry[0])) == (0.0, 0.0)
    many = read_region(tmp_path / "pages.tif", (10, 20, 60, 50))
    assert len(many.pages) == 3
    assert all(np.array_equal(got, page[20:50, 10:60])
               for got, page in zip(many.pages, pages, strict=True))


def test_fit_look_carries_a_map_of_the_channels_exactly():
    rng = np.random.default_rng(1)
    guide = rng.random((40, 60, 3)).astype(np.float32)
    mix = np.array([[0.6, 0.1, 0.0], [0.2, 0.7, 0.1], [0.0, 0.1, 0.5]], dtype=np.float32)
    target = 0.1 + guide @ mix
    offset, gain = fit_look(target, guide)
    carried = offset + np.einsum("rck,rckd->rcd", guide, gain)
    assert np.abs(carried - target).max() < 1e-4


# --- the zoom pictures --------------------------------------------------------------------


#: A section file larger than any working copy, with detail the working copy
#: cannot hold (1-px stripes) and a red marker.
FILE_SIZE = (4800, 3600)
MARKER = (1300, 1000, 1450, 1150)  # x0, y0, x1, y1 in file pixels


def _big_section(folder: Path) -> None:
    width, height = FILE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    disc = ((xx - width / 2) / (0.45 * width)) ** 2 + ((yy - height / 2) / (0.45 * height)) ** 2
    stripes = (xx % 2).astype(np.float32)
    gray = np.where(disc < 1, 90 + 60 * stripes, 5).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    x0, y0, x1, y1 = MARKER
    rgb[y0:y1, x0:x1] = (250, 40, 40)
    tifffile.imwrite(folder / "big.tif", rgb, tile=(512, 512))


@pytest.fixture(scope="module")
def big(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("native") / "images"
    folder.mkdir()
    _big_section(folder)
    return folder


def _stack(folder: Path, *, rotation: int = 0, flip: bool = False
           ) -> tuple[Workspace, StackState]:
    atlas = SyntheticAtlas()
    spec = JobSpec(image_folder=str(folder), model="m", preprocess="none",
                   inputs={"pixel_size_um": 2.0})
    ws = Workspace(spec=spec, image_folder=str(folder), emit=lambda _m: None,
                   atlas_loader=lambda _n: atlas)
    record = SliceState(id="big.tif", index_original=0, index_corrected=0, position_mm=0.15,
                        rotation_deg=rotation, flip=flip,
                        transform={"kind": "interactive", "params": [1, 0, 0, 0, 1, 0]})
    return ws, StackState(image_folder=str(folder), atlas=atlas.atlas_name, plane="coronal",
                          slices=[record])


def _content(picture: object) -> np.ndarray:
    return np.asarray(picture.image)[: picture.recipe["shown"][1], : picture.recipe["shown"][0]]  # type: ignore[attr-defined]


def _red(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] > 180) & (rgb[..., 1] < 110)


def _stripes(rgb: np.ndarray) -> float:
    """How strongly one period dominates a row near the top (the file's
    1-px stripes, magnified), against the row's mean spectrum."""
    row = rgb[3, :, 1].astype(np.float32)
    spectrum = np.abs(np.fft.rfft(row - row.mean()))[1:]
    return float(spectrum.max() / (spectrum.mean() + 1e-9))


@pytest.mark.parametrize("rotation,flip", [(0, False), (90, True)])
def test_a_section_zoom_reads_the_file_at_its_own_resolution(big: Path, rotation: int,
                                                              flip: bool):
    ws, state = _stack(big, rotation=rotation, flip=flip)
    (whole,) = look(ws, state, LookRequest(mode="section", long_edge=512))
    shown = _content(whole)
    ys, xs = np.nonzero(_red(shown))
    cx, cy = xs.mean() / shown.shape[1], ys.mean() / shown.shape[0]
    zoomed = redraw(whole.recipe, (cx - 0.03, cy - 0.03, cx + 0.03, cy + 0.03), ws, state)
    pixels = _content(zoomed)
    assert max(pixels.shape[:2]) == 512
    assert "from the image file" in zoomed.caption
    # The marker is in the middle of the zoom, as large as the file has it.
    zy, zx = np.nonzero(_red(pixels))
    assert abs(zx.mean() / pixels.shape[1] - 0.5) < 0.06
    assert abs(zy.mean() / pixels.shape[0] - 0.5) < 0.06
    # The file's 1-px stripes, which the working copy cannot hold, are there.
    across = np.rot90(pixels) if rotation % 180 else pixels
    assert _stripes(across) > 8.0


def test_an_overlay_zoom_draws_the_file_under_the_atlas_borders(big: Path):
    ws, state = _stack(big)
    (whole,) = look(ws, state, LookRequest(mode="overlay", long_edge=512))
    zoomed = redraw(whole.recipe, (0.35, 0.35, 0.45, 0.45), ws, state)
    pixels = _content(zoomed)
    assert max(pixels.shape[:2]) == 512
    assert zoomed.caption.endswith("from the image file")
    assert _stripes(pixels) > 8.0


def test_a_file_that_cannot_be_read_leaves_the_working_copy(big: Path, tmp_path: Path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "big.tif").write_bytes((big / "big.tif").read_bytes())
    ws, state = _stack(folder)
    (whole,) = look(ws, state, LookRequest(mode="section", long_edge=512))
    window = (0.4, 0.4, 0.46, 0.46)
    read = _content(redraw(whole.recipe, window, ws, state))
    (folder / "big.tif").unlink()  # the working copy is held; the file is gone
    zoomed = redraw(whole.recipe, window, ws, state)
    assert "from the image file" not in zoomed.caption
    held = _content(zoomed)
    assert max(held.shape[:2]) <= 512
    # The stripes' period, as the file shows it, is (nearly) absent from the
    # working copy's zoom: it holds at most its moire.
    file_row = np.abs(np.fft.rfft(read[3, :, 1] - read[3, :, 1].mean()))[1:]
    peak = int(np.argmax(file_row))
    held_row = np.abs(np.fft.rfft(np.asarray(Image.fromarray(held).resize(
        (read.shape[1], read.shape[0])))[3, :, 1].astype(np.float32)))[1:]
    assert held_row[max(0, peak - 1):peak + 2].max() < 0.25 * file_row[peak]
