"""The light look's pictures: a section on paper (``hosts/gui/paper.py``)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from langslice.hosts.gui.calm import is_brightfield
from langslice.hosts.gui.drawing import Style
from langslice.hosts.gui.paper import LINE, PAPER, paper
from langslice.hosts.gui.server import _style


def _slide() -> Image.Image:
    """A bright slide with a dark stained block, black ground around it."""
    pixels = np.zeros((120, 160, 3), np.uint8)
    pixels[10:110, 10:150] = (236, 232, 226)
    pixels[40:80, 50:110] = (90, 60, 120)
    return Image.fromarray(pixels, "RGB")


def _fluorescence() -> Image.Image:
    """A dark slide with a bright textured block."""
    rng = np.random.default_rng(0)
    pixels = np.zeros((120, 160), np.uint8)
    pixels[30:90, 40:120] = rng.integers(150, 250, (60, 80))
    pixels[30:36] //= 3
    pixels[84:90] //= 3
    pixels[:, 40:46] //= 3
    pixels[:, 114:120] //= 3
    return Image.fromarray(pixels, "L").convert("RGB")


def test_a_bright_slide_is_brightfield_and_a_dark_one_is_not() -> None:
    assert is_brightfield(_slide())
    assert not is_brightfield(_fluorescence())
    assert not is_brightfield(Image.new("RGB", (8, 8)))


def test_a_brightfield_slide_and_its_ground_become_paper() -> None:
    drawn = np.asarray(paper(_slide())).astype(int)
    assert np.abs(drawn[2, 2] - PAPER).max() <= 2
    assert np.abs(drawn[20, 20] - PAPER).max() <= 2
    stain = drawn[60, 80]
    assert stain.max() < 150
    assert stain[2] > stain[1]


def test_the_borders_are_ink_over_the_section() -> None:
    bare = _slide()
    pixels = np.asarray(bare).copy()
    pixels[60, 20:140] = (255, 255, 255)
    pixels[25, 20:140] = (20, 20, 20)
    lined = Image.fromarray(pixels, "RGB")
    drawn = np.asarray(paper(bare, lined=lined)).astype(int)
    plain = np.asarray(paper(bare)).astype(int)
    on_slide = drawn[25, 30]
    assert np.abs(on_slide - LINE).sum() < np.abs(plain[25, 30] - LINE).sum()
    assert (drawn[100, 30] == plain[100, 30]).all()


def test_a_darkfield_stain_is_ink_on_paper() -> None:
    drawn = np.asarray(paper(_fluorescence(), brightfield=False)).astype(int)
    assert np.abs(drawn[5, 5] - PAPER).max() <= 2
    assert drawn[30:90, 40:120].mean() < drawn[:20].mean() - 40


def test_the_style_a_picture_is_asked_in() -> None:
    assert _style({"style": ["paper"]}) is Style.PAPER
    assert _style({"style": ["calm"]}) is Style.CALM
    assert _style({}) is Style.RAW
    assert _style({"style": ["sepia"]}) is Style.RAW
