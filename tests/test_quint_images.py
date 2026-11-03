"""The image copies the QUINT export names, and their section numbers."""

from pathlib import Path

import numpy as np
from PIL import Image

from langslice.job.imports.matching import match_sections
from langslice.job.quint_images import (
    copy_filename,
    section_numbers,
    write_copies,
)


def test_section_numbers_come_from_the_filenames_when_they_all_carry_one():
    assert section_numbers(["a_s012.tif", "a_s004.tif", "b_s100.png"]) == [
        12,
        4,
        100,
    ]


def test_section_numbers_fall_back_to_the_stack_order():
    # one name without a number, two sharing one, a name with two
    assert section_numbers(["a_s012.tif", "b.tif"]) == [1, 2]
    assert section_numbers(["a_s012.tif", "b_s012.tif"]) == [1, 2]
    assert section_numbers(["a_s01_s02.tif", "b_s03.tif"]) == [1, 2]
    # microscope export names carry numbers, but not as _sNNN: the order counts
    assert section_numbers(
        ["M03_A_07_Overlay.png", "M03_B_07_Overlay.png"]
    ) == [1, 2]


def test_copy_names_carry_the_number_once():
    assert (
        copy_filename("M03_B_07_Overlay.png", 13)
        == "M03_B_07_Overlay_s013.png"
    )
    assert copy_filename("brain_s012.tif", 12) == "brain_s012.png"
    assert copy_filename("brain_s012.tif", 3) == "brain_s012_s003.png"
    assert copy_filename("x.tiff", 1234) == "x_s1234.png"


def test_an_import_finds_the_section_of_a_copy():
    sections = ["M03_B_07_Overlay.png", "M03_A_07_Overlay.png"]
    matched = match_sections(
        ["M03_A_07_Overlay_s002.png", "M03_B_07_Overlay_s001.png"], sections
    )
    assert matched == {
        0: ("M03_A_07_Overlay.png", "export copy"),
        1: ("M03_B_07_Overlay.png", "export copy"),
    }


def test_copies_are_written_and_the_old_ones_removed(tmp_path: Path):
    old = tmp_path / "a_s002.png"
    Image.new("L", (4, 3)).save(old)
    mine = tmp_path / "mine.png"  # a file no export named: kept
    Image.new("L", (4, 3)).save(mine)
    pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    written = write_copies(
        tmp_path, [("a_s001.png", pixels)], ["a_s002.png", "a_s001.png"]
    )
    assert written == [tmp_path / "a_s001.png"]
    with Image.open(written[0]) as picture:
        np.testing.assert_array_equal(np.asarray(picture), pixels)
    assert not old.exists() and mine.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "a_s001.png",
        "mine.png",
    ]
