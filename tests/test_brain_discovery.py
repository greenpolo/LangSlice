import os
from pathlib import Path

from langslice_harness.whole_brain.discovery import discover_slices


def test_discover_natural_sort(tmp_path: Path):
    """Filenames with numbers sort naturally, not lexicographically."""
    for name in ["slice_2.tif", "slice_10.tif", "slice_1.tif"]:
        (tmp_path / name).write_bytes(b"fake")
    result = discover_slices(str(tmp_path))
    assert [os.path.basename(r) for r in result] == [
        "slice_1.tif",
        "slice_2.tif",
        "slice_10.tif",
    ]


def test_discover_filters_extensions(tmp_path: Path):
    """Only image files are returned; other files are ignored."""
    (tmp_path / "slice_01.tif").write_bytes(b"fake")
    (tmp_path / "slice_02.png").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_bytes(b"fake")
    (tmp_path / "data.csv").write_bytes(b"fake")
    result = discover_slices(str(tmp_path))
    names = [os.path.basename(r) for r in result]
    assert len(names) == 2
    assert "notes.txt" not in names
    assert "data.csv" not in names


def test_discover_empty_folder(tmp_path: Path):
    """Empty folder returns empty list."""
    result = discover_slices(str(tmp_path))
    assert result == []


def test_discover_mixed_extensions(tmp_path: Path):
    """All supported image extensions are found."""
    for name in ["a.png", "b.jpg", "c.jpeg", "d.tif", "e.tiff"]:
        (tmp_path / name).write_bytes(b"fake")
    result = discover_slices(str(tmp_path))
    assert len(result) == 5
