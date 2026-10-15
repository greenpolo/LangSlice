"""The atlas disk cache: built once, read back memory-mapped with the same
values, keyed by atlas version, never fatal."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from langslice.core import oblique
from langslice.core.atlas import disk_cache
from tests.deformable_synthetic import SyntheticAtlas


class VersionedAtlas(SyntheticAtlas):
    """The synthetic atlas with what a BrainGlobe atlas says about itself."""

    def __init__(self, version: str = "3.0") -> None:
        super().__init__()
        self.metadata = {**self.metadata, "version": version}
        self.shape = self.template.shape


def _clear_memory() -> None:
    oblique._filtered_cache.clear()
    oblique._annotation_index_cache.clear()
    oblique._annotation_plane_cache.clear()


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv(disk_cache.ENV, str(tmp_path))
    _clear_memory()
    yield tmp_path
    _clear_memory()


def test_values_identical_with_and_without_the_disk_cache(cache_dir, monkeypatch):
    atlas = VersionedAtlas()
    args = (atlas, 0.12, "coronal", 3.0, -2.0)
    built = oblique.sample_oblique_plane(*args, volume="template", order=3)
    labels = oblique.sample_oblique_annotation(*args)
    names = sorted(p.name for p in cache_dir.rglob("*.npy"))
    assert names == ["annotation_ids.npy", "annotation_index.npy", "template_spline3.npy"]

    _clear_memory()  # a new process: everything read back from disk
    read = oblique.sample_oblique_plane(*args, volume="template", order=3)
    assert isinstance(oblique._filtered_cache[next(iter(oblique._filtered_cache))], np.memmap)
    assert np.array_equal(read, built)
    assert np.array_equal(oblique.sample_oblique_annotation(*args), labels)

    monkeypatch.setenv(disk_cache.ENV, "off")
    _clear_memory()
    assert np.array_equal(oblique.sample_oblique_plane(*args, volume="template", order=3), built)
    assert np.array_equal(oblique.sample_oblique_annotation(*args), labels)


def test_one_folder_per_atlas_version(cache_dir):
    first = disk_cache.atlas_folder(VersionedAtlas("3.0"))
    second = disk_cache.atlas_folder(VersionedAtlas("3.1"))
    assert first is not None and second is not None and first != second
    assert first.is_relative_to(cache_dir)


def test_the_first_write_removes_older_format_folders(cache_dir):
    folder = disk_cache.atlas_folder(VersionedAtlas())
    assert folder is not None
    old = folder.parent / "format0"
    other = folder.parent / "notes"
    old.mkdir(parents=True)
    other.mkdir()
    (old / "x.npy").write_bytes(b"old")
    disk_cache.cached_volume(VersionedAtlas(), "v", lambda: np.zeros(3))
    assert not old.exists() and other.exists() and folder.is_dir()


def test_not_cached_without_a_version_or_past_the_size_limit(cache_dir, monkeypatch):
    assert disk_cache.atlas_folder(SyntheticAtlas()) is None
    monkeypatch.setattr(disk_cache, "MAX_VOXELS", 10)
    assert disk_cache.atlas_folder(VersionedAtlas()) is None


def test_unwritable_or_corrupt_cache_falls_back_to_building(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    monkeypatch.setenv(disk_cache.ENV, str(blocker))
    atlas = VersionedAtlas()
    assert np.array_equal(disk_cache.cached_volume(atlas, "x", lambda: np.arange(4)), np.arange(4))

    monkeypatch.setenv(disk_cache.ENV, str(tmp_path / "cache"))
    folder = disk_cache.atlas_folder(atlas)
    assert folder is not None
    folder.mkdir(parents=True)
    (folder / "x.npy").write_bytes(b"truncated")
    rebuilt = disk_cache.cached_volume(atlas, "x", lambda: np.arange(4))
    assert np.array_equal(rebuilt, np.arange(4))
    assert np.array_equal(np.load(folder / "x.npy"), np.arange(4))  # repaired
    assert not list(folder.glob("*.part"))
