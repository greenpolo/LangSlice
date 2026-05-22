"""Tests for models/training-core/langslice_training/embeddings/cache.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from langslice_training.embeddings.cache import (
    AtlasEmbeddingCache,
    parse_atlas_path,
)


def _save_pair_file(
    out_dir: Path,
    *,
    atlas: str,
    plane: str,
    basename_to_emb: dict[str, torch.Tensor],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{atlas}_{plane}.pt"
    torch.save(
        {
            "atlas": atlas,
            "plane": plane,
            "embeddings": basename_to_emb,
        },
        path,
    )
    return path


def test_parse_atlas_path_canonical():
    parts = parse_atlas_path("atlas/allen_mouse_25um/coronal/9.60mm.jpg")
    assert parts is not None
    assert parts.atlas == "allen_mouse_25um"
    assert parts.plane == "coronal"
    assert parts.basename == "9.60mm.jpg"


def test_parse_atlas_path_png():
    parts = parse_atlas_path("atlas/whs_sd_rat_39um/sagittal/3.50mm.png")
    assert parts is not None
    assert parts.plane == "sagittal"
    assert parts.basename == "3.50mm.png"


def test_parse_atlas_path_with_leading_dirs():
    parts = parse_atlas_path("./data/atlas/allen_mouse_25um/horizontal/0.50mm.jpg")
    assert parts is not None
    assert parts.atlas == "allen_mouse_25um"
    assert parts.plane == "horizontal"
    assert parts.basename == "0.50mm.jpg"


def test_parse_atlas_path_windows_separator():
    parts = parse_atlas_path(r"data\atlas\allen_mouse_25um\coronal\9.60mm.jpg")
    assert parts is not None
    assert parts.atlas == "allen_mouse_25um"
    assert parts.basename == "9.60mm.jpg"


def test_parse_atlas_path_query_returns_none():
    assert parse_atlas_path("queries/some_brain_s1.jpg") is None


def test_parse_atlas_path_unknown_plane_returns_none():
    assert parse_atlas_path("atlas/allen_mouse_25um/diagonal/3.50mm.jpg") is None


def test_parse_atlas_path_missing_extension_returns_none():
    assert parse_atlas_path("atlas/allen_mouse_25um/coronal/3.50mm") is None


def test_cache_empty_dir_returns_none(tmp_path):
    """Lookup against a non-existent dir is allowed and returns None."""
    cache = AtlasEmbeddingCache(tmp_path / "does_not_exist")
    assert cache.pairs() == []
    assert cache.lookup("allen_mouse_25um", "coronal", "5.00mm.jpg") is None
    assert cache.lookup_by_path("atlas/allen_mouse_25um/coronal/5.00mm.jpg") is None


def test_cache_lookup_unknown_pair_returns_none(tmp_path):
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"5.00mm.jpg": torch.randn(1, 64)},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    # Different atlas — not cached.
    assert cache.lookup("whs_sd_rat_39um", "coronal", "5.00mm.jpg") is None
    # Different plane on the same atlas — not cached.
    assert cache.lookup("allen_mouse_25um", "sagittal", "5.00mm.jpg") is None


def test_cache_lookup_unknown_basename_returns_none(tmp_path):
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"5.00mm.jpg": torch.randn(1, 64)},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    # 8.00mm.jpg not in the cache → None.
    assert cache.lookup("allen_mouse_25um", "coronal", "8.00mm.jpg") is None


def test_cache_lookup_known_basename_returns_tensor(tmp_path):
    expected = torch.randn(1, 64)
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"5.00mm.jpg": expected},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    got = cache.lookup("allen_mouse_25um", "coronal", "5.00mm.jpg")
    assert got is not None
    assert torch.equal(got, expected)


def test_cache_lookup_distinguishes_off_grid_basenames(tmp_path):
    """0.62mm.jpg and 0.65mm.jpg get different cache entries (no qkey collision)."""
    a = torch.randn(1, 64)
    b = torch.randn(1, 64)
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"0.62mm.jpg": a, "0.65mm.jpg": b},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    got_a = cache.lookup("allen_mouse_25um", "coronal", "0.62mm.jpg")
    got_b = cache.lookup("allen_mouse_25um", "coronal", "0.65mm.jpg")
    assert got_a is not None and got_b is not None
    assert torch.equal(got_a, a)
    assert torch.equal(got_b, b)
    # Different objects in the cache, so the round-trip preserved both entries.
    assert not torch.equal(got_a, got_b)


def test_cache_lookup_returns_clone_not_view(tmp_path):
    """Mutating the returned tensor must not corrupt the cache."""
    expected = torch.zeros(1, 64)
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"5.00mm.jpg": expected},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    got1 = cache.lookup("allen_mouse_25um", "coronal", "5.00mm.jpg")
    assert got1 is not None
    got1.fill_(7.0)
    got2 = cache.lookup("allen_mouse_25um", "coronal", "5.00mm.jpg")
    assert got2 is not None
    # Cache returned an unmutated clone the second time.
    assert (got2 == 0).all().item()


def test_cache_lookup_by_path(tmp_path):
    expected = torch.randn(1, 64)
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"9.60mm.jpg": expected},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    got = cache.lookup_by_path("atlas/allen_mouse_25um/coronal/9.60mm.jpg")
    assert got is not None
    assert torch.equal(got, expected)


def test_cache_lookup_by_path_non_atlas(tmp_path):
    """Non-atlas paths return None (never trigger a snap)."""
    cache = AtlasEmbeddingCache(tmp_path)
    assert cache.lookup_by_path("queries/some_brain_s1.jpg") is None
    assert cache.lookup_by_path("/abs/path/with/no/atlas/segment.png") is None


def test_cache_pairs_lists_known(tmp_path):
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"5.00mm.jpg": torch.randn(1, 4)},
    )
    _save_pair_file(
        tmp_path,
        atlas="whs_sd_rat_39um",
        plane="sagittal",
        basename_to_emb={"2.00mm.jpg": torch.randn(1, 4)},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    pairs = cache.pairs()
    assert ("allen_mouse_25um", "coronal") in pairs
    assert ("whs_sd_rat_39um", "sagittal") in pairs


def test_cache_ignores_non_pair_files(tmp_path):
    """Files in cache_dir that don't match the <atlas>_<plane>.pt pattern are ignored."""
    (tmp_path / "README.txt").write_text("not a cache file", encoding="utf-8")
    (tmp_path / "garbage.pt").write_bytes(b"\x00\x01")
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"5.00mm.jpg": torch.randn(1, 4)},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    assert cache.pairs() == [("allen_mouse_25um", "coronal")]


def test_cache_has_pair(tmp_path):
    _save_pair_file(
        tmp_path,
        atlas="allen_mouse_25um",
        plane="coronal",
        basename_to_emb={"5.00mm.jpg": torch.randn(1, 4)},
    )
    cache = AtlasEmbeddingCache(tmp_path)
    assert cache.has_pair("allen_mouse_25um", "coronal")
    assert not cache.has_pair("allen_mouse_25um", "sagittal")


def test_cache_malformed_payload_raises(tmp_path):
    """A .pt file matching the filename pattern but missing 'embeddings' key raises."""
    bad = tmp_path / "allen_mouse_25um_coronal.pt"
    torch.save({"atlas": "allen_mouse_25um", "plane": "coronal"}, bad)
    cache = AtlasEmbeddingCache(tmp_path)
    with pytest.raises(RuntimeError, match="malformed"):
        cache.lookup("allen_mouse_25um", "coronal", "5.00mm.jpg")
