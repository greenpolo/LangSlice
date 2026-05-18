"""Tests for :class:`single_turn_rl.curriculum.BinDifficultyMap`.

The bin-difficulty map is a tiny ``(plane, ap_bin) -> mean_abs_err_pct``
dict-of-floats shared between the curriculum sampler (reader) and the
AdaRFT callback (writer). Tests cover the four properties the sampler/
callback rely on: empty lookups return ``None``, ``update`` overwrites,
``len`` matches the unique-key count, and ``keys`` returns a snapshot
that the caller can iterate without seeing concurrent writes.
"""

from __future__ import annotations

import pytest
from single_turn_rl.curriculum import BinDifficultyMap


def test_empty_map_returns_none_for_any_key() -> None:
    m = BinDifficultyMap()
    assert m.get("coronal", 0) is None
    assert m.get("sagittal", 19) is None


def test_update_then_get_round_trip() -> None:
    m = BinDifficultyMap()
    m.update("coronal", 5, 0.12)
    assert m.get("coronal", 5) == pytest.approx(0.12)


def test_update_overwrites_existing_key() -> None:
    """Same ``(plane, ap_bin)`` → most recent ``update`` wins (running-mean
    write-back semantics, not append)."""
    m = BinDifficultyMap()
    m.update("coronal", 5, 0.12)
    m.update("coronal", 5, 0.30)
    assert m.get("coronal", 5) == pytest.approx(0.30)


def test_plane_keys_independent() -> None:
    """Same bin number on different planes must not collide."""
    m = BinDifficultyMap()
    m.update("coronal", 5, 0.10)
    m.update("sagittal", 5, 0.90)
    assert m.get("coronal", 5) == pytest.approx(0.10)
    assert m.get("sagittal", 5) == pytest.approx(0.90)


def test_bin_keys_independent() -> None:
    """Different bin numbers on the same plane must not collide."""
    m = BinDifficultyMap()
    m.update("coronal", 0, 0.05)
    m.update("coronal", 19, 0.50)
    assert m.get("coronal", 0) == pytest.approx(0.05)
    assert m.get("coronal", 19) == pytest.approx(0.50)


def test_len_matches_unique_key_count() -> None:
    m = BinDifficultyMap()
    assert len(m) == 0
    m.update("coronal", 5, 0.10)
    assert len(m) == 1
    m.update("coronal", 6, 0.20)
    assert len(m) == 2
    # Re-updating an existing key doesn't grow length.
    m.update("coronal", 5, 0.30)
    assert len(m) == 2


def test_keys_returns_snapshot_not_live_view() -> None:
    """Mutating the returned list must not affect future lookups."""
    m = BinDifficultyMap()
    m.update("coronal", 5, 0.10)
    m.update("sagittal", 8, 0.40)
    keys = m.keys()
    assert set(keys) == {("coronal", 5), ("sagittal", 8)}
    keys.clear()
    # The map still holds both keys.
    assert len(m) == 2
    assert m.get("coronal", 5) == pytest.approx(0.10)


def test_get_with_int_like_ap_bin_coerces() -> None:
    """``get`` should coerce ap_bin via ``int(ap_bin)`` so a float bin index
    (from upstream arithmetic) still hits the right cell."""
    m = BinDifficultyMap()
    m.update("coronal", 7, 0.20)
    # Float that rounds to 7 via int() truncation.
    assert m.get("coronal", int(7.9)) == pytest.approx(0.20)


def test_update_coerces_score_to_float() -> None:
    """``update`` must coerce its score to float so int inputs survive lookup."""
    m = BinDifficultyMap()
    m.update("coronal", 5, 1)  # int input
    val = m.get("coronal", 5)
    assert val == pytest.approx(1.0)
    assert isinstance(val, float)
