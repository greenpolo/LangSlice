"""Unit tests for RLVR image-feeding pipeline.

Covers the SFT-matched preprocessing path: deterministic per-section CLAHE
assignment, lazy decode in ``RowDataset.__getitem__``, and exclusion of
underscore-prefixed plumbing keys from the public column set.
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

from langslice_training.rl.common.dataset import RowDataset, _should_apply_clahe


def test_should_apply_clahe_is_deterministic_per_section() -> None:
    sids = [f"sub_{i:03d}:sec_{j}" for i in range(5) for j in range(4)]
    first = [_should_apply_clahe(s, 0.25) for s in sids]
    second = [_should_apply_clahe(s, 0.25) for s in sids]
    assert first == second


def test_should_apply_clahe_fraction_matches_target_distribution() -> None:
    # Hash-based, so over many samples we should land near 25%.
    sids = [f"section_{i}" for i in range(2000)]
    truthy = sum(_should_apply_clahe(s, 0.25) for s in sids)
    fraction = truthy / len(sids)
    assert 0.20 < fraction < 0.30, f"expected ~0.25, got {fraction:.3f}"


def test_should_apply_clahe_edge_cases() -> None:
    sid = "any_section_id"
    assert _should_apply_clahe(sid, 0.0) is False
    assert _should_apply_clahe(sid, 1.0) is True


def test_row_dataset_strips_underscore_keys_from_column_names() -> None:
    rows = [
        {
            "prompt": [{"role": "user", "content": "hi"}],
            "kind": "single",
            "atlas_name": "allen",
            "_image_paths": ("/tmp/x.jpg",),
            "_apply_clahes": (False,),
            "_atlas_long_edge": 320,
        }
    ]
    ds = RowDataset(rows)
    for col in ds.column_names:
        assert not col.startswith("_"), f"underscore key leaked: {col}"
    # Both image columns are exposed so neither single nor group runs trip.
    assert "image" in ds.column_names
    assert "images" in ds.column_names


def test_row_dataset_lazy_decode_calls_preprocess_per_image() -> None:
    rows = [
        {
            "prompt": [],
            "kind": "single",
            "_image_paths": ("/tmp/a.jpg",),
            "_apply_clahes": (True,),
            "_atlas_long_edge": 256,
        },
        {
            "prompt": [],
            "kind": "group",
            "_image_paths": ("/tmp/b.jpg", "/tmp/c.jpg"),
            "_apply_clahes": (False, True),
            "_atlas_long_edge": 256,
        },
    ]
    ds = RowDataset(rows)
    sentinel = object()
    with patch(
        "langslice_training.rl.common.dataset.preprocess_query_image",
        return_value=sentinel,
    ) as mock_pp:
        single = ds[0]
        group = ds[1]
    # Single row: one decode, exposes "image"; "images" absent.
    assert single["image"] is sentinel
    assert "images" not in single
    # Group row: n decodes, exposes "images" list of length 2.
    assert group["images"] == [sentinel, sentinel]
    assert "image" not in group
    # Underscore plumbing never reaches the consumer.
    for k in single.keys() | group.keys():
        assert not k.startswith("_")
    # 1 single decode + 2 group decodes = 3 total.
    assert mock_pp.call_count == 3


def test_row_dataset_lazy_decode_passes_per_image_clahe_flag() -> None:
    rows = [
        {
            "prompt": [],
            "kind": "group",
            "_image_paths": ("/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"),
            "_apply_clahes": (True, False, True),
            "_atlas_long_edge": 320,
        }
    ]
    ds = RowDataset(rows)
    with patch(
        "langslice_training.rl.common.dataset.preprocess_query_image",
        return_value=object(),
    ) as mock_pp:
        ds[0]
    flag_args = [c.kwargs["apply_clahe"] for c in mock_pp.call_args_list]
    assert flag_args == [True, False, True]
    edge_args = [c.kwargs["atlas_long_edge"] for c in mock_pp.call_args_list]
    assert edge_args == [320, 320, 320]


def test_row_dataset_clahe_fraction_drives_assignment_distribution() -> None:
    # Surrogate end-to-end: feed deterministic section_ids through the helper
    # we use at row-construction time and ensure the bool matches what would
    # land in `_apply_clahes`.
    fractions = [0.0, 0.25, 0.5, 1.0]
    sids = [f"sec_{i}" for i in range(400)]
    counts = Counter()
    for frac in fractions:
        n_true = sum(_should_apply_clahe(s, frac) for s in sids)
        counts[frac] = n_true
    assert counts[0.0] == 0
    assert counts[1.0] == 400
    # Monotone in clahe_fraction.
    assert counts[0.25] < counts[0.5] < counts[1.0]
