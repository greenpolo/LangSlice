"""Unit tests for ``single_turn_rl.manifest_index``.

The tests build a synthetic manifest tree under ``tmp_path`` (shards +
allocations) so they exercise the real ``load_inventory_manifest`` code
path without ever touching the production ``data/manifest/`` tree on
disk. Each test asserts via ``tmp_path`` containment that this stays
true.

Coverage:
* round-trip from a 3-plane, 2-dataset synthetic manifest;
* exact lookup by composite key + ``pairs()`` enumeration;
* every keyword filter on :meth:`ManifestIndex.query` (including the
  bisect-on-position fast path), and the documented determinism of the
  return ordering;
* ``update_difficulty`` / ``live_difficulty`` happy + error paths;
* ``bulk_load_difficulty`` with orphan rows;
* ``persist_live_difficulty`` + ``restore_live_difficulty`` round-trip,
  default-path resolution, and atomic-write evidence;
* construction-time skipping of malformed rows.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import pytest
from single_turn_rl.manifest_index import (
    LiveDifficulty,
    ManifestIndex,
    Section,
)

# ---------------------------------------------------------------------------
# Synthetic manifest builders
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_row(
    *,
    plane: str,
    dataset: str,
    section_id: str,
    position_mm: float,
    atlas: str = "allen_mouse_25um",
    subject_id: str = "subj0",
    image_path: str | None = None,
    species: str = "mouse",
    imaging: str = "brightfield",
    staining: str = "Nissl",
    exclude_from_training: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one shard row with the QC app's expected schema."""
    axis = {"coronal": "ap", "sagittal": "ml", "horizontal": "dv"}[plane]
    row = {
        "image_path": image_path
        or f"data/datasets/{plane}/{dataset}/{subject_id}/{section_id}.png",
        "dataset": dataset,
        "subject_id": subject_id,
        "section_id": section_id,
        "species": species,
        "atlas": atlas,
        "orientation": plane,
        "slice_axis": axis,
        "position_mm": position_mm,
        "imaging": imaging,
        "staining": staining,
        "exclude_from_training": exclude_from_training,
    }
    if extra:
        row.update(extra)
    return row


def _build_manifest_tree(tmp_path: Path) -> Path:
    """Build a 3-plane × 2-dataset synthetic manifest under ``tmp_path``.

    Layout::

        tmp_path/manifest/shards/
            coronal/{ds_a,ds_b}.jsonl
            sagittal/{ds_a,ds_b}.jsonl
            horizontal/{ds_a,ds_b}.jsonl
        tmp_path/manifest/allocations/
            coronal/{eval,rlvr,sft}.jsonl
            ...

    Returns the manifest root.
    """
    manifest_root = tmp_path / "manifest"
    shards_root = manifest_root / "shards"

    # Each plane gets two datasets with three sections each, with
    # monotonic positions so bisect tests have something to slice.
    sections_per_dataset = 3
    plane_positions = {
        "coronal": [0.0, 1.5, 3.0],
        "sagittal": [0.5, 2.0, 3.5],
        "horizontal": [1.0, 2.5, 4.0],
    }
    for plane, positions in plane_positions.items():
        for dataset in ("ds_a", "ds_b"):
            rows: list[dict[str, Any]] = []
            for i in range(sections_per_dataset):
                rows.append(
                    _make_row(
                        plane=plane,
                        dataset=dataset,
                        section_id=f"{dataset}_s{i:02d}",
                        position_mm=positions[i],
                    )
                )
            _write_jsonl(shards_root / plane / f"{dataset}.jsonl", rows)

    # Allocations: stamp a single rlvr row + a single sft row per plane so
    # both code paths are exercised by ``query(split=...)``.
    alloc_root = manifest_root / "allocations"
    for plane in ("coronal", "sagittal", "horizontal"):
        _write_jsonl(
            alloc_root / plane / "rlvr.jsonl",
            [
                {
                    "section_id": "ds_a_s00",
                    "dataset": "ds_a",
                    "added_by": "test",
                    "added_at": "2026-01-01T00:00:00+00:00",
                }
            ],
        )
        _write_jsonl(
            alloc_root / plane / "sft.jsonl",
            [
                {
                    "section_id": "ds_b_s02",
                    "dataset": "ds_b",
                    "added_by": "test",
                    "added_at": "2026-01-01T00:00:00+00:00",
                }
            ],
        )
        _write_jsonl(alloc_root / plane / "eval.jsonl", [])
    return manifest_root


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> Path:
    return _build_manifest_tree(tmp_path)


@pytest.fixture
def index(synthetic_manifest: Path) -> ManifestIndex:
    return ManifestIndex.from_manifest_root(synthetic_manifest)


# ---------------------------------------------------------------------------
# Tmp-path containment guard
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_MANIFEST = REPO_ROOT / "data" / "manifest"


def _assert_under_tmp(path: Path, tmp_path: Path) -> None:
    """Belt-and-braces: every path the test touches must live under tmp_path."""
    abs_path = path.resolve()
    assert tmp_path.resolve() in abs_path.parents or abs_path == tmp_path.resolve(), (
        f"Path {abs_path} escaped tmp_path {tmp_path}"
    )


# ---------------------------------------------------------------------------
# Round-trip + structural tests
# ---------------------------------------------------------------------------


def test_round_trip_returns_expected_section_count(
    synthetic_manifest: Path, tmp_path: Path
) -> None:
    """3 planes × 2 datasets × 3 sections = 18 rows."""
    _assert_under_tmp(synthetic_manifest, tmp_path)
    idx = ManifestIndex.from_manifest_root(synthetic_manifest)
    assert len(idx) == 18


def test_no_real_manifest_paths_touched(
    synthetic_manifest: Path, tmp_path: Path
) -> None:
    """Every section's image_path must be tmp-path-relative; no shard read
    should have come from ``data/manifest/`` on the production tree."""
    idx = ManifestIndex.from_manifest_root(synthetic_manifest)
    for section in idx.query():
        # Synthetic image_path strings are intentionally tmp-relative
        # (built by ``_build_manifest_tree``); a path that starts with
        # the prod manifest prefix means the loader silently fell back
        # to the wrong source.
        assert not section.image_path.startswith(str(PROD_MANIFEST)), (
            f"Section image_path leaked production manifest: {section.image_path}"
        )


def test_section_by_key_exact_lookup(index: ManifestIndex) -> None:
    s = index.section_by_key("coronal", "ds_a", "ds_a_s01")
    assert s is not None
    assert s.plane == "coronal"
    assert s.dataset == "ds_a"
    assert s.section_id == "ds_a_s01"
    assert s.position_mm == pytest.approx(1.5)


def test_section_by_key_returns_none_for_unknown(index: ManifestIndex) -> None:
    assert index.section_by_key("coronal", "ds_a", "nope") is None
    assert index.section_by_key("coronal", "missing_ds", "ds_a_s00") is None
    assert index.section_by_key("nonsense", "ds_a", "ds_a_s00") is None


def test_pairs_returns_distinct_atlas_plane_pairs_sorted(
    index: ManifestIndex,
) -> None:
    expected = sorted(
        [
            ("allen_mouse_25um", "coronal"),
            ("allen_mouse_25um", "sagittal"),
            ("allen_mouse_25um", "horizontal"),
        ]
    )
    assert index.pairs() == expected


# ---------------------------------------------------------------------------
# Query: per-dimension filters
# ---------------------------------------------------------------------------


def test_query_filters_by_plane(index: ManifestIndex) -> None:
    out = index.query(plane="coronal")
    assert len(out) == 6  # 2 datasets × 3 sections
    assert all(s.plane == "coronal" for s in out)


def test_query_filters_by_atlas_and_plane_uses_bisect(
    index: ManifestIndex,
) -> None:
    out = index.query(atlas="allen_mouse_25um", plane="sagittal")
    assert len(out) == 6
    assert all(s.atlas == "allen_mouse_25um" and s.plane == "sagittal" for s in out)


def test_query_filters_by_split(index: ManifestIndex) -> None:
    rlvr_rows = index.query(split="rlvr")
    # One per plane: 3 rows total
    assert len(rlvr_rows) == 3
    assert all(s.split == "rlvr" for s in rlvr_rows)
    sft_rows = index.query(split="sft")
    assert len(sft_rows) == 3
    assert all(s.split == "sft" for s in sft_rows)
    # No-allocation rows have empty-string split
    no_split = index.query(split="")
    assert len(no_split) == 18 - 3 - 3


def test_query_filters_by_dataset(index: ManifestIndex) -> None:
    out = index.query(dataset="ds_a")
    assert len(out) == 9  # 3 planes × 3 sections
    assert all(s.dataset == "ds_a" for s in out)


def test_query_filters_by_position_range_full_scan(index: ManifestIndex) -> None:
    """Position-range without atlas+plane → falls back to full scan path."""
    out = index.query(position_range=(1.0, 2.5))
    for s in out:
        assert 1.0 <= s.position_mm <= 2.5


def test_query_filters_by_position_range_with_atlas_plane_uses_bisect(
    index: ManifestIndex,
) -> None:
    """Position-range with atlas+plane → bisect path. Coronal positions are
    [0.0, 1.5, 3.0] across each of two datasets, so [1.0, 2.0] should
    return exactly the 1.5-position rows (one per dataset)."""
    out = index.query(
        atlas="allen_mouse_25um", plane="coronal", position_range=(1.0, 2.0)
    )
    assert len(out) == 2
    assert all(s.position_mm == pytest.approx(1.5) for s in out)
    assert {s.dataset for s in out} == {"ds_a", "ds_b"}


def test_query_position_range_inclusive_bounds(index: ManifestIndex) -> None:
    """Bounds are inclusive [lo, hi]."""
    out = index.query(
        atlas="allen_mouse_25um", plane="coronal", position_range=(0.0, 0.0)
    )
    assert len(out) == 2  # both datasets have a 0.0 row
    out_three = index.query(
        atlas="allen_mouse_25um", plane="coronal", position_range=(0.0, 3.0)
    )
    assert len(out_three) == 6  # all coronal rows


def test_query_difficulty_range_excludes_unobserved(index: ManifestIndex) -> None:
    """When ``difficulty_range`` is set, sections without a live record are excluded."""
    index.update_difficulty("coronal", "ds_a", "ds_a_s00", 0.4, "test")
    index.update_difficulty("coronal", "ds_a", "ds_a_s01", 0.7, "test")
    # ds_a_s02 intentionally has no live record.
    out = index.query(plane="coronal", dataset="ds_a", difficulty_range=(0.0, 1.0))
    assert {s.section_id for s in out} == {"ds_a_s00", "ds_a_s01"}


def test_query_difficulty_range_none_includes_unobserved(
    index: ManifestIndex,
) -> None:
    """Default ``difficulty_range=None`` → unobserved rows are included."""
    index.update_difficulty("coronal", "ds_a", "ds_a_s00", 0.4, "test")
    out = index.query(plane="coronal", dataset="ds_a")
    assert len(out) == 3
    assert {s.section_id for s in out} == {
        "ds_a_s00",
        "ds_a_s01",
        "ds_a_s02",
    }


def test_query_exclude_zero_axes_drops_zero_position(
    index: ManifestIndex,
) -> None:
    full = index.query(plane="coronal")
    no_zero = index.query(plane="coronal", exclude_zero_axes=True)
    # 0.0 rows: one per dataset on coronal → 2 rows dropped.
    assert len(full) - len(no_zero) == 2
    assert all(s.position_mm != 0.0 for s in no_zero)


def test_query_default_includes_zero_axes(index: ManifestIndex) -> None:
    """``exclude_zero_axes`` defaults to False — zero positions are kept."""
    out = index.query(plane="coronal")
    assert any(s.position_mm == 0.0 for s in out)


def test_query_result_sorted_by_composite_key(index: ManifestIndex) -> None:
    out = index.query()
    keys = [(s.plane, s.dataset, s.section_id) for s in out]
    assert keys == sorted(keys)


def test_query_combined_filters(index: ManifestIndex) -> None:
    """Sanity check: combining several filters returns the AND of them."""
    out = index.query(
        atlas="allen_mouse_25um",
        plane="sagittal",
        dataset="ds_b",
        position_range=(1.0, 3.0),
    )
    # Sagittal positions: [0.5, 2.0, 3.5]; only 2.0 falls in [1.0, 3.0].
    assert len(out) == 1
    assert out[0].dataset == "ds_b"
    assert out[0].section_id == "ds_b_s01"
    assert out[0].position_mm == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Live difficulty: record / lookup / errors
# ---------------------------------------------------------------------------


def test_update_difficulty_records_score_and_metadata(
    index: ManifestIndex,
) -> None:
    index.update_difficulty("coronal", "ds_a", "ds_a_s00", 0.42, "slicebench")
    rec = index.live_difficulty("coronal", "ds_a", "ds_a_s00")
    assert rec is not None
    assert rec.score == pytest.approx(0.42)
    assert rec.source == "slicebench"
    # Loose ISO-8601 check: must contain at least date-time-T-time and end
    # with a UTC offset.
    assert "T" in rec.updated_at
    assert rec.updated_at.endswith("+00:00") or rec.updated_at.endswith("Z")


def test_update_difficulty_raises_on_unknown_key(index: ManifestIndex) -> None:
    with pytest.raises(KeyError):
        index.update_difficulty("coronal", "ds_a", "missing", 0.5, "test")


def test_update_difficulty_default_still_raises(index: ManifestIndex) -> None:
    """Defensive: callers that don't pass ``if_unknown`` get the historical
    KeyError contract — no behavior change for existing call sites."""
    with pytest.raises(KeyError):
        # Identical call to the test above but spelled out: the parameter
        # default must remain ``"raise"``.
        index.update_difficulty(
            "coronal", "ds_a", "missing", 0.5, "test"
        )


def test_update_difficulty_warn_on_unknown_returns_false_with_warning(
    index: ManifestIndex, capsys: pytest.CaptureFixture[str]
) -> None:
    """``if_unknown="warn"`` returns False, writes nothing, and emits stderr
    that names the orphan key."""
    result = index.update_difficulty(
        "coronal", "ds_a", "missing", 0.5, "test", if_unknown="warn"
    )
    assert result is False
    captured = capsys.readouterr()
    # The orphan key should appear in the warning so an operator can grep for
    # the offending section without re-running the trainer.
    assert "missing" in captured.err
    assert "ds_a" in captured.err
    # And no record was actually written.
    assert index.live_difficulty("coronal", "ds_a", "missing") is None


def test_update_difficulty_ignore_on_unknown_returns_false_silently(
    index: ManifestIndex, capsys: pytest.CaptureFixture[str]
) -> None:
    """``if_unknown="ignore"`` returns False with no stderr output."""
    result = index.update_difficulty(
        "coronal", "ds_a", "missing", 0.5, "test", if_unknown="ignore"
    )
    assert result is False
    captured = capsys.readouterr()
    assert captured.err == ""
    assert index.live_difficulty("coronal", "ds_a", "missing") is None


def test_update_difficulty_returns_true_on_known_key(
    index: ManifestIndex,
) -> None:
    """Successful writes return True regardless of ``if_unknown`` setting."""
    assert (
        index.update_difficulty(
            "coronal", "ds_a", "ds_a_s00", 0.5, "test"
        )
        is True
    )
    assert (
        index.update_difficulty(
            "coronal", "ds_a", "ds_a_s01", 0.5, "test", if_unknown="warn"
        )
        is True
    )
    assert (
        index.update_difficulty(
            "coronal", "ds_a", "ds_a_s02", 0.5, "test", if_unknown="ignore"
        )
        is True
    )


def test_live_difficulty_returns_none_when_unrecorded(
    index: ManifestIndex,
) -> None:
    assert index.live_difficulty("coronal", "ds_a", "ds_a_s00") is None


# ---------------------------------------------------------------------------
# bulk_load_difficulty
# ---------------------------------------------------------------------------


def test_bulk_load_difficulty_loads_known_rows(
    index: ManifestIndex, tmp_path: Path
) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ds_a_s00",
                        "difficulty_score": 0.31,
                        "source": "slicebench",
                    },
                    {
                        "plane": "sagittal",
                        "dataset": "ds_b",
                        "section_id": "ds_b_s01",
                        "difficulty_score": 0.62,
                        "source": "slicebench",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    n = index.bulk_load_difficulty(seed_path)
    assert n == 2
    rec = index.live_difficulty("coronal", "ds_a", "ds_a_s00")
    assert rec is not None
    assert rec.score == pytest.approx(0.31)
    assert rec.source == "slicebench"


def test_bulk_load_difficulty_skips_orphans(
    index: ManifestIndex, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ds_a_s00",
                        "difficulty_score": 0.31,
                        "source": "slicebench",
                    },
                    # Orphan: section doesn't exist in index.
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ghost",
                        "difficulty_score": 0.99,
                        "source": "slicebench",
                    },
                    # Invalid score type.
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ds_a_s01",
                        "difficulty_score": "not a number",
                        "source": "slicebench",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    n = index.bulk_load_difficulty(seed_path)
    assert n == 1
    captured = capsys.readouterr()
    # New warning shape: counts for orphan + invalid are reported separately
    # and the first up-to-5 offending orphan keys are listed.
    assert "1 orphan" in captured.err
    assert "1 invalid" in captured.err
    # The single orphan key must appear in the preview list.
    assert "ghost" in captured.err
    assert "ds_a" in captured.err


def test_bulk_load_difficulty_lists_first_five_orphans_and_truncates(
    index: ManifestIndex, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When there are more than 5 orphan keys, only the first 5 are listed
    and a ``(N more)`` suffix indicates how many were truncated."""
    seed_rows: list[dict[str, Any]] = []
    # 7 orphan rows so we exercise the "5 listed + 2 more" branch.
    for i in range(7):
        seed_rows.append(
            {
                "plane": "coronal",
                "dataset": "ds_a",
                "section_id": f"ghost_{i:02d}",
                "difficulty_score": 0.5,
                "source": "slicebench",
            }
        )
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps({"rows": seed_rows}), encoding="utf-8")
    n = index.bulk_load_difficulty(seed_path)
    assert n == 0
    captured = capsys.readouterr()
    assert "7 orphan" in captured.err
    assert "0 invalid" in captured.err
    # First five preview keys present.
    for i in range(5):
        assert f"ghost_{i:02d}" in captured.err
    # Last two are not in the preview but counted in the "(2 more)" tail.
    assert "ghost_05" not in captured.err
    assert "ghost_06" not in captured.err
    assert "(2 more)" in captured.err


# ---------------------------------------------------------------------------
# persist / restore round-trip
# ---------------------------------------------------------------------------


def test_persist_and_restore_round_trip(
    index: ManifestIndex, tmp_path: Path
) -> None:
    sidecar = tmp_path / "live_difficulty.json"
    index.update_difficulty("coronal", "ds_a", "ds_a_s00", 0.42, "slicebench")
    index.update_difficulty("sagittal", "ds_b", "ds_b_s02", 0.81, "live_rollout")
    n_written = index.persist_live_difficulty(sidecar)
    assert n_written == 2
    assert sidecar.is_file()

    # Build a fresh index from the same manifest and restore.
    fresh = ManifestIndex.from_manifest_root(
        tmp_path / "manifest", repo_root=tmp_path
    )
    n_restored = fresh.restore_live_difficulty(sidecar)
    assert n_restored == 2
    rec = fresh.live_difficulty("coronal", "ds_a", "ds_a_s00")
    assert rec is not None
    assert rec.score == pytest.approx(0.42)
    assert rec.source == "slicebench"
    rec2 = fresh.live_difficulty("sagittal", "ds_b", "ds_b_s02")
    assert rec2 is not None
    assert rec2.score == pytest.approx(0.81)


def test_persist_default_path_uses_repo_root(
    synthetic_manifest: Path, tmp_path: Path
) -> None:
    """Default persist path is ``_local/rlvr_progress/live_difficulty.json``
    under the inferred repo root."""
    # Pin repo_root to tmp_path so the test never writes outside tmp_path.
    idx = ManifestIndex.from_manifest_root(synthetic_manifest, repo_root=tmp_path)
    idx.update_difficulty("coronal", "ds_a", "ds_a_s00", 0.5, "test")
    n = idx.persist_live_difficulty()  # no path → default
    assert n == 1
    expected = tmp_path / "_local" / "rlvr_progress" / "live_difficulty.json"
    _assert_under_tmp(expected, tmp_path)
    assert expected.is_file()


def test_restore_skips_orphan_sections(
    index: ManifestIndex, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sidecar = tmp_path / "live.json"
    sidecar.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ds_a_s00",
                        "score": 0.42,
                        "source": "slicebench",
                        "updated_at": "2026-05-10T12:00:00+00:00",
                    },
                    # Orphan: doesn't exist in the index.
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ghost",
                        "score": 0.99,
                        "source": "slicebench",
                        "updated_at": "2026-05-10T12:00:00+00:00",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    n = index.restore_live_difficulty(sidecar)
    assert n == 1
    captured = capsys.readouterr()
    assert "1 orphan" in captured.err
    assert "0 invalid" in captured.err
    assert "ghost" in captured.err


def test_restore_distinguishes_orphan_and_invalid_counts(
    index: ManifestIndex, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mixed orphan + invalid rows are reported with separate counts."""
    sidecar = tmp_path / "live.json"
    sidecar.write_text(
        json.dumps(
            {
                "rows": [
                    # Good row.
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ds_a_s00",
                        "score": 0.42,
                        "source": "slicebench",
                        "updated_at": "2026-05-10T12:00:00+00:00",
                    },
                    # Orphan: section absent from index.
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ghost",
                        "score": 0.99,
                        "source": "slicebench",
                        "updated_at": "2026-05-10T12:00:00+00:00",
                    },
                    # Invalid: non-numeric score on a real key.
                    {
                        "plane": "coronal",
                        "dataset": "ds_a",
                        "section_id": "ds_a_s01",
                        "score": "not a number",
                        "source": "slicebench",
                        "updated_at": "2026-05-10T12:00:00+00:00",
                    },
                    # Invalid: not a dict at all.
                    "garbage",  # type: ignore[list-item]
                ]
            }
        ),
        encoding="utf-8",
    )
    n = index.restore_live_difficulty(sidecar)
    assert n == 1
    captured = capsys.readouterr()
    assert "1 orphan" in captured.err
    assert "2 invalid" in captured.err
    # Orphan key listed in preview.
    assert "ghost" in captured.err


def test_persist_writes_via_tmp_then_renames(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``persist_live_difficulty`` must write to a sibling .tmp file and then
    rename atomically. We assert this by intercepting ``os.replace``."""
    sidecar = tmp_path / "live.json"
    index.update_difficulty("coronal", "ds_a", "ds_a_s00", 0.5, "test")

    real_replace = os.replace
    seen: dict[str, Path] = {}

    def _spy_replace(src, dst):  # type: ignore[no-untyped-def]
        seen["src"] = Path(src)
        seen["dst"] = Path(dst)
        # The .tmp file must already exist before the rename happens.
        assert Path(src).is_file(), "tmp file missing at rename time"
        return real_replace(src, dst)

    monkeypatch.setattr(
        "single_turn_rl.manifest_index.os.replace", _spy_replace
    )
    index.persist_live_difficulty(sidecar)
    assert seen["src"].name == "live.json.tmp"
    assert seen["dst"] == sidecar
    assert sidecar.is_file()


# ---------------------------------------------------------------------------
# Construction-time row validation
# ---------------------------------------------------------------------------


def test_skips_rows_missing_required_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rows missing required string fields or with un-mappable slice_axis
    are skipped during construction with a stderr warning."""
    manifest_root = tmp_path / "manifest"
    shards = manifest_root / "shards"
    # One good row + several bad ones in the same shard.
    bad_rows: list[dict[str, Any]] = [
        _make_row(
            plane="coronal", dataset="ds_a", section_id="ok", position_mm=1.0
        ),
        # Missing section_id.
        {
            "image_path": "x.png",
            "dataset": "ds_a",
            "subject_id": "subj",
            "atlas": "allen_mouse_25um",
            "orientation": "coronal",
            "slice_axis": "ap",
            "position_mm": 0.5,
        },
        # slice_axis can't be mapped to a plane.
        _make_row(
            plane="coronal",
            dataset="ds_a",
            section_id="bad_axis",
            position_mm=2.0,
            extra={"slice_axis": "garbage"},
        ),
        # Non-numeric position_mm.
        _make_row(
            plane="coronal",
            dataset="ds_a",
            section_id="bad_pos",
            position_mm=0.0,
            extra={"position_mm": "not a number"},
        ),
        # Empty string atlas.
        _make_row(
            plane="coronal",
            dataset="ds_a",
            section_id="empty_atlas",
            position_mm=1.0,
            atlas="",
        ),
    ]
    _write_jsonl(shards / "coronal" / "ds_a.jsonl", bad_rows)
    # Empty allocation tree so the loader doesn't choke.
    for plane in ("coronal", "sagittal", "horizontal"):
        for split in ("eval", "rlvr", "sft"):
            (manifest_root / "allocations" / plane / f"{split}.jsonl").parent.mkdir(
                parents=True, exist_ok=True
            )
            (manifest_root / "allocations" / plane / f"{split}.jsonl").touch()

    idx = ManifestIndex.from_manifest_root(manifest_root)
    sections = idx.query()
    assert [s.section_id for s in sections] == ["ok"]
    captured = capsys.readouterr()
    assert "skipped" in captured.err


# ---------------------------------------------------------------------------
# Real-loader sanity
# ---------------------------------------------------------------------------


def test_loader_does_not_walk_real_manifest(
    synthetic_manifest: Path, tmp_path: Path
) -> None:
    """The loader must read only from tmp_path's shards, never from the
    production manifest. We verify this by counting sections — if the loader
    accidentally fell back to the prod manifest, the count would balloon
    well beyond the 18 synthetic rows."""
    idx = ManifestIndex.from_manifest_root(synthetic_manifest)
    assert len(idx) == 18, (
        "Section count mismatch — loader may have walked real shards"
    )
    # Belt + braces: assert the manifest_root field on the index points
    # under tmp_path.
    assert idx._manifest_root is not None
    _assert_under_tmp(idx._manifest_root, tmp_path)


def test_section_dataclass_is_frozen() -> None:
    """The Section dataclass must be frozen (hashable, immutable)."""
    s = Section(
        plane="coronal",
        dataset="ds_a",
        subject_id="subj",
        section_id="s00",
        atlas="allen_mouse_25um",
        position_mm=1.0,
        image_path="x.png",
        species="mouse",
        imaging="brightfield",
        staining="Nissl",
        orientation="coronal",
        exclude_from_training=False,
        split="rlvr",
    )
    # frozen=True dataclasses raise FrozenInstanceError on attribute write.
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.position_mm = 2.0  # type: ignore[misc]
    # And usable as a dict key.
    assert s.key == ("coronal", "ds_a", "s00")
    assert {s: 1}[s] == 1


def test_live_difficulty_dataclass_is_frozen() -> None:
    rec = LiveDifficulty(score=0.5, source="test", updated_at="2026-05-10T00:00:00+00:00")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.score = 0.6  # type: ignore[misc]
