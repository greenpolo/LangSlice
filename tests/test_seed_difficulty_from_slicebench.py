"""Unit tests for ``tools/seed_difficulty_from_slicebench.py``.

Tests build a synthetic manifest tree under ``tmp_path`` (same pattern as
``tests/test_manifest_index.py``) and stub the atlas-extent lookup so the
real BrainGlobe atlases are never loaded. The CLI must:

* prefer ``per_section`` MAE over ``per_coord_bin`` when both are present;
* fall back to ``per_coord_bin`` MAE expanded across every matching
  manifest section when ``per_section`` is absent;
* skip sections that exist in the manifest but not in slicebench (their
  difficulty stays NULL — the curriculum's uniform fallback handles them);
* skip sections in slicebench but not in the manifest with a stderr warning
  (slicebench may use a different section pool — orphans must be noted but
  not fatal);
* error out when neither block is present (a wrong path shouldn't silently
  emit an empty seed file);
* round-trip end-to-end through ``ManifestIndex.bulk_load_difficulty`` —
  the live difficulty after load must equal the per-row score the CLI
  emitted, even though the JSON crosses a process / module boundary;
* atomically write via ``.tmp`` then ``os.replace``;
* respect ``--source-label`` (default ``"slicebench"``);
* use slicebench's ``_bin_index`` correctly when expanding per_coord_bin.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from single_turn_rl.manifest_index import ManifestIndex

# ---------------------------------------------------------------------------
# Module-under-test loader
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = REPO_ROOT / "tools" / "seed_difficulty_from_slicebench.py"


def _load_tool_module() -> Any:
    """Import the tool by file path so the test doesn't depend on a package
    layout for ``tools/``. Re-imports each call so per-test monkeypatches
    of cached state stay scoped (the tool keeps a module-level extent
    cache that would otherwise leak across tests)."""
    spec = importlib.util.spec_from_file_location(
        "seed_difficulty_from_slicebench", TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tool() -> Any:
    """Fresh copy of the tool module per test (avoids extent-cache leakage)."""
    return _load_tool_module()


# ---------------------------------------------------------------------------
# Synthetic manifest builders (mirrors ``tests/test_manifest_index.py``)
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
) -> dict[str, Any]:
    """Build a minimal shard row that ``load_inventory_manifest`` accepts."""
    axis = {"coronal": "ap", "sagittal": "ml", "horizontal": "dv"}[plane]
    return {
        "image_path": (
            f"data/datasets/{plane}/{dataset}/{subject_id}/{section_id}.png"
        ),
        "dataset": dataset,
        "subject_id": subject_id,
        "section_id": section_id,
        "species": "mouse",
        "atlas": atlas,
        "orientation": plane,
        "slice_axis": axis,
        "position_mm": position_mm,
        "imaging": "brightfield",
        "staining": "Nissl",
        "exclude_from_training": False,
    }


def _build_manifest_tree(tmp_path: Path) -> Path:
    """Three planes × two datasets × three sections each = 18 rows."""
    manifest_root = tmp_path / "manifest"
    shards_root = manifest_root / "shards"

    plane_positions = {
        "coronal": [0.0, 1.5, 3.0],
        "sagittal": [0.5, 2.0, 3.5],
        "horizontal": [1.0, 2.5, 4.0],
    }
    for plane, positions in plane_positions.items():
        for dataset in ("ds_a", "ds_b"):
            rows = [
                _make_row(
                    plane=plane,
                    dataset=dataset,
                    section_id=f"{dataset}_s{i:02d}",
                    position_mm=positions[i],
                )
                for i in range(3)
            ]
            _write_jsonl(shards_root / plane / f"{dataset}.jsonl", rows)

    # Empty allocation tree so the loader doesn't choke. The CLI doesn't
    # consult split membership so empty allocations are sufficient.
    alloc_root = manifest_root / "allocations"
    for plane in ("coronal", "sagittal", "horizontal"):
        for split in ("eval", "rlvr", "sft"):
            (alloc_root / plane / f"{split}.jsonl").parent.mkdir(
                parents=True, exist_ok=True,
            )
            (alloc_root / plane / f"{split}.jsonl").touch()
    return manifest_root


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> Path:
    return _build_manifest_tree(tmp_path)


@pytest.fixture
def index(synthetic_manifest: Path) -> ManifestIndex:
    return ManifestIndex.from_manifest_root(synthetic_manifest)


# Synthetic atlas extents — chosen for clean quintile boundaries so per_coord_bin
# expansion is unambiguous to verify by hand.
SYNTHETIC_EXTENTS: dict[tuple[str, str], float] = {
    ("allen_mouse_25um", "coronal"): 10.0,
    ("allen_mouse_25um", "sagittal"): 5.0,
    ("allen_mouse_25um", "horizontal"): 8.0,
}


@pytest.fixture
def stub_extents(monkeypatch: pytest.MonkeyPatch, tool: Any) -> None:
    """Bypass BrainGlobe atlas loading by replacing ``_axis_extent_mm``."""
    cache: dict[tuple[str, str], float] = dict(SYNTHETIC_EXTENTS)

    def _fake_extent(atlas_name: str, plane: str) -> float:
        try:
            return cache[(atlas_name, plane)]
        except KeyError as exc:
            raise KeyError(
                f"Test atlas extent missing for ({atlas_name!r}, {plane!r})"
            ) from exc

    monkeypatch.setattr(tool, "_axis_extent_mm", _fake_extent)


# ---------------------------------------------------------------------------
# Tmp-path containment guard
# ---------------------------------------------------------------------------


def _assert_under_tmp(path: Path, tmp_path: Path) -> None:
    abs_path = path.resolve()
    assert tmp_path.resolve() in abs_path.parents or abs_path == tmp_path.resolve(), (
        f"Path {abs_path} escaped tmp_path {tmp_path}"
    )


# ---------------------------------------------------------------------------
# 1. per_section path: difficulty_score = mae_mm / axis_extent
# ---------------------------------------------------------------------------


def test_per_section_path_emits_one_row_per_entry(
    tool: Any, index: ManifestIndex, stub_extents: None,
) -> None:
    """When per_section is present, every entry produces one output row."""
    summary = {
        "per_section": {
            "ds_a_s00": {
                "plane": "coronal", "dataset": "ds_a",
                "atlas": "allen_mouse_25um", "mae_mm": 2.0,
            },
            "ds_b_s01": {
                "plane": "sagittal", "dataset": "ds_b",
                "atlas": "allen_mouse_25um", "mae_mm": 1.0,
            },
        },
    }
    rows = tool.build_rows(summary, index, source_label="slicebench")
    by_key = {(r["plane"], r["dataset"], r["section_id"]): r for r in rows}
    assert len(rows) == 2
    # 2.0 / 10.0 = 0.2 (coronal extent); 1.0 / 5.0 = 0.2 (sagittal extent).
    assert by_key[("coronal", "ds_a", "ds_a_s00")]["difficulty_score"] == pytest.approx(0.2)
    assert by_key[("sagittal", "ds_b", "ds_b_s01")]["difficulty_score"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 2. per_coord_bin path: bin MAE expanded across matching sections
# ---------------------------------------------------------------------------


def test_per_coord_bin_path_emits_row_per_matching_section(
    tool: Any, index: ManifestIndex, stub_extents: None,
) -> None:
    """When only per_coord_bin is present, every section in the manifest
    that maps to a bin with MAE data gets a row."""
    # Coronal extent = 10.0, n_bins=5 → bin width = 2.0.
    # Coronal positions [0.0, 1.5, 3.0] → bins [0, 0, 1].
    # Setting only q1+q2 MAE so we verify the "skip if no MAE in bin" branch
    # (no other plane gets data → only coronal rows emitted).
    summary = {
        "per_coord_bin": {
            "coronal": {
                "q1": {"mae_mm": 2.5},   # bin 0 → score 0.25
                "q2": {"mae_mm": 1.5},   # bin 1 → score 0.15
            },
        },
    }
    rows = tool.build_rows(summary, index, source_label="slicebench")
    # 2 datasets × 3 coronal sections = 6 rows. (sagittal/horizontal absent.)
    assert len(rows) == 6
    assert all(r["plane"] == "coronal" for r in rows)

    by_section = {(r["dataset"], r["section_id"]): r["difficulty_score"] for r in rows}
    # ds_*_s00 (pos 0.0) and ds_*_s01 (pos 1.5) both fall in bin 0 (q1).
    assert by_section[("ds_a", "ds_a_s00")] == pytest.approx(0.25)
    assert by_section[("ds_a", "ds_a_s01")] == pytest.approx(0.25)
    assert by_section[("ds_b", "ds_b_s00")] == pytest.approx(0.25)
    assert by_section[("ds_b", "ds_b_s01")] == pytest.approx(0.25)
    # ds_*_s02 (pos 3.0) lands in bin 1 (q2).
    assert by_section[("ds_a", "ds_a_s02")] == pytest.approx(0.15)
    assert by_section[("ds_b", "ds_b_s02")] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# 3. per_section preferred over per_coord_bin when both present
# ---------------------------------------------------------------------------


def test_per_section_preferred_over_per_coord_bin(
    tool: Any, index: ManifestIndex, stub_extents: None,
) -> None:
    """Mixed summary — per_section data must win, per_coord_bin is ignored."""
    summary = {
        "per_section": {
            "ds_a_s00": {
                "plane": "coronal", "dataset": "ds_a",
                "atlas": "allen_mouse_25um", "mae_mm": 4.0,  # → 0.4
            },
        },
        "per_coord_bin": {
            "coronal": {
                "q1": {"mae_mm": 0.5},  # would imply 0.05 — must be ignored.
            },
        },
    }
    rows = tool.build_rows(summary, index, source_label="slicebench")
    # Only the per_section row should be emitted; per_coord_bin shouldn't fire.
    assert len(rows) == 1
    assert rows[0]["section_id"] == "ds_a_s00"
    assert rows[0]["difficulty_score"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 4. Sections in manifest but not in slicebench: skipped (no row written)
# ---------------------------------------------------------------------------


def test_sections_only_in_manifest_are_skipped(
    tool: Any, index: ManifestIndex, stub_extents: None,
) -> None:
    """Round-0 only writes rows for sections that have an estimate. The
    curriculum's uniform fallback covers the rest, so any section absent
    from per_section must produce no row (and no orphan warning either —
    that's the trainer's pool, not slicebench's)."""
    summary = {
        "per_section": {
            "ds_a_s00": {
                "plane": "coronal", "dataset": "ds_a",
                "atlas": "allen_mouse_25um", "mae_mm": 1.0,
            },
        },
    }
    rows = tool.build_rows(summary, index, source_label="slicebench")
    assert len(rows) == 1
    assert rows[0]["section_id"] == "ds_a_s00"


# ---------------------------------------------------------------------------
# 5. Sections in slicebench but not in manifest: skipped + stderr warning
# ---------------------------------------------------------------------------


def test_orphans_in_slicebench_are_skipped_with_stderr_warning(
    tool: Any,
    index: ManifestIndex,
    stub_extents: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Slicebench may use a different section pool than the manifest the
    trainer reads (e.g. an older eval cut). Orphan rows must be dropped
    rather than passed through, and the operator must see a clear
    warning naming the offending keys."""
    summary = {
        "per_section": {
            "ds_a_s00": {
                "plane": "coronal", "dataset": "ds_a",
                "atlas": "allen_mouse_25um", "mae_mm": 1.0,
            },
            # Doesn't exist in the synthetic manifest — should be reported.
            "ghost_section": {
                "plane": "coronal", "dataset": "ds_ghost",
                "atlas": "allen_mouse_25um", "mae_mm": 5.0,
            },
        },
    }
    rows = tool.build_rows(
        summary, index, source_label="slicebench",
        summary_path_for_logs="fake/summary.json",
    )
    assert len(rows) == 1
    assert rows[0]["section_id"] == "ds_a_s00"
    captured = capsys.readouterr()
    assert "1 orphan" in captured.err
    assert "ghost_section" in captured.err
    assert "ds_ghost" in captured.err


# ---------------------------------------------------------------------------
# 6. End-to-end: CLI output round-trips through bulk_load_difficulty
# ---------------------------------------------------------------------------


def test_round_trip_through_bulk_load_difficulty(
    tool: Any,
    synthetic_manifest: Path,
    index: ManifestIndex,
    stub_extents: None,
    tmp_path: Path,
) -> None:
    """The live difficulty after ``bulk_load_difficulty`` reads the CLI's
    output must equal the score the CLI computed — verifies the JSON shape
    contract end-to-end."""
    summary = {
        "per_section": {
            "ds_a_s00": {
                "plane": "coronal", "dataset": "ds_a",
                "atlas": "allen_mouse_25um", "mae_mm": 2.0,  # 0.2
            },
            "ds_b_s02": {
                "plane": "sagittal", "dataset": "ds_b",
                "atlas": "allen_mouse_25um", "mae_mm": 1.25,  # 0.25
            },
        },
    }
    rows = tool.build_rows(summary, index, source_label="slicebench-tiny-round0")
    out_path = tmp_path / "seeded_difficulty.json"
    tool.write_seed_atomic(rows, out_path)
    _assert_under_tmp(out_path, tmp_path)

    n = index.bulk_load_difficulty(out_path)
    assert n == 2
    rec = index.live_difficulty("coronal", "ds_a", "ds_a_s00")
    assert rec is not None
    assert rec.score == pytest.approx(0.2)
    assert rec.source == "slicebench-tiny-round0"
    rec2 = index.live_difficulty("sagittal", "ds_b", "ds_b_s02")
    assert rec2 is not None
    assert rec2.score == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 7. --source-label is recorded as the row's `source` field
# ---------------------------------------------------------------------------


def test_source_label_recorded_per_row(
    tool: Any, index: ManifestIndex, stub_extents: None,
) -> None:
    """Custom ``--source-label`` lands on every emitted row's ``source`` field."""
    summary = {
        "per_section": {
            "ds_a_s00": {
                "plane": "coronal", "dataset": "ds_a",
                "atlas": "allen_mouse_25um", "mae_mm": 1.0,
            },
        },
    }
    rows = tool.build_rows(summary, index, source_label="slicebench-round-3")
    assert len(rows) == 1
    assert rows[0]["source"] == "slicebench-round-3"


def test_source_label_default_is_slicebench(
    tool: Any, index: ManifestIndex, stub_extents: None, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``--source-label`` is the literal string ``"slicebench"``."""
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({
            "per_section": {
                "ds_a_s00": {
                    "plane": "coronal", "dataset": "ds_a",
                    "atlas": "allen_mouse_25um", "mae_mm": 1.0,
                },
            },
        }),
        encoding="utf-8",
    )
    output = tmp_path / "seed.json"

    # Patch the loader inside main() to use our synthetic index.
    def _fake_from_manifest_root(_root: Path) -> ManifestIndex:
        return index

    monkeypatch.setattr(
        tool.ManifestIndex, "from_manifest_root", _fake_from_manifest_root,
    )
    rc = tool.main([
        "--slicebench-summary", str(summary_path),
        "--manifest-root", str(tmp_path / "manifest"),
        "--output", str(output),
    ])
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rows"]
    for row in payload["rows"]:
        assert row["source"] == "slicebench"


# ---------------------------------------------------------------------------
# 8. Errors cleanly when neither per_section nor per_coord_bin present
# ---------------------------------------------------------------------------


def test_errors_when_neither_per_section_nor_per_coord_bin_present(
    tool: Any, index: ManifestIndex, stub_extents: None,
) -> None:
    """A summary missing both blocks must raise — silently emitting an
    empty seed would mask a wrong ``--slicebench-summary``."""
    summary = {"overall": {"n": 100, "mae_mm": 1.5}}
    with pytest.raises(ValueError, match="per_section.*per_coord_bin"):
        tool.build_rows(summary, index, source_label="slicebench")


def test_cli_returns_nonzero_when_neither_block_present(
    tool: Any, synthetic_manifest: Path, stub_extents: None, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, index: ManifestIndex,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI surfaces the build-time error as a non-zero exit + stderr."""
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"overall": {"n": 0}}), encoding="utf-8",
    )
    output = tmp_path / "seed.json"
    monkeypatch.setattr(
        tool.ManifestIndex, "from_manifest_root",
        lambda _root: index,
    )
    rc = tool.main([
        "--slicebench-summary", str(summary_path),
        "--manifest-root", str(synthetic_manifest),
        "--output", str(output),
    ])
    assert rc != 0
    captured = capsys.readouterr()
    assert "per_section" in captured.err and "per_coord_bin" in captured.err
    # And no seed file should have been written.
    assert not output.exists()


# ---------------------------------------------------------------------------
# 9. Atomic write: writes via .tmp then renames
# ---------------------------------------------------------------------------


def test_write_seed_atomic_uses_tmp_file_then_replace(
    tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``write_seed_atomic`` must write to a sibling .tmp file and then
    ``os.replace`` it into place. We assert this by intercepting
    ``os.replace`` and verifying the .tmp file exists at rename time —
    same pattern as ``test_persist_writes_via_tmp_then_renames``."""
    rows = [
        {
            "plane": "coronal", "dataset": "ds_a",
            "section_id": "ds_a_s00", "difficulty_score": 0.5,
            "source": "slicebench",
        },
    ]
    output = tmp_path / "seed.json"
    real_replace = os.replace
    seen: dict[str, Path] = {}

    def _spy_replace(src, dst):  # type: ignore[no-untyped-def]
        seen["src"] = Path(src)
        seen["dst"] = Path(dst)
        assert Path(src).is_file(), "tmp file missing at rename time"
        return real_replace(src, dst)

    monkeypatch.setattr(tool.os, "replace", _spy_replace)
    tool.write_seed_atomic(rows, output)
    assert seen["src"].name == "seed.json.tmp"
    assert seen["dst"] == output
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {"rows": rows}


# ---------------------------------------------------------------------------
# 10. _bin_index correctness: synthetic position falls in expected bin
# ---------------------------------------------------------------------------


def test_bin_index_imported_from_slicebench_returns_expected_quintile(
    tool: Any,
) -> None:
    """The CLI's bin assignment must agree with slicebench's authoritative
    binning. Verify with explicit positions on a 10mm extent + n_bins=5
    (each quintile = 2mm wide)."""
    bin_index = tool._bin_index
    assert bin_index(0.0, 10.0, 5) == 0    # q1
    assert bin_index(1.99, 10.0, 5) == 0   # q1
    assert bin_index(2.0, 10.0, 5) == 1    # q2
    assert bin_index(3.99, 10.0, 5) == 1   # q2
    assert bin_index(4.0, 10.0, 5) == 2    # q3
    assert bin_index(6.0, 10.0, 5) == 3    # q4
    assert bin_index(8.0, 10.0, 5) == 4    # q5
    # Exactly at the upper bound clamps to q5 (not a sixth bin).
    assert bin_index(10.0, 10.0, 5) == 4
    # Zero / negative extent collapses to q1.
    assert bin_index(5.0, 0.0, 5) == 0


def test_per_coord_bin_uses_slicebench_bin_index_correctly(
    tool: Any, index: ManifestIndex, stub_extents: None,
) -> None:
    """Sanity: the per_coord_bin path's section-to-bin assignment matches
    what ``_bin_index`` returns directly."""
    # Horizontal extent = 8.0; positions [1.0, 2.5, 4.0] → bins [0, 1, 2].
    summary = {
        "per_coord_bin": {
            "horizontal": {
                "q1": {"mae_mm": 4.0},  # 0.5
                "q2": {"mae_mm": 2.0},  # 0.25
                "q3": {"mae_mm": 1.0},  # 0.125
            },
        },
    }
    rows = tool.build_rows(summary, index, source_label="slicebench")
    by_section = {
        (r["dataset"], r["section_id"]): r["difficulty_score"] for r in rows
    }
    # s00 @ 1.0 mm → bin 0 (q1) → 0.5
    assert by_section[("ds_a", "ds_a_s00")] == pytest.approx(0.5)
    assert by_section[("ds_b", "ds_b_s00")] == pytest.approx(0.5)
    # s01 @ 2.5 mm → bin 1 (q2) → 0.25
    assert by_section[("ds_a", "ds_a_s01")] == pytest.approx(0.25)
    # s02 @ 4.0 mm → bin 2 (q3) → 0.125
    assert by_section[("ds_a", "ds_a_s02")] == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# Defensive: per_section entries missing required fields are skipped silently
# (per the orphan-vs-invalid distinction)
# ---------------------------------------------------------------------------


def test_per_section_invalid_entries_are_counted_as_invalid(
    tool: Any,
    index: ManifestIndex,
    stub_extents: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Per-section rows that are non-dicts, missing mae_mm, or missing
    plane/dataset are skipped and counted in the invalid bucket."""
    summary = {
        "per_section": {
            "good": {
                "plane": "coronal", "dataset": "ds_a", "section_id": "ds_a_s00",
                "atlas": "allen_mouse_25um", "mae_mm": 1.0,
            },
            "missing_mae": {
                "plane": "coronal", "dataset": "ds_a", "section_id": "ds_a_s01",
                "atlas": "allen_mouse_25um",
            },
            "no_atlas": {
                "plane": "coronal", "dataset": "ds_a", "section_id": "ds_a_s02",
                "mae_mm": 1.0,
            },
            "garbage_string": "not-a-dict",  # type: ignore[dict-item]
        },
    }
    rows = tool.build_rows(
        summary, index, source_label="slicebench",
        summary_path_for_logs="fake/summary.json",
    )
    assert len(rows) == 1
    assert rows[0]["section_id"] == "ds_a_s00"
    captured = capsys.readouterr()
    # 3 invalid: missing_mae, no_atlas, garbage_string.
    assert "3 invalid" in captured.err


# ---------------------------------------------------------------------------
# CLI smoke: --help works without raising
# ---------------------------------------------------------------------------


def test_cli_help_smoke() -> None:
    """``python tools/seed_difficulty_from_slicebench.py --help`` exits 0
    with a usage line — covers the argparse plumbing without exercising the
    rest of the pipeline."""
    proc = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--help"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "--slicebench-summary" in proc.stdout
    assert "--manifest-root" in proc.stdout
    assert "--output" in proc.stdout
    assert "--source-label" in proc.stdout
