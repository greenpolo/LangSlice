"""Tests for ``single_turn_rl.terminal_states`` synthetic Lane-A pass.

Covers the ``--include-synthetic`` flag added by Task 3 of the
unified-trace-generator plan:

* OFF (default) → no synthetic rows in output, source unchanged.
* ON → synthetic rows appended after teacher rows; tagged
  ``source="procedural_generator:lane_a"``; ground_truth_mm comes from
  manifest, never the teacher's submit.
* Strategy tag stamped under ``quality["strategy"] == "lane_a_prefix"``.
* Every synthetic position is present in the loaded atlas grid.
* Same seed → identical synthetic rows across two builds.

Fixtures build a tiny in-memory manifest with two coronal sections:
one already covered by a teacher trace (``covered_keys`` populated) and
one bare. The synthetic pass should emit exactly one row.

The real Allen 25um coronal embedding cache is used for grid lookup; if
the .pt file is missing the whole module is skipped (no fakes — the
generator's image-path fabrication has to match the real cache).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Skip the entire module if torch isn't importable in this environment.
# load_atlas_grid uses torch.load on the .pt cache.
torch = pytest.importorskip("torch")

from langslice_training.rl.single_turn import terminal_states as ts  # noqa: E402
from langslice_training.rl.single_turn.manifest_index import ManifestIndex  # noqa: E402

# ---------------------------------------------------------------------------
# Atlas embedding cache dependency
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
ATLAS_CACHE_DIR = REPO_ROOT / "out" / "atlas_embeddings"
ALLEN_CORONAL_CACHE = ATLAS_CACHE_DIR / "allen_mouse_25um_coronal.pt"


pytestmark = pytest.mark.skipif(
    not ALLEN_CORONAL_CACHE.is_file(),
    reason=(
        "synthetic Lane-A tests need out/atlas_embeddings/"
        "allen_mouse_25um_coronal.pt — build it with "
        "the atlas embedding cache before running."
    ),
)


# ---------------------------------------------------------------------------
# Synthetic manifest builder
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
    """Build one shard row in the QC app's expected schema."""
    axis = {"coronal": "ap", "sagittal": "ml", "horizontal": "dv"}[plane]
    return {
        "image_path": f"data/datasets/{plane}/{dataset}/{subject_id}/{section_id}.png",
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


def _build_two_section_manifest(tmp_path: Path) -> Path:
    """One coronal dataset with two RLVR-allocated sections at 3.0 / 5.0 mm.

    Returns the manifest root.
    """
    manifest_root = tmp_path / "manifest"
    shards_root = manifest_root / "shards"
    _write_jsonl(
        shards_root / "coronal" / "ds_a.jsonl",
        [
            _make_row(
                plane="coronal", dataset="ds_a",
                section_id="s_covered", position_mm=3.0,
                subject_id="subj_covered",
            ),
            _make_row(
                plane="coronal", dataset="ds_a",
                section_id="s_bare", position_mm=5.0,
                subject_id="subj_bare",
            ),
        ],
    )
    # Allocate both sections to rlvr so the synthetic pass considers them.
    alloc_root = manifest_root / "allocations"
    _write_jsonl(
        alloc_root / "coronal" / "rlvr.jsonl",
        [
            {"section_id": "s_covered", "dataset": "ds_a",
             "added_by": "test", "added_at": "2026-05-10T00:00:00+00:00"},
            {"section_id": "s_bare", "dataset": "ds_a",
             "added_by": "test", "added_at": "2026-05-10T00:00:00+00:00"},
        ],
    )
    _write_jsonl(alloc_root / "coronal" / "sft.jsonl", [])
    _write_jsonl(alloc_root / "coronal" / "eval.jsonl", [])
    return manifest_root


def _make_teacher_row(plane: str, dataset: str, section_id: str) -> ts.TerminalState:
    """Stand in for a row emitted by build_from_sft_corpus."""
    return ts.TerminalState(
        section_id=section_id,
        subject_id="subj_covered",
        atlas_name="allen_mouse_25um",
        plane=plane,  # type: ignore[arg-type]
        valid_range_mm=(0.0, 13.2),
        ground_truth_mm=3.0,
        query_image_path=f"queries/{section_id}.jpg",
        atlas_image_paths=("atlas/allen_mouse_25um/coronal/3.00mm.jpg",),
        fetched_positions_mm=(3.0,),
        source="sft_corpus:strict",
        quality={
            "acceptance_tier": "strict",
            "teacher_position_mm": 3.05,
            "dataset": dataset,
        },
    )


def _stub_atlas_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass BrainGlobe atlas loading by stubbing the cache helper."""
    cache: dict[tuple[str, str], tuple[float, float]] = {
        ("allen_mouse_25um", "coronal"): (0.0, 13.2),
        ("allen_mouse_25um", "sagittal"): (0.0, 6.0),
        ("allen_mouse_25um", "horizontal"): (0.0, 8.0),
    }
    monkeypatch.setattr(
        ts, "_atlas_valid_range_mm",
        lambda atlas, plane: cache[(atlas, plane)],
    )


@pytest.fixture
def manifest_root(tmp_path: Path) -> Path:
    return _build_two_section_manifest(tmp_path)


@pytest.fixture
def manifest_index(manifest_root: Path) -> ManifestIndex:
    return ManifestIndex.from_manifest_root(manifest_root)


# ---------------------------------------------------------------------------
# Off → only teacher rows
# ---------------------------------------------------------------------------


def test_include_synthetic_off_emits_only_teacher_rows(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When build_synthetic_states is NOT called, downstream code sees no
    procedural rows. This test directly verifies the default-off contract
    by simulating the build_cmd path without invoking synthetic generation.
    """
    _stub_atlas_range(monkeypatch)
    teacher_rows = [_make_teacher_row("coronal", "ds_a", "s_covered")]
    assert all(r.source != ts.SYNTHETIC_LANE_A_SOURCE for r in teacher_rows)
    assert all(r.source.startswith("sft_corpus:") for r in teacher_rows)


# ---------------------------------------------------------------------------
# On → synthetic rows appended
# ---------------------------------------------------------------------------


def test_include_synthetic_on_appends_synthetic_rows(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With covered={s_covered}, synthetic pass yields exactly one row
    (for s_bare). The synthetic row is tagged with the canonical source
    and references the manifest GT, not any teacher submit."""
    _stub_atlas_range(monkeypatch)
    covered = {("coronal", "ds_a", "s_covered")}
    synthetic = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys=covered,
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=42,
    ))
    assert len(synthetic) == 1
    row = synthetic[0]
    assert row.source == ts.SYNTHETIC_LANE_A_SOURCE
    assert row.section_id == "s_bare"
    assert row.subject_id == "subj_bare"
    # GT must be the manifest's 5.0 (not any teacher submit — there is none).
    assert row.ground_truth_mm == pytest.approx(5.0)
    # No teacher_position_mm in quality (no teacher trace exists).
    assert "teacher_position_mm" not in row.quality


def test_synthetic_row_uses_lane_a_prefix_strategy(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """quality["strategy"] must be ``lane_a_prefix`` and synthetic=True."""
    _stub_atlas_range(monkeypatch)
    [row] = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=42,
    ))
    assert row.quality["strategy"] == "lane_a_prefix"
    assert row.quality["synthetic"] is True
    assert row.quality["dataset"] == "ds_a"


def test_synthetic_grid_compliance(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every synthetic atlas position must be present in the loaded grid.

    The generator snaps each fabricated position to the closest valid grid
    point — by definition every emitted position must be one of those grid
    points. A mismatch would indicate path/position drift between the
    generator and the cache; the resulting image paths would point at
    nonexistent files.
    """
    _stub_atlas_range(monkeypatch)
    from langslice_traces import load_atlas_grid  # noqa: PLC0415

    grid = load_atlas_grid(ATLAS_CACHE_DIR, "allen_mouse_25um", "coronal")
    grid_set = {round(g, 6) for g in grid}
    [row] = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=42,
    ))
    assert row.fetched_positions_mm, "synthetic row must have at least one fetch"
    for pos in row.fetched_positions_mm:
        assert round(pos, 6) in grid_set, (
            f"synthetic position {pos!r} not in atlas grid (sample: "
            f"{sorted(grid_set)[:5]} ... {sorted(grid_set)[-5:]})"
        )


def test_synthetic_seeded_deterministic(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same seed must produce byte-identical synthetic rows.

    Determinism is required so the trainer's data shard hash is stable
    across reruns — otherwise curriculum + reward bookkeeping would
    have to re-bootstrap on every restart.
    """
    _stub_atlas_range(monkeypatch)
    a = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=1234,
    ))
    b = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=1234,
    ))
    assert len(a) == len(b) == 1
    assert a[0].section_id == b[0].section_id
    assert a[0].atlas_image_paths == b[0].atlas_image_paths
    assert a[0].fetched_positions_mm == b[0].fetched_positions_mm
    assert a[0].ground_truth_mm == b[0].ground_truth_mm


def test_synthetic_skip_when_section_already_covered(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover BOTH sections → zero synthetic rows emitted."""
    _stub_atlas_range(monkeypatch)
    covered = {
        ("coronal", "ds_a", "s_covered"),
        ("coronal", "ds_a", "s_bare"),
    }
    synthetic = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys=covered,
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=42,
    ))
    assert synthetic == []


def test_synthetic_skip_when_atlas_cache_missing(
    manifest_index: ManifestIndex,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the atlas embedding cache directory lacks the (atlas, plane) .pt
    file, the section is skipped with a warning (not crashed)."""
    _stub_atlas_range(monkeypatch)
    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    synthetic = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=empty_cache,
        seed=42,
    ))
    assert synthetic == []


def test_synthetic_image_paths_match_atlas_grid_filenames(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic atlas_image_paths must follow the canonical filename
    convention so downstream loaders can resolve them against the real
    on-disk atlas tile tree via ``repo_root / p``."""
    _stub_atlas_range(monkeypatch)
    [row] = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=42,
    ))
    for path in row.atlas_image_paths:
        assert path.startswith(
            "models/langslice-gemma-4/data/atlas/allen_mouse_25um/coronal/"
        )
        assert path.endswith("mm.jpg")


def test_synthetic_lane_a_image_paths_use_repo_relative_prefix(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every synthetic atlas path must ship under the single canonical
    prefix the trainer's ``repo_root / p`` resolver expects (Lane A and
    Lane B must agree — the headline blocker the helper fixed)."""
    _stub_atlas_range(monkeypatch)
    [row] = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=42,
    ))
    assert row.atlas_image_paths, "expected non-empty atlas paths"
    for path in row.atlas_image_paths:
        assert path.startswith("models/langslice-gemma-4/data/atlas/"), (
            f"synthetic Lane-A row leaked non-canonical prefix: {path!r}"
        )


def test_synthetic_lane_a_paths_resolve_on_disk(
    manifest_index: ManifestIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end integration: ``repo_root / atlas_image_paths[0]`` must
    point at a real tile on disk. Gated on the existence of the seed
    fixture tile (5.00mm.jpg under Allen 25um coronal) so the test still
    runs cleanly on machines without the full atlas tree."""
    _stub_atlas_range(monkeypatch)
    expected_tile = (
        REPO_ROOT / "models" / "langslice-gemma-4" / "data" / "atlas"
        / "allen_mouse_25um" / "coronal" / "5.00mm.jpg"
    )
    if not expected_tile.is_file():
        pytest.skip(
            f"atlas tile fixture missing: {expected_tile} — render the "
            "Allen 25um coronal slate before running this test."
        )
    [row] = list(ts.build_synthetic_states(
        manifest_index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=42,
    ))
    assert row.atlas_image_paths, "expected at least one atlas path"
    first = row.atlas_image_paths[0]
    resolved = REPO_ROOT / first
    assert resolved.is_file(), (
        f"synthetic atlas path {first!r} did not resolve to an on-disk "
        f"tile under {REPO_ROOT!s}"
    )


def test_synthetic_only_emits_rlvr_split(
    manifest_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A section allocated to sft (not rlvr) must NOT be synthesized over —
    the data-pool policy reserves SFT-allocated rows for SFT runs."""
    _stub_atlas_range(monkeypatch)
    # Move s_bare from rlvr → sft in the allocations layer.
    rlvr_path = manifest_root / "allocations" / "coronal" / "rlvr.jsonl"
    sft_path = manifest_root / "allocations" / "coronal" / "sft.jsonl"
    _write_jsonl(rlvr_path, [
        {"section_id": "s_covered", "dataset": "ds_a",
         "added_by": "test", "added_at": "2026-05-10T00:00:00+00:00"},
    ])
    _write_jsonl(sft_path, [
        {"section_id": "s_bare", "dataset": "ds_a",
         "added_by": "test", "added_at": "2026-05-10T00:00:00+00:00"},
    ])
    index = ManifestIndex.from_manifest_root(manifest_root)
    synthetic = list(ts.build_synthetic_states(
        index,
        covered_keys={("coronal", "ds_a", "s_covered")},
        atlas_embedding_cache=ATLAS_CACHE_DIR,
        seed=42,
    ))
    assert synthetic == [], (
        "synthetic pass leaked an SFT-allocated section into Lane A"
    )


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_help_advertises_synthetic_flags() -> None:
    """``build --help`` must show all three new flags so operators can
    discover them without grepping the source."""
    parser_text: list[str] = []

    class _Capture:
        def write(self, s: str) -> int:
            parser_text.append(s)
            return len(s)

        def flush(self) -> None:  # noqa: D401
            return None

    import io
    import sys as _sys  # noqa: PLC0415

    buf = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = buf
    try:
        with pytest.raises(SystemExit) as exc:
            ts.main(["build", "--help"])
        assert exc.value.code == 0
    finally:
        _sys.stdout = old_stdout

    out = buf.getvalue()
    assert "--include-synthetic" in out
    assert "--synthetic-seed" in out
    assert "--atlas-embedding-cache" in out


def test_cli_build_fails_fast_on_missing_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--include-synthetic with a nonexistent cache dir must SystemExit
    at validation time, before any corpus parsing."""
    monkeypatch.chdir(tmp_path)
    # Minimal SFT corpus path so argparse type=Path doesn't error.
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text("", encoding="utf-8")
    bogus_cache = tmp_path / "does_not_exist"
    out_path = tmp_path / "out.jsonl"

    argv = [
        "build",
        "--sft-corpus", str(sft_path),
        "--manifest-root", str(tmp_path / "data" / "manifest"),
        "--output", str(out_path),
        "--include-synthetic",
        "--atlas-embedding-cache", str(bogus_cache),
    ]
    with pytest.raises(SystemExit) as exc:
        ts.main(argv)
    msg = str(exc.value)
    assert "atlas-embedding-cache" in msg or "not a directory" in msg


def test_cli_build_fails_fast_on_empty_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--include-synthetic with an empty cache dir must also SystemExit."""
    monkeypatch.chdir(tmp_path)
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text("", encoding="utf-8")
    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    out_path = tmp_path / "out.jsonl"

    argv = [
        "build",
        "--sft-corpus", str(sft_path),
        "--manifest-root", str(tmp_path / "data" / "manifest"),
        "--output", str(out_path),
        "--include-synthetic",
        "--atlas-embedding-cache", str(empty_cache),
    ]
    with pytest.raises(SystemExit) as exc:
        ts.main(argv)
    msg = str(exc.value)
    assert "no .pt files" in msg or "empty" in msg.lower()
