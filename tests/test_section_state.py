"""Unit tests for ``single_turn_rl.section_state``.

The synthetic-manifest pattern from :mod:`test_manifest_index` is reused
so the tests exercise the real :class:`ManifestIndex` code path (Lane B
reads from it) without ever touching the production ``data/manifest/``
tree on disk. ``rlvr.atlas_grid``-driven snapping is verified in
isolation by stubbing the BrainGlobe range helper to a known
``(pos_lo, pos_hi)`` — no real atlas image files are required.

Coverage:

* :class:`SectionState` is frozen and round-trips through
  :meth:`from_section`.
* Slate generator returns the requested number of positions, snaps them
  to the 0.05 mm grid, and uses the documented repo-relative image-path
  convention.
* JSON persistence round-trips and creates parent directories on demand.
* :func:`get_or_build_canonical_slate` skips the build when the JSON is
  cached and writes through when it isn't.
* :func:`iter_section_states` carries per-section ``difficulty_score``,
  honours the ``require_*_on_disk`` guards, and stays under tmp_path.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from single_turn_rl import section_state as ss
from single_turn_rl.manifest_index import ManifestIndex

# ---------------------------------------------------------------------------
# Synthetic manifest (mirrors test_manifest_index._build_manifest_tree)
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
) -> dict[str, Any]:
    axis = {"coronal": "ap", "sagittal": "ml", "horizontal": "dv"}[plane]
    return {
        "image_path": image_path
        or f"data/datasets/{plane}/{dataset}/{subject_id}/{section_id}.png",
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
    """Build a 3-plane, 2-dataset synthetic manifest (parallel to the
    Lane A index test). Positions are spaced so query() returns a
    well-defined ordered list."""
    manifest_root = tmp_path / "manifest"
    shards_root = manifest_root / "shards"
    plane_positions = {
        "coronal": [0.5, 1.5, 3.0],
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
                    subject_id=f"subj_{dataset}",
                )
                for i in range(3)
            ]
            _write_jsonl(shards_root / plane / f"{dataset}.jsonl", rows)

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
        _write_jsonl(alloc_root / plane / "sft.jsonl", [])
        _write_jsonl(alloc_root / plane / "eval.jsonl", [])
    return manifest_root


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> Path:
    return _build_manifest_tree(tmp_path)


@pytest.fixture
def index(synthetic_manifest: Path, tmp_path: Path) -> ManifestIndex:
    # Pin repo_root to tmp_path so any sidecar writes from the index stay
    # under tmp_path (the Lane A guard pattern).
    return ManifestIndex.from_manifest_root(synthetic_manifest, repo_root=tmp_path)


# ---------------------------------------------------------------------------
# Atlas-range stub (avoids loading BrainGlobe atlases in unit tests)
# ---------------------------------------------------------------------------


def _stub_atlas_range(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> None:
    cache: dict[tuple[str, str], tuple[float, float]] = {
        ("allen_mouse_25um", "coronal"): (0.0, 13.2),
        ("allen_mouse_25um", "sagittal"): (0.0, 6.0),
        ("allen_mouse_25um", "horizontal"): (0.0, 8.0),
    }
    if extra is not None:
        cache.update(extra)
    monkeypatch.setattr(
        ss, "_atlas_valid_range_mm", lambda atlas, plane: cache[(atlas, plane)]
    )


def _seed_atlas_files(
    base: Path, slate: ss.CanonicalSlate
) -> None:
    """Create empty stub files for each slate path under ``base``."""
    for rel in slate.image_paths:
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


# ---------------------------------------------------------------------------
# Tmp-path containment guard (mirrors test_manifest_index)
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_MANIFEST = REPO_ROOT / "data" / "manifest"


def _assert_under_tmp(path: Path, tmp_path: Path) -> None:
    abs_path = path.resolve()
    assert tmp_path.resolve() in abs_path.parents or abs_path == tmp_path.resolve(), (
        f"Path {abs_path} escaped tmp_path {tmp_path}"
    )


# ---------------------------------------------------------------------------
# 1. SectionState dataclass shape + from_section
# ---------------------------------------------------------------------------


def test_section_state_dataclass_is_frozen_and_has_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SectionState is frozen and exposes the spec'd field set."""
    _stub_atlas_range(monkeypatch)
    state = ss.SectionState(
        section_id="s00",
        subject_id="subj",
        atlas_name="allen_mouse_25um",
        plane="coronal",
        dataset="ds_a",
        valid_range_mm=(0.0, 13.2),
        ground_truth_mm=4.37,
        query_image_path="data/datasets/coronal/ds_a/subj/s00.png",
        atlas_slate_paths=(
            "models/langslice-gemma-4/data/atlas/"
            "allen_mouse_25um/coronal/0.65mm.jpg",
        ),
        atlas_slate_positions_mm=(0.65,),
        difficulty_score=0.42,
    )
    # Frozen → assignment raises.
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.ground_truth_mm = 1.0  # type: ignore[misc]
    # Field set exactly matches the spec (catches accidental drift).
    # ``source`` / ``quality`` are Task 4 additions for the randomized
    # Lane B path; both default empty so the deterministic constructor
    # above doesn't need to pass them.
    expected_fields = {
        "section_id",
        "subject_id",
        "atlas_name",
        "plane",
        "dataset",
        "valid_range_mm",
        "ground_truth_mm",
        "query_image_path",
        "atlas_slate_paths",
        "atlas_slate_positions_mm",
        "difficulty_score",
        "source",
        "quality",
    }
    assert {f.name for f in dataclasses.fields(state)} == expected_fields


def test_from_section_builds_correctly(
    index: ManifestIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """from_section copies fields verbatim and uses the cached valid range."""
    _stub_atlas_range(monkeypatch)
    section = index.section_by_key("coronal", "ds_a", "ds_a_s00")
    assert section is not None
    slate = ss.build_canonical_slate(
        section.atlas, section.plane, n_positions=3,
    )
    state = ss.SectionState.from_section(section, slate=slate, difficulty_score=0.3)
    assert state.section_id == section.section_id
    assert state.subject_id == section.subject_id
    assert state.dataset == section.dataset
    assert state.atlas_name == section.atlas
    assert state.plane == section.plane
    assert state.ground_truth_mm == pytest.approx(section.position_mm)
    assert state.query_image_path == section.image_path
    assert state.valid_range_mm == (0.0, 13.2)
    assert state.atlas_slate_paths == slate.image_paths
    assert state.atlas_slate_positions_mm == slate.positions_mm
    assert state.difficulty_score == pytest.approx(0.3)


def test_from_section_difficulty_none_passthrough(
    index: ManifestIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """from_section preserves None difficulty (cold-start sections)."""
    _stub_atlas_range(monkeypatch)
    section = index.section_by_key("coronal", "ds_a", "ds_a_s00")
    assert section is not None
    slate = ss.build_canonical_slate(
        section.atlas, section.plane, n_positions=3,
    )
    state = ss.SectionState.from_section(section, slate=slate, difficulty_score=None)
    assert state.difficulty_score is None


# ---------------------------------------------------------------------------
# 2-3. build_canonical_slate position counts
# ---------------------------------------------------------------------------


def test_build_canonical_slate_returns_nine_positions_evenly_spaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``n_positions=9`` over [0, 13.2] yields 9 evenly-spaced positions
    (within a 0.05 mm snap tolerance)."""
    _stub_atlas_range(monkeypatch)
    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=9,
    )
    assert len(slate.positions_mm) == 9
    # Pre-snap step is 13.2/8 = 1.65 mm; post-snap step is within
    # ±0.05 mm of that (snap rounds to the nearest grid point).
    expected_step = 13.2 / 8
    deltas = [
        slate.positions_mm[i + 1] - slate.positions_mm[i] for i in range(8)
    ]
    for d in deltas:
        assert abs(d - expected_step) <= ss._GRID_STEP_MM + 1e-9
    # First/last bracket the valid range (snapped inward to stay in-range).
    assert slate.positions_mm[0] >= 0.0
    assert slate.positions_mm[-1] <= 13.2


def test_build_canonical_slate_returns_five_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_range(monkeypatch)
    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=5,
    )
    assert len(slate.positions_mm) == 5
    expected_step = 13.2 / 4
    for i in range(4):
        d = slate.positions_mm[i + 1] - slate.positions_mm[i]
        assert abs(d - expected_step) <= ss._GRID_STEP_MM + 1e-9


# ---------------------------------------------------------------------------
# 4. Snap-to-grid invariant
# ---------------------------------------------------------------------------


def test_build_canonical_slate_positions_snap_to_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every slate position is a multiple of 0.05 mm.

    The on-disk pre-rendered atlas grid only exists at 0.05 mm steps;
    a slate position off the grid would reference a missing file.
    """
    _stub_atlas_range(monkeypatch)
    # Pick a range whose endpoints are NOT round multiples of 0.05 so
    # the snap step is observable.
    extra = {("synthetic", "coronal"): (0.0, 12.7)}
    _stub_atlas_range(monkeypatch, extra=extra)
    slate = ss.build_canonical_slate("synthetic", "coronal", n_positions=9)
    for pos in slate.positions_mm:
        # Every position is a 0.05-mm multiple: pos / step ≈ integer.
        ratio = pos / ss._GRID_STEP_MM
        assert abs(ratio - round(ratio)) < 1e-6, (
            f"Position {pos} is not on the 0.05 mm grid"
        )


# ---------------------------------------------------------------------------
# 5. Image-path naming convention
# ---------------------------------------------------------------------------


def test_build_canonical_slate_image_paths_use_repo_relative_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_range(monkeypatch)
    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "sagittal", n_positions=4,
    )
    assert len(slate.image_paths) == 4
    for pos, path in zip(slate.positions_mm, slate.image_paths, strict=True):
        # Repo-relative convention (canonical per
        # langslice_traces.CANONICAL_ATLAS_ROOT):
        # models/langslice-gemma-4/data/atlas/<atlas>/<plane>/<X.XX>mm.jpg
        expected = (
            f"models/langslice-gemma-4/data/atlas/allen_mouse_25um/"
            f"sagittal/{pos:.2f}mm.jpg"
        )
        assert path == expected


# ---------------------------------------------------------------------------
# 6. save_canonical_slate writes spec'd JSON shape + creates parents
# ---------------------------------------------------------------------------


def test_save_canonical_slate_writes_spec_json_and_creates_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_atlas_range(monkeypatch)
    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=9,
    )
    root = tmp_path / "data_root"  # parents do NOT exist yet
    target = ss.save_canonical_slate(slate, root)
    expected = root / "atlas" / "allen_mouse_25um" / "coronal" / "_slate_9.json"
    assert target == expected
    assert target.is_file()
    _assert_under_tmp(target, tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["atlas_name"] == "allen_mouse_25um"
    assert payload["plane"] == "coronal"
    assert payload["n_positions"] == 9
    assert payload["positions_mm"] == list(slate.positions_mm)
    assert payload["image_paths"] == list(slate.image_paths)
    assert payload["valid_range_mm"] == [0.0, 13.2]


# ---------------------------------------------------------------------------
# 7-8. load_canonical_slate round-trip + missing-file → None
# ---------------------------------------------------------------------------


def test_load_canonical_slate_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_atlas_range(monkeypatch)
    original = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=9,
    )
    ss.save_canonical_slate(original, tmp_path)
    loaded = ss.load_canonical_slate(
        tmp_path, "allen_mouse_25um", "coronal", n_positions=9,
    )
    assert loaded is not None
    assert loaded.atlas_name == original.atlas_name
    assert loaded.plane == original.plane
    assert loaded.positions_mm == original.positions_mm
    assert loaded.image_paths == original.image_paths


def test_load_canonical_slate_returns_none_when_missing(tmp_path: Path) -> None:
    out = ss.load_canonical_slate(
        tmp_path, "allen_mouse_25um", "coronal", n_positions=9,
    )
    assert out is None


# ---------------------------------------------------------------------------
# 9-10. get_or_build_canonical_slate cache hit / miss
# ---------------------------------------------------------------------------


def test_get_or_build_canonical_slate_uses_cache_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the JSON is on disk, subsequent calls do not call build_canonical_slate."""
    _stub_atlas_range(monkeypatch)
    # Prime the cache via a normal build+save round-trip.
    primed = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=5,
    )
    ss.save_canonical_slate(primed, tmp_path)

    call_count = {"n": 0}

    def _spy_build(*args: Any, **kwargs: Any) -> ss.CanonicalSlate:
        call_count["n"] += 1
        return primed

    monkeypatch.setattr(ss, "build_canonical_slate", _spy_build)
    out = ss.get_or_build_canonical_slate(
        "allen_mouse_25um", "coronal", root=tmp_path, n_positions=5,
    )
    assert out.positions_mm == primed.positions_mm
    assert call_count["n"] == 0, (
        "build_canonical_slate should not have been called when the cache is warm"
    )


def test_get_or_build_canonical_slate_builds_and_saves_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_atlas_range(monkeypatch)
    # Cache deliberately empty.
    expected_path = (
        tmp_path / "atlas" / "allen_mouse_25um" / "coronal" / "_slate_5.json"
    )
    assert not expected_path.exists()
    out = ss.get_or_build_canonical_slate(
        "allen_mouse_25um", "coronal", root=tmp_path, n_positions=5,
    )
    assert len(out.positions_mm) == 5
    # Should have been persisted as a side effect.
    assert expected_path.is_file()
    _assert_under_tmp(expected_path, tmp_path)
    payload = json.loads(expected_path.read_text(encoding="utf-8"))
    assert payload["n_positions"] == 5


# ---------------------------------------------------------------------------
# 11-12. iter_section_states yields one row per section + carries difficulty
# ---------------------------------------------------------------------------


def _seed_query_files(base: Path, index: ManifestIndex, plane: str) -> None:
    """Create empty stub files for every section's query image under ``base``."""
    for section in index.query(plane=plane):
        target = base / section.image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")


def test_iter_section_states_yields_one_per_matching_section(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    # Build slate first so we can pre-seed the atlas image stubs.
    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=5,
    )
    _seed_atlas_files(tmp_path, slate)

    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            n_positions=5,
            repo_root=tmp_path,
        )
    )
    coronal_sections = index.query(plane="coronal", atlas="allen_mouse_25um")
    assert len(states) == len(coronal_sections)
    # Composite key parity (state.dataset is unique to SectionState; the
    # rest comes from Section).
    state_keys = {(s.plane, s.dataset, s.section_id) for s in states}
    section_keys = {(s.plane, s.dataset, s.section_id) for s in coronal_sections}
    assert state_keys == section_keys


def test_iter_section_states_carries_difficulty_score(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=3,
    )
    _seed_atlas_files(tmp_path, slate)
    # Stamp a difficulty on one section; leave the others at None.
    index.update_difficulty("coronal", "ds_a", "ds_a_s00", 0.42, "test")

    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            n_positions=3,
            repo_root=tmp_path,
        )
    )
    by_key = {(s.dataset, s.section_id): s for s in states}
    stamped = by_key[("ds_a", "ds_a_s00")]
    assert stamped.difficulty_score == pytest.approx(0.42)
    other = by_key[("ds_a", "ds_a_s01")]
    assert other.difficulty_score is None


# ---------------------------------------------------------------------------
# 13. require_query_on_disk drops sections with missing query images
# ---------------------------------------------------------------------------


def test_iter_section_states_skips_when_query_image_missing(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_atlas_range(monkeypatch)
    # Seed query images for every section EXCEPT one.
    coronal = index.query(plane="coronal", atlas="allen_mouse_25um")
    skipped = coronal[0]
    for section in coronal:
        if section.section_id == skipped.section_id and section.dataset == skipped.dataset:
            continue
        target = tmp_path / section.image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")

    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=3,
    )
    _seed_atlas_files(tmp_path, slate)

    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            n_positions=3,
            require_query_on_disk=True,
            repo_root=tmp_path,
        )
    )
    yielded_keys = {(s.dataset, s.section_id) for s in states}
    assert (skipped.dataset, skipped.section_id) not in yielded_keys
    assert len(states) == len(coronal) - 1

    # And: when the guard is OFF, the missing-file row is included again.
    states_unfiltered = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            n_positions=3,
            require_query_on_disk=False,
            repo_root=tmp_path,
        )
    )
    assert len(states_unfiltered) == len(coronal)


# ---------------------------------------------------------------------------
# 14. require_atlas_on_disk drops everything when slate image missing
# ---------------------------------------------------------------------------


def test_iter_section_states_skips_when_atlas_slate_image_missing(
    index: ManifestIndex, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    # Deliberately do NOT seed any atlas slate image files.
    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            n_positions=3,
            require_atlas_on_disk=True,
            repo_root=tmp_path,
        )
    )
    assert states == []
    captured = capsys.readouterr()
    assert "atlas slate incomplete" in captured.err

    # Turn the guard off → all sections yield (slate paths reference
    # missing files but nobody verifies that).
    states_unfiltered = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            n_positions=3,
            require_atlas_on_disk=False,
            require_query_on_disk=False,
            repo_root=tmp_path,
        )
    )
    coronal = index.query(plane="coronal", atlas="allen_mouse_25um")
    assert len(states_unfiltered) == len(coronal)


# ---------------------------------------------------------------------------
# 15. tmp-path containment guard (no real data/manifest touched)
# ---------------------------------------------------------------------------


def test_iter_section_states_does_not_touch_real_manifest(
    synthetic_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The yielded SectionStates must reference only tmp_path-relative
    image paths — a leak from the production manifest would surface as
    a path that doesn't share the tmp_path prefix once resolved."""
    _stub_atlas_range(monkeypatch)
    idx = ManifestIndex.from_manifest_root(synthetic_manifest, repo_root=tmp_path)
    _seed_query_files(tmp_path, idx, "coronal")
    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=3,
    )
    _seed_atlas_files(tmp_path, slate)

    states = list(
        ss.iter_section_states(
            index=idx,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            n_positions=3,
            repo_root=tmp_path,
        )
    )
    assert states, "expected at least one yielded section state"
    for state in states:
        # Synthetic image_path strings are tmp-relative by construction;
        # a path under PROD_MANIFEST means the loader silently fell back.
        assert not state.query_image_path.startswith(str(PROD_MANIFEST))
        # Resolved query path must live under tmp_path.
        _assert_under_tmp((tmp_path / state.query_image_path), tmp_path)
        # Atlas slate paths must be repo-relative under the canonical
        # models/langslice-gemma-4/data/atlas/ tree so the trainer's
        # repo_root / p resolver finds the on-disk tile.
        for p in state.atlas_slate_paths:
            assert p.startswith("models/langslice-gemma-4/data/atlas/"), (
                f"slate path {p} doesn't follow canonical atlas convention"
            )

    # And the slate JSON cache landed under tmp_path too.
    cache_dir = tmp_path / "atlas"
    _assert_under_tmp(cache_dir, tmp_path)
    assert any(cache_dir.rglob("_slate_*.json"))
