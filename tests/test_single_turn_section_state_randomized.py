"""Tests for ``single_turn_rl.section_state`` randomized Lane-B path.

Covers the ``--randomized-slate`` flag added by Task 4 of the
unified-trace-generator plan:

* OFF (default) → deterministic canonical slate; no procedural-generator
  ``source`` tag on emitted rows.
* ON → every section gets a fresh randomized Lane B slate, tagged
  ``source="procedural_generator:lane_b"`` and stamped with
  ``quality["strategy"] == "lane_b_broad_slate"``.
* Bracket invariant: ``min(positions) <= gt <= max(positions)`` for every
  emitted row (the generator's hard contract).
* Off-center bias: ~Uniform(0.1, 0.9) GT fraction across many rows.
* Grid compliance: every position is an exact element of the loaded grid.
* Determinism: same seed → identical slates.
* Graceful skip when the atlas cache file is missing.

The real Allen 25um coronal embedding cache is used for grid lookup; the
randomized-Lane-B tests that need it are guarded per-test with
``pytest.skipif`` so CLI parse-time tests still run when the cache is
absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Defer torch import to test bodies — needed by load_atlas_grid but not by
# the CLI-help / parse-time tests.

from single_turn_rl import section_state as ss
from single_turn_rl.manifest_index import ManifestIndex

# ---------------------------------------------------------------------------
# Atlas embedding cache dependency (per-test skip, not module-wide)
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
ATLAS_CACHE_DIR = REPO_ROOT / "out" / "atlas_embeddings"
ALLEN_CORONAL_CACHE = ATLAS_CACHE_DIR / "allen_mouse_25um_coronal.pt"

_requires_atlas_cache = pytest.mark.skipif(
    not ALLEN_CORONAL_CACHE.is_file(),
    reason=(
        "randomized Lane-B tests need out/atlas_embeddings/"
        "allen_mouse_25um_coronal.pt — build it with "
        "the atlas embedding cache before running."
    ),
)


# ---------------------------------------------------------------------------
# Synthetic manifest fixture
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


def _build_three_section_manifest(tmp_path: Path) -> Path:
    """Three coronal RLVR sections at well-spaced positions so the
    randomized slate has room to draw on either side of GT."""
    manifest_root = tmp_path / "manifest"
    shards_root = manifest_root / "shards"
    rows = [
        _make_row(
            plane="coronal", dataset="ds_a",
            section_id="s00", position_mm=4.0, subject_id="subj_a",
        ),
        _make_row(
            plane="coronal", dataset="ds_a",
            section_id="s01", position_mm=6.5, subject_id="subj_b",
        ),
        _make_row(
            plane="coronal", dataset="ds_a",
            section_id="s02", position_mm=9.0, subject_id="subj_c",
        ),
    ]
    _write_jsonl(shards_root / "coronal" / "ds_a.jsonl", rows)

    alloc_root = manifest_root / "allocations"
    _write_jsonl(
        alloc_root / "coronal" / "rlvr.jsonl",
        [
            {"section_id": "s00", "dataset": "ds_a",
             "added_by": "test", "added_at": "2026-05-10T00:00:00+00:00"},
            {"section_id": "s01", "dataset": "ds_a",
             "added_by": "test", "added_at": "2026-05-10T00:00:00+00:00"},
            {"section_id": "s02", "dataset": "ds_a",
             "added_by": "test", "added_at": "2026-05-10T00:00:00+00:00"},
        ],
    )
    _write_jsonl(alloc_root / "coronal" / "sft.jsonl", [])
    _write_jsonl(alloc_root / "coronal" / "eval.jsonl", [])
    return manifest_root


def _stub_atlas_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass BrainGlobe atlas loading by stubbing the cache helper."""
    cache: dict[tuple[str, str], tuple[float, float]] = {
        ("allen_mouse_25um", "coronal"): (0.0, 13.2),
        ("allen_mouse_25um", "sagittal"): (0.0, 6.0),
        ("allen_mouse_25um", "horizontal"): (0.0, 8.0),
    }
    monkeypatch.setattr(
        ss, "_atlas_valid_range_mm", lambda atlas, plane: cache[(atlas, plane)]
    )


def _seed_query_files(base: Path, index: ManifestIndex, plane: str) -> None:
    for section in index.query(plane=plane):
        target = base / section.image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")


@pytest.fixture
def manifest_root(tmp_path: Path) -> Path:
    return _build_three_section_manifest(tmp_path)


@pytest.fixture
def index(manifest_root: Path, tmp_path: Path) -> ManifestIndex:
    return ManifestIndex.from_manifest_root(manifest_root, repo_root=tmp_path)


# ---------------------------------------------------------------------------
# randomized=False keeps the deterministic source tag
# ---------------------------------------------------------------------------


def test_randomized_off_emits_deterministic_slates(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With randomized=False the iterator follows the existing canonical
    slate path and the procedural source tag is never stamped."""
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    slate = ss.build_canonical_slate(
        "allen_mouse_25um", "coronal", n_positions=5,
    )
    for rel in slate.image_paths:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")

    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            n_positions=5,
            repo_root=tmp_path,
            randomized=False,
        )
    )
    assert states, "expected at least one deterministic state"
    for state in states:
        assert state.source != ss.RANDOMIZED_LANE_B_SOURCE
        # Default (deterministic) source is empty string.
        assert state.source == ""
        # All sections in this fixture share the same canonical slate.
        assert state.atlas_slate_positions_mm == slate.positions_mm


# ---------------------------------------------------------------------------
# randomized=True stamps the procedural-generator source/strategy tags
# ---------------------------------------------------------------------------


@_requires_atlas_cache
def test_randomized_on_emits_lane_b_strategy(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every randomized row carries the canonical source + strategy."""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    import random as _random

    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            repo_root=tmp_path,
            randomized=True,
            rng=_random.Random(42),
            atlas_embedding_cache_dir=ATLAS_CACHE_DIR,
        )
    )
    assert states, "expected randomized states for all 3 fixture sections"
    for state in states:
        assert state.source == ss.RANDOMIZED_LANE_B_SOURCE
        assert state.quality is not None
        assert state.quality["strategy"] == "lane_b_broad_slate"
        assert "gt_fraction_in_slate" in state.quality


# ---------------------------------------------------------------------------
# Bracket invariant: GT lives inside every slate
# ---------------------------------------------------------------------------


@_requires_atlas_cache
def test_randomized_slate_contains_gt(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For 50 generated rows the slate must bracket GT inclusive."""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    import random as _random

    rng = _random.Random(7)
    seen = 0
    # Loop over many independent rolls (re-draw the same 3 sections ~17×).
    for _ in range(17):
        for state in ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            repo_root=tmp_path,
            randomized=True,
            rng=rng,
            atlas_embedding_cache_dir=ATLAS_CACHE_DIR,
        ):
            positions = state.atlas_slate_positions_mm
            gt = state.ground_truth_mm
            assert positions, (
                f"randomized row for {state.section_id} has empty slate"
            )
            assert min(positions) <= gt <= max(positions), (
                f"GT {gt} not bracketed by slate {positions!r} "
                f"(section={state.section_id})"
            )
            seen += 1
            if seen >= 50:
                break
        if seen >= 50:
            break
    assert seen >= 50, f"only checked {seen} rows; need at least 50"


# ---------------------------------------------------------------------------
# Off-center bias: ~60% of rows have |gt_fraction - 0.5| > 0.1
# ---------------------------------------------------------------------------


@_requires_atlas_cache
def test_randomized_slate_gt_not_centered(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generator samples gt_fraction ~ Uniform(0.1, 0.9), so the
    expected fraction of |gt_fraction - 0.5| > 0.1 is 0.75 in the
    uniform case. We require >=60% to leave headroom for clip-shift +
    rescue insertion edge cases."""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    import random as _random

    rng = _random.Random(11)
    fractions: list[float] = []
    while len(fractions) < 200:
        for state in ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            repo_root=tmp_path,
            randomized=True,
            rng=rng,
            atlas_embedding_cache_dir=ATLAS_CACHE_DIR,
        ):
            assert state.quality is not None
            fractions.append(float(state.quality["gt_fraction_in_slate"]))
            if len(fractions) >= 200:
                break
    off_center = sum(1 for f in fractions if abs(f - 0.5) > 0.1)
    ratio = off_center / len(fractions)
    assert ratio >= 0.60, (
        f"expected >=60% off-center rows, got {ratio:.3f} "
        f"({off_center}/{len(fractions)})"
    )


# ---------------------------------------------------------------------------
# Grid compliance: every emitted position is an exact grid element
# ---------------------------------------------------------------------------


@_requires_atlas_cache
def test_randomized_grid_compliance(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    from langslice_traces import load_atlas_grid  # noqa: PLC0415
    import random as _random

    grid = load_atlas_grid(ATLAS_CACHE_DIR, "allen_mouse_25um", "coronal")
    grid_set = {round(g, 6) for g in grid}

    rng = _random.Random(13)
    rows_checked = 0
    while rows_checked < 30:
        for state in ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            repo_root=tmp_path,
            randomized=True,
            rng=rng,
            atlas_embedding_cache_dir=ATLAS_CACHE_DIR,
        ):
            for pos in state.atlas_slate_positions_mm:
                assert round(pos, 6) in grid_set, (
                    f"randomized position {pos!r} not in atlas grid "
                    f"(section={state.section_id})"
                )
            rows_checked += 1
            if rows_checked >= 30:
                break


# ---------------------------------------------------------------------------
# Atlas-prefix unification: Lane B paths land under the canonical root.
# ---------------------------------------------------------------------------


@_requires_atlas_cache
def test_lane_b_randomized_image_paths_use_repo_relative_prefix(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every randomized Lane B slate path must use the canonical
    ``models/langslice-gemma-4/data/atlas/...`` prefix so the trainer's
    ``repo_root / p`` resolver finds the on-disk tile. (Lane A and
    Lane B used to disagree; the canonical helper unifies them.)"""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    import random as _random

    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            repo_root=tmp_path,
            randomized=True,
            rng=_random.Random(99),
            atlas_embedding_cache_dir=ATLAS_CACHE_DIR,
        )
    )
    assert states, "expected randomized Lane B states for the fixture"
    for state in states:
        for path in state.atlas_slate_paths:
            assert path.startswith(
                "models/langslice-gemma-4/data/atlas/"
            ), (
                f"randomized Lane B leaked non-canonical prefix: {path!r} "
                f"(section={state.section_id})"
            )


def test_randomized_lane_b_paths_resolve_on_disk(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end integration: ``repo_root / atlas_slate_paths[0]`` must
    point at a real on-disk atlas tile. Gated on the existence of the
    canonical Allen 25um coronal seed tile + the embedding cache so the
    test still runs cleanly on machines without the full atlas tree."""
    pytest.importorskip("torch")
    if not ALLEN_CORONAL_CACHE.is_file():
        pytest.skip(
            f"atlas embedding cache missing: {ALLEN_CORONAL_CACHE} — "
            "build the atlas embedding cache before running."
        )
    expected_tile = (
        REPO_ROOT / "models" / "langslice-gemma-4" / "data" / "atlas"
        / "allen_mouse_25um" / "coronal" / "5.00mm.jpg"
    )
    if not expected_tile.is_file():
        pytest.skip(
            f"atlas tile fixture missing: {expected_tile} — render the "
            "Allen 25um coronal slate before running this test."
        )
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    import random as _random

    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            repo_root=REPO_ROOT,
            randomized=True,
            require_query_on_disk=False,
            rng=_random.Random(123),
            atlas_embedding_cache_dir=ATLAS_CACHE_DIR,
        )
    )
    assert states, "expected at least one randomized Lane B state"
    first_state = states[0]
    assert first_state.atlas_slate_paths, "slate must have at least one path"
    first_path = first_state.atlas_slate_paths[0]
    resolved = REPO_ROOT / first_path
    assert resolved.is_file(), (
        f"randomized Lane B path {first_path!r} did not resolve to an "
        f"on-disk tile under {REPO_ROOT!s}"
    )


# ---------------------------------------------------------------------------
# Determinism: same seed → identical output (slate paths + positions)
# ---------------------------------------------------------------------------


@_requires_atlas_cache
def test_randomized_seeded_deterministic(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    import random as _random

    def _build() -> list[ss.SectionState]:
        rng = _random.Random(2026)
        return list(
            ss.iter_section_states(
                index=index,
                plane="coronal",
                atlas="allen_mouse_25um",
                slate_root=tmp_path,
                repo_root=tmp_path,
                randomized=True,
                rng=rng,
                atlas_embedding_cache_dir=ATLAS_CACHE_DIR,
            )
        )

    a = _build()
    b = _build()
    assert len(a) == len(b) > 0
    for x, y in zip(a, b, strict=True):
        assert x.section_id == y.section_id
        assert x.atlas_slate_paths == y.atlas_slate_paths
        assert x.atlas_slate_positions_mm == y.atlas_slate_positions_mm
        assert x.ground_truth_mm == y.ground_truth_mm
        assert x.quality == y.quality


# ---------------------------------------------------------------------------
# Graceful skip when the atlas cache file is missing
# ---------------------------------------------------------------------------


def test_randomized_skip_when_atlas_cache_missing(
    index: ManifestIndex,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the requested (atlas, plane) lacks a .pt cache, the iterator
    skips (with a stderr warning) instead of crashing the trainer."""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")
    import random as _random

    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            repo_root=tmp_path,
            randomized=True,
            rng=_random.Random(0),
            atlas_embedding_cache_dir=empty_cache,
        )
    )
    assert states == []
    captured = capsys.readouterr()
    assert "randomized slate skip" in captured.err


# ---------------------------------------------------------------------------
# CLI surface (parse-time only — no atlas cache needed)
# ---------------------------------------------------------------------------


def test_cli_help_advertises_randomized_flags() -> None:
    """``build --help`` must show all three new flags so operators can
    discover them without grepping the source."""
    import io
    import sys as _sys

    buf = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = buf
    try:
        with pytest.raises(SystemExit) as exc:
            ss.main(["build", "--help"])
        assert exc.value.code == 0
    finally:
        _sys.stdout = old_stdout

    out = buf.getvalue()
    assert "--randomized-slate" in out
    assert "--randomized-seed" in out
    assert "--atlas-embedding-cache" in out


def test_cli_build_fails_fast_on_missing_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--randomized-slate with a nonexistent cache dir must SystemExit
    at validation time."""
    monkeypatch.chdir(tmp_path)
    bogus_cache = tmp_path / "does_not_exist"
    out_path = tmp_path / "out.jsonl"

    argv = [
        "build",
        "--manifest-root", str(tmp_path / "data" / "manifest"),
        "--output", str(out_path),
        "--randomized-slate",
        "--atlas-embedding-cache", str(bogus_cache),
    ]
    with pytest.raises(SystemExit) as exc:
        ss.main(argv)
    msg = str(exc.value)
    assert "atlas-embedding-cache" in msg or "not a directory" in msg


def test_cli_build_fails_fast_on_empty_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--randomized-slate with an empty cache dir must also SystemExit."""
    monkeypatch.chdir(tmp_path)
    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    out_path = tmp_path / "out.jsonl"

    argv = [
        "build",
        "--manifest-root", str(tmp_path / "data" / "manifest"),
        "--output", str(out_path),
        "--randomized-slate",
        "--atlas-embedding-cache", str(empty_cache),
    ]
    with pytest.raises(SystemExit) as exc:
        ss.main(argv)
    msg = str(exc.value)
    assert "no .pt files" in msg or "empty" in msg.lower()


# ---------------------------------------------------------------------------
# Randomized Lane A (multi-step prefix) path
# ---------------------------------------------------------------------------


@_requires_atlas_cache
def test_randomized_lane_a_strategy_yields_multi_step(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_randomized_section_state(strategy="lane_a_prefix")`` flattens
    the multi-step prefix into a single (positions, paths) pair, every path
    routed through ``_repo_relative_atlas_path`` so the trainer's
    ``repo_root / p`` resolver lands on disk."""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")

    import random as _random

    from langslice_traces import generate_trace, load_atlas_grid

    section = next(iter(index.query(plane="coronal", atlas="allen_mouse_25um")))
    grid = load_atlas_grid(ATLAS_CACHE_DIR, "allen_mouse_25um", "coronal")

    rng_state = _random.Random(2027)
    state = ss.build_randomized_section_state(
        section,
        atlas_grid=grid,
        rng=rng_state,
        difficulty_score=None,
        strategy="lane_a_prefix",
    )
    assert state is not None, "expected a non-None state for lane_a_prefix"

    # Re-run generate_trace with the same seed so we can compare flattened
    # outputs (positions + path order must match the generator exactly).
    rng_check = _random.Random(2027)
    trace = generate_trace(
        image_path=section.image_path,
        ground_truth_mm=float(section.position_mm),
        plane=section.plane,  # type: ignore[arg-type]
        atlas_name=section.atlas,
        atlas_version="",
        subject_id=section.subject_id,
        grid=grid,
        strategy="lane_a_prefix",
        rng=rng_check,
    )
    assert trace.tool_steps, (
        "if state is non-None we expect generate_trace to produce >=1 tool step"
    )

    expected_positions: list[float] = []
    expected_paths: list[str] = []
    for step in trace.tool_steps:
        expected_positions.extend(float(p) for p in step.call_args.get("positions_mm", []))
        expected_paths.extend(step.result_image_paths)

    # Flat lengths match each other and the generator's flatten.
    assert len(state.atlas_slate_positions_mm) == len(state.atlas_slate_paths)
    assert len(state.atlas_slate_positions_mm) == len(expected_positions)
    assert tuple(float(p) for p in state.atlas_slate_positions_mm) == tuple(expected_positions)

    # Paths are routed through canonical_atlas_repo_path.
    for path in state.atlas_slate_paths:
        assert path.startswith("models/langslice-gemma-4/data/atlas/"), (
            f"lane_a_prefix path not normalized to canonical root: {path!r}"
        )

    # Quality stamp records the actual number of tool steps the generator emitted.
    assert state.quality is not None
    assert state.quality["n_tool_steps"] == len(trace.tool_steps)


@_requires_atlas_cache
def test_randomized_lane_a_quality_stamp(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lane A rows must carry the strategy/source/synthetic stamp the
    downstream curriculum/eval splits filter on."""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")

    import random as _random

    from langslice_traces import load_atlas_grid

    section = next(iter(index.query(plane="coronal", atlas="allen_mouse_25um")))
    grid = load_atlas_grid(ATLAS_CACHE_DIR, "allen_mouse_25um", "coronal")

    state = ss.build_randomized_section_state(
        section,
        atlas_grid=grid,
        rng=_random.Random(99),
        difficulty_score=None,
        strategy="lane_a_prefix",
    )
    assert state is not None
    assert state.source == ss.RANDOMIZED_LANE_A_SOURCE
    assert state.source == "procedural_generator:lane_a"
    assert state.quality is not None
    assert state.quality["strategy"] == "lane_a_prefix"
    assert state.quality["n_tool_steps"] >= 1
    assert state.quality["synthetic"] is True


@_requires_atlas_cache
def test_randomized_lane_a_seeded_deterministic(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same ``Random(seed)`` → identical flattened lists across two calls."""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")

    import random as _random

    from langslice_traces import load_atlas_grid

    section = next(iter(index.query(plane="coronal", atlas="allen_mouse_25um")))
    grid = load_atlas_grid(ATLAS_CACHE_DIR, "allen_mouse_25um", "coronal")

    a = ss.build_randomized_section_state(
        section,
        atlas_grid=grid,
        rng=_random.Random(2026),
        difficulty_score=None,
        strategy="lane_a_prefix",
    )
    b = ss.build_randomized_section_state(
        section,
        atlas_grid=grid,
        rng=_random.Random(2026),
        difficulty_score=None,
        strategy="lane_a_prefix",
    )
    assert a is not None and b is not None
    assert a.atlas_slate_positions_mm == b.atlas_slate_positions_mm
    assert a.atlas_slate_paths == b.atlas_slate_paths
    assert a.quality == b.quality


@_requires_atlas_cache
def test_iter_section_states_randomized_lane_a(
    index: ManifestIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end via ``iter_section_states(randomized=True,
    strategy="lane_a_prefix", ...)`` — each yielded SectionState carries
    the Lane A stamp."""
    pytest.importorskip("torch")
    _stub_atlas_range(monkeypatch)
    _seed_query_files(tmp_path, index, "coronal")

    import random as _random

    states = list(
        ss.iter_section_states(
            index=index,
            plane="coronal",
            atlas="allen_mouse_25um",
            slate_root=tmp_path,
            repo_root=tmp_path,
            randomized=True,
            rng=_random.Random(31),
            atlas_embedding_cache_dir=ATLAS_CACHE_DIR,
            strategy="lane_a_prefix",
        )
    )
    assert states, "expected randomized Lane A states for the fixture"
    for state in states:
        assert state.source == ss.RANDOMIZED_LANE_A_SOURCE
        assert state.quality is not None
        assert state.quality["strategy"] == "lane_a_prefix"
        assert state.quality["synthetic"] is True
        assert state.quality["n_tool_steps"] >= 1
        # Every flat path lands under the canonical atlas root.
        for path in state.atlas_slate_paths:
            assert path.startswith("models/langslice-gemma-4/data/atlas/")
        # And the parallel-pair invariant holds.
        assert len(state.atlas_slate_paths) == len(state.atlas_slate_positions_mm)
        assert len(state.atlas_slate_paths) >= 1
