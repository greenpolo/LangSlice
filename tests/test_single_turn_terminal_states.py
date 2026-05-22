"""Unit tests for ``single_turn_rl.terminal_states``.

Covers (after the 2026-05-10 RLVR review fixes):
* Atlas filename parsing: positions come from the on-disk filenames the
  production tool returned, NOT from ``tool_call.args.positions_mm``
  (which can be snapped/deduped before render).
* :class:`ManifestGTLookup`: indexes shards by ``(plane, subject_id)``,
  matches the section by word-bounded substring against the query stem,
  drops rows that don't resolve.
* Walker fails closed on unparseable atlas filenames.
* Builder uses true GT from manifest, stashes teacher submit under
  ``quality["teacher_position_mm"]`` for diagnostic only.
* Default ``--repo-root`` is the current working directory (aligns with
  trainer's ``--repo-root .`` default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langslice_training.rl.single_turn import terminal_states as ts

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _seed(path: Path, content: bytes = b"\x89PNG\r\n\x1a\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _stub_atlas_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass BrainGlobe atlas loading by stubbing the cache helper."""
    cache: dict[tuple[str, str], tuple[float, float]] = {
        ("allen_mouse_25um", "coronal"): (0.0, 13.2),
        ("allen_mouse_25um", "sagittal"): (0.0, 6.0),
        ("allen_mouse_25um", "horizontal"): (0.0, 8.0),
    }
    monkeypatch.setattr(ts, "_atlas_valid_range_mm",
                        lambda atlas, plane: cache[(atlas, plane)])


# ---------------------------------------------------------------------------
# _position_mm_from_atlas_path
# ---------------------------------------------------------------------------


def test_position_mm_from_atlas_path_basic() -> None:
    assert ts._position_mm_from_atlas_path(
        "atlas/allen_mouse_25um/coronal/4.00mm.jpg"
    ) == pytest.approx(4.0)


def test_position_mm_from_atlas_path_decimal() -> None:
    assert ts._position_mm_from_atlas_path(
        "atlas/x/coronal/13.25mm.jpg"
    ) == pytest.approx(13.25)


def test_position_mm_from_atlas_path_zero() -> None:
    assert ts._position_mm_from_atlas_path(
        "atlas/x/coronal/0.05mm.jpg"
    ) == pytest.approx(0.05)


def test_position_mm_from_atlas_path_integer_no_decimal() -> None:
    assert ts._position_mm_from_atlas_path(
        "atlas/x/coronal/2mm.jpg"
    ) == pytest.approx(2.0)


def test_position_mm_from_atlas_path_handles_backslashes() -> None:
    # Windows-style separators in the path.
    assert ts._position_mm_from_atlas_path(
        "atlas\\allen_mouse_25um\\coronal\\4.00mm.jpg"
    ) == pytest.approx(4.0)


def test_position_mm_from_atlas_path_png_and_jpeg() -> None:
    assert ts._position_mm_from_atlas_path("a/4.00mm.png") == pytest.approx(4.0)
    assert ts._position_mm_from_atlas_path("a/4.00mm.jpeg") == pytest.approx(4.0)


def test_position_mm_from_atlas_path_returns_none_on_garbage() -> None:
    assert ts._position_mm_from_atlas_path("queries/single_x.jpg") is None
    assert ts._position_mm_from_atlas_path("atlas/x/coronal/no_position.jpg") is None
    assert ts._position_mm_from_atlas_path("") is None


# ---------------------------------------------------------------------------
# _section_matches_stem (word-boundary matching)
# ---------------------------------------------------------------------------


def test_section_matches_stem_short_numeric() -> None:
    """Section ``"2"`` must match ``..._2`` at end of stem (ad_bxd case)."""
    stem = "single_ad_bxd_CKT-1-AB1-42_CKT-1-AB1-42-03-14-16_03_2"
    assert ts._section_matches_stem("2", stem)
    assert ts._section_matches_stem("3", stem) is False


def test_section_matches_stem_does_not_substring_match_within_token() -> None:
    """Section ``"7"`` must NOT match a stem where 7 only appears inside a longer number."""
    stem = "single_x_subj_272414770"  # contains 7s, but no standalone _7_
    assert ts._section_matches_stem("7", stem) is False


def test_section_matches_stem_long_section_id() -> None:
    """Multi-token section parts (deepslice) match as a contiguous block."""
    stem = "single_deepslice_gt_NM01_641_2002_2565_NM01_s102_10x_M"
    assert ts._section_matches_stem("641_2002_2565_NM01_s102_10x_M", stem)
    assert ts._section_matches_stem("641_2002_2567_NM01_s104_10x_A", stem) is False


def test_section_matches_stem_internal_token() -> None:
    stem = "a_42_b"
    assert ts._section_matches_stem("42", stem)
    assert ts._section_matches_stem("4", stem) is False


def test_section_matches_stem_empty_section() -> None:
    assert ts._section_matches_stem("", "any_stem") is False


# ---------------------------------------------------------------------------
# ManifestGTLookup
# ---------------------------------------------------------------------------


def _seed_shard(manifest_root: Path, plane: str, dataset: str, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(manifest_root / "shards" / plane / f"{dataset}.jsonl", rows)


def test_manifest_lookup_resolves_simple_subject(tmp_path: Path) -> None:
    manifest = tmp_path / "data" / "manifest"
    _seed_shard(manifest, "coronal", "allen_connectivity", [
        {"section_id": "272414403:272414770", "subject_id": "272414403",
         "atlas": "allen_mouse_25um", "position_mm": 5.5},
    ])
    lookup = ts.ManifestGTLookup(manifest)
    out = lookup.lookup(
        "coronal", "272414403",
        "single_allen_connectivity_272414403_272414770",
    )
    assert out is not None
    dataset, sid, pos = out
    assert dataset == "allen_connectivity"
    assert sid == "272414403:272414770"
    assert pos == pytest.approx(5.5)


def test_manifest_lookup_resolves_subject_with_slashes(tmp_path: Path) -> None:
    """ad_bxd-style subject_id with embedded ``/`` and a single-digit section."""
    manifest = tmp_path / "data" / "manifest"
    _seed_shard(manifest, "coronal", "ad_bxd", [
        {"section_id": "CKT-1-AB1-42/CKT-1-AB1-42-03-14-16_03:2",
         "subject_id": "CKT-1-AB1-42/CKT-1-AB1-42-03-14-16_03",
         "atlas": "allen_mouse_25um", "position_mm": 2.815},
        {"section_id": "CKT-1-AB1-42/CKT-1-AB1-42-03-14-16_03:5",
         "subject_id": "CKT-1-AB1-42/CKT-1-AB1-42-03-14-16_03",
         "atlas": "allen_mouse_25um", "position_mm": 4.512},
    ])
    lookup = ts.ManifestGTLookup(manifest)
    out = lookup.lookup(
        "coronal", "CKT-1-AB1-42/CKT-1-AB1-42-03-14-16_03",
        "single_ad_bxd_CKT-1-AB1-42_CKT-1-AB1-42-03-14-16_03_2",
    )
    assert out is not None
    dataset, sid, _ = out
    assert dataset == "ad_bxd"
    assert sid.endswith(":2")


def test_manifest_lookup_returns_none_for_unknown_subject(tmp_path: Path) -> None:
    manifest = tmp_path / "data" / "manifest"
    _seed_shard(manifest, "coronal", "ds", [
        {"section_id": "A:1", "subject_id": "A",
         "atlas": "allen_mouse_25um", "position_mm": 1.0},
    ])
    lookup = ts.ManifestGTLookup(manifest)
    assert lookup.lookup("coronal", "B", "single_ds_B_1") is None


def test_manifest_lookup_returns_none_when_section_not_in_stem(tmp_path: Path) -> None:
    manifest = tmp_path / "data" / "manifest"
    _seed_shard(manifest, "coronal", "ds", [
        {"section_id": "A:99", "subject_id": "A",
         "atlas": "allen_mouse_25um", "position_mm": 1.0},
    ])
    lookup = ts.ManifestGTLookup(manifest)
    # Stem says section "1", manifest has "99" — shouldn't match.
    assert lookup.lookup("coronal", "A", "single_ds_A_1") is None


def test_manifest_lookup_handles_missing_plane_dir(tmp_path: Path) -> None:
    """Missing plane directory yields None (no crash)."""
    lookup = ts.ManifestGTLookup(tmp_path / "data" / "manifest")
    assert lookup.lookup("coronal", "A", "single_ds_A_1") is None


def test_manifest_lookup_tie_breaks_by_dataset_in_stem(tmp_path: Path) -> None:
    """When multiple datasets carry the same (subject, section), prefer the dataset
    whose name appears in the query stem."""
    manifest = tmp_path / "data" / "manifest"
    # Same subject + same section number across two datasets.
    _seed_shard(manifest, "coronal", "first_ds", [
        {"section_id": "X:42", "subject_id": "X",
         "atlas": "allen_mouse_25um", "position_mm": 1.0},
    ])
    _seed_shard(manifest, "coronal", "second_ds", [
        {"section_id": "X:42", "subject_id": "X",
         "atlas": "allen_mouse_25um", "position_mm": 9.0},
    ])
    lookup = ts.ManifestGTLookup(manifest)
    out = lookup.lookup("coronal", "X", "single_second_ds_X_42")
    assert out is not None
    dataset, _, pos = out
    assert dataset == "second_ds"
    assert pos == pytest.approx(9.0)


def test_manifest_lookup_skips_backup_files(tmp_path: Path) -> None:
    """Backup files (``.bak.<ts>...``) don't end in ``.jsonl`` so the
    glob skips them; verify that real shards still load."""
    manifest = tmp_path / "data" / "manifest"
    plane_dir = manifest / "shards" / "coronal"
    plane_dir.mkdir(parents=True)
    _write_jsonl(plane_dir / "real.jsonl", [
        {"section_id": "S:1", "subject_id": "S",
         "atlas": "allen_mouse_25um", "position_mm": 1.0},
    ])
    # Pretend there's a backup file.
    (plane_dir / "real.jsonl.bak.20260507_204212.before_image_path_update").write_text(
        "garbage-not-jsonl\n", encoding="utf-8",
    )
    lookup = ts.ManifestGTLookup(manifest)
    assert lookup.lookup("coronal", "S", "single_real_S_1") is not None


# ---------------------------------------------------------------------------
# TerminalState round-trip
# ---------------------------------------------------------------------------


def test_terminal_state_jsonl_roundtrip(tmp_path: Path) -> None:
    state = ts.TerminalState(
        section_id="A:1", subject_id="A",
        atlas_name="allen_mouse_25um", plane="coronal",
        valid_range_mm=(0.0, 13.2), ground_truth_mm=4.37,
        query_image_path="models/langslice-gemma-4/data/queries/q.jpg",
        atlas_image_paths=("models/langslice-gemma-4/data/atlas/x/coronal/4.00mm.jpg",),
        fetched_positions_mm=(4.0,),
        source="sft_corpus:strict",
        quality={"acceptance_tier": "strict", "teacher_position_mm": 4.42},
    )
    out_path = tmp_path / "out.jsonl"
    n = ts.write_terminal_states([state], out_path)
    assert n == 1
    [round_tripped] = list(ts.read_terminal_states(out_path))
    assert round_tripped == state


# ---------------------------------------------------------------------------
# _walk_trace_to_terminal
# ---------------------------------------------------------------------------


def _fetch_step(image_paths: list[str]) -> dict[str, Any]:
    """A fetch step: positions_mm in args is intentionally set to a
    different value than the image filenames so the test verifies that
    the walker uses filename-derived positions, NOT tool_call.args."""
    return {
        "tool_call": {"name": "fetch_atlas",
                      "args": {"positions_mm": [9999.0] * len(image_paths)}},
        "tool_result": {"image_paths": image_paths, "text": "..."},
    }


def _submit_step(position_mm: float) -> dict[str, Any]:
    return {"submit": {"name": "submit_estimate",
                       "args": {"position_mm": position_mm, "reasoning": "..."}}}


def test_walk_trace_derives_positions_from_atlas_filenames() -> None:
    """Verifies the P1b fix: the walker ignores tool_call.args.positions_mm
    and parses snapped positions from the on-disk filenames the production
    tool actually returned."""
    trace = [
        _fetch_step([
            "atlas/allen_mouse_25um/sagittal/2.00mm.jpg",
            "atlas/allen_mouse_25um/sagittal/3.00mm.jpg",
        ]),
        _fetch_step([
            "atlas/allen_mouse_25um/sagittal/2.50mm.jpg",
        ]),
        _submit_step(2.7),
    ]
    out = ts._walk_trace_to_terminal(trace)
    assert out is not None
    paths, positions, args = out
    assert paths == [
        "atlas/allen_mouse_25um/sagittal/2.00mm.jpg",
        "atlas/allen_mouse_25um/sagittal/3.00mm.jpg",
        "atlas/allen_mouse_25um/sagittal/2.50mm.jpg",
    ]
    assert positions == [2.0, 3.0, 2.5]
    assert args["position_mm"] == 2.7


def test_walk_trace_fails_closed_on_unparseable_atlas_filename() -> None:
    """Unparseable filename → return None rather than ship a mislabeled row."""
    trace = [
        _fetch_step(["atlas/allen_mouse_25um/sagittal/no_position.jpg"]),
        _submit_step(2.0),
    ]
    assert ts._walk_trace_to_terminal(trace) is None


def test_walk_trace_returns_none_when_no_submit() -> None:
    trace = [_fetch_step(["atlas/x/coronal/2.00mm.jpg"])]
    assert ts._walk_trace_to_terminal(trace) is None


def test_walk_trace_returns_none_for_malformed_submit() -> None:
    trace = [{"submit": "not a dict"}]
    assert ts._walk_trace_to_terminal(trace) is None


def test_walk_trace_skips_non_fetch_atlas_tool_calls() -> None:
    trace = [
        {"tool_call": {"name": "some_other_tool", "args": {}},
         "tool_result": {"image_paths": ["atlas/x/coronal/99.00mm.jpg"]}},
        _fetch_step(["atlas/x/coronal/2.00mm.jpg"]),
        _submit_step(2.0),
    ]
    out = ts._walk_trace_to_terminal(trace)
    assert out is not None
    paths, positions, _ = out
    assert paths == ["atlas/x/coronal/2.00mm.jpg"]
    assert positions == [2.0]


# ---------------------------------------------------------------------------
# build_from_sft_corpus (with manifest GT lookup)
# ---------------------------------------------------------------------------


def _sft_row(
    *,
    subject_id: str,
    section_basename: str,
    fetches: list[list[str]],
    submit_position: float | None,
    tier: str = "strict",
    plane: str = "coronal",
    kind: str = "single_slice",
) -> dict[str, Any]:
    """Build an SFT-corpus-shaped row.

    ``fetches`` is a list of per-fetch image-path lists. Each path follows
    the production convention ``atlas/<atlas>/<plane>/<X.XX>mm.jpg`` so the
    walker can derive positions from filenames.
    """
    trace: list[dict[str, Any]] = [_fetch_step(paths) for paths in fetches]
    if submit_position is not None:
        trace.append(_submit_step(submit_position))
    return {
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": plane,
        "subject_id": subject_id,
        "system_prompt_kind": kind,
        "query_image_paths": [f"queries/{section_basename}.jpg"],
        "user_prompt_text": "Determine this slice's position.",
        "trace": trace,
        "quality": {
            "acceptance_tier": tier,
            "max_error_mm": 0.05,
            "tolerance_mm": 0.085,
        },
    }


def _seed_sft_corpus(
    root: Path, rows: list[dict[str, Any]], *, seed_images: bool = True,
) -> Path:
    corpus_path = root / "sft.jsonl"
    _write_jsonl(corpus_path, rows)
    if seed_images:
        for row in rows:
            for q in row.get("query_image_paths", []):
                _seed(root / q)
            for step in row.get("trace", []):
                tr = step.get("tool_result") or {}
                for p in tr.get("image_paths", []):
                    _seed(root / p)
    return corpus_path


def _build_one(
    tmp_path: Path,
    sft_row: dict[str, Any],
    manifest_rows: list[dict[str, Any]] | None = None,
    *,
    plane: str = "coronal",
    dataset: str = "ds",
    monkeypatch: pytest.MonkeyPatch | None = None,
    **kwargs: Any,
) -> list[ts.TerminalState]:
    if monkeypatch is not None:
        _stub_atlas_range(monkeypatch)
    corpus = _seed_sft_corpus(tmp_path, [sft_row])
    manifest = tmp_path / "data" / "manifest"
    if manifest_rows is not None:
        _seed_shard(manifest, plane, dataset, manifest_rows)
    return list(ts.build_from_sft_corpus(
        corpus,
        manifest_root=manifest,
        repo_root=tmp_path,  # so paths in JSONL are tmp_path-relative
        **kwargs,
    ))


def test_build_uses_manifest_gt_not_teacher_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The teacher submitted 4.42, but the manifest says the true GT is 4.37.
    The emitted ground_truth_mm must be 4.37; teacher 4.42 stays in quality."""
    out = _build_one(
        tmp_path,
        _sft_row(
            subject_id="S1",
            section_basename="single_ds_S1_99",
            fetches=[["atlas/allen_mouse_25um/coronal/4.00mm.jpg"]],
            submit_position=4.42,
        ),
        manifest_rows=[
            {"section_id": "S1:99", "subject_id": "S1",
             "atlas": "allen_mouse_25um", "position_mm": 4.37},
        ],
        monkeypatch=monkeypatch,
    )
    assert len(out) == 1
    state = out[0]
    assert state.ground_truth_mm == pytest.approx(4.37)  # manifest, NOT teacher
    assert state.quality["teacher_position_mm"] == pytest.approx(4.42)
    assert state.quality["dataset"] == "ds"
    assert state.section_id == "S1:99"


def test_build_drops_rows_with_no_manifest_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row whose section isn't in the manifest must be dropped, not silently
    fall back to the teacher's submit (which would cap policy accuracy)."""
    out = _build_one(
        tmp_path,
        _sft_row(
            subject_id="orphan",
            section_basename="single_ds_orphan_42",
            fetches=[["atlas/allen_mouse_25um/coronal/4.00mm.jpg"]],
            submit_position=4.42,
        ),
        manifest_rows=[
            # Manifest has a different subject; orphan won't resolve.
            {"section_id": "S1:99", "subject_id": "S1",
             "atlas": "allen_mouse_25um", "position_mm": 4.37},
        ],
        monkeypatch=monkeypatch,
    )
    assert out == []


def test_build_collects_atlas_paths_and_positions_in_filename_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _build_one(
        tmp_path,
        _sft_row(
            subject_id="S1",
            section_basename="single_ds_S1_99",
            fetches=[
                ["atlas/allen_mouse_25um/coronal/2.00mm.jpg",
                 "atlas/allen_mouse_25um/coronal/3.00mm.jpg"],
                ["atlas/allen_mouse_25um/coronal/2.50mm.jpg"],
            ],
            submit_position=2.55,
        ),
        manifest_rows=[
            {"section_id": "S1:99", "subject_id": "S1",
             "atlas": "allen_mouse_25um", "position_mm": 2.50},
        ],
        monkeypatch=monkeypatch,
    )
    assert len(out) == 1
    state = out[0]
    assert state.atlas_image_paths[-3:] == (
        "atlas/allen_mouse_25um/coronal/2.00mm.jpg",
        "atlas/allen_mouse_25um/coronal/3.00mm.jpg",
        "atlas/allen_mouse_25um/coronal/2.50mm.jpg",
    )
    # Positions parsed from filenames, not from tool_call.args (which were 9999).
    assert state.fetched_positions_mm == (2.0, 3.0, 2.5)


def test_build_filters_out_group_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _build_one(
        tmp_path,
        _sft_row(
            subject_id="g1",
            section_basename="group_x",
            fetches=[["atlas/allen_mouse_25um/coronal/2.00mm.jpg"]],
            submit_position=2.0,
            kind="group",
        ),
        manifest_rows=[
            {"section_id": "g1:x", "subject_id": "g1",
             "atlas": "allen_mouse_25um", "position_mm": 2.0},
        ],
        monkeypatch=monkeypatch,
    )
    assert out == []


def test_build_filters_by_tier_strict_drops_rescued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_range(monkeypatch)
    rows = [
        _sft_row(subject_id="S1", section_basename="single_ds_S1_1",
                 fetches=[["atlas/allen_mouse_25um/coronal/2.00mm.jpg"]],
                 submit_position=2.0, tier="strict"),
        _sft_row(subject_id="S2", section_basename="single_ds_S2_2",
                 fetches=[["atlas/allen_mouse_25um/coronal/2.00mm.jpg"]],
                 submit_position=2.0, tier="rescued"),
    ]
    corpus = _seed_sft_corpus(tmp_path, rows)
    manifest = tmp_path / "data" / "manifest"
    _seed_shard(manifest, "coronal", "ds", [
        {"section_id": "S1:1", "subject_id": "S1",
         "atlas": "allen_mouse_25um", "position_mm": 2.0},
        {"section_id": "S2:2", "subject_id": "S2",
         "atlas": "allen_mouse_25um", "position_mm": 2.0},
    ])
    strict_only = list(ts.build_from_sft_corpus(
        corpus, manifest_root=manifest, repo_root=tmp_path, tier="strict",
    ))
    assert {r.subject_id for r in strict_only} == {"S1"}

    rescued_inclusive = list(ts.build_from_sft_corpus(
        corpus, manifest_root=manifest, repo_root=tmp_path, tier="rescued",
    ))
    assert {r.subject_id for r in rescued_inclusive} == {"S1", "S2"}


def test_build_drops_rows_with_no_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _build_one(
        tmp_path,
        _sft_row(
            subject_id="S1",
            section_basename="single_ds_S1_99",
            fetches=[["atlas/allen_mouse_25um/coronal/2.00mm.jpg"]],
            submit_position=None,  # never submits
        ),
        manifest_rows=[
            {"section_id": "S1:99", "subject_id": "S1",
             "atlas": "allen_mouse_25um", "position_mm": 2.0},
        ],
        monkeypatch=monkeypatch,
    )
    assert out == []


def test_build_drops_rows_with_unparseable_atlas_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walker fails closed when an atlas filename has no parseable position."""
    out = _build_one(
        tmp_path,
        _sft_row(
            subject_id="S1",
            section_basename="single_ds_S1_99",
            fetches=[["atlas/allen_mouse_25um/coronal/garbage.jpg"]],
            submit_position=2.0,
        ),
        manifest_rows=[
            {"section_id": "S1:99", "subject_id": "S1",
             "atlas": "allen_mouse_25um", "position_mm": 2.0},
        ],
        monkeypatch=monkeypatch,
    )
    assert out == []


def test_build_drops_rows_with_missing_atlas_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_range(monkeypatch)
    row = _sft_row(
        subject_id="S1",
        section_basename="single_ds_S1_99",
        fetches=[["atlas/allen_mouse_25um/coronal/2.00mm.jpg",
                  "atlas/allen_mouse_25um/coronal/3.00mm.jpg"]],
        submit_position=2.5,
    )
    corpus = _seed_sft_corpus(tmp_path, [row], seed_images=False)
    _seed(tmp_path / "queries/single_ds_S1_99.jpg")
    _seed(tmp_path / "atlas/allen_mouse_25um/coronal/2.00mm.jpg")
    # 3.00mm.jpg deliberately not seeded.
    manifest = tmp_path / "data" / "manifest"
    _seed_shard(manifest, "coronal", "ds", [
        {"section_id": "S1:99", "subject_id": "S1",
         "atlas": "allen_mouse_25um", "position_mm": 2.5},
    ])
    out = list(ts.build_from_sft_corpus(
        corpus, manifest_root=manifest, repo_root=tmp_path,
    ))
    assert out == []


def test_build_respects_max_rows_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_atlas_range(monkeypatch)
    rows = [
        _sft_row(
            subject_id=f"S{i}",
            section_basename=f"single_ds_S{i}_99",
            fetches=[["atlas/allen_mouse_25um/coronal/2.00mm.jpg"]],
            submit_position=2.0,
        )
        for i in range(5)
    ]
    corpus = _seed_sft_corpus(tmp_path, rows)
    manifest = tmp_path / "data" / "manifest"
    _seed_shard(manifest, "coronal", "ds", [
        {"section_id": f"S{i}:99", "subject_id": f"S{i}",
         "atlas": "allen_mouse_25um", "position_mm": 2.0}
        for i in range(5)
    ])
    out = list(ts.build_from_sft_corpus(
        corpus, manifest_root=manifest, repo_root=tmp_path, max_rows=2,
    ))
    assert len(out) == 2


def test_build_unknown_tier_raises() -> None:
    with pytest.raises(ValueError, match="tier must be one of"):
        list(ts.build_from_sft_corpus(
            Path("/nonexistent.jsonl"), tier="bogus",  # type: ignore[arg-type]
        ))


# ---------------------------------------------------------------------------
# Default --repo-root behavior (P2 fix)
# ---------------------------------------------------------------------------


def test_default_repo_root_uses_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When --repo-root is omitted, paths are relative to Path.cwd() so they
    align with the trainer's --repo-root . default."""
    _stub_atlas_range(monkeypatch)
    # Layout: tmp_path/models/langslice-gemma-4/data/{queries,atlas}/...
    sft_data_root = tmp_path / "models" / "langslice-gemma-4" / "data"
    row = _sft_row(
        subject_id="S1",
        section_basename="single_ds_S1_99",
        fetches=[["atlas/allen_mouse_25um/coronal/2.00mm.jpg"]],
        submit_position=2.0,
    )
    corpus = _seed_sft_corpus(sft_data_root, [row])
    manifest = tmp_path / "data" / "manifest"
    _seed_shard(manifest, "coronal", "ds", [
        {"section_id": "S1:99", "subject_id": "S1",
         "atlas": "allen_mouse_25um", "position_mm": 2.0},
    ])
    # Run from tmp_path (the "repo root") with NO --repo-root → defaults to cwd.
    monkeypatch.chdir(tmp_path)
    out = list(ts.build_from_sft_corpus(
        corpus, manifest_root=manifest,  # repo_root=None → cwd
    ))
    assert len(out) == 1
    state = out[0]
    # Paths should be tmp_path-relative (i.e. start with models/...).
    assert state.query_image_path.startswith("models/langslice-gemma-4/data/queries/")
    assert all(
        p.startswith("models/langslice-gemma-4/data/atlas/")
        for p in state.atlas_image_paths
    )
