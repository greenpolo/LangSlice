"""Unit tests for iSFT.path_rewriter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "models" / "langslice-gemma-4" / "training"))

from iSFT.path_rewriter import (  # noqa: E402
    build_unified_corpus,
)


def _write_image(path: Path, color=(127, 64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path, format="JPEG")


def _make_row(
    *,
    subject_id: str,
    query_path: str,
    atlas_paths: list[str],
    submit_position: float = 5.0,
) -> dict:
    return {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "v0.0.1",
        "plane": "coronal",
        "subject_id": subject_id,
        "system_prompt_kind": "single_slice",
        "query_image_paths": [query_path],
        "user_prompt_text": "Estimate position.",
        "trace": [
            {
                "tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [4.0]}},
                "tool_result": {
                    "image_paths": list(atlas_paths),
                    "text": "ok",
                },
            },
            {
                "submit": {
                    "name": "submit_estimate",
                    "args": {"position_mm": submit_position, "reasoning": "ok."},
                },
            },
        ],
        "quality": {
            "accuracy": "tight", "max_error_mm": 0.05,
            "fetch_calls": 1, "preprocessing": "raw",
            "query_long_edge_px": 8, "acceptance_tier": "strict",
            "rescue_threshold_mm": 0.9, "submitted_position_mm": submit_position,
            "estimated_tokens": 100, "trim": {},
        },
    }


# ──────────────────────────────────────────────────────────────────────────
# Smoke: single corpus → unified tree
# ──────────────────────────────────────────────────────────────────────────

def test_query_images_under_data_datasets_bypass_staging(tmp_path: Path) -> None:
    """Slice images that already live under ``<repo>/data/datasets/`` should
    be emitted as canonical repo-relative paths and NOT copied into
    ``output_dir/queries/``. Two wins handled by the shortcut:

    1. No JPEG copy onto slow NTFS — the JPEG decode at training time reads
       from the langslice-data-fast volume.
    2. The query embedding cache (keyed on canonical ``data/datasets/...``
       paths) actually hits during ``lookup_by_path`` instead of missing
       100% of the time as it does with the ``queries/<hash>_<name>`` shape.
    """
    # Stand up a fake repo root with a pyproject.toml so the rewriter's
    # walk-up succeeds and the shortcut fires.
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    src_img = repo / "data" / "datasets" / "coronal" / "ds_a" / "subjA" / "section_001.jpg"
    _write_image(src_img)

    base_dir = repo / "out" / "base_corpus"
    base_dir.mkdir(parents=True)
    _write_image(base_dir / "atlas" / "p0500.jpg")
    base_jsonl = base_dir / "examples.jsonl"
    # Query path is repo-relative to the data/datasets/ tree.
    base_jsonl.write_text(
        json.dumps(_make_row(
            subject_id="subjA",
            query_path="../../data/datasets/coronal/ds_a/subjA/section_001.jpg",
            atlas_paths=["atlas/p0500.jpg"],
        )) + "\n",
        encoding="utf-8",
    )

    out_dir = repo / "out" / "round"
    out_jsonl = out_dir / "round_0_corpus.jsonl"
    stats = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=base_dir / "doesnt_exist",
        iterative_jsonls=[],
        output_dir=out_dir,
        output_jsonl=out_jsonl,
    )
    assert stats["rows_kept"] == 1
    # The query was NOT staged into queries/ — no copy, no symlink.
    queries_dir = out_dir / "queries"
    assert not queries_dir.exists() or not any(queries_dir.iterdir())
    # The rewritten path is the canonical repo-relative one (cache-key match).
    row = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert row["query_image_paths"] == [
        "data/datasets/coronal/ds_a/subjA/section_001.jpg"
    ]
    # Atlas images STILL get staged (snap-to-grid + cache-key shape).
    atlas_paths = row["trace"][0]["tool_result"]["image_paths"]
    assert all(p.startswith("atlas/") for p in atlas_paths)


def test_unified_corpus_stages_query_and_atlas_images(tmp_path: Path) -> None:
    base_dir = tmp_path / "base_corpus"
    base_dir.mkdir()
    # Place query + atlas images under base_dir.
    _write_image(base_dir / "queries" / "subjA.jpg")
    _write_image(base_dir / "atlas" / "coronal" / "p0500.jpg")
    base_jsonl = base_dir / "examples.jsonl"
    base_jsonl.write_text(
        json.dumps(_make_row(
            subject_id="subjA",
            query_path="queries/subjA.jpg",
            atlas_paths=["atlas/coronal/p0500.jpg"],
        )) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out_jsonl = out_dir / "round_0_corpus.jsonl"
    stats = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=base_dir / "doesnt_exist",
        iterative_jsonls=[],
        output_dir=out_dir,
        output_jsonl=out_jsonl,
    )
    assert stats["rows_kept"] == 1
    assert stats["rows_dropped_missing_images"] == 0
    # Both images staged into out_dir.
    assert (out_dir / "queries").is_dir()
    assert (out_dir / "atlas").is_dir()
    # Rewritten JSONL points at unified-tree paths.
    line = out_jsonl.read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row["query_image_paths"][0].startswith("queries/")
    for p in row["trace"][0]["tool_result"]["image_paths"]:
        assert p.startswith("atlas/")
        assert (out_dir / p).is_file() or (out_dir / p).is_symlink()


def test_unified_corpus_unions_two_jsonls_with_disjoint_roots(tmp_path: Path) -> None:
    """The wave-1 bug was that base and iterative had different roots; verify fix."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    _write_image(base_dir / "queries" / "base_a.jpg")
    _write_image(base_dir / "atlas" / "p0500.jpg")
    base_jsonl = base_dir / "examples.jsonl"
    base_jsonl.write_text(
        json.dumps(_make_row(
            subject_id="base_a",
            query_path="queries/base_a.jpg",
            atlas_paths=["atlas/p0500.jpg"],
        )) + "\n",
        encoding="utf-8",
    )

    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    _write_image(iter_dir / "queries" / "iter_b.jpg")
    _write_image(iter_dir / "atlas" / "coronal" / "p0700.jpg")
    iter_jsonl = iter_dir / "round_0.jsonl"
    iter_jsonl.write_text(
        json.dumps(_make_row(
            subject_id="iter_b",
            query_path="queries/iter_b.jpg",
            atlas_paths=["atlas/coronal/p0700.jpg"],
        )) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out_jsonl = out_dir / "round_0_corpus.jsonl"
    stats = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=iter_dir,
        iterative_jsonls=[iter_jsonl],
        output_dir=out_dir,
        output_jsonl=out_jsonl,
    )
    assert stats["rows_input"] == 2
    assert stats["rows_kept"] == 2
    assert stats["rows_dropped_missing_images"] == 0

    # Both rows' images now resolve relative to out_dir.
    rows = [
        json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    for row in rows:
        for q in row["query_image_paths"]:
            assert (out_dir / q).is_file() or (out_dir / q).is_symlink()
        for step in row["trace"][:-1]:
            for p in step["tool_result"]["image_paths"]:
                assert (out_dir / p).is_file() or (out_dir / p).is_symlink()


def test_unified_corpus_drops_rows_with_missing_images(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    # Note: NO image written for missing.jpg.
    base_jsonl = base_dir / "examples.jsonl"
    base_jsonl.write_text(
        json.dumps(_make_row(
            subject_id="ghost",
            query_path="queries/missing.jpg",
            atlas_paths=["atlas/missing.jpg"],
        )) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    stats = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=base_dir / "iter",
        iterative_jsonls=[],
        output_dir=out_dir,
        output_jsonl=out_dir / "out.jsonl",
    )
    assert stats["rows_input"] == 1
    assert stats["rows_kept"] == 0
    assert stats["rows_dropped_missing_images"] == 1


def test_unified_corpus_dedups_shared_image_across_rows(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    _write_image(base_dir / "queries" / "qA.jpg")
    _write_image(base_dir / "queries" / "qB.jpg")
    _write_image(base_dir / "atlas" / "shared.jpg")
    rows = [
        _make_row(
            subject_id="A", query_path="queries/qA.jpg",
            atlas_paths=["atlas/shared.jpg"],
        ),
        _make_row(
            subject_id="B", query_path="queries/qB.jpg",
            atlas_paths=["atlas/shared.jpg"],  # same atlas image
        ),
    ]
    base_jsonl = base_dir / "examples.jsonl"
    base_jsonl.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    stats = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=base_dir / "iter",
        iterative_jsonls=[],
        output_dir=out_dir,
        output_jsonl=out_dir / "out.jsonl",
    )
    assert stats["rows_kept"] == 2
    # Only ONE atlas image staged (the second cache-hits).
    atlas_files = list((out_dir / "atlas").iterdir())
    assert len(atlas_files) == 1


def test_unified_corpus_handles_absolute_image_paths(tmp_path: Path) -> None:
    """Some iterative rows store absolute paths; the rewriter must absorb them."""
    src_dir = tmp_path / "weird"
    _write_image(src_dir / "img.jpg")
    abs_path = str((src_dir / "img.jpg").resolve()).replace("\\", "/")
    _write_image(src_dir / "atlas.jpg")
    abs_atlas = str((src_dir / "atlas.jpg").resolve()).replace("\\", "/")

    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    iter_jsonl = iter_dir / "round_0.jsonl"
    iter_jsonl.write_text(
        json.dumps(_make_row(
            subject_id="abs_a",
            query_path=abs_path,
            atlas_paths=[abs_atlas],
        )) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    stats = build_unified_corpus(
        base_corpus=tmp_path / "no_base.jsonl",
        iterative_corpus_dir=iter_dir,
        iterative_jsonls=[iter_jsonl],
        output_dir=out_dir,
        output_jsonl=out_dir / "out.jsonl",
    )
    assert stats["rows_kept"] == 1


def test_unified_corpus_extra_image_roots_resolves_repo_relative(tmp_path: Path) -> None:
    """Some legacy paths are repo-root-relative; extra_image_roots covers them."""
    repo_root = tmp_path / "repo"
    _write_image(repo_root / "data" / "something.jpg")
    _write_image(repo_root / "data" / "atlas.jpg")
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    iter_jsonl = iter_dir / "round_0.jsonl"
    # Path is relative to repo_root, NOT to iter_dir.
    iter_jsonl.write_text(
        json.dumps(_make_row(
            subject_id="ra",
            query_path="data/something.jpg",
            atlas_paths=["data/atlas.jpg"],
        )) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    stats = build_unified_corpus(
        base_corpus=tmp_path / "no_base.jsonl",
        iterative_corpus_dir=iter_dir,
        iterative_jsonls=[iter_jsonl],
        output_dir=out_dir,
        output_jsonl=out_dir / "out.jsonl",
        extra_image_roots=[repo_root],
    )
    assert stats["rows_kept"] == 1


def test_unified_corpus_base_sample_n_caps_distilled_rows(tmp_path: Path) -> None:
    """When base_sample_n is set, only that many distilled rows pass through."""
    base_jsonl = tmp_path / "distilled.jsonl"
    rows = []
    for i in range(20):
        _write_image(tmp_path / f"q_{i}.jpg")
        _write_image(tmp_path / f"a_{i}.jpg")
        rows.append(_make_row(
            subject_id=f"sub_{i}",
            query_path=f"q_{i}.jpg",
            atlas_paths=[f"a_{i}.jpg"],
        ))
    base_jsonl.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    stats = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=tmp_path / "iter",
        iterative_jsonls=[],
        output_dir=out_dir,
        output_jsonl=out_dir / "out.jsonl",
        base_sample_n=5,
        base_sample_seed=42,
    )
    assert stats["rows_input"] == 5  # capped at sample size
    assert stats["rows_kept"] == 5

    # Same seed → deterministic
    stats2 = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=tmp_path / "iter",
        iterative_jsonls=[],
        output_dir=tmp_path / "out2",
        output_jsonl=tmp_path / "out2" / "out.jsonl",
        base_sample_n=5,
        base_sample_seed=42,
    )
    rows1 = (out_dir / "out.jsonl").read_text(encoding="utf-8")
    rows2 = (tmp_path / "out2" / "out.jsonl").read_text(encoding="utf-8")
    assert rows1 == rows2

    # Sample N larger than dataset → use everything
    stats3 = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=tmp_path / "iter",
        iterative_jsonls=[],
        output_dir=tmp_path / "out3",
        output_jsonl=tmp_path / "out3" / "out.jsonl",
        base_sample_n=1000,
        base_sample_seed=42,
    )
    assert stats3["rows_input"] == 20


def test_unified_corpus_skips_malformed_rows(tmp_path: Path) -> None:
    """Malformed JSON gets dropped; semantically-malformed rows are passed
    through (path_rewriter is path-resolution only, not schema validation;
    _validate_row is responsible for catching missing required fields)."""
    base_jsonl = tmp_path / "examples.jsonl"
    # First row is invalid JSON, second is valid JSON but missing fields.
    base_jsonl.write_text(
        "{ not json }\n" + json.dumps({"_": "missing fields"}) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    stats = build_unified_corpus(
        base_corpus=base_jsonl,
        iterative_corpus_dir=tmp_path / "iter",
        iterative_jsonls=[],
        output_dir=out_dir,
        output_jsonl=out_dir / "out.jsonl",
    )
    assert stats["rows_input"] == 2
    # Malformed JSON dropped; the {"_": ...} row has no images so it slips
    # through path-resolution (image-less rows aren't a path_rewriter concern).
    assert stats["rows_dropped_missing_images"] == 1
