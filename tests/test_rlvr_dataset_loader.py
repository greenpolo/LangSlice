"""Smoke tests for ``langslice_training.rl.common.dataset.load_rlvr_allocation``.

The loader joins per-plane RLVR allocation files with their dataset shards and
produces ``SingleSliceExample`` rows with absolute image paths. Tests here use
synthetic mini-allocations + shards in tmp_path so they don't depend on the
state of the live ``data/manifest`` tree.
"""

from __future__ import annotations

import json
from pathlib import Path

from langslice_training.rl.common.dataset import SingleSliceExample, load_rlvr_allocation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _seed_image(repo_root: Path, rel_path: str) -> Path:
    abs_path = repo_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header — only existence matters
    return abs_path


def test_load_rlvr_allocation_joins_shard_to_allocation(tmp_path: Path) -> None:
    repo_root = tmp_path
    manifest_root = repo_root / "data" / "manifest"

    _seed_image(repo_root, "data/datasets/coronal/allen/sub_a/sec_1.jpg")
    _seed_image(repo_root, "data/datasets/coronal/allen/sub_b/sec_2.jpg")
    _write_jsonl(
        manifest_root / "shards" / "coronal" / "allen.jsonl",
        [
            {
                "section_id": "sub_a:sec_1",
                "subject_id": "sub_a",
                "atlas": "allen_mouse_25um",
                "image_path": "data/datasets/coronal/allen/sub_a/sec_1.jpg",
                "position_mm": 4.5,
            },
            {
                "section_id": "sub_b:sec_2",
                "subject_id": "sub_b",
                "atlas": "allen_mouse_25um",
                "image_path": "data/datasets/coronal/allen/sub_b/sec_2.jpg",
                "position_mm": 9.1,
            },
        ],
    )
    _write_jsonl(
        manifest_root / "allocations" / "coronal" / "rlvr.jsonl",
        [
            {"section_id": "sub_a:sec_1", "dataset": "allen"},
            {"section_id": "sub_b:sec_2", "dataset": "allen"},
        ],
    )

    examples = load_rlvr_allocation(manifest_root, repo_root=repo_root, planes=("coronal",))

    assert len(examples) == 2
    by_subject = {e.subject_id: e for e in examples}
    assert isinstance(by_subject["sub_a"], SingleSliceExample)
    assert by_subject["sub_a"].plane == "coronal"
    assert by_subject["sub_a"].atlas_name == "allen_mouse_25um"
    assert by_subject["sub_a"].ap_mm == 4.5
    assert by_subject["sub_a"].image_path.is_absolute()
    assert by_subject["sub_a"].image_path.is_file()


def test_tombstone_drops_section_from_active_set(tmp_path: Path) -> None:
    repo_root = tmp_path
    manifest_root = repo_root / "data" / "manifest"

    _seed_image(repo_root, "data/datasets/coronal/allen/sub_a/sec_1.jpg")
    _write_jsonl(
        manifest_root / "shards" / "coronal" / "allen.jsonl",
        [
            {
                "section_id": "sub_a:sec_1",
                "subject_id": "sub_a",
                "atlas": "allen_mouse_25um",
                "image_path": "data/datasets/coronal/allen/sub_a/sec_1.jpg",
                "position_mm": 4.5,
            },
        ],
    )
    _write_jsonl(
        manifest_root / "allocations" / "coronal" / "rlvr.jsonl",
        [
            {"section_id": "sub_a:sec_1", "dataset": "allen"},
            {"section_id": "sub_a:sec_1", "dataset": "allen", "action": "remove"},
        ],
    )

    examples = load_rlvr_allocation(manifest_root, repo_root=repo_root, planes=("coronal",))
    assert examples == []


def test_missing_image_dropped_with_warning(tmp_path: Path, capsys) -> None:
    repo_root = tmp_path
    manifest_root = repo_root / "data" / "manifest"

    _write_jsonl(
        manifest_root / "shards" / "coronal" / "allen.jsonl",
        [
            {
                "section_id": "sub_a:sec_1",
                "subject_id": "sub_a",
                "atlas": "allen_mouse_25um",
                "image_path": "data/datasets/coronal/allen/sub_a/missing.jpg",
                "position_mm": 4.5,
            },
        ],
    )
    _write_jsonl(
        manifest_root / "allocations" / "coronal" / "rlvr.jsonl",
        [{"section_id": "sub_a:sec_1", "dataset": "allen"}],
    )

    examples = load_rlvr_allocation(manifest_root, repo_root=repo_root, planes=("coronal",))
    assert examples == []
    err = capsys.readouterr().err
    assert "image not on disk" in err


def test_section_id_missing_from_shard_dropped_with_warning(tmp_path: Path, capsys) -> None:
    repo_root = tmp_path
    manifest_root = repo_root / "data" / "manifest"

    _seed_image(repo_root, "data/datasets/coronal/allen/sub_a/sec_1.jpg")
    _write_jsonl(
        manifest_root / "shards" / "coronal" / "allen.jsonl",
        [
            {
                "section_id": "sub_a:sec_1",
                "subject_id": "sub_a",
                "atlas": "allen_mouse_25um",
                "image_path": "data/datasets/coronal/allen/sub_a/sec_1.jpg",
                "position_mm": 4.5,
            },
        ],
    )
    _write_jsonl(
        manifest_root / "allocations" / "coronal" / "rlvr.jsonl",
        [
            {"section_id": "sub_a:sec_1", "dataset": "allen"},
            {"section_id": "sub_a:GHOST", "dataset": "allen"},
        ],
    )

    examples = load_rlvr_allocation(manifest_root, repo_root=repo_root, planes=("coronal",))
    assert len(examples) == 1
    err = capsys.readouterr().err
    assert "no matching shard" in err
