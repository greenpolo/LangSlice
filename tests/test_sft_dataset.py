"""Tests for models/langslice-gemma-4/training/sft/dataset.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sft.dataset import (
    DatasetValidationError,
    Example,
    load_examples,
    split_subject_aware,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sft_traces"


def test_load_single_slice_minimal_returns_one_example() -> None:
    examples = load_examples(FIXTURES / "single_slice_minimal.jsonl")
    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, Example)
    assert ex.bucket == 1
    assert ex.atlas_name == "allen_mouse_25um"
    assert ex.system_prompt_kind == "single_slice"
    assert ex.query_image_paths == ["query.png"]
    assert ex.subject_id == "test_subj_01"
    assert len(ex.trace) == 2
    assert ex.trace[0]["tool_call"]["name"] == "fetch_atlas"
    assert ex.trace[1]["submit"]["name"] == "submit_estimate"
    assert ex.trace[1]["submit"]["args"]["reasoning"]


def test_load_thinking_submit_allows_optional_reasoning(tmp_path: Path) -> None:
    (tmp_path / "query.png").write_bytes(b"q")
    row = {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": "coronal",
        "subject_id": "thinking_subj",
        "system_prompt_kind": "single_slice",
        "query_image_paths": ["query.png"],
        "user_prompt_text": "Estimate.",
        "thinking_mode": True,
        "trace": [
            {
                "submit": {
                    "name": "submit_estimate",
                    "args": {"position_mm": 5.0},
                },
                "thinking": "I should sanity-check neighbors first.",
            }
        ],
    }
    path = tmp_path / "thinking_submit.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    examples = load_examples(path)
    assert len(examples) == 1
    assert examples[0].thinking_mode is True
    assert examples[0].trace[-1]["submit"]["args"]["position_mm"] == pytest.approx(5.0)


def test_load_terminal_fetch_tool_call_without_tool_result(tmp_path: Path) -> None:
    for name in ("query.png", "a3.png"):
        (tmp_path / name).write_bytes(b"x")
    row = {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": "coronal",
        "subject_id": "fetch_terminal_subj",
        "system_prompt_kind": "single_slice",
        "query_image_paths": ["query.png"],
        "user_prompt_text": "Estimate.",
        "thinking_mode": True,
        "trace": [
            {
                "tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [3.0]}},
                "tool_result": {"image_paths": ["a3.png"], "text": "Atlas at 3.00 mm"},
            },
            {
                "tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [5.0]}},
                "thinking": "Need a closer atlas comparison.",
            },
        ],
    }
    path = tmp_path / "terminal_fetch.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    examples = load_examples(path)
    assert len(examples) == 1
    assert "tool_call" in examples[0].trace[-1]
    assert "tool_result" not in examples[0].trace[-1]
    assert examples[0].thinking_mode is True


def test_terminal_fetch_requires_thinking_mode(tmp_path: Path) -> None:
    for name in ("query.png", "a3.png"):
        (tmp_path / name).write_bytes(b"x")
    row = {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": "coronal",
        "subject_id": "fetch_terminal_subj",
        "system_prompt_kind": "single_slice",
        "query_image_paths": ["query.png"],
        "user_prompt_text": "Estimate.",
        "trace": [
            {
                "tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [3.0]}},
                "tool_result": {"image_paths": ["a3.png"], "text": "Atlas at 3.00 mm"},
            },
            {
                "tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [5.0]}},
            },
        ],
    }
    path = tmp_path / "terminal_fetch_no_thinking.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="terminal tool_call.*thinking"):
        load_examples(path)


def test_malformed_examples_raise_validation_errors() -> None:
    with pytest.raises(DatasetValidationError) as exc:
        load_examples(FIXTURES / "malformed_examples.jsonl")
    msg = str(exc.value)
    # First defective row is line 1 (bucket != 1)
    assert "line 1" in msg


def test_missing_image_path_is_rejected(tmp_path: Path) -> None:
    row = {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "CCFv3",
        "plane": "coronal",
        "subject_id": "x",
        "system_prompt_kind": "single_slice",
        "query_image_paths": ["missing.png"],
        "user_prompt_text": "x",
        "trace": [
            {
                "submit": {
                    "name": "submit_estimate",
                    "args": {"position_mm": 5.0, "reasoning": "x"},
                }
            }
        ],
    }
    path = tmp_path / "missing_image.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="image not found"):
        load_examples(path)


def test_repo_root_fallback_resolves_canonical_data_datasets_paths(
    tmp_path: Path,
) -> None:
    """A row whose ``query_image_paths`` is the canonical repo-relative
    ``data/datasets/...`` path (emitted by the path_rewriter's shortcut for
    sources living under data/datasets) must validate even though the JSONL
    sits in a different directory tree.

    Without the repo-root fallback in ``_require_existing_image``, the
    rewriter's shortcut would break validation 100% of the time.
    """
    # Fake repo layout: pyproject.toml at root, image under data/datasets/.
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    img_rel = "data/datasets/coronal/ds_a/subjA/section_001.jpg"
    img_abs = repo / img_rel
    img_abs.parent.mkdir(parents=True)
    img_abs.write_bytes(b"\xff\xd8\xff\xe0\x00fakeJPEG")  # just enough header

    # JSONL lives under repo/out/round/ — NOT next to the image.
    jsonl_dir = repo / "out" / "round"
    jsonl_dir.mkdir(parents=True)
    row = {
        "bucket": 1,
        "atlas_name": "allen_mouse_25um",
        "atlas_version": "v0.0.1",
        "plane": "coronal",
        "subject_id": "subjA",
        "system_prompt_kind": "single_slice",
        "query_image_paths": [img_rel],
        "user_prompt_text": "Estimate position.",
        "trace": [
            {
                "submit": {
                    "name": "submit_estimate",
                    "args": {"position_mm": 5.0, "reasoning": "x"},
                }
            }
        ],
    }
    jsonl = jsonl_dir / "row.jsonl"
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    # Validation must succeed via the repo-root walk-up fallback. The naive
    # `(root / rel)` resolution would point at out/round/data/datasets/... which
    # doesn't exist; the fallback rewalks to repo and finds it there.
    examples = load_examples(jsonl)
    assert len(examples) == 1
    assert examples[0].query_image_paths == [img_rel]


def _make_examples(subject_ids: list[str]) -> list[Example]:
    """Tiny helper for split tests — uses the single_slice fixture's shape."""
    template_path = FIXTURES / "single_slice_minimal.jsonl"
    base = load_examples(template_path)[0]
    return [
        Example(
            bucket=base.bucket,
            atlas_name=base.atlas_name,
            atlas_version=base.atlas_version,
            plane=base.plane,
            subject_id=sid,
            system_prompt_kind=base.system_prompt_kind,
            query_image_paths=base.query_image_paths,
            user_prompt_text=base.user_prompt_text,
            trace=base.trace,
        )
        for sid in subject_ids
    ]


def test_split_subject_aware_no_subject_in_both_partitions() -> None:
    # 10 subjects, each contributing 3 examples. Holdout fraction 0.3.
    subjects = [f"subj_{i:02d}" for i in range(10)]
    examples = []
    for sid in subjects:
        examples.extend(_make_examples([sid] * 3))

    train, eval_ = split_subject_aware(examples, holdout_fraction=0.3, seed=0)

    train_subjects = {ex.subject_id for ex in train}
    eval_subjects = {ex.subject_id for ex in eval_}
    assert train_subjects.isdisjoint(eval_subjects), (
        f"subject leakage between partitions: "
        f"{train_subjects & eval_subjects}"
    )
    # Holdout fraction is deterministic under fixed seed: round(10 * 0.3) = 3
    assert len(eval_subjects) == 3
    # No examples lost
    assert len(train) + len(eval_) == len(examples)


def test_split_subject_aware_deterministic_with_seed() -> None:
    examples = _make_examples([f"subj_{i:02d}" for i in range(10)])
    train_a, eval_a = split_subject_aware(examples, holdout_fraction=0.3, seed=42)
    train_b, eval_b = split_subject_aware(examples, holdout_fraction=0.3, seed=42)
    assert [ex.subject_id for ex in train_a] == [ex.subject_id for ex in train_b]
    assert [ex.subject_id for ex in eval_a] == [ex.subject_id for ex in eval_b]


def test_split_subject_aware_rejects_invalid_holdout_fraction() -> None:
    examples = _make_examples([f"subj_{i:02d}" for i in range(4)])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="holdout_fraction must be in"):
            split_subject_aware(examples, holdout_fraction=bad, seed=0)


def test_split_subject_aware_rejects_holdout_consuming_all_subjects() -> None:
    examples = _make_examples(["subj_a", "subj_b"])
    # 0.99 of 2 subjects rounds to 2, leaving zero for train.
    with pytest.raises(ValueError, match="would consume all"):
        split_subject_aware(examples, holdout_fraction=0.99, seed=0)
