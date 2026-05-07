"""Tests for models/langslice-gemma-4/training/sft/dataset.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft.dataset import DatasetValidationError, Example, load_examples

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
        "trace": [{"submit": {"name": "submit_estimate", "args": {"position_mm": 5.0, "reasoning": "x"}}}],
    }
    path = tmp_path / "missing_image.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="image not found"):
        load_examples(path)


from sft.dataset import split_subject_aware


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
    # Holdout fraction is approximate (subject-level, not row-level)
    assert 2 <= len(eval_subjects) <= 4
    # No examples lost
    assert len(train) + len(eval_) == len(examples)


def test_split_subject_aware_deterministic_with_seed() -> None:
    examples = _make_examples([f"subj_{i:02d}" for i in range(10)])
    train_a, eval_a = split_subject_aware(examples, holdout_fraction=0.3, seed=42)
    train_b, eval_b = split_subject_aware(examples, holdout_fraction=0.3, seed=42)
    assert [ex.subject_id for ex in train_a] == [ex.subject_id for ex in train_b]
    assert [ex.subject_id for ex in eval_a] == [ex.subject_id for ex in eval_b]
