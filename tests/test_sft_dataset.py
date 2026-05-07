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
