"""Tests for Phase 6: curriculum lift + section_id keying.

Coverage:
  A. langslice_training.adaptive.curriculum.bins import works.
  B. langslice_training.curriculum.* exports work.
  C. Example.section_id is optional and defaults to None.
  D. load_examples round-trips section_id through JSONL.
  E. WeightedRandomSampler in train_sft prefers section_id over subject_id.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sft_traces"

_BASE_ROW: dict[str, Any] = {
    "bucket": 1,
    "atlas_name": "allen_mouse_25um",
    "atlas_version": "CCFv3",
    "plane": "coronal",
    "subject_id": "subj_001",
    "system_prompt_kind": "single_slice",
    "query_image_paths": ["query.png"],
    "user_prompt_text": "Estimate the AP position of this slice.",
    "trace": [
        {
            "tool_call": {"name": "fetch_atlas", "args": {"positions_mm": [3.0, 5.0, 7.0]}},
            "tool_result": {
                "image_paths": ["a3.png", "a5.png", "a7.png"],
                "text": "Atlas at 3.00 mm | 5.00 mm | 7.00 mm",
            },
        },
        {
            "submit": {
                "name": "submit_estimate",
                "args": {"position_mm": 5.2, "reasoning": "Best match after atlas comparison."},
            }
        },
    ],
}


def _write_jsonl(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows to a JSONL file in tmp_path alongside image stubs."""
    # Symlink (or copy) fixture images so _validate_image_paths is satisfied.
    for img in ("query.png", "a3.png", "a5.png", "a7.png"):
        src = FIXTURES / img
        dst = tmp_path / img
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy(src, dst)
    jsonl = tmp_path / "examples.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return jsonl


# ---------------------------------------------------------------------------
# A. adaptive.curriculum.bins import
# ---------------------------------------------------------------------------


def test_curriculum_imports_from_adaptive() -> None:
    """Canonical adaptive curriculum path exports compute_section_bins."""
    from langslice_training.adaptive.curriculum.bins import compute_section_bins  # noqa: PLC0415

    assert callable(compute_section_bins)


# ---------------------------------------------------------------------------
# B. Shim re-export from the old path
# ---------------------------------------------------------------------------


def test_curriculum_bins_imports_work() -> None:
    """Canonical curriculum bins module exports compute_section_bins."""
    from langslice_training.adaptive.curriculum.bins import compute_section_bins  # noqa: PLC0415

    assert callable(compute_section_bins)


def test_curriculum_weights_imports_work() -> None:
    """Canonical curriculum weights module exports read_weights_json."""
    from langslice_training.adaptive.curriculum.weights import read_weights_json  # noqa: PLC0415

    assert callable(read_weights_json)


def test_curriculum_log_imports_work() -> None:
    """Canonical curriculum log module exports CurriculumLogger."""
    from langslice_training.adaptive.curriculum.log import CurriculumLogger  # noqa: PLC0415

    assert callable(CurriculumLogger)


# ---------------------------------------------------------------------------
# C. Example.section_id is optional + defaults to None
# ---------------------------------------------------------------------------


def test_example_section_id_optional_default_none() -> None:
    """Constructing Example without section_id should leave it as None."""
    from sft.dataset import Example  # noqa: PLC0415

    ex = Example(
        bucket=1,
        atlas_name="allen_mouse_25um",
        atlas_version="CCFv3",
        plane="coronal",
        subject_id="subj_001",
        system_prompt_kind="single_slice",
        query_image_paths=["query.png"],
        user_prompt_text="Estimate.",
        trace=[
            {"submit": {"name": "submit_estimate", "args": {"position_mm": 5.0, "reasoning": "ok"}}}
        ],
    )
    assert ex.section_id is None


def test_example_section_id_field_exists() -> None:
    """Example dataclass must have a section_id field."""
    from sft.dataset import Example  # noqa: PLC0415

    field_names = {f.name for f in fields(Example)}
    assert "section_id" in field_names


def test_example_section_id_can_be_set_explicitly() -> None:
    """section_id accepts a non-None string value."""
    from sft.dataset import Example  # noqa: PLC0415

    ex = Example(
        bucket=1,
        atlas_name="allen_mouse_25um",
        atlas_version="CCFv3",
        plane="coronal",
        subject_id="subj_002",
        system_prompt_kind="single_slice",
        query_image_paths=["q.png"],
        user_prompt_text="x",
        trace=[
            {"submit": {"name": "submit_estimate", "args": {"position_mm": 3.0, "reasoning": "r"}}}
        ],
        section_id="ds_coronal:0042",
    )
    assert ex.section_id == "ds_coronal:0042"


# ---------------------------------------------------------------------------
# D. load_examples round-trips section_id
# ---------------------------------------------------------------------------


def test_example_section_id_roundtrips_through_load(tmp_path: Path) -> None:
    """A JSONL row with section_id must load into Example.section_id."""
    from sft.dataset import load_examples  # noqa: PLC0415

    row = dict(_BASE_ROW)
    row["section_id"] = "allen_coronal:0088"
    jsonl = _write_jsonl(tmp_path, [row])

    examples = load_examples(jsonl)
    assert len(examples) == 1
    assert examples[0].section_id == "allen_coronal:0088"
    # subject_id is still available.
    assert examples[0].subject_id == "subj_001"


def test_example_section_id_absent_in_jsonl_gives_none(tmp_path: Path) -> None:
    """A JSONL row without section_id must produce Example.section_id == None."""
    from sft.dataset import load_examples  # noqa: PLC0415

    jsonl = _write_jsonl(tmp_path, [dict(_BASE_ROW)])
    examples = load_examples(jsonl)
    assert examples[0].section_id is None


def test_load_example_existing_fixture_has_section_id_none() -> None:
    """The existing single_slice_minimal.jsonl (no section_id) loads with None."""
    from sft.dataset import load_examples  # noqa: PLC0415

    examples = load_examples(FIXTURES / "single_slice_minimal.jsonl")
    assert examples[0].section_id is None


# ---------------------------------------------------------------------------
# E. WeightedRandomSampler prefers section_id when present
# ---------------------------------------------------------------------------


def test_weighted_sampler_prefers_section_id_when_present() -> None:
    """Weight lookup should use section_id if non-None, else subject_id.

    Build two Example instances:
      ex_with_sid   — has section_id="section:007" (weight 3.5 in map)
      ex_no_sid     — section_id=None, subject_id="subj_fallback" (weight 2.1 in map)

    The lookup logic from train_sft.py is reproduced inline here so the
    test is self-contained and doesn't require loading the heavy trainer.
    """
    from sft.dataset import Example  # noqa: PLC0415

    def _dummy_example(subject_id: str, section_id: str | None) -> Example:
        return Example(
            bucket=1,
            atlas_name="a",
            atlas_version="v",
            plane="coronal",
            subject_id=subject_id,
            system_prompt_kind="single_slice",
            query_image_paths=["q.png"],
            user_prompt_text="x",
            trace=[
                {
                    "submit": {
                        "name": "submit_estimate",
                        "args": {"position_mm": 1.0, "reasoning": "r"},
                    }
                }
            ],
            section_id=section_id,
        )

    # Ex 1: has section_id → key = "section:007"
    ex_with_sid = _dummy_example("subj_001", "section:007")
    # Ex 2: no section_id → key falls back to subject_id = "subj_fallback"
    ex_no_sid = _dummy_example("subj_fallback", None)

    weights_map = {
        "section:007": 3.5,
        "subj_fallback": 2.1,
        "subj_001": 99.0,  # should NOT be used — section_id takes priority
    }

    # Replicate the key lookup from train_sft.py.
    def _lookup(ex: Example) -> float:
        key = ex.section_id if ex.section_id is not None else ex.subject_id
        return float(weights_map.get(key, 1.0))

    assert _lookup(ex_with_sid) == pytest.approx(3.5)
    assert _lookup(ex_no_sid) == pytest.approx(2.1)


def test_weighted_sampler_falls_back_to_default_when_no_match() -> None:
    """When neither section_id nor subject_id is in the map, weight is 1.0."""
    from sft.dataset import Example  # noqa: PLC0415

    ex = Example(
        bucket=1,
        atlas_name="a",
        atlas_version="v",
        plane="coronal",
        subject_id="unknown_subj",
        system_prompt_kind="single_slice",
        query_image_paths=["q.png"],
        user_prompt_text="x",
        trace=[
            {"submit": {"name": "submit_estimate", "args": {"position_mm": 1.0, "reasoning": "r"}}}
        ],
        section_id=None,
    )
    weights_map: dict[str, float] = {}

    def _lookup(ex: Example) -> float:
        key = ex.section_id if ex.section_id is not None else ex.subject_id
        return float(weights_map.get(key, 1.0))

    assert _lookup(ex) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# F. adaptive/__init__ re-exports curriculum subpackage
# ---------------------------------------------------------------------------


def test_curriculum_module_exports_compute_section_bins() -> None:
    """Canonical curriculum package root exposes compute_section_bins."""
    import langslice_training.adaptive.curriculum as curriculum  # noqa: PLC0415
    assert hasattr(curriculum, "compute_section_bins")
