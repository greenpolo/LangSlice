"""Tests for ``curriculum.log.CurriculumLogger``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langslice_training.adaptive.curriculum.log import CurriculumLogger


def _row(round_idx: int = 0, section_id: str = "subj_a:s1") -> dict:
    return {
        "round": round_idx,
        "section_id": section_id,
        "bin_idx": 2,
        "plane": "coronal",
        "abs_err_mm": 0.42,
        "accuracy_pct": 96.81,
    }


def test_append_writes_jsonl_with_timestamp(tmp_path: Path) -> None:
    log_path = tmp_path / "subdir" / "curriculum_log.jsonl"
    logger = CurriculumLogger(log_path)
    logger.append(_row())

    assert log_path.is_file()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["round"] == 0
    assert parsed["section_id"] == "subj_a:s1"
    assert parsed["bin_idx"] == 2
    assert parsed["plane"] == "coronal"
    assert parsed["abs_err_mm"] == 0.42
    assert parsed["accuracy_pct"] == 96.81
    assert "ts" in parsed and isinstance(parsed["ts"], str)


def test_append_idempotent_across_calls(tmp_path: Path) -> None:
    """Multiple appends produce one JSONL line each, in order."""
    log_path = tmp_path / "log.jsonl"
    logger = CurriculumLogger(log_path)

    rows = [
        _row(round_idx=k, section_id=f"subj:s{k}")
        for k in range(5)
    ]
    for r in rows:
        logger.append(r)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    parsed = [json.loads(line) for line in lines]
    assert [p["round"] for p in parsed] == [0, 1, 2, 3, 4]
    assert [p["section_id"] for p in parsed] == [f"subj:s{k}" for k in range(5)]


def test_caller_supplied_ts_preserved(tmp_path: Path) -> None:
    """A caller-provided ``ts`` field is preserved verbatim, not overwritten."""
    log_path = tmp_path / "log.jsonl"
    logger = CurriculumLogger(log_path)

    row = _row()
    row["ts"] = "2026-05-09T12:34:56+00:00"
    logger.append(row)

    parsed = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["ts"] == "2026-05-09T12:34:56+00:00"


def test_missing_required_field_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    logger = CurriculumLogger(log_path)
    incomplete = {
        "round": 0,
        "section_id": "subj:s1",
        # bin_idx missing
        "plane": "coronal",
        "abs_err_mm": 0.5,
        "accuracy_pct": 90.0,
    }
    with pytest.raises(ValueError, match="missing required field"):
        logger.append(incomplete)
    # Nothing should have been written
    assert not log_path.exists()


def test_appends_to_existing_file(tmp_path: Path) -> None:
    """Re-instantiating the logger over the same path keeps appending, not truncating."""
    log_path = tmp_path / "log.jsonl"
    CurriculumLogger(log_path).append(_row(round_idx=0))
    CurriculumLogger(log_path).append(_row(round_idx=1))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rounds = [json.loads(line)["round"] for line in lines]
    assert rounds == [0, 1]
