from __future__ import annotations

from pathlib import Path

from langslice_training.sft.filtering import best_of_n
from langslice_training.sft.path_rewriter import build_unified_corpus
from langslice_training.sft.seen_ledger import SeenLedger
from langslice_training.sft.trace_format import convert_trace


def test_seen_ledger_roundtrip(tmp_path: Path) -> None:
    ledger_path = tmp_path / "seen.jsonl"
    ledger = SeenLedger(ledger_path=ledger_path, manifest_root=None)
    ledger.mark_seen("section-a", round_idx=0, source="synthetic")
    ledger.mark_seen("section-b", round_idx=1, source="real_rollout")

    reloaded = SeenLedger(ledger_path=ledger_path, manifest_root=None)
    reloaded.load()

    assert reloaded.is_seen("section-a")
    assert reloaded.is_seen("section-b")
    assert reloaded.seen_count() == 2


def test_best_of_n_selects_lowest_error_per_section() -> None:
    rows = [
        {"section_id": "s1", "abs_err_mm": 0.4},
        {"section_id": "s1", "abs_err_mm": 0.2},
        {"section_id": "s2", "abs_err_mm": 0.3},
    ]

    picked = best_of_n(rows)

    assert picked == [
        {"section_id": "s1", "abs_err_mm": 0.2},
        {"section_id": "s2", "abs_err_mm": 0.3},
    ]


def test_shared_module_symbols_import() -> None:
    assert callable(build_unified_corpus)
    assert callable(convert_trace)
