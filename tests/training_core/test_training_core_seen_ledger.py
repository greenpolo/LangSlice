from __future__ import annotations

from pathlib import Path

from langslice_training.sft.seen_ledger import SeenLedger


def test_seen_ledger_roundtrip_no_manifest(tmp_path: Path) -> None:
    ledger_path = tmp_path / "seen.jsonl"
    ledger = SeenLedger(ledger_path)
    ledger.load()
    ledger.mark_seen("sec-1", round_idx=0, source="synthetic")
    ledger.mark_seen("sec-2", round_idx=1, source="real_rollout")

    loaded = SeenLedger(ledger_path)
    loaded.load()

    assert loaded.is_seen("sec-1")
    assert loaded.is_seen("sec-2")
    assert loaded.seen_count() == 2
