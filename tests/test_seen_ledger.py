"""Unit tests for iSFT/seen_ledger.py.

All tests are pure-Python and require no GPU, Docker, or real manifest files.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

# The seen_ledger module lives under models/.../iSFT/ which is already on
# sys.path via conftest.py or directly importable when running from the
# training directory.  We add it explicitly so tests run from the repo root.
import sys

_TRAINING = (
    Path(__file__).resolve().parents[1]
    / "models" / "langslice-gemma-4" / "training"
)
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from iSFT.seen_ledger import SeenLedger, StaleLedgerError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ledger(tmp_path: Path, *, manifest_root: Path | None = None) -> SeenLedger:
    return SeenLedger(tmp_path / "ledger.jsonl", manifest_root=manifest_root)


def _make_fake_manifest(tmp_path: Path) -> Path:
    """Create a minimal fake manifest tree with one shard + one override file."""
    root = tmp_path / "manifest"
    shard_dir = root / "shards" / "coronal"
    shard_dir.mkdir(parents=True)
    override_dir = root / "overrides" / "coronal"
    override_dir.mkdir(parents=True)
    (shard_dir / "fake_dataset.jsonl").write_text(
        '{"section_id": "s1"}\n', encoding="utf-8"
    )
    (override_dir / "fake_dataset.json").write_text("{}", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Tests — SeenLedger
# ---------------------------------------------------------------------------

class TestSeenLedger:
    def test_load_empty_file_returns_empty_set(self, tmp_path: Path) -> None:
        """Fresh / non-existent file → empty seen set, no error."""
        ledger = _make_ledger(tmp_path)
        ledger.load()
        assert ledger.seen_count() == 0
        assert not ledger.is_seen("anything")

    def test_mark_and_query(self, tmp_path: Path) -> None:
        """mark_seen → is_seen returns True."""
        ledger = _make_ledger(tmp_path)
        ledger.load()
        ledger.mark_seen("sec_001", round_idx=0, source="synthetic")
        assert ledger.is_seen("sec_001")
        assert not ledger.is_seen("sec_002")
        assert ledger.seen_count() == 1

    def test_ledger_persists_across_load(self, tmp_path: Path) -> None:
        """Write entries in one instance; fresh instance load sees them."""
        ledger1 = _make_ledger(tmp_path)
        ledger1.load()
        ledger1.mark_seen("sec_A", round_idx=0, source="real_rollout")
        ledger1.mark_seen("sec_B", round_idx=1, source="synthetic")

        # New instance, same path.
        ledger2 = _make_ledger(tmp_path)
        ledger2.load()
        assert ledger2.is_seen("sec_A")
        assert ledger2.is_seen("sec_B")
        assert not ledger2.is_seen("sec_C")
        assert ledger2.seen_count() == 2

    def test_fingerprint_mismatch_raises(self, tmp_path: Path) -> None:
        """Write ledger with manifest, touch shard file mtime → StaleLedgerError."""
        manifest_root = _make_fake_manifest(tmp_path)

        # Write the ledger against the current manifest.
        ledger1 = SeenLedger(tmp_path / "ledger.jsonl", manifest_root=manifest_root)
        ledger1.load()
        ledger1.mark_seen("sec_001", round_idx=0, source="synthetic")

        # Simulate manifest change: touch the shard file with a new mtime.
        shard_file = manifest_root / "shards" / "coronal" / "fake_dataset.jsonl"
        # Use a future mtime to ensure mtime changes regardless of resolution.
        future_mtime = time.time() + 10.0
        os.utime(str(shard_file), (future_mtime, future_mtime))

        # Loading now should raise StaleLedgerError.
        ledger2 = SeenLedger(tmp_path / "ledger.jsonl", manifest_root=manifest_root)
        with pytest.raises(StaleLedgerError, match="fingerprint mismatch"):
            ledger2.load()

    def test_fingerprint_matches_no_change(self, tmp_path: Path) -> None:
        """Write ledger with manifest; reload without changes → no error."""
        manifest_root = _make_fake_manifest(tmp_path)
        ledger1 = SeenLedger(tmp_path / "ledger.jsonl", manifest_root=manifest_root)
        ledger1.load()
        ledger1.mark_seen("sec_001", round_idx=0, source="real_rollout")

        # Reload against same manifest — fingerprint should match.
        ledger2 = SeenLedger(tmp_path / "ledger.jsonl", manifest_root=manifest_root)
        ledger2.load()  # Must not raise.
        assert ledger2.is_seen("sec_001")

    def test_no_manifest_root_uses_degenerate_fingerprint(self, tmp_path: Path) -> None:
        """manifest_root=None → degenerate 'no-manifest' fingerprint; works fine."""
        ledger1 = SeenLedger(tmp_path / "ledger.jsonl", manifest_root=None)
        ledger1.load()
        ledger1.mark_seen("sec_X", round_idx=0, source="synthetic")

        ledger2 = SeenLedger(tmp_path / "ledger.jsonl", manifest_root=None)
        ledger2.load()  # Must not raise.
        assert ledger2.is_seen("sec_X")

    def test_mark_seen_source_written_to_disk(self, tmp_path: Path) -> None:
        """The 'source' field is persisted in the JSONL entries."""
        ledger = _make_ledger(tmp_path)
        ledger.load()
        ledger.mark_seen("sec_Y", round_idx=3, source="real_rollout")

        path = tmp_path / "ledger.jsonl"
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # First line is the header, second is the entry.
        assert len(lines) == 2
        entry = json.loads(lines[1])
        assert entry["source"] == "real_rollout"
        assert entry["round_idx"] == 3
        assert entry["section_id"] == "sec_Y"

    def test_seen_property_returns_set(self, tmp_path: Path) -> None:
        """The ``seen`` property returns the in-memory set."""
        ledger = _make_ledger(tmp_path)
        ledger.load()
        ledger.mark_seen("sec_P", round_idx=0, source="synthetic")
        assert "sec_P" in ledger.seen
        assert isinstance(ledger.seen, set)
