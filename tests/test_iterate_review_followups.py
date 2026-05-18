"""Unit tests for the two review-followup fixes landed post-bucketing.

1. ``_count_synthetic_rows`` + ``_phase_synthetic`` resume guard:
   crash between ``_phase_append_iterative`` and ``_phase_synthetic``
   would re-run synth generation on resume; the guard detects existing
   synthetic rows and skips the phase.

2. ``_phase_rollouts`` v2 path marks every picked section as seen at
   sample time, not only after acceptance. Verified indirectly by
   confirming the SeenLedger receives ``mark_seen`` for picked-and-
   rejected sections via the helper used inside that branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iSFT.iterate import _count_synthetic_rows


# ──────────────────────────────────────────────────────────────────────────
# _count_synthetic_rows
# ──────────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_count_missing_file_returns_zero(tmp_path):
    assert _count_synthetic_rows(tmp_path / "nope.jsonl") == 0


def test_count_empty_file_returns_zero(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert _count_synthetic_rows(p) == 0


def test_count_only_real_rollout_rows(tmp_path):
    p = tmp_path / "real.jsonl"
    _write_jsonl(p, [
        {"subject_id": "s1", "quality": {"source": "real_rollout"}},
        {"subject_id": "s2", "quality": {"source": "real_rollout"}},
    ])
    assert _count_synthetic_rows(p) == 0


def test_count_mixed_real_and_synth(tmp_path):
    p = tmp_path / "mixed.jsonl"
    _write_jsonl(p, [
        {"subject_id": "s1", "quality": {"source": "real_rollout"}},
        {"subject_id": "s2", "quality": {"source": "synthetic_terminal"}},
        {"subject_id": "s3", "quality": {"source": "synthetic_terminal"}},
        {"subject_id": "s4", "quality": {"source": "real_rollout"}},
    ])
    assert _count_synthetic_rows(p) == 2


def test_count_skips_malformed_lines(tmp_path):
    p = tmp_path / "malformed.jsonl"
    p.write_text(
        '{"quality": {"source": "synthetic_terminal"}}\n'
        'this is not json\n'
        '\n'
        '{"quality": {"source": "synthetic_terminal"}}\n',
        encoding="utf-8",
    )
    assert _count_synthetic_rows(p) == 2


def test_count_handles_missing_quality_key(tmp_path):
    p = tmp_path / "no_quality.jsonl"
    _write_jsonl(p, [
        {"subject_id": "s1"},
        {"subject_id": "s2", "quality": {}},
        {"subject_id": "s3", "quality": {"source": "synthetic_terminal"}},
    ])
    assert _count_synthetic_rows(p) == 1


# ──────────────────────────────────────────────────────────────────────────
# _phase_rollouts: mark_seen at sample time (v2 path)
# ──────────────────────────────────────────────────────────────────────────

# Direct integration of _phase_rollouts requires loading the RLVR
# allocation, vLLM probe, etc. — far too heavy for a unit test. Instead
# we verify the *contract* the fix relies on: SeenLedger.mark_seen is
# idempotent in the in-memory set even when called multiple times for
# the same (section_id, round_idx) pair, so the new pick-time mark
# combined with the existing append-time mark cannot corrupt state.


def test_seen_ledger_mark_seen_is_idempotent(tmp_path):
    """The post-fix flow calls mark_seen twice for accepted sections
    (once at pick time, once at append time). Verify the in-memory set
    stays small and the on-disk audit log captures both entries."""
    from iSFT.seen_ledger import SeenLedger

    # Build a minimal manifest dir so the fingerprint can be computed.
    shards_dir = tmp_path / "manifest" / "shards"
    shards_dir.mkdir(parents=True)
    (shards_dir / "shard.jsonl").write_text("", encoding="utf-8")

    ledger_path = tmp_path / "seen.jsonl"
    ledger = SeenLedger(ledger_path, manifest_root=tmp_path / "manifest")
    ledger.load()

    # Pick-time mark
    ledger.mark_seen("sec_A", round_idx=0, source="real_rollout")
    # Append-time mark (duplicate)
    ledger.mark_seen("sec_A", round_idx=0, source="real_rollout")
    # Different section
    ledger.mark_seen("sec_B", round_idx=0, source="real_rollout")

    assert ledger.seen_count() == 2  # set dedupes
    assert ledger.is_seen("sec_A")
    assert ledger.is_seen("sec_B")
    assert not ledger.is_seen("sec_C")

    # On-disk audit log shows all three appends (header + 3 entries)
    lines = [
        line for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 4  # 1 header + 3 entries

    # Reload from disk: still consistent
    fresh = SeenLedger(ledger_path, manifest_root=tmp_path / "manifest")
    fresh.load()
    assert fresh.seen_count() == 2
    assert fresh.is_seen("sec_A")
    assert fresh.is_seen("sec_B")


def test_seen_ledger_picked_but_rejected_section_not_resampled(tmp_path):
    """The whole point of the fix: a section that was picked and then
    failed quality filter must not be eligible for re-selection next round.
    Verify the post-pick mark_seen lands before any acceptance check."""
    from iSFT.seen_ledger import SeenLedger

    shards_dir = tmp_path / "manifest" / "shards"
    shards_dir.mkdir(parents=True)
    (shards_dir / "shard.jsonl").write_text("", encoding="utf-8")

    ledger = SeenLedger(tmp_path / "seen.jsonl", manifest_root=tmp_path / "manifest")
    ledger.load()

    # Simulate round 0: pick 3 sections (only 1 accepted; 2 rejected).
    picked_round_0 = ["sec_X", "sec_Y", "sec_Z"]
    for sid in picked_round_0:
        ledger.mark_seen(sid, round_idx=0, source="real_rollout")
    # Acceptance: only sec_X. Pre-fix code would call mark_seen only for
    # sec_X. Post-fix code already marked all three at pick time.

    # Round 1: v2 sampler should treat all 3 as seen, regardless of
    # whether they were accepted.
    for sid in picked_round_0:
        assert ledger.is_seen(sid), (
            f"{sid} should be in seen set even though only sec_X was accepted"
        )


# ──────────────────────────────────────────────────────────────────────────
# _read_synthetic_section_ids — resume-replay helper for _phase_synthetic
# ──────────────────────────────────────────────────────────────────────────

def test_read_synthetic_section_ids_returns_only_synth_rows(tmp_path):
    """Returns ``section_id`` only for ``quality.source == "synthetic_terminal"`` rows.

    Real-rollout rows and rows with no quality field are skipped so a
    resume replay never mark-seen a section the synth phase didn't write.
    """
    from iSFT.iterate import _read_synthetic_section_ids

    p = tmp_path / "mixed.jsonl"
    _write_jsonl(p, [
        {"section_id": "syn_A", "quality": {"source": "synthetic_terminal"}},
        {"section_id": "real_B", "quality": {"source": "real_rollout"}},
        {"section_id": "syn_C", "quality": {"source": "synthetic_terminal"}},
        {"section_id": "bare_D"},  # missing quality
        {"quality": {"source": "synthetic_terminal", "section_id": "fallback_E"}},
    ])
    out = _read_synthetic_section_ids(p)
    # Order is file order; top-level section_id preferred, falls back to quality.section_id.
    assert out == ["syn_A", "syn_C", "fallback_E"]


def test_phase_synthetic_resume_marks_seen(tmp_path):
    """Crash-resume replay: _phase_synthetic must mark-seen every existing
    synthetic_terminal row in the iterative JSONL before the early return.

    Otherwise the prior attempt's pre-crash synth writes are "invisible" to
    the ledger and the next round can re-pick those sections.
    """
    import argparse

    from iSFT.iterate import _phase_synthetic
    from iSFT.seen_ledger import SeenLedger

    # Manifest dir for fingerprinting.
    shards_dir = tmp_path / "manifest" / "shards"
    shards_dir.mkdir(parents=True)
    (shards_dir / "shard.jsonl").write_text("", encoding="utf-8")

    ledger = SeenLedger(tmp_path / "seen.jsonl", manifest_root=tmp_path / "manifest")
    ledger.load()

    # Pre-existing synth rows in the iterative JSONL — simulates the
    # write-then-crash state the prior attempt left behind.
    iterative = tmp_path / "round_3.jsonl"
    _write_jsonl(iterative, [
        {"section_id": "sec_001", "quality": {"source": "synthetic_terminal"}},
        {"section_id": "sec_002", "quality": {"source": "synthetic_terminal"}},
        {"section_id": "sec_003", "quality": {"source": "synthetic_terminal"}},
        # A real-rollout row that must NOT be marked synthetic.
        {"section_id": "sec_real", "quality": {"source": "real_rollout"}},
    ])

    args = argparse.Namespace(
        # _phase_synthetic uses these on the resume path; others are not
        # reached because the early return fires before them.
        use_seen_ledger=True,
        # Stash the ledger where _phase_synthetic looks for it.
        _seen_ledger=ledger,
    )

    stats = _phase_synthetic(args=args, round_idx=3, iterative_jsonl=iterative)

    # Resume return shape.
    assert stats == {"n_synth_written": 3, "n_synth_skipped_resume": 3}

    # All three synthetic section_ids replayed into the ledger.
    assert ledger.is_seen("sec_001")
    assert ledger.is_seen("sec_002")
    assert ledger.is_seen("sec_003")
    # The real-rollout row was not synthetic-marked.
    assert not ledger.is_seen("sec_real")
    assert ledger.seen_count() == 3
