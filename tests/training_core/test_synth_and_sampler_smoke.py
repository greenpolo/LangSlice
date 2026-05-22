"""Smoke tests for code that lost dedicated unit tests in the
training-core cleanup (2026-05-22):

- ``synthdata.sft_corpus.generate_synthetic_rows`` (formerly tested by the
  deleted ``test_synth_corpus.py``, ~822 lines).
- ``synthdata.atlas_signature.compose_visible_clues`` (formerly tested by
  the deleted ``test_atlas_signature.py``, ~214 lines).
- ``sft.train_sft.ShapeBucketBatchSampler`` (half-tested by the deleted
  ``test_bucketed_sampler_and_double_lora.py``; the iSFT half went away
  with iSFT but this sampler is still load-bearing for PDBS=4 training).

These are happy-path smokes — they catch import regressions, the
fallback-when-atlas-missing path, and shape-pure batching invariants.
"""

from __future__ import annotations

import random
import types


def test_compose_visible_clues_returns_fallback_for_unknown_atlas() -> None:
    """Unknown atlas → graceful fallback, no exception."""
    from synthdata.atlas_signature import compose_visible_clues

    result = compose_visible_clues("nonexistent_atlas_xyz", "coronal", 5.0)
    assert "atlas signature unavailable" in result


def test_generate_synthetic_rows_emits_one_row_per_spec_in_region_dump_mode() -> None:
    """region_dump mode (default) emits exactly one row per SectionSpec."""
    from synthdata.sft_corpus import SectionSpec, generate_synthetic_rows

    specs = [
        SectionSpec(
            plane="coronal",
            dataset="ds_a",
            subject_id="S1",
            section_id="S1:001",
            image_path="models/langslice-gemma-4/data/queries/foo.jpg",
            ground_truth_mm=5.0,
            atlas_name="allen_mouse_25um",
            atlas_version="v1.2",
        ),
    ]
    rows = generate_synthetic_rows(
        specs,
        multi_turn_floor=0.0,  # all answer-only — no atlas-image lookups
        rng=random.Random(0),
        grid=[4.0, 4.5, 5.0, 5.5, 6.0],  # synthetic grid avoids disk
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["plane"] == "coronal"
    assert row["subject_id"] == "S1"
    assert len(row["trace"]) == 1, "answer-only row should be submit-only"
    assert row["trace"][0]["submit"]["args"]["position_mm"] == 5.0


def test_shape_bucket_batch_sampler_yields_shape_pure_batches() -> None:
    """Each yielded batch is mono-shape (all-answer-only or all-multi-turn)."""
    from sft.train_sft import ShapeBucketBatchSampler

    examples = (
        [types.SimpleNamespace(trace=[{}])] * 6        # 6 answer-only (len 1)
        + [types.SimpleNamespace(trace=[{}] * 3)] * 4  # 4 multi-turn (len 3)
    )
    sampler = ShapeBucketBatchSampler(examples, batch_size=2, seed=42)
    batches = list(iter(sampler))

    assert len(batches) == 5  # 6//2 + 4//2 = 3 + 2
    for batch in batches:
        trace_lens = {len(examples[i].trace) for i in batch}
        assert len(trace_lens) == 1, f"mixed-shape batch: {batch}"


def test_shape_bucket_batch_sampler_drops_incomplete_tail() -> None:
    """Incomplete tail batches are dropped to keep batch size uniform."""
    from sft.train_sft import ShapeBucketBatchSampler

    examples = [types.SimpleNamespace(trace=[{}])] * 5  # 5 answer-only rows
    sampler = ShapeBucketBatchSampler(examples, batch_size=2, seed=0)
    batches = list(iter(sampler))

    assert len(batches) == 2  # 5 // 2 = 2 full, last row dropped
