"""Unit tests for the Phase 3 bucketed-shape sampler and the iSFT
double-LoRA assertion.  Both are pure-Python additions landed
post-review (no Docker, no TRL, no GPU)."""

from __future__ import annotations

import argparse
import types

import pytest
from iSFT.iterate import _assert_no_double_lora
from sft.train_sft import ShapeBucketBatchSampler


def _example(trace_len: int):
    return types.SimpleNamespace(trace=[{}] * trace_len)


# ──────────────────────────────────────────────────────────────────────────
# ShapeBucketBatchSampler
# ──────────────────────────────────────────────────────────────────────────

def test_bucketing_separates_answer_only_and_multi_turn():
    # 6 answer-only (trace=1) + 4 multi-turn (trace=3)
    examples = [_example(1)] * 6 + [_example(3)] * 4
    sampler = ShapeBucketBatchSampler(examples, batch_size=2, seed=42)
    batches = list(iter(sampler))
    # All batches must be shape-pure (every index in a batch has same trace len)
    for batch in batches:
        trace_lens = {len(examples[i].trace) for i in batch}
        assert len(trace_lens) == 1, f"mixed-shape batch {batch}: lens={trace_lens}"
    # 6//2 + 4//2 = 5 batches
    assert len(batches) == 5


def test_bucketing_drops_incomplete_tail():
    # 5 single-bucket rows, batch_size=2 → 2 batches, drop 1 row
    examples = [_example(1)] * 5
    sampler = ShapeBucketBatchSampler(examples, batch_size=2, seed=0)
    batches = list(iter(sampler))
    assert len(batches) == 2
    all_picked = sorted(i for b in batches for i in b)
    assert len(all_picked) == 4  # one row dropped per epoch


def test_bucketing_no_collisions_within_batch():
    examples = [_example(1)] * 8 + [_example(2)] * 8
    sampler = ShapeBucketBatchSampler(examples, batch_size=4, seed=1)
    batches = list(iter(sampler))
    for batch in batches:
        assert len(set(batch)) == len(batch), f"duplicate idx in batch {batch}"


def test_bucketing_seed_advances_each_epoch():
    examples = [_example(1)] * 8 + [_example(2)] * 8
    sampler = ShapeBucketBatchSampler(examples, batch_size=2, seed=7)
    epoch1 = list(iter(sampler))
    epoch2 = list(iter(sampler))
    # Different epochs should produce different orderings (PRNG advanced)
    assert epoch1 != epoch2, "seed did not advance between epochs"


def test_bucketing_single_bucket_collapses_to_shuffle():
    # All multi-turn — bucketing should still produce correct batches
    examples = [_example(3)] * 10
    sampler = ShapeBucketBatchSampler(examples, batch_size=2, seed=0)
    batches = list(iter(sampler))
    assert len(batches) == 5
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(10))


def test_bucketing_len_matches_iter():
    examples = [_example(1)] * 7 + [_example(3)] * 9
    sampler = ShapeBucketBatchSampler(examples, batch_size=3, seed=0)
    assert len(sampler) == len(list(iter(sampler)))


# ──────────────────────────────────────────────────────────────────────────
# _assert_no_double_lora
# ──────────────────────────────────────────────────────────────────────────

def _make_args(*, vllm_lora_mode: bool, sft_initial_adapter):
    return argparse.Namespace(
        vllm_lora_mode=vllm_lora_mode,
        sft_initial_adapter=sft_initial_adapter,
    )


def test_assert_blocks_lora_mode_plus_initial_adapter(tmp_path):
    adapter_path = tmp_path / "adapter"
    args = _make_args(vllm_lora_mode=True, sft_initial_adapter=adapter_path)
    with pytest.raises(RuntimeError, match="double-LoRA"):
        _assert_no_double_lora(args)


def test_assert_allows_lora_mode_without_initial_adapter():
    args = _make_args(vllm_lora_mode=True, sft_initial_adapter=None)
    _assert_no_double_lora(args)  # no raise


def test_assert_allows_initial_adapter_without_lora_mode(tmp_path):
    args = _make_args(vllm_lora_mode=False, sft_initial_adapter=tmp_path / "a")
    _assert_no_double_lora(args)  # no raise


def test_assert_allows_neither():
    args = _make_args(vllm_lora_mode=False, sft_initial_adapter=None)
    _assert_no_double_lora(args)  # no raise
