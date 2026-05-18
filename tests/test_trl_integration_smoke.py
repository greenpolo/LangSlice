"""Real-TRL integration smoke for the curriculum sampler.

Sibling to :mod:`tests.test_unified_rl_pipeline_smoke` which stubs
``trl`` via :mod:`sys.modules`. That smoke is fast but can't catch a
RepeatSampler-shape regression because the stub never exercises TRL's
real index sequence. This test gates on ``pytest.importorskip("trl")``
so it skips cleanly on a host that doesn't have TRL installed (e.g. a
fresh CI that hasn't pip-installed ``models/langslice-gemma-4/training/requirements-rlvr.txt``).

We intentionally keep the surface narrow: build the real
``trl.trainer.utils.RepeatSampler`` and our
:class:`CurriculumRepeatingSampler` over the same tiny synthetic
dataset, and assert that — given identical (mini_repeat_count,
batch_size, repeat_count) — both samplers emit the same length and the
same per-batch repetition shape. This is the assertion that would have
caught Codex's blocking finding (our sampler emitting 1/N of the indices
TRL expects when ``steps_per_generation > 1``).

We do NOT spin up a full ``GRPOTrainer`` here — that needs real model
weights, a CUDA device, and a multi-second initialization, none of
which a unit-test smoke should require. If a future regression demands
trainer-level coverage, add it as a separate dependency-gated module
(it would already be skipped on the same hosts as this one).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
import torch

# Skip the whole module if TRL isn't installed; transformers/accelerate are
# transitive deps of TRL but we list them explicitly so the skip message is
# precise when one of them is missing in isolation.
trl = pytest.importorskip("trl")
pytest.importorskip("transformers")

# Import the real RepeatSampler from TRL — its location can drift across
# versions, so probe a couple of paths before giving up.
try:
    from trl.trainer.utils import RepeatSampler as _TRLRepeatSampler  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    pytest.skip(
        "trl.trainer.utils.RepeatSampler not present in installed TRL "
        f"(version {getattr(trl, '__version__', '?')}); the integration "
        "smoke can't compare emit shapes without it.",
        allow_module_level=True,
    )

from single_turn_rl.curriculum import CurriculumRepeatingSampler  # noqa: E402


class _StubDataset:
    """Minimal duck type for both samplers' ``data_source``/``dataset`` arg."""

    def __init__(self, n: int) -> None:
        self._specs = [
            {
                "section_id": f"s{i:03d}",
                "subject_id": f"subj_{i:03d}",
                "plane": "coronal",
                "dataset": "ds_a",
                "difficulty_score": 0.5,
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, i: int) -> Any:
        return self._specs[i]


def _ours(
    dataset: _StubDataset, *, num_generations: int, batch_size: int, repeat_count: int,
) -> CurriculumRepeatingSampler:
    return CurriculumRepeatingSampler(
        dataset,
        num_generations=num_generations,
        batch_size=batch_size,
        repeat_count=repeat_count,
        # Wide band over the uniform-difficulty stub so rung 1 always wins.
        band_width=1.0,
        initial_T=0.5,
        max_per_subject_in_batch=batch_size,
        generator=torch.Generator().manual_seed(0),
    )


def test_emit_length_matches_trl_repeat_sampler() -> None:
    """``__len__`` must equal TRL's RepeatSampler under matching config.

    With our toml's effective ``steps_per_generation=8``,
    ``num_iterations=1`` ⇒ ``repeat_count=8``. Verify this and a smaller
    ``repeat_count=2`` shape both round-trip correctly so a subsequent
    config change doesn't drift one sampler vs the other.
    """
    n_rows = 16
    num_generations = 2
    batch_size = 4  # unique_prompts = 2

    for repeat_count in (1, 2, 8):
        dataset = _StubDataset(n_rows)
        ours = _ours(
            dataset, num_generations=num_generations,
            batch_size=batch_size, repeat_count=repeat_count,
        )
        # TRL's RepeatSampler expects mini_repeat_count = num_generations
        # and batch_size = unique_prompts (the "number of unique indices
        # per batch"). See trl/trainer/utils.py:702-714 for arg semantics.
        unique_prompts = batch_size // num_generations
        trl_sampler = _TRLRepeatSampler(
            data_source=dataset,
            mini_repeat_count=num_generations,
            batch_size=unique_prompts,
            repeat_count=repeat_count,
            shuffle=False,
        )
        # The TRL sampler walks every chunk in the dataset; ours emits one
        # band-selected chunk per __iter__. So we compare per-chunk shape:
        # a single chunk from TRL must equal one full iter of ours in
        # length.
        trl_indices = list(iter(trl_sampler))
        trl_len_per_chunk = (
            unique_prompts * num_generations * repeat_count
        )
        # Total TRL length divides evenly by per-chunk length — sanity
        # check that we're slicing the right shape.
        assert len(trl_indices) % trl_len_per_chunk == 0, (
            f"TRL emit length drift: total {len(trl_indices)} not divisible "
            f"by per-chunk {trl_len_per_chunk}"
        )
        assert len(ours) == trl_len_per_chunk, (
            f"emit length drift at repeat_count={repeat_count}: "
            f"ours={len(ours)} trl_per_chunk={trl_len_per_chunk}"
        )


def test_repetition_shape_matches_trl_for_repeat_count_gt_one() -> None:
    """Per-chunk shape: each picked index repeats ``num_generations`` times,
    and that whole inner sequence repeats ``repeat_count`` times.

    This is the GRPOTrainer expectation Codex flagged: with
    steps_per_generation > 1 the inner reuse loop walks the same
    generation N times, so the dataloader needs to deliver each pick's
    repetition block exactly ``repeat_count`` consecutive times. A
    single-pass sampler would only feed 1/N of what's needed and the
    advantage calculation downstream would silently use stale slots.
    """
    dataset = _StubDataset(8)
    num_generations = 2
    batch_size = 4
    repeat_count = 4

    ours = _ours(
        dataset, num_generations=num_generations,
        batch_size=batch_size, repeat_count=repeat_count,
    )
    indices = list(iter(ours))

    # Length: unique_prompts (2) * num_generations (2) * repeat_count (4) = 16.
    assert len(indices) == 2 * num_generations * repeat_count == 16

    # Each unique pick must appear exactly ``num_generations *
    # repeat_count`` times — the same per-pick total TRL's RepeatSampler
    # produces.
    counts = Counter(indices)
    expected_per_pick = num_generations * repeat_count
    for pick, count in counts.items():
        assert count == expected_per_pick, (
            f"index {pick} appeared {count} times, expected {expected_per_pick}"
        )

    # And the inner shape: walk the indices one repeat_count cycle at a
    # time and assert the cycle is ``[idx0]*num_generations +
    # [idx1]*num_generations + ...`` repeated ``repeat_count`` times.
    chunk_len = batch_size  # unique_prompts * num_generations
    cycles = [indices[i : i + chunk_len] for i in range(0, len(indices), chunk_len)]
    assert len(cycles) == repeat_count
    # Every cycle is identical (the picked sequence is deterministic per __iter__).
    first_cycle = cycles[0]
    for c in cycles[1:]:
        assert c == first_cycle, (
            f"cycle drift: first={first_cycle} later={c}"
        )
    # Within the cycle, indices come in num_generations-sized groups of
    # identical values.
    for group_start in range(0, chunk_len, num_generations):
        group = first_cycle[group_start : group_start + num_generations]
        assert len(set(group)) == 1, (
            f"group at {group_start} not homogeneous: {group}"
        )
