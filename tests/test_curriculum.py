"""Unit tests for ``single_turn_rl.curriculum``.

The sampler + callback pair has zero TRL/Unsloth runtime dependency at the
unit-test layer: the dataset is stubbed via dict specs (the same shape
:mod:`single_turn_rl.dataset` builds), the schedule + bin-difficulty map
are faked, and the trainer state is a tiny ``SimpleNamespace`` with a
``log_history`` list.

Coverage map:

* Sampler shape (1-2): ``__len__`` and per-batch index count.
* GRPO repetition (3): each unique prompt repeats ``num_generations`` times.
* Band selection (4): all picked indices land in
  ``[T - band_width, T + band_width]`` when ladder rung 1 succeeds.
* Ladder rungs (5-8): each of the four fallback levels exercised in
  isolation by shaping the dataset around the constraints.
* Subject cap (9): no subject exceeds the cap on rung 1.
* AdaRFT update (10-14): zero-step at ``R == target_reward``, monotonic
  moves up/down, clamping at ``d_min`` / ``d_max``, and EMA smoothing.
* Reproducibility (15): two samplers with the same seed emit the same
  index stream.
* AP-bin refresh: sampler re-reads :class:`BinDifficultyMap` between
  iterations; missing keys fall back to the construction-time score.
* Callback: empty/missing reward is a no-op; reward present forwards to
  ``sampler.update`` and (optionally) writes per-bin running means back
  into the shared :class:`BinDifficultyMap`; plane filter narrows the
  write-back; reward rescaling clamps negative raw rewards before AdaRFT.
"""

from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from single_turn_rl import adaptive_reward as ar
from single_turn_rl.curriculum import (
    AdaRFTCurriculumCallback,
    BinDifficultyMap,
    CurriculumRepeatingSampler,
)

# ---------------------------------------------------------------------------
# Fake AdaptiveRewardSchedule + recent_errors hook
# ---------------------------------------------------------------------------


class _FakeSchedule:
    """Minimal stand-in for :class:`AdaptiveRewardSchedule`.

    The callback only calls :meth:`per_bin_difficulty`. Tests seed
    ``_bin_means`` directly to mimic a populated rolling buffer
    without going through the real adaptive_reward module.
    """

    def __init__(
        self,
        per_bin_means: dict[tuple[str, int], float] | None = None,
    ) -> None:
        self._bin_means = per_bin_means or {}

    def per_bin_difficulty(self, plane: str, ap_bin: int) -> float | None:
        return self._bin_means.get((plane, ap_bin))


class _RecentErrorsBuffer:
    """A tuple-shaped surrogate for ``adaptive_reward._RECENT_ERRORS``.

    The callback calls the module-level ``recent_errors`` symbol; the
    ``fake_buffer`` fixture monkeypatches that symbol on
    ``single_turn_rl.curriculum`` to point at this buffer's
    :meth:`snapshot` so we don't have to mutate the real module's
    rolling deque.
    """

    def __init__(self) -> None:
        self.entries: list[tuple[float, str, int]] = []

    def add(
        self,
        abs_err_pct: float,
        plane: str,
        ap_bin: int,
    ) -> None:
        self.entries.append((abs_err_pct, plane, ap_bin))

    def snapshot(self) -> list[tuple[float, str, int]]:
        return list(self.entries)


@pytest.fixture
def fake_buffer(monkeypatch: pytest.MonkeyPatch) -> _RecentErrorsBuffer:
    """Provide a controlled rolling-error buffer to the curriculum module."""
    buf = _RecentErrorsBuffer()
    monkeypatch.setattr("single_turn_rl.curriculum.recent_errors", buf.snapshot)
    return buf


def _spec(
    *,
    section_id: str,
    subject_id: str,
    difficulty_score: float | None,
    plane: str = "coronal",
    dataset: str = "ds_a",
    ap_bin: int | None = 10,
    ground_truth_mm: float = 5.0,
    valid_range_mm: tuple[float, float] = (0.0, 10.0),
) -> dict[str, Any]:
    """Build a minimal spec dict matching what
    :mod:`single_turn_rl.dataset`'s ``_spec_from_state`` emits — only the
    fields the curriculum sampler actually reads."""
    return {
        "section_id": section_id,
        "subject_id": subject_id,
        "plane": plane,
        "dataset": dataset,
        "difficulty_score": difficulty_score,
        "ap_bin": ap_bin,
        "ground_truth_mm": ground_truth_mm,
        "valid_range_mm": valid_range_mm,
    }


class _StubDataset:
    """Duck-typed stand-in for :class:`single_turn_rl.dataset.RowDataset`.

    The sampler reads ``_specs`` directly; ``__len__`` is just for
    ergonomics in tests that want ``len(dataset)``.
    """

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        self._specs = specs

    def __len__(self) -> int:
        return len(self._specs)


def _make_dataset_uniform_difficulty(
    n: int,
    *,
    difficulty: float = 0.5,
    plane: str = "coronal",
    dataset: str = "ds_a",
) -> _StubDataset:
    """All rows at one difficulty + each row has its own subject (no cap pressure)."""
    specs = [
        _spec(
            section_id=f"s{i:03d}",
            subject_id=f"subj_{i:03d}",
            difficulty_score=difficulty,
            plane=plane,
            dataset=dataset,
            ap_bin=i % 20,
        )
        for i in range(n)
    ]
    return _StubDataset(specs)


def _make_dataset_with_difficulty_grid(
    *,
    n_per_bin: int = 4,
    difficulties: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> _StubDataset:
    """A row at each difficulty in ``difficulties`` repeated ``n_per_bin`` times,
    with a unique subject per row so the cap never bites."""
    specs: list[dict[str, Any]] = []
    counter = 0
    for d in difficulties:
        for _ in range(n_per_bin):
            specs.append(
                _spec(
                    section_id=f"s{counter:03d}",
                    subject_id=f"subj_{counter:03d}",
                    difficulty_score=d,
                    ap_bin=counter % 20,
                )
            )
            counter += 1
    return _StubDataset(specs)


# ---------------------------------------------------------------------------
# 1. __len__ shape
# ---------------------------------------------------------------------------


def test_len_returns_unique_prompts_times_num_generations() -> None:
    dataset = _make_dataset_uniform_difficulty(20)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=4,
        batch_size=8,  # unique_prompts = 8 / 4 = 2
        initial_T=0.5,
    )
    assert sampler.unique_prompts == 2
    assert len(sampler) == 2 * 4


# ---------------------------------------------------------------------------
# 2. __iter__ yields one batch
# ---------------------------------------------------------------------------


def test_iter_yields_unique_prompts_times_num_generations() -> None:
    dataset = _make_dataset_uniform_difficulty(20)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=4,
        batch_size=8,
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    indices = list(iter(sampler))
    assert len(indices) == 8


# ---------------------------------------------------------------------------
# 3. GRPO grouping — each unique prompt repeats num_generations times
# ---------------------------------------------------------------------------


def test_each_unique_prompt_appears_num_generations_times() -> None:
    """Plan invariant verified at trl/trainer/grpo_trainer.py:2235-2241."""
    dataset = _make_dataset_uniform_difficulty(20)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=4,
        batch_size=12,  # unique_prompts = 3
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    indices = list(iter(sampler))
    # Indices must come in groups of ``num_generations`` identical values.
    assert len(indices) == 12
    for group_start in (0, 4, 8):
        group = indices[group_start : group_start + 4]
        assert len(set(group)) == 1, f"group {group_start} not homogeneous: {group}"
    # And the three group leaders must be distinct (sampling without replacement).
    leaders = [indices[0], indices[4], indices[8]]
    assert len(set(leaders)) == 3


# ---------------------------------------------------------------------------
# 4. Band selection — all picks land within [T - band_width, T + band_width]
# ---------------------------------------------------------------------------


def test_picks_lie_in_band_when_ladder_level_one_succeeds() -> None:
    dataset = _make_dataset_with_difficulty_grid(
        n_per_bin=10,
        difficulties=(0.1, 0.3, 0.5, 0.7, 0.9),
    )
    T = 0.5
    band = 0.15
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=8,  # unique_prompts = 4
        band_width=band,
        initial_T=T,
        generator=torch.Generator().manual_seed(0),
    )
    indices = list(iter(sampler))
    assert sampler.last_ladder_level == 1
    # Every chosen index must have a difficulty in [T-band, T+band].
    for idx in indices:
        d = sampler._difficulty_for_index(idx)
        assert (T - band) <= d <= (T + band), f"idx {idx} d={d} outside band"


# ---------------------------------------------------------------------------
# 5-8. Ladder levels 1-4
# ---------------------------------------------------------------------------


def test_ladder_level_1_when_band_has_enough() -> None:
    """Plenty of in-band candidates → no fallback."""
    dataset = _make_dataset_uniform_difficulty(50, difficulty=0.5)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=8,
        band_width=0.15,
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    list(iter(sampler))
    assert sampler.last_ladder_level == 1


def test_ladder_level_2_when_band_must_widen() -> None:
    """No rows in the tight band, but enough within a 50%-widened band.

    Setup: ``unique_prompts = 4``. Tight band of ±0.05 around T=0.5
    is [0.45, 0.55]; widened ±0.075 band is [0.425, 0.575]. Rows at
    difficulty 0.56 sit OUTSIDE the tight band but INSIDE the widened
    band, so rung 1 yields zero candidates and rung 2 succeeds.
    """
    dataset = _make_dataset_uniform_difficulty(8, difficulty=0.56)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=8,  # unique_prompts = 4
        band_width=0.05,
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    list(iter(sampler))
    assert sampler.last_ladder_level == 2


def test_ladder_level_3_when_subject_cap_must_relax() -> None:
    """Enough rows in band, but subject cap is the binding constraint.

    Setup: ``unique_prompts = 4`` and ``max_per_subject_in_batch = 1``,
    but only 2 subjects exist (each with many rows). Rung 1 caps at 2
    picks (one per subject); rung 2 widens the band but the cap still
    binds; rung 3 lifts the cap and we reach 4.
    """
    # 6 rows split across 2 subjects, ALL at difficulty 0.5 so band
    # selection would otherwise succeed.
    specs: list[dict[str, Any]] = []
    for i in range(6):
        specs.append(
            _spec(
                section_id=f"s{i:03d}",
                subject_id=f"subj_{i % 2}",  # only 2 unique subjects
                difficulty_score=0.5,
                ap_bin=i % 20,
            )
        )
    dataset = _StubDataset(specs)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=8,  # unique_prompts = 4
        band_width=0.15,
        initial_T=0.5,
        max_per_subject_in_batch=1,
        generator=torch.Generator().manual_seed(0),
    )
    list(iter(sampler))
    assert sampler.last_ladder_level == 3


def test_rung_4_raises_when_dataset_smaller_than_unique_prompts() -> None:
    """When the dataset has fewer distinct rows than unique_prompts, rung 4
    must raise rather than silently repeat picks.

    Repeating prompts across distinct GRPO group slots would collapse the
    group-relative advantage assumption (TRL expects identical prompts only
    within a num_generations group, never across them).
    """
    # 3 rows, but unique_prompts=4 (batch_size=8 / num_generations=2).
    dataset = _make_dataset_uniform_difficulty(3, difficulty=0.95)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=8,
        band_width=0.05,
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    with pytest.raises(RuntimeError, match="rung 4"):
        list(iter(sampler))


def test_ladder_level_4_when_uniform_backfill_fires() -> None:
    """No rows survive even the widened band → uniform backfill.

    Setup: every row at difficulty 0.95. T=0.5, band=0.05; widened
    band ±0.075 still excludes 0.95. Subject cap relaxation can't help
    because there are no in-band candidates to begin with. Falls back
    to rung 4.
    """
    # All rows at difficulty 0.95; each row gets a unique subject so cap
    # isn't a confound.
    dataset = _make_dataset_uniform_difficulty(8, difficulty=0.95)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=8,  # unique_prompts = 4
        band_width=0.05,
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    list(iter(sampler))
    assert sampler.last_ladder_level == 4


# ---------------------------------------------------------------------------
# 9. Subject cap honoured at ladder rung 1
# ---------------------------------------------------------------------------


def test_subject_cap_honoured_at_rung_1() -> None:
    """When rung 1 succeeds, no subject exceeds ``max_per_subject_in_batch``."""
    # Lots of rows at difficulty 0.5 across 4 subjects.
    specs = [
        _spec(
            section_id=f"s{i:03d}",
            subject_id=f"subj_{i % 4}",  # 4 unique subjects
            difficulty_score=0.5,
            ap_bin=i % 20,
        )
        for i in range(40)
    ]
    dataset = _StubDataset(specs)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=8,  # unique_prompts = 4
        band_width=0.15,
        initial_T=0.5,
        max_per_subject_in_batch=2,
        generator=torch.Generator().manual_seed(0),
    )
    indices = list(iter(sampler))
    assert sampler.last_ladder_level == 1
    # Pull the unique-prompt slots (every num_generations-th index is a
    # group leader, but a leader's repeats also share the same subject,
    # so simplest is to count subject occurrences over the unique
    # prompts only).
    unique = indices[::2]  # one per group-of-2
    subjects = [sampler._subject_ids[i] for i in unique]
    counts: dict[str, int] = {}
    for subj in subjects:
        counts[subj] = counts.get(subj, 0) + 1
    for subj, c in counts.items():
        assert c <= 2, f"subject {subj} has {c} slots (cap was 2)"


# ---------------------------------------------------------------------------
# 10. update(target_reward) → no movement
# ---------------------------------------------------------------------------


def test_update_with_target_reward_does_not_move_T() -> None:
    """tanh(0) = 0 → smoothed = 0 → T unchanged."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        initial_T=0.5,
    )
    sampler.update(0.5)
    assert sampler.T == pytest.approx(0.5)
    assert sampler.last_step == pytest.approx(0.0)
    assert sampler.last_smoothed_step == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. update(1.0) → T moves toward d_max
# ---------------------------------------------------------------------------


def test_update_high_reward_moves_T_up() -> None:
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        d_min=0.0,
        d_max=1.0,
        initial_T=0.5,
    )
    initial_T = sampler.T
    sampler.update(1.0)
    expected_raw = 0.05 * math.tanh(2.0 * (1.0 - 0.5))
    expected_smoothed = (1.0 - 0.7) * expected_raw  # prev_step is 0.0
    assert sampler.T > initial_T
    assert sampler.T == pytest.approx(initial_T + expected_smoothed)
    assert sampler.last_step == pytest.approx(expected_raw)


# ---------------------------------------------------------------------------
# 12. update(0.0) → T moves toward d_min
# ---------------------------------------------------------------------------


def test_update_low_reward_moves_T_down() -> None:
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        d_min=0.0,
        d_max=1.0,
        initial_T=0.5,
    )
    sampler.update(0.0)
    expected_raw = 0.05 * math.tanh(2.0 * (0.0 - 0.5))  # negative
    assert expected_raw < 0
    assert sampler.T < 0.5


# ---------------------------------------------------------------------------
# 13. T clamped to [d_min, d_max]
# ---------------------------------------------------------------------------


def test_update_clamps_at_d_max() -> None:
    """Many high-reward updates → T saturates at d_max, never exceeds it."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=10.0,  # steep tanh → big steps
        eta=0.5,  # large step size
        ema_decay=0.0,  # no smoothing
        d_min=0.0,
        d_max=1.0,
        initial_T=0.95,
    )
    for _ in range(50):
        sampler.update(1.0)
    assert sampler.T == pytest.approx(1.0)


def test_update_clamps_at_d_min() -> None:
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=10.0,
        eta=0.5,
        ema_decay=0.0,
        d_min=0.0,
        d_max=1.0,
        initial_T=0.05,
    )
    for _ in range(50):
        sampler.update(0.0)
    assert sampler.T == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 14. EMA smoothing — two updates with same R move T less than 2 * raw
# ---------------------------------------------------------------------------


def test_ema_smoothing_dampens_back_to_back_updates() -> None:
    """Smoothed step on the first update is (1-decay) * raw, not raw itself."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        initial_T=0.5,
    )
    sampler.update(1.0)
    raw = sampler.last_step
    smoothed = sampler.last_smoothed_step
    assert smoothed == pytest.approx((1.0 - 0.7) * raw)
    assert smoothed < raw

    sampler2 = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        initial_T=0.5,
    )
    sampler2.update(1.0)
    sampler2.update(1.0)
    cumulative_T_move = sampler2.T - 0.5
    assert cumulative_T_move < 2.0 * raw


# ---------------------------------------------------------------------------
# 15. Reproducibility — same seed → same stream
# ---------------------------------------------------------------------------


def test_two_samplers_same_seed_same_stream() -> None:
    dataset = _make_dataset_uniform_difficulty(20)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    s1 = CurriculumRepeatingSampler(
        dataset,
        num_generations=3,
        batch_size=9,
        initial_T=0.5,
        generator=g1,
    )
    s2 = CurriculumRepeatingSampler(
        dataset,
        num_generations=3,
        batch_size=9,
        initial_T=0.5,
        generator=g2,
    )
    out1 = list(iter(s1)) + list(iter(s1))
    out2 = list(iter(s2)) + list(iter(s2))
    assert out1 == out2


# ---------------------------------------------------------------------------
# Sampler ctor validation
# ---------------------------------------------------------------------------


def test_ctor_rejects_batch_size_not_divisible_by_num_generations() -> None:
    dataset = _make_dataset_uniform_difficulty(4)
    with pytest.raises(ValueError, match="must be divisible by num_generations"):
        CurriculumRepeatingSampler(
            dataset,
            num_generations=4,
            batch_size=10,
        )


def test_ctor_rejects_empty_dataset() -> None:
    dataset = _StubDataset([])
    with pytest.raises(ValueError, match="non-empty"):
        CurriculumRepeatingSampler(
            dataset,
            num_generations=2,
            batch_size=2,
        )


def test_cold_start_difficulty_treated_as_T() -> None:
    """A row with ``difficulty_score=None`` always passes the band filter."""
    specs = [
        _spec(
            section_id=f"s{i:03d}",
            subject_id=f"subj_{i:03d}",
            difficulty_score=None,
            ap_bin=i % 20,
        )
        for i in range(8)
    ]
    dataset = _StubDataset(specs)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=8,  # unique_prompts = 4
        band_width=0.01,  # extremely tight band
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    list(iter(sampler))
    # Cold-start rows take T's value, so even a 0.01 band catches them.
    assert sampler.last_ladder_level == 1


# ---------------------------------------------------------------------------
# Callback — TrainerState stub
# ---------------------------------------------------------------------------


def _make_trainer_state(log_history: list[dict[str, Any]] | None = None) -> Any:
    """Tiny stand-in for ``transformers.TrainerState``."""
    return SimpleNamespace(log_history=log_history if log_history is not None else [])


# ---------------------------------------------------------------------------
# 16. on_log with empty log_history → no-op
# ---------------------------------------------------------------------------


def test_callback_on_log_empty_history_is_noop(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        initial_T=0.5,
    )
    schedule = _FakeSchedule()
    bin_map = BinDifficultyMap()
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=bin_map,
        schedule=schedule,  # type: ignore[arg-type]
        write_back=True,
    )

    state = _make_trainer_state(log_history=[])
    # Should not raise even though log_history is empty.
    cb.on_log(args=None, state=state, control=None)
    assert sampler.T == pytest.approx(0.5)  # unchanged
    assert len(bin_map) == 0  # nothing written back


# ---------------------------------------------------------------------------
# 17. on_log with reward → forwards to sampler.update
# ---------------------------------------------------------------------------


def test_callback_on_log_forwards_reward_to_sampler(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        initial_T=0.5,
    )
    real_update = sampler.update
    spy = MagicMock(side_effect=real_update)
    sampler.update = spy  # type: ignore[method-assign]

    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=BinDifficultyMap(),
        schedule=_FakeSchedule(),  # type: ignore[arg-type]
        write_back=False,
    )
    state = _make_trainer_state(log_history=[{"step": 1, "reward": 0.8}])
    cb.on_log(args=None, state=state, control=None)
    spy.assert_called_once_with(0.8)


# ---------------------------------------------------------------------------
# 18. write_back=True → calls bin_difficulty.update per unique (plane, ap_bin)
# ---------------------------------------------------------------------------


def test_callback_writes_back_unique_bins(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    """Snapshot of recent errors → one update per unique ``(plane, ap_bin)``."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        initial_T=0.5,
    )
    # Seed the fake buffer with three observations for two unique keys.
    fake_buffer.add(0.04, "coronal", 5)
    fake_buffer.add(0.05, "coronal", 5)  # dupe key
    fake_buffer.add(0.03, "coronal", 7)

    schedule = _FakeSchedule(
        per_bin_means={
            ("coronal", 5): 0.045,
            ("coronal", 7): 0.030,
        }
    )
    bin_map = BinDifficultyMap()
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=bin_map,
        schedule=schedule,  # type: ignore[arg-type]
        write_back=True,
    )
    state = _make_trainer_state(log_history=[{"reward": 0.5}])
    cb.on_log(args=None, state=state, control=None)

    # Two unique keys → two writes.
    assert len(bin_map) == 2
    assert bin_map.get("coronal", 5) == pytest.approx(0.045)
    assert bin_map.get("coronal", 7) == pytest.approx(0.030)


# ---------------------------------------------------------------------------
# 19. write_back=False → bin map untouched
# ---------------------------------------------------------------------------


def test_callback_write_back_false_skips_bin_update(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        initial_T=0.5,
    )
    fake_buffer.add(0.04, "coronal", 5)

    bin_map = BinDifficultyMap()
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=bin_map,
        schedule=_FakeSchedule(
            per_bin_means={("coronal", 5): 0.04},
        ),  # type: ignore[arg-type]
        write_back=False,
    )
    state = _make_trainer_state(log_history=[{"reward": 0.5}])
    cb.on_log(args=None, state=state, control=None)

    # Sampler still updates (T moves, last_step set) but no bin write.
    assert len(bin_map) == 0


# ---------------------------------------------------------------------------
# 20. Plane filter — only configured plane is written back
# ---------------------------------------------------------------------------


def test_callback_plane_filter_restricts_write_back(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        initial_T=0.5,
    )
    # Mix of planes in the buffer.
    fake_buffer.add(0.04, "coronal", 3)
    fake_buffer.add(0.05, "sagittal", 8)
    fake_buffer.add(0.03, "horizontal", 12)

    bin_map = BinDifficultyMap()
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=bin_map,
        schedule=_FakeSchedule(
            per_bin_means={
                ("coronal", 3): 0.04,
                ("sagittal", 8): 0.05,
                ("horizontal", 12): 0.03,
            },
        ),  # type: ignore[arg-type]
        plane="coronal",
        write_back=True,
    )
    state = _make_trainer_state(log_history=[{"reward": 0.5}])
    cb.on_log(args=None, state=state, control=None)

    # Only the coronal bin should be written.
    assert len(bin_map) == 1
    assert bin_map.get("coronal", 3) == pytest.approx(0.04)
    assert bin_map.get("sagittal", 8) is None
    assert bin_map.get("horizontal", 12) is None


# ---------------------------------------------------------------------------
# Bonus — defensive guards
# ---------------------------------------------------------------------------


def test_callback_falls_back_to_logs_when_history_empty(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    """If state.log_history is empty but ``logs`` carries a reward, use ``logs``."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        initial_T=0.5,
    )
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=BinDifficultyMap(),
        schedule=_FakeSchedule(),  # type: ignore[arg-type]
        write_back=False,
    )
    state = _make_trainer_state(log_history=[])
    cb.on_log(args=None, state=state, control=None, logs={"reward": 1.0})
    # T should have moved (raw step = 0.05 * tanh(2 * 0.5) > 0).
    assert sampler.T > 0.5


def test_callback_handles_non_numeric_reward_gracefully(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    """A reward field whose value isn't coercible to float must not crash."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        initial_T=0.5,
    )
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=BinDifficultyMap(),
        schedule=_FakeSchedule(),  # type: ignore[arg-type]
        write_back=False,
    )
    state = _make_trainer_state(log_history=[{"reward": "not_a_float"}])
    cb.on_log(args=None, state=state, control=None)
    # T unchanged because reward couldn't be parsed.
    assert sampler.T == pytest.approx(0.5)


def test_callback_skips_bins_with_no_running_mean(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    """If ``per_bin_difficulty`` returns ``None``, skip rather than write None."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        initial_T=0.5,
    )
    fake_buffer.add(0.04, "coronal", 5)
    fake_buffer.add(0.03, "coronal", 7)
    bin_map = BinDifficultyMap()
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=bin_map,
        schedule=_FakeSchedule(
            per_bin_means={
                # Only bin 5 has a running mean.
                ("coronal", 5): 0.04,
            },
        ),  # type: ignore[arg-type]
        write_back=True,
    )
    state = _make_trainer_state(log_history=[{"reward": 0.5}])
    cb.on_log(args=None, state=state, control=None)
    # Only the bin with a non-None running mean is written.
    assert len(bin_map) == 1
    assert bin_map.get("coronal", 5) == pytest.approx(0.04)


def test_callback_emits_adarft_metrics_into_logs(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    """The callback must annotate the ``logs`` dict with adarft/* metrics."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        initial_T=0.5,
    )
    bin_map = BinDifficultyMap()
    bin_map.update("coronal", 5, 0.04)
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=bin_map,
        schedule=_FakeSchedule(),  # type: ignore[arg-type]
        write_back=False,
    )
    state = _make_trainer_state(log_history=[{"reward": 0.8}])
    logs: dict[str, Any] = {"reward": 0.8}
    cb.on_log(args=None, state=state, control=None, logs=logs)
    assert "adarft/T" in logs
    assert "adarft/ladder_level" in logs
    assert "adarft/normalized_reward" in logs
    assert logs["adarft/ap_bin_observed_count"] == 1


# ---------------------------------------------------------------------------
# Optional — exercise the real adaptive_reward path when it imports cleanly
# ---------------------------------------------------------------------------


def test_real_adaptive_reward_buffer_uses_deque() -> None:
    """Sanity check that the real buffer shape hasn't drifted to e.g. a list."""
    assert isinstance(ar._RECENT_ERRORS, deque)


# ---------------------------------------------------------------------------
# Bin-map live difficulty refresh
# ---------------------------------------------------------------------------


def test_sampler_refreshes_difficulties_from_bin_map_per_iter() -> None:
    """The sampler must re-read live difficulty from the bin map before each batch.

    Setup: two specs, one initially in-band (d=0.5) and one out-of-band
    (d=0.95). Tight band [0.45, 0.55] around T=0.5 → first batch picks
    only the in-band row. Then we mutate the bin map so the previously
    out-of-band row is now in-band (d=0.5) and the previously in-band
    row is now out-of-band (d=0.95). The next batch must reflect that
    swap — proving the sampler doesn't keep using the construction-time
    snapshot.
    """
    specs = [
        _spec(
            section_id="s001",
            subject_id="subj_a",
            difficulty_score=0.5,
            plane="coronal",
            dataset="ds_a",
            ap_bin=3,
        ),
        _spec(
            section_id="s002",
            subject_id="subj_b",
            difficulty_score=0.95,
            plane="coronal",
            dataset="ds_a",
            ap_bin=7,
        ),
    ]
    dataset = _StubDataset(specs)
    bin_map = BinDifficultyMap()
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=1,
        batch_size=1,  # unique_prompts = 1
        band_width=0.05,  # tight band [0.45, 0.55]
        initial_T=0.5,
        max_per_subject_in_batch=1,
        generator=torch.Generator().manual_seed(0),
        bin_difficulty=bin_map,
    )

    # First batch: s001 (d=0.5) is in-band, s002 (d=0.95) is not.
    indices = list(iter(sampler))
    assert len(indices) == 1
    assert indices[0] == 0  # s001 spec lives at index 0
    # Sanity: s002 is currently out-of-band.
    assert sampler._difficulty_for_index(1) == pytest.approx(0.95)

    # Mutate the bin map: swap which bin is in-band.
    bin_map.update("coronal", 3, 0.95)
    bin_map.update("coronal", 7, 0.5)

    # Next batch must reflect the swap.
    indices2 = list(iter(sampler))
    assert len(indices2) == 1
    assert indices2[0] == 1  # s002 spec lives at index 1
    assert sampler._difficulty_for_index(0) == pytest.approx(0.95)
    assert sampler._difficulty_for_index(1) == pytest.approx(0.5)


def test_sampler_without_bin_map_keeps_frozen_difficulties() -> None:
    """Backwards compat: with no bin_difficulty, behaviour is unchanged.

    The sampler must NOT mutate ``_difficulty_scores`` between iterations
    when no back-reference is provided.
    """
    specs = [
        _spec(
            section_id=f"s{i:03d}",
            subject_id=f"subj_{i:03d}",
            difficulty_score=0.5,
            plane="coronal",
            dataset="ds_a",
            ap_bin=i % 20,
        )
        for i in range(4)
    ]
    dataset = _StubDataset(specs)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=4,
        band_width=0.15,
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
        # bin_difficulty intentionally omitted.
    )
    snapshot_before = list(sampler._difficulty_scores)
    list(iter(sampler))
    list(iter(sampler))
    assert sampler._difficulty_scores == snapshot_before


def test_sampler_skips_refresh_for_specs_missing_bin_keys() -> None:
    """Specs without a derivable ``(plane, ap_bin)`` key skip the refresh.

    A spec missing ``plane`` (or with no way to compute ap_bin) must keep
    its construction-time difficulty rather than crash on the lookup.
    """
    # Mix one well-formed spec with one missing the plane field.
    specs = [
        _spec(
            section_id="s001",
            subject_id="subj_a",
            difficulty_score=0.4,
            plane="coronal",
            dataset="ds_a",
            ap_bin=5,
        ),
        # Plane missing → no refresh key.
        {
            "section_id": "s002",
            "subject_id": "subj_b",
            "plane": "",  # empty → key derivation returns None
            "dataset": "ds_b",
            "difficulty_score": 0.5,
            "ap_bin": None,
            "ground_truth_mm": None,
            "valid_range_mm": None,
        },
    ]
    dataset = _StubDataset(specs)
    bin_map = BinDifficultyMap()
    bin_map.update("coronal", 5, 0.7)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=1,
        batch_size=1,
        band_width=0.5,  # wide enough to catch everything
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
        bin_difficulty=bin_map,
    )
    # Trigger the refresh.
    list(iter(sampler))
    # Well-formed spec picked up the live score.
    assert sampler._difficulty_for_index(0) == pytest.approx(0.7)
    # Malformed spec kept its construction-time value.
    assert sampler._difficulty_for_index(1) == pytest.approx(0.5)


def test_sampler_derives_bin_key_from_gt_when_ap_bin_missing() -> None:
    """When ``ap_bin`` isn't pre-set but ``ground_truth_mm`` + ``valid_range_mm``
    are present, the sampler should derive the bin key on the fly."""
    specs = [
        {
            "section_id": "s001",
            "subject_id": "subj_a",
            "plane": "coronal",
            "dataset": "ds_a",
            "difficulty_score": 0.4,
            "ap_bin": None,  # not pre-computed
            "ground_truth_mm": 5.0,
            "valid_range_mm": (0.0, 10.0),
        },
    ]
    dataset = _StubDataset(specs)
    bin_map = BinDifficultyMap()
    # ap_pct = 0.5 → bin = int(0.5 * 20) = 10.
    bin_map.update("coronal", 10, 0.88)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=1,
        batch_size=1,
        band_width=0.5,
        initial_T=0.5,
        generator=torch.Generator().manual_seed(0),
        bin_difficulty=bin_map,
    )
    list(iter(sampler))
    assert sampler._difficulty_for_index(0) == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Reward rescale before AdaRFT update
# ---------------------------------------------------------------------------


def test_callback_rescales_negative_reward_to_zero(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    """A negative raw reward (format penalty) must be clamped to 0 before update."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        d_min=0.0,
        d_max=1.0,
        initial_T=0.5,
    )
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=BinDifficultyMap(),
        schedule=_FakeSchedule(),  # type: ignore[arg-type]
        write_back=False,
    )
    state = _make_trainer_state(log_history=[{"reward": -1.0}])
    cb.on_log(args=None, state=state, control=None)
    expected_raw_step = 0.05 * math.tanh(2.0 * (0.0 - 0.5))
    assert sampler.last_step == pytest.approx(expected_raw_step)


def test_callback_rescales_in_range_reward_passes_through(
    fake_buffer: _RecentErrorsBuffer,
) -> None:
    """A reward already in [0, 1] flows through the clamp unchanged."""
    dataset = _make_dataset_uniform_difficulty(4)
    sampler = CurriculumRepeatingSampler(
        dataset,
        num_generations=2,
        batch_size=2,
        target_reward=0.5,
        alpha=2.0,
        eta=0.05,
        ema_decay=0.7,
        initial_T=0.5,
    )
    cb = AdaRFTCurriculumCallback(
        sampler,
        bin_difficulty=BinDifficultyMap(),
        schedule=_FakeSchedule(),  # type: ignore[arg-type]
        write_back=False,
    )
    state = _make_trainer_state(log_history=[{"reward": 0.8}])
    cb.on_log(args=None, state=state, control=None)
    expected_raw_step = 0.05 * math.tanh(2.0 * (0.8 - 0.5))
    assert sampler.last_step == pytest.approx(expected_raw_step)
