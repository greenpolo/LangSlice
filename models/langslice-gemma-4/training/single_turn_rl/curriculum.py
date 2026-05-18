"""Adaptive AdaRFT-style curriculum sampler + trainer callback.

Sibling to :mod:`single_turn_rl.adaptive_reward` (Layer 4) and
:mod:`single_turn_rl.section_state` (Layer 2). The pieces here are the
"control loop" that closes the difficulty/reward feedback cycle:

* :class:`CurriculumRepeatingSampler` is a ``torch.utils.data.Sampler``
  subclass that picks each batch from the rows whose ``difficulty_score``
  falls in a band ``[T - band_width, T + band_width]`` around a moving
  target ``T``. It repeats every unique prompt ``num_generations`` times in
  the index stream so TRL's group-relative advantage stays well-defined
  (``trl/trainer/grpo_trainer.py:2235-2241`` requires identical prompts in
  consecutive slots within a group).
* :class:`AdaRFTCurriculumCallback` is a :class:`transformers.TrainerCallback`
  subclass hooked on ``on_log`` (NOT ``on_step_end`` — TRL only flushes
  reward into ``state.log_history`` at ``logging_steps``, so step-end would
  see stale data). It pulls the most-recent flushed reward, advances the
  sampler's target ``T`` via the AdaRFT update, and (optionally) writes
  per-bin live difficulty back into the shared :class:`BinDifficultyMap`
  so the next batch's band query sees the freshest signal.

AdaRFT update
-------------
The target moves toward harder material when reward is high and toward
easier material when reward is low::

    raw_step  = eta * tanh(alpha * (R - beta))
    smoothed  = ema_decay * prev_step + (1 - ema_decay) * raw_step
    T         = clip(T + smoothed, d_min, d_max)

``R`` MUST be pre-rescaled to ``[0, 1]`` by the caller. The single-turn
reward path uses ``accuracy_pct / 100`` semantics (or any other
[0,1]-bounded scalar) — passing the raw GRPO reward (which can dip to -1
on format failures) would push the schedule toward ``d_min`` indefinitely
and wedge the curriculum at trivial difficulty.

AP-bin difficulty pooling
-------------------------
The corpus has ~14k unique sections seen 1x per epoch, so per-section
observations don't accumulate signal in a 200-entry rolling buffer.
Observations are pooled by AP-coordinate bin (sections at the same AP
look visually similar across brains), which gives ~10 observations per
bin with the same buffer. The sampler reads bin-level difficulty via a
shared :class:`BinDifficultyMap`; the callback writes per-bin running
means into the same map.

Subject-cap fallback ladder
---------------------------
A single batch can't lean too hard on one brain (subject) — the
``num_generations`` repetition would otherwise hand the policy 4-8
identical prompts from the same subject, collapsing exploration. The
sampler enforces ``max_per_subject_in_batch`` at ladder level 1; if the
band can't supply enough candidates under that cap it relaxes
constraints in this order, escalating only as far as needed:

1. band ± ``band_width`` AND ``max_per_subject_in_batch`` (default).
2. band widened 50% (one-shot for this batch — does NOT mutate the
   instance attr).
3. band widened 50% AND subject cap relaxed to infinity (one-shot).
4. uniform backfill from the entire dataset (last resort, e.g. the very
   first round when no row has a non-cold-start difficulty).

Each batch records the highest level it had to reach in
``last_ladder_level`` so the trainer log can flag persistent ladder-4
escalations (a sign the curriculum is starved at the current ``T`` and
should be looked at).

Cold-start handling
-------------------
A row whose ``difficulty_score`` is ``None`` is treated as if its
difficulty equals the current target ``T``. This bootstraps the
curriculum: round 0 (no observations yet) trivially passes the band
filter for every row, so the sampler doesn't deadlock waiting for live
difficulty signal that hasn't accumulated yet. With the AP-bin keying,
dataset construction also seeds ``difficulty_score = ap_pct`` so
cold-start rows already have a meaningful position-based prior.

DDP / multi-GPU note
--------------------
The sampler holds in-process state (``T``, ``last_smoothed_step``).
Under DDP each rank would maintain its own copy and the curricula
would diverge. Multi-GPU support requires a rank-0-only update +
broadcast — out of scope for this layer.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler
from transformers import TrainerCallback

from .adaptive_reward import (
    AdaptiveRewardSchedule,
    compute_ap_bin,
    recent_errors,
)

logger = logging.getLogger(__name__)


_BAND_WIDEN_FACTOR: float = 1.5
"""Multiplicative factor applied to band_width at ladder rung 2."""


# ---------------------------------------------------------------------------
# Bin-difficulty map (shared between sampler and callback)
# ---------------------------------------------------------------------------


class BinDifficultyMap:
    """In-process ``(plane, ap_bin) -> mean_abs_err_pct`` map.

    Shared between :class:`CurriculumRepeatingSampler` (reader) and
    :class:`AdaRFTCurriculumCallback` (writer). Kept deliberately small —
    no persistence, no serialization, no plane validation — because the
    AP-bin pool is a transient run-time signal, not a curated artefact.
    The persisted-difficulty sidecar still lives on the manifest index
    for per-section bootstrapping; this map only handles live observations.
    """

    def __init__(self) -> None:
        self._scores: dict[tuple[str, int], float] = {}

    def update(self, plane: str, ap_bin: int, score: float) -> None:
        """Record a new running-mean score for a ``(plane, ap_bin)`` key."""
        self._scores[(str(plane), int(ap_bin))] = float(score)

    def get(self, plane: str, ap_bin: int) -> float | None:
        """Return the score for ``(plane, ap_bin)``, or ``None`` if absent."""
        return self._scores.get((str(plane), int(ap_bin)))

    def __len__(self) -> int:
        return len(self._scores)

    def keys(self) -> list[tuple[str, int]]:
        """Snapshot of all observed ``(plane, ap_bin)`` keys."""
        return list(self._scores.keys())


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def _spec_difficulty(spec: Any) -> float | None:
    """Pull ``difficulty_score`` out of either a dict spec or a dataclass.

    The sampler is duck-typed: ``RowDataset._specs`` carries dict specs
    (Lane A SFT-trace path) and dataclass-based ``SectionState`` rows
    (Lane B index-driven path) both expose ``difficulty_score``. Centralise
    the lookup so callers can mix the two without the sampler caring.
    """
    if isinstance(spec, dict):
        return spec.get("difficulty_score")
    return getattr(spec, "difficulty_score", None)


def _spec_subject(spec: Any) -> str:
    """Pull ``subject_id`` out of either a dict spec or a dataclass.

    Returns an empty string if the field is absent — a missing subject
    can't anchor the cap, but the sampler should keep working rather
    than blow up on a malformed spec.
    """
    if isinstance(spec, dict):
        return str(spec.get("subject_id", ""))
    val = getattr(spec, "subject_id", "")
    return str(val) if val is not None else ""


def _spec_ap_bin_key(spec: Any) -> tuple[str, int] | None:
    """Extract the ``(plane, ap_bin)`` key for live-difficulty refresh.

    Prefers a precomputed ``ap_bin`` field on the spec; falls back to
    deriving it from ``ground_truth_mm`` + ``valid_range_mm`` when those
    are available. Returns ``None`` when ``plane`` is missing or the bin
    can't be derived — those specs keep their construction-time
    difficulty during refresh.
    """
    if isinstance(spec, dict):
        plane = spec.get("plane")
        ap_bin = spec.get("ap_bin")
        gt = spec.get("ground_truth_mm")
        vr = spec.get("valid_range_mm")
    else:
        plane = getattr(spec, "plane", None)
        ap_bin = getattr(spec, "ap_bin", None)
        gt = getattr(spec, "ground_truth_mm", None)
        vr = getattr(spec, "valid_range_mm", None)
    if not plane:
        return None
    if ap_bin is None:
        if gt is None or vr is None:
            return None
        try:
            ap_bin = compute_ap_bin(float(gt), (float(vr[0]), float(vr[1])))
        except (TypeError, ValueError, IndexError):
            return None
    return (str(plane), int(ap_bin))


class CurriculumRepeatingSampler(Sampler[int]):
    """Difficulty-band sampler that repeats each unique prompt N times.

    See module docstring for the AdaRFT formula and the subject-cap
    fallback ladder. The sampler is intended to plug into TRL's
    ``GRPOTrainer._get_train_dataloader`` via the curriculum-trainer
    subclass selector (Task 6) — it doesn't import TRL itself so the
    unit tests stay light.

    Each ``__iter__`` yields exactly ``unique_prompts * num_generations``
    indices (one batch's worth). TRL invokes the iterator multiple
    times per epoch; the sampler's stateful ``T`` is updated externally
    via :meth:`update` between batches.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        num_generations: int,
        batch_size: int,
        target_reward: float = 0.5,
        alpha: float = 2.0,
        eta: float = 0.05,
        ema_decay: float = 0.7,
        d_min: float = 0.0,
        d_max: float = 1.0,
        band_width: float = 0.15,
        initial_T: float = 0.5,
        max_per_subject_in_batch: int = 2,
        min_visits_per_bin: int = 0,
        quota_slots_per_batch: int = 0,
        generator: torch.Generator | None = None,
        bin_difficulty: BinDifficultyMap | None = None,
        repeat_count: int = 1,
    ) -> None:
        # Validate everything up front — a misconfigured sampler would only
        # surface its bug after the first ``next(iter(sampler))`` call,
        # which under the trainer would mean a confusing error mid-step-1.
        if num_generations <= 0:
            raise ValueError(f"num_generations must be positive, got {num_generations}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size % num_generations != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_generations "
                f"({num_generations}) — TRL's GRPO contract requires identical "
                "groups of size num_generations within a batch"
            )
        if d_max < d_min:
            raise ValueError(
                f"d_max ({d_max}) must be >= d_min ({d_min}) for the AdaRFT clamp"
            )
        if band_width < 0:
            raise ValueError(f"band_width must be non-negative, got {band_width}")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {ema_decay}")
        if max_per_subject_in_batch <= 0:
            raise ValueError(
                f"max_per_subject_in_batch must be positive, got "
                f"{max_per_subject_in_batch}"
            )
        if repeat_count < 1:
            raise ValueError(f"repeat_count must be >= 1, got {repeat_count}")
        if min_visits_per_bin < 0:
            raise ValueError(
                f"min_visits_per_bin must be non-negative, got {min_visits_per_bin}"
            )
        if quota_slots_per_batch < 0:
            raise ValueError(
                f"quota_slots_per_batch must be non-negative, got {quota_slots_per_batch}"
            )

        # Skip the parent ``Sampler.__init__`` — it only emits a deprecation
        # warning when ``data_source`` is non-None (the arg is removed in
        # torch 2.2). Calling ``super().__init__()`` with no args is fine and
        # keeps subclasses clean of that warning.
        super().__init__()

        self.dataset = dataset
        # Read the underlying spec list once at construction. Both the SFT
        # ``RowDataset`` and the Lane B ``SectionState`` iterable expose
        # ``_specs`` (or, for a Lane B build, ``self`` is iterable and the
        # caller pre-materialises the list). Duck-type the lookup so this
        # sampler stays decoupled from either dataset class.
        specs = getattr(dataset, "_specs", None)
        if specs is None:
            specs = list(dataset)
        self._specs: Sequence[Any] = specs
        if len(self._specs) == 0:
            raise ValueError("CurriculumRepeatingSampler: dataset must be non-empty")

        self.num_generations: int = int(num_generations)
        self.batch_size: int = int(batch_size)
        self.unique_prompts: int = self.batch_size // self.num_generations
        # ``repeat_count`` mirrors TRL's RepeatSampler arg; for GRPOTrainer the
        # caller must pass ``num_iterations * steps_per_generation`` to match
        # trl/trainer/grpo_trainer.py:921. With the TOML's
        # gradient_accumulation_steps=4 (and no explicit steps_per_generation),
        # GRPOConfig defaults steps_per_generation to 4, so a value of 1 here
        # would emit only 1/4 of the indices the inner-loop reuses, breaking
        # the group-relative-advantage layout downstream.
        self.repeat_count: int = int(repeat_count)

        # AdaRFT hyperparameters
        self.target_reward: float = float(target_reward)
        self.alpha: float = float(alpha)
        self.eta: float = float(eta)
        self.ema_decay: float = float(ema_decay)
        self.d_min: float = float(d_min)
        self.d_max: float = float(d_max)
        self.band_width: float = float(band_width)
        self.max_per_subject_in_batch: int = int(max_per_subject_in_batch)
        # Coverage quota: every bin gets at least ``min_visits_per_bin`` picks
        # before AdaRFT is allowed to concentrate. While any bin is below
        # quota, the sampler reserves ``quota_slots_per_batch`` slots per
        # batch for picks from the most-deficit bins. Defaults of 0/0 keep
        # the legacy pure-AdaRFT behavior — set both in the TOML to enable.
        self.min_visits_per_bin: int = int(min_visits_per_bin)
        self.quota_slots_per_batch: int = int(quota_slots_per_batch)

        # Schedule state — ``T`` clamped to bounds at construction so a
        # caller passing initial_T outside [d_min, d_max] still gets a
        # legal starting target rather than silently drifting into bounds
        # on the first update.
        self.T: float = max(self.d_min, min(self.d_max, float(initial_T)))
        self.last_step: float = 0.0
        self.last_smoothed_step: float = 0.0

        # Per-batch diagnostic. 1-4 mapping the four ladder levels in the
        # module docstring; 0 means "no batch sampled yet".
        self.last_ladder_level: int = 0

        # Per-batch + cumulative AP-bin coverage diagnostics. Populated in
        # ``__iter__`` after the ladder picks. ``last_picked_bins`` is the
        # bin index of each pick this batch (-1 for picks without a
        # derivable ``(plane, ap_bin)`` key); ``bin_histogram`` is the
        # cumulative count of picks per bin index across the whole run.
        # Used by the AdaRFT callback to surface band-focus vs coverage —
        # ``T`` alone tells us where the difficulty band sits but not which
        # AP bins actually got gradient.
        self.last_picked_bins: list[int] = []
        self.bin_histogram: Counter[int] = Counter()

        # Generator for reproducibility. Defer construction of a default
        # generator until first use so two samplers built without an
        # explicit generator don't accidentally share one.
        self._generator: torch.Generator | None = generator

        # Pre-cache subject ids and difficulty scores per index so the
        # per-batch hot path doesn't re-walk the spec dicts. Indices into
        # ``self._specs``; values are parallel.
        self._subject_ids: list[str] = [_spec_subject(s) for s in self._specs]
        self._difficulty_scores: list[float | None] = [
            _spec_difficulty(s) for s in self._specs
        ]

        # Optional bin-difficulty back-reference for live difficulty refresh.
        # When set, ``__iter__`` re-reads ``bin_difficulty.get(plane, ap_bin)``
        # for each spec at the start of every batch so the band query sees
        # the freshest per-bin signal the AdaRFT callback wrote back.
        # Without this, ``self._difficulty_scores`` would stay frozen at
        # construction time and only ``T`` would move within a run — half-
        # closing the curriculum loop.
        self._bin_difficulty: BinDifficultyMap | None = bin_difficulty
        # Pre-cache the ``(plane, ap_bin)`` keys so the per-batch refresh
        # doesn't re-walk the spec dicts. ``None`` entries mark specs that
        # can't be refreshed (no plane / no ap-bin derivable); those skip
        # the refresh and keep their construction-time difficulty.
        self._bin_keys: list[tuple[str, int] | None] = [
            _spec_ap_bin_key(s) for s in self._specs
        ]

    # ------------------------------------------------------------------ AdaRFT update

    def update(self, recent_reward_mean: float) -> None:
        """Advance the curriculum target ``T`` per the AdaRFT formula.

        The ``recent_reward_mean`` argument MUST already be rescaled to
        ``[0, 1]`` (e.g. ``accuracy_pct / 100``). Passing the raw GRPO
        scalar (which can dip to ``-1`` on format failures) will skew the
        update toward ``d_min`` and wedge the curriculum at trivial
        difficulty. The ``AdaRFTCurriculumCallback`` performs this
        rescaling before forwarding the trainer's logged reward here.
        """
        r = float(recent_reward_mean)
        raw = self.eta * math.tanh(self.alpha * (r - self.target_reward))
        smoothed = self.ema_decay * self.last_smoothed_step + (1.0 - self.ema_decay) * raw
        self.last_step = raw
        self.last_smoothed_step = smoothed
        self.T = max(self.d_min, min(self.d_max, self.T + smoothed))

    # ------------------------------------------------------------------ band query helper

    def _difficulty_for_index(self, idx: int) -> float:
        """Effective difficulty for ``idx``: cold-start rows take T's value.

        ``None`` rows always pass the band filter — which is what we want
        for a freshly-seeded run that hasn't observed any section yet.
        Once the curriculum callback writes back live bin difficulties via
        :meth:`BinDifficultyMap.update`, those values flow into the
        sampler's cached ``_difficulty_scores`` on the next batch refresh
        and the cold-start treatment falls away.
        """
        d = self._difficulty_scores[idx]
        return self.T if d is None else float(d)

    def _candidates_in_band(self, half_width: float) -> list[int]:
        """Indices whose difficulty lies within ``[T - hw, T + hw]``."""
        lo = self.T - half_width
        hi = self.T + half_width
        return [
            i
            for i in range(len(self._specs))
            if lo <= self._difficulty_for_index(i) <= hi
        ]

    # ------------------------------------------------------------------ sampling

    def _generator_or_default(self) -> torch.Generator:
        """Lazily construct (and cache) a default generator if none was set."""
        if self._generator is None:
            self._generator = torch.Generator()
        return self._generator

    def _shuffle_indices(self, indices: list[int]) -> list[int]:
        """Stable shuffle backed by the configured generator."""
        if not indices:
            return indices
        gen = self._generator_or_default()
        perm = torch.randperm(len(indices), generator=gen).tolist()
        return [indices[i] for i in perm]

    def _pick_with_subject_cap(
        self,
        candidates: list[int],
        n_needed: int,
        subject_cap: int | None,
    ) -> list[int]:
        """Sample ``n_needed`` indices honouring an optional per-subject cap.

        Walks a shuffled view of ``candidates`` greedily, skipping any
        index whose subject already has ``subject_cap`` slots in the
        running pick. ``subject_cap=None`` disables the cap entirely.
        """
        shuffled = self._shuffle_indices(candidates)
        picked: list[int] = []
        subject_counts: dict[str, int] = {}
        for idx in shuffled:
            if len(picked) >= n_needed:
                break
            subj = self._subject_ids[idx]
            if subject_cap is not None and subject_counts.get(subj, 0) >= subject_cap:
                continue
            picked.append(idx)
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
        return picked

    def _deficit_bins(self) -> list[int]:
        """Bins below the visit quota, most-deficit first.

        Considers only bins that have at least one spec in our dataset
        (bins that don't appear in ``self._bin_keys`` are unreachable and
        would force the quota into an infinite-deficit state). Returns []
        when the quota is disabled (``min_visits_per_bin == 0``).
        """
        if self.min_visits_per_bin <= 0:
            return []
        bins_in_dataset: set[int] = set()
        for key in self._bin_keys:
            if key is not None:
                bins_in_dataset.add(key[1])
        deficit = [
            b for b in bins_in_dataset
            if self.bin_histogram[b] < self.min_visits_per_bin
        ]
        deficit.sort(key=lambda b: self.bin_histogram[b])
        return deficit

    def _fill_quota(self, n_needed: int) -> list[int]:
        """Pick up to ``quota_slots_per_batch`` indices from deficit bins.

        Each pick targets a different deficit bin (most-deficit first), so
        a single batch can advance the quota on multiple under-visited
        bins instead of piling all quota slots into one bin. Honors the
        subject cap. Returns [] when no deficit bins remain.
        """
        if self.quota_slots_per_batch <= 0:
            return []
        deficit = self._deficit_bins()
        if not deficit:
            return []
        n_to_pick = min(self.quota_slots_per_batch, n_needed, len(deficit))
        picks: list[int] = []
        subject_counts: dict[str, int] = {}
        for target_bin in deficit:
            if len(picks) >= n_to_pick:
                break
            bin_indices = [
                i for i, key in enumerate(self._bin_keys)
                if key is not None and key[1] == target_bin
            ]
            for idx in self._shuffle_indices(bin_indices):
                if idx in picks:
                    continue
                subj = self._subject_ids[idx]
                if subject_counts.get(subj, 0) >= self.max_per_subject_in_batch:
                    continue
                picks.append(idx)
                subject_counts[subj] = subject_counts.get(subj, 0) + 1
                break
        return picks

    def _select_unique_prompts(self) -> tuple[list[int], int]:
        """Run the four-rung fallback ladder, returning (picks, level).

        ``picks`` has length ``self.unique_prompts``. ``level`` is 1-4 per
        the module docstring's ladder definition. The function never
        returns fewer than ``unique_prompts`` indices: at the bottom rung
        it backfills uniformly from the entire dataset.

        When the quota is enabled (``min_visits_per_bin > 0`` and
        ``quota_slots_per_batch > 0``), the function first reserves up to
        ``quota_slots_per_batch`` slots for picks from under-visited bins.
        The remaining slots are filled by the rung ladder, excluding any
        index already chosen by the quota. ``level=0`` is returned when
        the quota alone satisfied ``n_needed``.
        """
        n_needed = self.unique_prompts
        quota_picks = self._fill_quota(n_needed)
        quota_set = set(quota_picks)
        n_remaining = n_needed - len(quota_picks)
        if n_remaining <= 0:
            return quota_picks[:n_needed], 0

        # Rung 1: band ± band_width AND subject cap.
        candidates_1 = [
            i for i in self._candidates_in_band(self.band_width)
            if i not in quota_set
        ]
        rung_picks = self._pick_with_subject_cap(
            candidates_1, n_remaining, self.max_per_subject_in_batch
        )
        if len(rung_picks) >= n_remaining:
            return quota_picks + rung_picks[:n_remaining], 1

        # Rung 2: widen the band 50% (one-shot — do NOT mutate self.band_width).
        widened_hw = self.band_width * _BAND_WIDEN_FACTOR
        candidates_2 = [
            i for i in self._candidates_in_band(widened_hw)
            if i not in quota_set
        ]
        rung_picks = self._pick_with_subject_cap(
            candidates_2, n_remaining, self.max_per_subject_in_batch
        )
        if len(rung_picks) >= n_remaining:
            return quota_picks + rung_picks[:n_remaining], 2

        # Rung 3: keep the widened band but drop the subject cap.
        rung_picks = self._pick_with_subject_cap(
            candidates_2, n_remaining, subject_cap=None
        )
        if len(rung_picks) >= n_remaining:
            return quota_picks + rung_picks[:n_remaining], 3
        # Carry the rung-3 picks into the rung-4 backfill so we don't lose them.
        picks = list(quota_picks) + list(rung_picks)

        # Rung 4: uniform backfill from the entire dataset. Two-stage so
        # the operator can still see what the band picked vs what we had
        # to invent — but for index purposes we just stitch.
        picked_set = set(picks)
        all_indices = [i for i in range(len(self._specs)) if i not in picked_set]
        backfill = self._shuffle_indices(all_indices)
        for idx in backfill:
            if len(picks) >= n_needed:
                break
            picks.append(idx)
        if len(picks) < n_needed:
            # The rung-4 backfill exhausted every distinct index in the
            # dataset and still fell short of ``unique_prompts``. Repeating
            # picks would silently corrupt GRPO's group-relative-advantage
            # assumption (identical prompts in different groups would
            # collapse the advantage signal), so fail loudly with an
            # operator-actionable message rather than continue.
            raise RuntimeError(
                "CurriculumRepeatingSampler rung 4: dataset has only "
                f"{len(self._specs)} distinct rows but the batch needs "
                f"{n_needed} unique prompts (batch_size={self.batch_size}, "
                f"num_generations={self.num_generations}). Repeating prompts "
                "across distinct slots would corrupt GRPO group-relative "
                "advantage. Reduce batch_size or num_generations, or grow the "
                "dataset to at least unique_prompts rows."
            )
        return picks[:n_needed], 4

    # ------------------------------------------------------------------ Sampler protocol

    def __len__(self) -> int:
        """Length matching TRL's RepeatSampler shape contract.

        Returns ``unique_prompts * num_generations * repeat_count`` so the
        emit length aligns with the parent ``RepeatSampler.__len__`` (see
        ``trl/trainer/utils.py:746``). When ``repeat_count == 1`` (legacy
        Lane A path with ``steps_per_generation==num_iterations==1``) the
        value matches the previous "one batch's worth" semantics; when the
        trainer config bumps either knob, the dataloader's index stream
        gets the same number of slots TRL's own sampler would have emitted.
        """
        return self.unique_prompts * self.num_generations * self.repeat_count

    def _refresh_difficulties(self) -> None:
        """Re-read live per-bin difficulty from the shared bin map.

        The AdaRFT callback writes back per-bin observations into
        :class:`BinDifficultyMap` between batches; the sampler must
        re-pull those before each band query or it would keep sampling
        against the construction-time snapshot. No-op when the sampler
        wasn't constructed with a ``bin_difficulty`` back-reference.
        """
        if self._bin_difficulty is None:
            return
        for i, key in enumerate(self._bin_keys):
            if key is None:
                continue
            live = self._bin_difficulty.get(*key)
            if live is not None:
                self._difficulty_scores[i] = live

    def __iter__(self) -> Iterator[int]:
        """Yield indices in the GRPO-compatible repetition shape.

        Picks ``unique_prompts`` distinct rows via the fallback ladder,
        records the achieved ladder level on ``self.last_ladder_level``,
        and yields the repeated index sequence in the same shape TRL's
        ``RepeatSampler`` produces (``trl/trainer/utils.py:739-743``):
        the inner sequence ``[idx0]*num_generations + [idx1]*num_generations
        + ...`` is then repeated ``repeat_count`` times so the dataloader
        delivers enough slots for ``num_iterations * steps_per_generation``
        inner reuses.
        """
        self._refresh_difficulties()
        picks, level = self._select_unique_prompts()
        self.last_ladder_level = level
        picked_bins = [
            self._bin_keys[i][1] if self._bin_keys[i] is not None else -1
            for i in picks
        ]
        self.last_picked_bins = picked_bins
        for bin_idx in picked_bins:
            self.bin_histogram[bin_idx] += 1
        for _ in range(self.repeat_count):
            for idx in picks:
                for _ in range(self.num_generations):
                    yield idx


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class AdaRFTCurriculumCallback(TrainerCallback):
    """Trainer callback that feeds reward back into the sampler + bin map.

    Hooks ``on_log`` because TRL only flushes per-batch reward into
    ``state.log_history`` at ``logging_steps`` cadence. Step-end fires
    every step but only sees stale data between flushes.

    Behaviour per ``on_log`` call:

    1. Read ``state.log_history[-1]`` if non-empty; pull the ``"reward"``
       field. If absent, no-op (TRL's first ``on_log`` can fire before
       any reward has been computed — e.g. for the LR schedule).
    2. Call :meth:`CurriculumRepeatingSampler.update` to advance ``T``.
    3. If ``write_back=True``, walk a snapshot of
       :data:`adaptive_reward._RECENT_ERRORS`, dedupe by
       ``(plane, ap_bin)``, and call :meth:`BinDifficultyMap.update`
       for each unique key with the running mean from
       :meth:`AdaptiveRewardSchedule.per_bin_difficulty`.
    4. Log a one-line diagnostic via the module logger so trackio /
       tensorboard runs can grep ``T`` / ``ladder`` over time even
       without direct access to the callback's state.
    """

    def __init__(
        self,
        sampler: CurriculumRepeatingSampler,
        *,
        bin_difficulty: BinDifficultyMap,
        schedule: AdaptiveRewardSchedule,
        plane: str | None = None,
        write_back: bool = True,
    ) -> None:
        self.sampler = sampler
        self.bin_difficulty: BinDifficultyMap = bin_difficulty
        self.schedule: AdaptiveRewardSchedule = schedule
        self.plane = plane
        self.write_back = write_back

    # noqa annotations match the surrounding TrainerCallback subclasses; the
    # ``args`` / ``state`` / ``control`` kwargs come from
    # ``transformers.TrainerCallback`` and are deliberately untyped here for
    # a minimal callback signature.
    def on_log(  # noqa: ANN001, ANN003
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):
        """Pull the most-recent flushed reward and advance the curriculum."""
        # Prefer ``state.log_history`` (TRL's GRPOTrainer flushes the per-batch
        # mean reward there at every ``logging_steps`` boundary). Fall back to
        # ``logs`` for resilience — that's the dict the trainer just emitted
        # and it should carry the same ``reward`` key.
        history = getattr(state, "log_history", None) or []
        reward: float | None = None
        if history:
            entry = history[-1]
            if isinstance(entry, dict) and "reward" in entry:
                try:
                    reward = float(entry["reward"])
                except (TypeError, ValueError):
                    reward = None
        if reward is None and isinstance(logs, dict) and "reward" in logs:
            try:
                reward = float(logs["reward"])
            except (TypeError, ValueError):
                reward = None
        if reward is None:
            # Nothing to update — pre-warmup, or this on_log fired for an
            # eval-only metric that didn't carry a reward. Don't pretend.
            return control

        # Rescale the raw GRPO reward to AdaRFT's expected [0, 1] domain.
        # The raw reward lives in [format_penalty, 1.0] (typically [-1, 1])
        # — feeding the negative tail directly into tanh would saturate the
        # update toward d_min and wedge the curriculum at trivial difficulty
        # whenever a single format-fail dragged a small batch's mean below
        # zero. Clamping into [0, 1] preserves the "didn't work" signal
        # (collapses both OOR=0 and format=-1 to 0) without the wedge.
        normalized_reward = max(0.0, min(1.0, reward))
        self.sampler.update(normalized_reward)

        if self.write_back:
            self._write_back_difficulties()

        # Surface the AdaRFT internals as trainer metrics so wandb/trackio
        # can plot ``T`` / ladder / smoothed step over time. Two write paths
        # because transformers ``Trainer.log`` calls integrations with the
        # ``logs`` kwarg (a dict separate from the entry it just appended
        # to ``state.log_history``); writing to both keeps the metric visible
        # to live integrations AND to downstream readers of log_history.
        # Cumulative bin coverage — how many distinct AP bins have been
        # sampled at least once across the run, and the share of picks
        # that came from the top-3 most-sampled bins. ``coverage_count``
        # tells us if curriculum focus is broadening; ``top3_share`` flags
        # over-concentration (e.g., 0.8 means 80% of picks hit just 3 bins).
        histogram = self.sampler.bin_histogram
        total_picks = sum(histogram.values())
        if total_picks > 0:
            top3_sum = sum(c for _, c in histogram.most_common(3))
            top3_share = top3_sum / total_picks
        else:
            top3_share = 0.0
        coverage_count = sum(1 for b, c in histogram.items() if b >= 0 and c > 0)
        # Coverage quota deficit — how many bins remain under the quota.
        # 0 means quota has been met for every reachable bin.
        deficit_count = len(self.sampler._deficit_bins())  # noqa: SLF001

        adarft_metrics = {
            "adarft/T": float(self.sampler.T),
            "adarft/raw_step": float(self.sampler.last_step),
            "adarft/smoothed_step": float(self.sampler.last_smoothed_step),
            "adarft/ladder_level": int(self.sampler.last_ladder_level),
            "adarft/normalized_reward": float(normalized_reward),
            "adarft/ap_bin_observed_count": len(self.bin_difficulty),
            "adarft/ap_bin_coverage_count": int(coverage_count),
            "adarft/ap_bin_top3_share": float(top3_share),
            "adarft/ap_bin_deficit_count": int(deficit_count),
        }
        if isinstance(logs, dict):
            logs.update(adarft_metrics)
        if history and isinstance(history[-1], dict):
            history[-1].update(adarft_metrics)

        # Diagnostic — one line per ``on_log`` call. Cheap and grep-friendly.
        # Logs both the raw and normalized reward so a wedge at d_min (raw R
        # frequently negative, normalized R clamped to 0) is diagnosable from
        # the trainer log without re-deriving the rescale.
        logger.info(
            "[adarft] T=%.4f step=%.4f smoothed=%.4f ladder=%d "
            "reward_raw=%.4f reward_norm=%.4f bins=%d "
            "ap_cov=%d ap_top3=%.2f ap_deficit=%d ap_last=%s",
            self.sampler.T,
            self.sampler.last_step,
            self.sampler.last_smoothed_step,
            self.sampler.last_ladder_level,
            reward,
            normalized_reward,
            len(self.bin_difficulty),
            coverage_count,
            top3_share,
            deficit_count,
            self.sampler.last_picked_bins,
        )
        return control

    def _write_back_difficulties(self) -> None:
        """Push live per-bin difficulty into the shared bin map.

        Walks a snapshot of :data:`adaptive_reward._RECENT_ERRORS` so a
        concurrent reward call can't mutate the deque under us. Dedupes
        by ``(plane, ap_bin)`` and queries
        :meth:`AdaptiveRewardSchedule.per_bin_difficulty` for the
        running mean.
        """
        snapshot = recent_errors()
        seen: set[tuple[str, int]] = set()
        for _abs_err_pct, plane, ap_bin in snapshot:
            key = (plane, ap_bin)
            if key in seen:
                continue
            seen.add(key)
            if self.plane is not None and plane != self.plane:
                continue
            running_mean = self.schedule.per_bin_difficulty(plane, ap_bin)
            if running_mean is None:
                # Defensive — a key in the snapshot must have at least one
                # observation, but if filtering ever drops them all, skip
                # rather than write None.
                continue
            self.bin_difficulty.update(plane, ap_bin, float(running_mean))


__all__ = [
    "AdaRFTCurriculumCallback",
    "BinDifficultyMap",
    "CurriculumRepeatingSampler",
]
