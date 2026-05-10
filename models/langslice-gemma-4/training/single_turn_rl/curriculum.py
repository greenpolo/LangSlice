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
  per-section live difficulty back to the :class:`ManifestIndex` so the
  next batch's band query sees the freshest signal.

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
difficulty signal that hasn't accumulated yet.

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
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler
from transformers import TrainerCallback

from .adaptive_reward import AdaptiveRewardSchedule, recent_errors
from .manifest_index import ManifestIndex

logger = logging.getLogger(__name__)


_BAND_WIDEN_FACTOR: float = 1.5
"""Multiplicative factor applied to band_width at ladder rung 2."""


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
        generator: torch.Generator | None = None,
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

        # AdaRFT hyperparameters
        self.target_reward: float = float(target_reward)
        self.alpha: float = float(alpha)
        self.eta: float = float(eta)
        self.ema_decay: float = float(ema_decay)
        self.d_min: float = float(d_min)
        self.d_max: float = float(d_max)
        self.band_width: float = float(band_width)
        self.max_per_subject_in_batch: int = int(max_per_subject_in_batch)

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

    # ------------------------------------------------------------------ AdaRFT update

    def update(self, recent_reward_mean: float) -> None:
        """Advance the curriculum target ``T`` per the AdaRFT formula.

        The caller is responsible for passing a reward already rescaled to
        ``[0, 1]`` (e.g. ``accuracy_pct / 100``). Passing the raw GRPO
        scalar (which can dip to ``-1`` on format failures) will skew the
        update toward ``d_min`` and wedge the curriculum at trivial
        difficulty.
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
        Once the curriculum callback writes back live difficulties via
        :meth:`ManifestIndex.update_difficulty`, those values flow into
        the next dataset rebuild's specs and the cold-start treatment
        falls away.
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

    def _select_unique_prompts(self) -> tuple[list[int], int]:
        """Run the four-rung fallback ladder, returning (picks, level).

        ``picks`` has length ``self.unique_prompts``. ``level`` is 1-4 per
        the module docstring's ladder definition. The function never
        returns fewer than ``unique_prompts`` indices: at the bottom rung
        it backfills uniformly from the entire dataset.
        """
        n_needed = self.unique_prompts

        # Rung 1: band ± band_width AND subject cap.
        candidates_1 = self._candidates_in_band(self.band_width)
        picks = self._pick_with_subject_cap(
            candidates_1, n_needed, self.max_per_subject_in_batch
        )
        if len(picks) >= n_needed:
            return picks[:n_needed], 1

        # Rung 2: widen the band 50% (one-shot — do NOT mutate self.band_width).
        widened_hw = self.band_width * _BAND_WIDEN_FACTOR
        candidates_2 = self._candidates_in_band(widened_hw)
        picks = self._pick_with_subject_cap(
            candidates_2, n_needed, self.max_per_subject_in_batch
        )
        if len(picks) >= n_needed:
            return picks[:n_needed], 2

        # Rung 3: keep the widened band but drop the subject cap.
        picks = self._pick_with_subject_cap(candidates_2, n_needed, subject_cap=None)
        if len(picks) >= n_needed:
            return picks[:n_needed], 3

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
            # Can only happen if the dataset itself is smaller than
            # unique_prompts; the constructor doesn't reject that case
            # because a smoke run might want batch_size=2 over a 1-row
            # dataset. Pad by repeating the picks we have.
            if not picks:
                # Truly empty — should be impossible because the ctor
                # rejects an empty dataset, but guard anyway.
                raise RuntimeError(
                    "CurriculumRepeatingSampler: no candidates available even "
                    "after uniform backfill — dataset state is inconsistent."
                )
            j = 0
            while len(picks) < n_needed:
                picks.append(picks[j % len(picks)])
                j += 1
        return picks[:n_needed], 4

    # ------------------------------------------------------------------ Sampler protocol

    def __len__(self) -> int:
        """One batch's worth of indices.

        TRL invokes ``iter(sampler)`` once per batch, so the sampler
        advertises a single batch of length ``unique_prompts *
        num_generations``. ``DataLoader`` uses this for the progress bar
        only — the iterator itself is the source of truth.
        """
        return self.unique_prompts * self.num_generations

    def __iter__(self) -> Iterator[int]:
        """Yield one batch worth of indices, repeated for GRPO grouping.

        Picks ``unique_prompts`` distinct rows via the fallback ladder,
        records the achieved ladder level on ``self.last_ladder_level``,
        and yields each pick ``num_generations`` times in a row so TRL's
        group-relative-advantage computation sees identical prompts in
        consecutive slots within a group.
        """
        picks, level = self._select_unique_prompts()
        self.last_ladder_level = level
        for idx in picks:
            for _ in range(self.num_generations):
                yield idx


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class AdaRFTCurriculumCallback(TrainerCallback):
    """Trainer callback that feeds reward back into the sampler + manifest.

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
       ``(plane, dataset, section_id)``, and call
       :meth:`ManifestIndex.update_difficulty` for each unique key with
       the running mean from
       :meth:`AdaptiveRewardSchedule.per_section_difficulty`. Uses
       ``if_unknown="warn"`` so stale section keys (e.g. after a
       manifest rebuild between runs) don't crash training.
    4. Log a one-line diagnostic via the module logger so trackio /
       tensorboard runs can grep ``T`` / ``ladder`` over time even
       without direct access to the callback's state.
    """

    def __init__(
        self,
        sampler: CurriculumRepeatingSampler,
        *,
        manifest_index: ManifestIndex,
        schedule: AdaptiveRewardSchedule,
        plane: str | None = None,
        write_back: bool = True,
    ) -> None:
        self.sampler = sampler
        self.manifest_index: ManifestIndex = manifest_index
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

        self.sampler.update(reward)

        if self.write_back:
            self._write_back_difficulties()

        # Diagnostic — one line per ``on_log`` call. Cheap and grep-friendly.
        logger.info(
            "[adarft] T=%.4f step=%.4f smoothed=%.4f ladder=%d reward=%.4f",
            self.sampler.T,
            self.sampler.last_step,
            self.sampler.last_smoothed_step,
            self.sampler.last_ladder_level,
            reward,
        )
        return control

    def _write_back_difficulties(self) -> None:
        """Push live per-section difficulty into the manifest index.

        Walks a snapshot of :data:`adaptive_reward._RECENT_ERRORS` so a
        concurrent reward call can't mutate the deque under us. Dedupes
        by ``(plane, dataset, section_id)`` and queries
        :meth:`AdaptiveRewardSchedule.per_section_difficulty` for the
        running mean — that method also reads ``_RECENT_ERRORS`` but the
        cost is small (linear scan filtered by key) and the alternative
        (carrying a parallel running-mean structure) duplicates state.
        """
        snapshot = recent_errors()
        seen: set[tuple[str, str, str]] = set()
        for _abs_err, _axis, plane, dataset, section_id in snapshot:
            key = (plane, dataset, section_id)
            if key in seen:
                continue
            seen.add(key)
            if self.plane is not None and plane != self.plane:
                continue
            mean = self.schedule.per_section_difficulty(key)
            if mean is None:
                # Defensive — a key in the snapshot must have at least one
                # observation, but if filtering out axis_span<=0 ate them
                # all (per_section_difficulty's inner filter), skip
                # rather than write None.
                continue
            self.manifest_index.update_difficulty(
                plane,
                dataset,
                section_id,
                score=float(mean),
                source="live_rollout",
                if_unknown="warn",
            )


__all__ = [
    "AdaRFTCurriculumCallback",
    "CurriculumRepeatingSampler",
]
