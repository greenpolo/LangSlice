# adaptive/ — shared adaptive training primitives

Cross-pipeline primitives that more than one trainer needs to keep up with
each other: a quantile-based difficulty schedule and a coordinate-bin
curriculum sampler. Lifted here from their original homes so SFT, iSFT,
and single-turn RL can all import the same logic without copying it.

For top-level orientation see [`../README.md`](../README.md).

## What's in here

```
adaptive/
├── schedule.py         ← AdaptiveSchedule (lifted from single_turn_rl/adaptive_reward.py)
├── curriculum/         ← coordinate-bin sampler (lifted from training/curriculum/)
│   ├── bins.py
│   ├── weights.py
│   ├── sampler.py
│   ├── log.py
│   ├── cli.py
│   └── README.md
└── __init__.py         ← re-exports: AdaptiveSchedule, make_error_buffer,
                          ErrorObservation, curriculum
```

The package was carved out in two Phase commits:
- `1b2c929` lifted `AdaptiveSchedule` out of `single_turn_rl/adaptive_reward.py`
- `67c7307` moved `training/curriculum/` to `training/adaptive/curriculum/`

Old import paths still work via lazy shims in `single_turn_rl/adaptive_reward.py`
and `training/curriculum/__init__.py`; new code should use the `adaptive.*`
paths directly.

## The two pieces

### `schedule.py` — quantile-based difficulty schedule

A **stateless** dataclass that observes a stream of `(abs_error_mm, plane_extent_mm)`
pairs and computes a tightening reward shape: `sigma_frac` at the 0.5-quantile of
recent errors, `cutoff_frac` at the 0.95-quantile. The deque holding observations
is owned by the caller — pass it in via `make_error_buffer()` — so iSFT and RL
don't accidentally share global state.

Two consumers today:
- `single_turn_rl/adaptive_reward.py` — RL reward-shaping (long-standing user).
- `iSFT/iterate.py` `_kept_rollouts_for_filter` — sets the accept threshold for
  rollouts in expert iteration (`--adaptive-accept` flag, Phase 5).

### `curriculum/` — coordinate-bin difficulty sampler

Per-plane × per-quintile bins; per-bin MAE → per-section sampling weights.
See [`curriculum/README.md`](curriculum/README.md) for the full mechanic,
the EMA/cap/floor formula, and how the weights file is consumed.

Three consumers:
- `sft/train_sft.py` — `--curriculum-weights` activates `CurriculumSFTTrainer`'s
  `WeightedRandomSampler`.
- `iSFT/iterate.py` — `--curriculum-weights-dir` reads each round's
  `round_<k>_weights.json` to bias prompt sampling AND passes it through to
  the SFT subprocess.
- `single_turn_rl/` — `--curriculum-weights` on the RL trainer.

## Pitfalls / things to know

1. **`AdaptiveSchedule` is stateless on purpose.** Older code (`adaptive_reward.py`
   in single_turn_rl) kept the buffer at module level via `_RECENT_ERRORS`.
   That made iSFT + RL contention possible if both imported the schedule.
   Always pass a buffer in via `make_error_buffer()`; never reach into the
   shim's `_RECENT_ERRORS`.

2. **Curriculum + bucketed sampler are mutually exclusive in iSFT.**
   When iSFT runs `train_sft.py --bucketed-shape-sampler` (the default for
   synthetic-row training), the per-row WeightedRandomSampler is bypassed.
   Curriculum still applies at section-selection time upstream (the v2
   sampler in `_phase_rollouts`); just not inside the trainer. This is the
   resolution of Risk #9 in the iSFT speed-upgrade plan.

3. **Section_id vs subject_id keying.** Weights files are keyed by `section_id`
   (Phase 6 decision). `train_sft.py` falls back to `subject_id` for legacy
   rows that pre-date Phase 6 and have no `section_id` field. New writers
   should always emit `section_id` at the top level of the JSONL row (the
   `iSFT/trace_format.py:build_row` helper does this).

## Where the call paths go

```
single_turn_rl/train_grpo.py ──► single_turn_rl/adaptive_reward.py (shim)
                                  └► adaptive.schedule.AdaptiveSchedule

iSFT/iterate.py _phase_filter ──► adaptive.schedule.AdaptiveSchedule
                                  (--adaptive-accept flag)

sft/train_sft.py ──► adaptive.curriculum.weights.read_weights_json
                     ──► adaptive.curriculum.sampler.CurriculumSFTTrainer

iSFT/iterate.py ──► (still uses legacy curriculum.* shim path; works via
                     curriculum/__init__.py __getattr__ delegation)
```

The single legacy import path (iSFT calling `curriculum.weights`) is the
only known shim consumer. Cleaning it up is cosmetic — see the post-Phase-6
review's Minor finding #3.
