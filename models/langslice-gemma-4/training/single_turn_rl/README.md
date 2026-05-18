# Single-Turn GRPO Trainer (Unified RL pipeline)

This is the single-turn lane of LangSlice RL — sibling of the parked
multi-turn `rlvr/` trainer. Where `rlvr/` asks the policy to learn the
whole tool-use loop and gets one final-coordinate reward to credit-assign
across landmarks + broad sweep + narrow sweep + submit timing,
`single_turn_rl/` trains only the final coordinate decision: given a
query image and a deterministic atlas slate, output
`{"position_mm": <number>}` and get scored.

As of 2026-05-10 (Task 6) the pipeline is **unified**: a single CLI wires
the adaptive curriculum (`CurriculumRepeatingSampler` + `AdaRFTCurriculumCallback`),
the adaptive bell-curve reward (`AdaptiveRewardSchedule`), and the
index-driven dataset (`ManifestIndex` + `SectionState`) into one
in-training control loop. Static fallback paths (per-trace
`terminal_states.jsonl`, fixed-bell reward, `WeightedRandomSampler`) stay
fully supported as opt-in flags.

The motivating diagnosis:

- SFT already made good answers sampleable (best-of-4 MAE 1.15 mm vs
  greedy 2.88 mm). RL should compress that pass@k advantage into pass@1.
- The 300-step multi-turn GRPO run (`out/rlvr/stage1-n8-overnight`) had
  flat reward + KL drift + 55% failure rate because the multi-turn
  episode shape made the clean reward hard to optimize.
- Single-turn RL trains the same submit-decision the production agent
  reaches but without the tool-loop credit-assignment problem.

**Status:** plumbing scaffold (2026-05-10). No training run yet; gate any
real run on the diagnostics in [Required diagnostics before training](#required-diagnostics-before-training).

## Unified pipeline (default)

Three orthogonal modes, each with an adaptive default and a static opt-out.

| flag | default | static fallback |
|---|---|---|
| `--curriculum-mode` | `adaptive` (`CurriculumRepeatingSampler` + AdaRFT `T` update + per-section live difficulty write-back) | `static_weights` (legacy `WeightedRandomSampler`, requires `--curriculum-weights`) or `none` (vanilla random sampling) |
| `--reward-mode` | `adaptive` (bell sigma/cutoff scale to recent achievement quantiles via `AdaptiveRewardSchedule`) | `static` (fixed `cutoff_frac`/`sigma_frac` from `[reward]` in TOML) |
| `--data-source` | `index` (pull rows from `data/manifest` via `ManifestIndex`, render via canonical `SectionState` slate over the full ~25k+ RLVR allocation) | `terminal_states` (per-trace JSONL from an external trace-factory, requires `--terminal-states PATH`) |

`--data-source` defaults to `index` because RL operates on the full RLVR
allocation; the SFT corpus is reserved for SFT runs only. Use
`terminal_states` for runs against an external trace-factory artifact
(e.g., the agent-built trace generator's output).

The four mutual-exclusion + required-companion rules (enforced by the CLI):

1. `--curriculum-mode adaptive` is incompatible with `--curriculum-weights`.
2. `--curriculum-mode static_weights` requires `--curriculum-weights PATH`.
3. `--data-source terminal_states` requires `--terminal-states PATH`.
4. `--data-source index` ignores `--terminal-states` (warning).

Adaptive-mode knobs override the TOML `[adaptive]` section per key. The
defaults match the spec — see `configs/grpo_single_turn_terminal.toml`.

**Note on Lane A + adaptive curriculum.** When running
`--data-source terminal_states` with `--curriculum-mode adaptive`, the
per-section difficulty write-back path is silently a no-op: terminal-state
specs don't carry the `(plane, dataset, section_id)` keys the
`ManifestIndex` is keyed on, so the sampler's per-iter refresh and the
callback's manifest write-back both find no eligible rows. The AdaRFT
`T`-update still works (driven by the trainer's logged reward), so the
curriculum's global difficulty target keeps moving — but per-section
banding stays at construction-time cold-start. For the full closed-loop
curriculum, use `--data-source index`.

### Recommended workflow

1. (Optional) Seed difficulty from a slicebench summary so the AdaRFT
   curriculum doesn't start cold:

   ```powershell
   python models/langslice-gemma-4/training/tools/seed_difficulty_from_slicebench.py `
     --slicebench-summary _local/slicebench/round0_summary.json `
     --output _local/rlvr_progress/seeded_difficulty.json
   ```

2. Train with the unified pipeline defaults (adaptive everywhere,
   terminal-states data — the production smoke path):

   ```powershell
   langslice-single-turn-rl `
     --config models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml `
     --sft-model out/sft/docker-sft-1011-merged-bf16 `
     --output-dir out/rlvr_single_turn/unified_smoke `
     --terminal-states out/single_turn_rl/terminal_states.jsonl `
     --difficulty-seed _local/rlvr_progress/seeded_difficulty.json
   ```

   Opt into the full-manifest index path (Lane B) once smoke-on-iron has
   validated the curriculum loop:

   ```powershell
   langslice-single-turn-rl `
     --config models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml `
     --sft-model out/sft/docker-sft-1011-merged-bf16 `
     --output-dir out/rlvr_single_turn/index_smoke `
     --data-source index `
     --difficulty-seed _local/rlvr_progress/seeded_difficulty.json
   ```

3. Static fallback if any of the adaptive layers misbehave (the legacy
   path is preserved verbatim for comparison runs):

   ```powershell
   langslice-single-turn-rl `
     --config models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml `
     --sft-model out/sft/docker-sft-1011-merged-bf16 `
     --output-dir out/rlvr_single_turn/legacy_smoke `
     --curriculum-mode static_weights `
     --curriculum-weights out/single_turn_rl/round_0_weights.json `
     --reward-mode static `
     --data-source terminal_states `
     --terminal-states out/single_turn_rl/terminal_states.jsonl
   ```

### Trainer-class composition

`_select_trainer_cls` builds the trainer subclass as **mixins**, layered
in this order (outermost first):

```
splice → adarft → curriculum_static_weights → CurriculumGRPOTrainer →
GRPOTrainer
```

* `splice` mixin overrides `__init__` (wraps the processor with the
  sidecar-emitting proxy); always outermost when active so it can pop
  `atlas_cache` before delegating.
* `adarft` mixin overrides `_get_train_sampler` (returns the
  `CurriculumRepeatingSampler` directly — bypasses TRL's default
  `RepeatSampler` because that sampler doesn't wrap arbitrary samplers).
* `curriculum_static_weights` is the legacy
  `curriculum.sampler.CurriculumGRPOTrainer` which overrides
  `_get_train_dataloader` for the `WeightedRandomSampler` path. Mutually
  exclusive with `adarft` (CLI mutex enforces this).

Six effective combinations (3 curriculum modes × splice on/off) are
materialised lazily — no class is built until its combination is
selected.

## Files

| File | Purpose |
|---|---|
| `prompts.py` | Strict-JSON system + user prompt builder. |
| `rewards.py` | TRL-shaped reward: parse JSON, score via `rlvr.rewards.normalized_bell_reward`, format/OOR penalties. |
| `adaptive_reward.py` | Rolling-error buffer + `AdaptiveRewardSchedule` + adaptive bell reward factory. |
| `manifest_index.py` | In-memory wrapper over `_local/qc_app/app.py:load_inventory_manifest`; bisect-backed `(plane, atlas, position_range)` queries + per-section live difficulty + JSON sidecar persistence. |
| `section_state.py` | Lane B `SectionState` dataclass + canonical-slate builder + `iter_section_states` generator. |
| `curriculum.py` | `CurriculumRepeatingSampler` (AdaRFT band selection + GRPO repetition) + `AdaRFTCurriculumCallback` (`on_log` writes back per-section difficulty). |
| `terminal_states.py` | Lane A `TerminalState` dataclass + JSONL I/O + SFT-corpus walker CLI. |
| `dataset.py` | Lazy-decode `RowDataset` over both Lane A (terminal states) and Lane B (`SectionState`) row specs; Gemma 4 image-before-text ordering. |
| `train_grpo.py` | Unified GRPO driver: CLI mutex, mixin composition, callback wiring. |
| `eval_single_turn.py` | Held-out N-gen eval (greedy / best-of-N MAE, parse failure %, OOR %, reward stats, per-plane). |

## Reuse from `rlvr/`

This package imports the following from the parked multi-turn trainer
(read-only; no edits to `rlvr/`):

- `rlvr.dataset.preprocess_query_image` — SFT-matched query preprocessing
  (atlas-aware downsample).
- `rlvr.dataset._atlas_in_plane_long_edge` — cached atlas long-edge lookup.
- `rlvr.dataset.canonicalize_atlas_name`, `species_from_atlas_name`.
- `rlvr.rewards.normalized_bell_reward` — axis-normalized truncated Gaussian.
- `rlvr.train_grpo._install_optional_dep_stubs` — TRL optional-dep gate.
- `rlvr.train_grpo._filter_grpo_config_for_installed_trl` — GRPOConfig key
  filter for the installed TRL build.
- `rlvr.train_grpo._adapter_base_model_name` — PEFT adapter base resolver.

If `rlvr/` is ever deleted, these helpers must be promoted to a shared
module first (or duplicated here).

## Ground truth

The terminal-state walker joins each SFT row to the manifest shard at
`data/manifest/shards/<plane>/<dataset>.jsonl` (via
`ManifestGTLookup`) and uses the shard's `position_mm` as the
training reward target. Rows that don't resolve are dropped — the
walker never silently falls back to the SFT teacher's submit, since
that would cap policy accuracy at the teacher's accuracy and turn
RLVR into imitation. The teacher's submit is preserved for diagnostic
comparison under `quality["teacher_position_mm"]`.

The matcher uses `(plane, subject_id)` to narrow shard candidates,
then word-bounded substring match on the section part (after the
`:` in `section_id`) against the query filename's stem. Single-digit
sections (e.g. `ad_bxd` slice "7") and multi-token sections (e.g.
`deepslice_gt` `641_2002_2565_NM01_s102_10x_M`) are both handled.
When multiple shards carry the same `(subject_id, section)` (rare),
the dataset name in the query stem breaks the tie.

Atlas-image positions in the captions come from the on-disk filenames
(`<X.XX>mm.jpg`) the production tool returned, NOT from
`tool_call.args.positions_mm`. The production tool can snap or dedupe
positions before rendering, so reading positions from the args list
risks labeling slice *i* with a coordinate that's actually for a
different snap point. The walker fails closed (drops the row) on any
unparseable atlas filename.

## Output contract

The policy is asked for **exactly** one JSON object:

```json
{"position_mm": 4.37}
```

Anything else — surrounding prose, markdown fences, missing key, extra
keys, non-numeric value, NaN, infinite — earns `format_penalty` (–1.0 by
default). A clean parse with `position_mm` outside `valid_range_mm`
earns `out_of_range_reward` (0.0 by default — neither penalize nor
credit). In-range answers earn the axis-normalized truncated-Gaussian
bell reward.

## Run order

### 1. Build the terminal-state JSONL

Run from the repo root with `$env:PYTHONPATH = "models/langslice-gemma-4/training"`.
`--manifest-root` defaults to `data/manifest` and `--repo-root` defaults
to the current working directory; both align with the trainer's
`--repo-root .` default.

```powershell
python -m single_turn_rl.terminal_states build `
  --sft-corpus models/langslice-gemma-4/data/sft_examples.jsonl `
  --output out/single_turn_rl/terminal_states.jsonl `
  --tier strict
```

Run from the repo root with
`$env:PYTHONPATH = "models/langslice-gemma-4/training"`. The walker:

- filters to `system_prompt_kind == "single_slice"`,
- filters by `acceptance_tier`,
- cuts each trace immediately before the first `submit_estimate` step,
- collects every `fetch_atlas` call's `image_paths` in chronological
  order,
- looks up `valid_range_mm` from the BrainGlobe atlas (cached per
  `(atlas, plane)` pair),
- writes one `TerminalState` row per accepted trace.

### 2. Hold out an eval slice

```powershell
# Example: hold out every 5th subject's terminal states.
python -m single_turn_rl.terminal_states build `
  --sft-corpus models/langslice-gemma-4/data/sft_examples.jsonl `
  --output out/single_turn_rl/terminal_eval.jsonl `
  --tier strict `
  --max-rows 200
```

(Subject-level holdout via `eval_holdout_every` lives in the trainer's
`build_datasets` and operates on whatever JSONL you pass — cleaner is to
build a single JSONL and rely on the trainer's holdout, but for explicit
slicebench-style eval you can build a separate held-out JSONL.)

### 3. Required diagnostics before training

Before any GRPO run with `max_steps > 100`, the plan requires:

```powershell
python -m single_turn_rl.eval_single_turn `
  --model out/sft/docker-sft-1011-merged-bf16 `
  --eval-states out/single_turn_rl/terminal_eval.jsonl `
  --num-generations 1 4 8
```

Proceed only if best-of-N MAE is meaningfully better than greedy MAE on
the eval slice. RL is most likely to help when there is a clear
pass@k-to-pass@1 gap. Abort/redesign if reward std is near zero for most
groups.

### 4. Train

Plain GRPO (uniform sampling, live SigLIP):

```powershell
langslice-single-turn-rl `
  --config models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml `
  --sft-model out/sft/docker-sft-1011-merged-bf16 `
  --terminal-states out/single_turn_rl/terminal_states.jsonl `
  --output-dir out/rlvr_single_turn/terminal_smoke
```

With curriculum sampling (re-weights training examples by per-bin
slicebench MAE) **and** the atlas-embedding splice (skips SigLIP for
cached atlas reference images):

```powershell
langslice-single-turn-rl `
  --config models/langslice-gemma-4/training/configs/grpo_single_turn_terminal.toml `
  --sft-model out/sft/docker-sft-1011-merged-bf16 `
  --terminal-states out/single_turn_rl/terminal_states.jsonl `
  --output-dir out/rlvr_single_turn/terminal_smoke `
  --curriculum-weights out/single_turn_rl/round_0_weights.json `
  --atlas-embedding-cache out/atlas_embeddings
```

The two optional features are independent — pass either or both.

#### Curriculum

When `--curriculum-weights` points at an existing JSON file:

- The train dataset becomes a `WeightedRowDataset`, sampled with
  `torch.utils.data.WeightedRandomSampler` via the
  `curriculum.sampler.CurriculumGRPOTrainer` subclass. Default weights are
  uniform (1.0) until `update_weighted_dataset()` mutates them.
- A per-rollout reward log lands at
  `<output_dir>/curriculum_log.jsonl` with one row per
  `(round, section_id, bin_idx, plane, abs_err_mm, accuracy_pct)`.
  `data.curriculum_round` (TOML) is recorded in each row so a later
  weight-update pass can scope the read.
- Section bins use `slicebench.score._bin_index` (5 quintiles per plane),
  same as `rlvr/`'s curriculum hookup.

When the path doesn't exist, the trainer logs a warning and falls back
to uniform sampling — no behavioral difference from omitting the flag.

#### Atlas-embedding splice

When `--atlas-embedding-cache` points at a directory of
`<atlas>_<plane>.pt` files (built once via `python -m embeddings.precompute`):

- Every `(atlas, plane)` pair the training set uses is checked against
  the cache; missing pairs are warned (splice falls back to live SigLIP
  for those images).
- After PEFT wrap, `embeddings.splice.install_atlas_splice(model)` is
  called — this monkey-patches the inner Gemma 4 model's
  `get_image_features` and registers a forward pre-hook that strips
  three sidecar tensors (`precomputed_image_mask`,
  `precomputed_cached_flat`, `precomputed_cached_patch_counts`) from
  forward kwargs.
- A `SingleTurnGRPOTrainer` subclass wraps the processor with a
  sidecar-emitting proxy that:
  1. stashes per-row `image_paths` from each generation batch,
  2. when TRL's rollout code calls `processor(images=..., text=...)`,
     looks up each image's path via `cache.lookup_by_path` and appends
     the three sidecars to the processor output,
  3. those sidecars flow into `forward_kwargs` (via TRL's existing
     "everything except `input_ids` / `attention_mask`" rule) and into
     the splice hook.

**Pre-flight gate (mandatory):** before activating the splice, run
`python -m embeddings._verify_cache --cache-dir out/atlas_embeddings
--corpus-root models/langslice-gemma-4/data --model
unsloth/gemma-4-E4B-it`. Bit-exact correctness is the gate; subtly-wrong
cached embeddings silently corrupt training.

**Cache invalidates** when (a) base SigLIP weights change (e.g. expert
iteration produces a new merged-bf16 base) or (b) atlas JPGs change.
Re-run precompute after either.

Stop early if any of the plan's stop-rule conditions hold after 25-50
steps:

- reward mean flat + reward std low,
- KL grows monotonically without eval improvement,
- completion length rises instead of staying near JSON answer length,
- parse failures increase,
- held-out single-turn MAE worsens vs the SFT initialization.

### 5. Re-eval and gate

```powershell
python -m single_turn_rl.eval_single_turn `
  --model out/rlvr_single_turn/terminal_smoke `
  --sft-model out/sft/docker-sft-1011-merged-bf16 `
  --eval-states out/single_turn_rl/terminal_eval.jsonl `
  --num-generations 1 4 8
```

A checkpoint passes only if **both** gates hold:

- single-turn held-out eval improves pass@1 MAE or failure rate, and
- production SliceBench eval does not regress (ideally improves
  num_gen1 while preserving the existing best-of-4 ceiling).

## Hyperparameters

See `configs/grpo_single_turn_terminal.toml`. Key differences from
`grpo_pilot.toml`:

| key | rlvr (multi-turn) | single_turn_rl |
|---|---|---|
| `max_completion_length` | 3072 | 128 |
| `mask_truncated_completions` | true | **false** |
| `learning_rate` | 5e-6 | 1e-6 |
| `loss_type` | `dr_grpo` | `dapo` |
| `temperature` | 0.9 | 1.0 |
| `top_p` | 0.95 | 1.0 |
| `max_steps` | 300 | 100 |
| `stop_tool_names` | set | absent (no tool loop) |
| `max_tool_calling_iterations` | 12 | absent |

Smoke grid the plan suggests before an overnight run:

- `scale_rewards="batch"` vs `"none"`
- `learning_rate=1e-6` vs `3e-6`
- `num_generations=8` vs `16` if VRAM allows
- optional small `beta` if KL drifts again and memory permits
  reference-model loading.

## Procedural-trace expansions (2026-05-10)

Two opt-in flags landed that consume `langslice_traces.generator` to expand training pools beyond what teacher traces alone can cover. See `../langslice_traces/README.md` for the factory design + realism story.

### Lane A: `--include-synthetic` (terminal_states.py)

Expands the Lane A pool from 1716 → ~27,562 by synthesizing realistic prefixes for RLVR-split sections without a teacher trace.

```powershell
python -m single_turn_rl.terminal_states build `
  --sft-corpus models/langslice-gemma-4/data/sft_examples.jsonl `
  --output out/single_turn_rl/terminal_states.jsonl `
  --tier strict `
  --include-synthetic `
  --synthetic-seed 1337 `
  --atlas-embedding-cache out/atlas_embeddings
```

Synthetic rows are tagged `source="procedural_generator:lane_a"` (importable as `terminal_states.SYNTHETIC_LANE_A_SOURCE`). The synthesizer enforces sort/grid-compliance and matches the empirical step-0 GT correlation (~0.94 vs corpus 0.98), step-0 integer rate (~46% vs corpus 49.5%), and the n_fetch distribution.

Splits default to `("rlvr",)` per the hard data-pool policy — SFT-allocated sections are never synthesized over.

### Lane B: `--randomized-slate` (section_state.py)

Replaces the deterministic 9-position canonical slate with a per-row randomized broad slate. GT lies inside the range but is never centered (>60% of slates have `|gt_fraction - 0.5| > 0.1`).

```powershell
python -m single_turn_rl.section_state ... `
  --randomized-slate `
  --randomized-seed 1337 `
  --atlas-embedding-cache out/atlas_embeddings
```

Randomized rows tagged `source="procedural_generator:lane_b"` (`section_state.RANDOMIZED_LANE_B_SOURCE`). The deterministic builder is preserved (default-off) for repro experiments.

## Not implemented (deferred)

These are specified in the plan but out of scope for the current scaffold:

- **Lane B secondary local slate.** A tight non-GT-centered slate near the SFT model's greedy estimate, appended to the broad slate. Adds production-style local refinement.
- **Lane C: next-action single-turn GRPO.** Reward design needs more
  work (shaped reward for fetch validity / coverage / bracketing in
  addition to submit coordinate). The matching `lane_c_intermediate` generator strategy is also deferred — built alongside the Lane C trainer when that lane comes online.
- **Production grid format (`send_individually=False`).** Production
  default is `send_individually=True`, so individual atlas images
  already match. If production switches to grids, regenerate terminal
  states with the production grid builder.
- **Curriculum weight-update loop.** The trainer reads weights JSON +
  writes a per-rollout log; the offline weight update (read log →
  per-bin MAE → next round's weights JSON via
  `curriculum.weights.compute_weights`) is the same pattern as
  `../iSFT/iterate.py` for SFT and is left for the
  caller to wire.

## Why this is separate from `rlvr/`

The plan explicitly says to keep this lane separate so we can compare
against the failed multi-turn run and delete cleanly if it
underperforms. Don't refactor the two trainers into a shared base class
until at least one of them produces a transferable checkpoint.
