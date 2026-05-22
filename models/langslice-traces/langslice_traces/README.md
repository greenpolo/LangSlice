# langslice_traces

> Moved from `src/langslice_traces/` on 2026-05-11 during training package cleanup.

The factory layer that sits between **raw trace data on disk** and the **HF chat-template messages a trainer eats**. One place that knows how to turn a langslice-native trace (real teacher or procedurally synthesized) into a training example in any of the four modes consumers need.

Built across two passes (Pass 2 commits `e47b45d..a1661fd`, the procedural generator + realism work `f65738a..613be9c`). Pass 3 — wiring consumers — fully landed:
* Lane A + Lane B in `models/training-core/langslice_training/rl/single_turn/`

## Two factory paths

```
                                  ┌───────────────────┐
                                  │   CanonicalTrace   │ ◄── unified contract
                                  └─────────┬──────────┘
                                            │
                                            ▼
                              ┌──────────────────────────┐
                              │      4 renderers          │
                              │  sft_full / answer_only   │
                              │  rl_prefix / isft_prefix  │
                              └──────────┬───────────────┘
                                         ▼
                                   RenderedExample
                                   (messages, tools,
                                    image_paths, target_mm, label_mm)
```

Two ways to produce a `CanonicalTrace`:

**A. Parse an existing teacher trace** (the 1716 Gemini-authored rows at `models/langslice-gemma-4/data/sft_examples.jsonl`):
```python
from langslice_traces import iter_canonical_traces
for trace in iter_canonical_traces(Path("models/langslice-gemma-4/data/sft_examples.jsonl")):
    ...
```

**B. Procedurally synthesize for slices without a teacher trace** (the other ~25k+ in the RLVR allocation):
```python
from langslice_traces import generate_trace, load_atlas_grid
import random

grid = load_atlas_grid(Path("out/atlas_embeddings"), "allen_mouse_25um", "coronal")
trace = generate_trace(
    image_path="data/datasets/coronal/foo/bar.jpg",
    ground_truth_mm=5.2,
    plane="coronal",
    atlas_name="allen_mouse_25um",
    atlas_version="CCFv3",
    subject_id="foo",
    grid=grid,
    strategy="lane_a_prefix",  # or "lane_b_broad_slate"
    rng=random.Random(0),
)
```

Both paths produce `CanonicalTrace` instances that flow into the same renderers.

## Module map

| File | Role |
|---|---|
| `schema.py` | `CanonicalTrace`, `ToolStep`, `FinalAnswer`, `RenderedExample`, `RenderMetadata`. `final_answer` is `Optional[FinalAnswer]` — procedurally-generated prefixes leave it `None`. |
| `parser.py` | `parse_canonical_trace(row)` + `iter_canonical_traces(jsonl_path)`. Reads the on-disk SFT corpus shape. |
| `generator.py` | `generate_trace(strategy=...)` for procedural synthesis + `load_atlas_grid()` + `canonical_atlas_repo_path()`. |
| `_empirical.py` | Frozen distributions extracted from the 1716 corpus (n_fetch per step, span, roundness, step-1 offset, anchor σ). Plus `sample()` / `sample_roundness()` helpers. |
| `renderers/_common.py` | Shared helpers mirrored from `models/langslice-gemma-4/training/sft/render.py` — system prompt, tool schemas, image hydration, image-token injection. |
| `renderers/_prefix.py` | The shared "everything up to but not including the terminal submit" walk used by 3 of the 4 renderers. |
| `renderers/sft_full.py` | `render_sft_full(canonical)` — byte-identical to the SFT trainer's `render_example`. Pinned by a golden-equality test. |
| `renderers/sft_answer_only.py` | `render_sft_answer_only(canonical, prefer_ground_truth=...)` — no tool steps, just system + user + terminal submit. |
| `renderers/rl_prefix.py` | `render_rl_prefix(canonical, ground_truth_mm=...)` — prefix only, sets `target_mm` for RL reward. |
| `renderers/isft_prefix.py` | `render_isft_prefix(canonical)` — same prefix shape, sets `label_mm` from teacher (or leaves `None` for generated traces). |
| `iterator.py` | `TraceIterator(corpus, mode, ...)` — streams a corpus through a renderer with optional manifest join and augmentations. |
| `augmentations.py` | `AtlasFetchJitter` — perturbs a teacher's fetch positions and snaps to the pre-embedded grid. `ClaheMixToggle` is reserved but deferred. |
| `manifest.py`, `accuracy.py`, `quality.py`, `image.py`, `trace_ops.py`, `reducer.py`, `constants.py` | Primitives lifted from the various pipelines that consumed traces before this package existed. |

## The two generator strategies

### `lane_a_prefix` — realistic teacher-style prefix

Two tool steps with image-informed step-0 and GT-near-but-not-leaky step-1, matching the statistical shape of the 1716 corpus.

| Property | Real corpus | `lane_a_prefix` |
|---|---|---|
| n_tool_steps | 1: ~1%, 2: ~99% | sampled from `P_N_TOOL_STEPS` |
| n_positions per fetch | varied (3:19% / 4:24% / 5:37% / 6:6% / ...) | sampled from `P_NFETCH_STEP*` |
| r(step-0 center, GT) | 0.98 | **0.94** (cohort, n=1000) |
| Step-0 integer rate | 49.5% | **46.4%** |
| Step-1 center == submit | 49.5% | varies via `P_CENTER_OFFSET_STEP1` |
| Positions sorted | 99.97% | **100%** (enforced) |

Used by Lane A trainer when `--include-synthetic` is set on `terminal_states.py build`.

### `lane_b_broad_slate` — randomized broad slate

One tool step. Width sampled from the empirical step-0 span distribution clamped to `[3, 12]` mm; GT placed at a uniform random fraction in `[0.1, 0.9]` of the slate (never centered, never an exact position); n positions sampled from `P_NFETCH_LANE_B` (uniform over `{7, 8, 9}`).

Hard invariant: `min(positions) <= gt <= max(positions)` always. Never `gt in positions` (no leakage).

Used by Lane B trainer when `--randomized-slate` is set on `section_state.py`.

## Realism guarantees

The procedural generator is designed so a model trained on a mix of real and synthesized traces can't learn to distinguish them and shortcut on a generator-specific tell:

1. **No fixed step-0 signature.** Step 0 is anchored at `gt + N(0, SIGMA_ANCHOR_MM ≈ 0.89mm)`, so the cohort r(center, GT) matches the real corpus.
2. **GT not the step-1 midpoint.** Step-1 center is offset from GT via `P_CENTER_OFFSET_STEP1`, so the "submit = mean(step-1 positions)" shortcut doesn't generalize from synthesized rows.
3. **Round-number ladder.** Per-position roundness draws from `P_ROUNDNESS_STEP*` (49.5% integers in step 0, 52% tenths in step 1), then snap to the nearest grid point at that tier.
4. **Positions always sorted.** Real corpus has 0.03% unsorted; generator now enforces sort post-snap.
5. **Path format canonical.** All emitted image paths flow through `canonical_atlas_repo_path()` so they resolve under `repo_root / p` consistently across the SFT-teacher, Lane A synthetic, and Lane B randomized sources.

The frozen distributions in `_empirical.py` are extracted from the corpus once (date stamped in the module docstring); regenerate via `python models/langslice-traces/scripts/extract_empirical_distributions.py` if the SFT corpus is materially changed.

## Consumer wiring (Lane A + Lane B)

Both wirings are **opt-in** — default behavior is unchanged.

### Lane A: expand the pool from 1716 → ~27,562

```powershell
$env:PYTHONPATH = "models/langslice-gemma-4/training"
python -m langslice_training.rl.single_turn.terminal_states build `
  --sft-corpus models/langslice-gemma-4/data/sft_examples.jsonl `
  --output out/rl_single_turn/terminal_states.jsonl `
  --tier strict `
  --include-synthetic `
  --synthetic-seed 1337 `
  --atlas-embedding-cache out/atlas_embeddings
```

After walking the 1716 SFT-traced rows, every RLVR-split section that doesn't already have a Lane A row gets a synthesized prefix via `generate_trace(strategy="lane_a_prefix")`. Synthetic rows are tagged `source="procedural_generator:lane_a"` (importable as `terminal_states.SYNTHETIC_LANE_A_SOURCE`) so downstream curriculum/eval can split by source. **Splits default to `("rlvr",)`** — SFT-allocated sections are never synthesized over (per the hard data-pool policy).

### Lane B: randomized per-row slate instead of the deterministic 9-pos canonical slate

```powershell
python -m langslice_training.rl.single_turn.section_state ... `
  --randomized-slate `
  --randomized-seed 1337 `
  --atlas-embedding-cache out/atlas_embeddings
```

Each section gets a fresh randomized broad slate. Synthetic rows tagged `source="procedural_generator:lane_b"` (`section_state.RANDOMIZED_LANE_B_SOURCE`). The deterministic `build_canonical_slate` is preserved for repro experiments — call without the flag.

## TraceIterator (corpus → rendered examples)

```python
from langslice_traces import TraceIterator
from langslice_traces.renderers._common import AtlasMetaCache

cache = AtlasMetaCache()
it = TraceIterator(
    corpus=Path("models/langslice-gemma-4/data/sft_examples.jsonl"),
    mode="isft_prefix",
    seed=42,
    augmentations={"atlas_fetch_jitter": {"sigma_mm": 0.05, "max_calls_jittered": 2}},
    atlas_meta_cache=cache,
    grid_resolver=lambda atlas_plane: load_atlas_grid(Path("out/atlas_embeddings"), *atlas_plane),
)
for example in it:
    # example is RenderedExample: messages, tools, image_paths, target_mm, label_mm
    ...
```

`mode` is one of `"sft_full" | "sft_answer_only" | "rl_prefix" | "isft_prefix"`.

For `mode="rl_prefix"`, pass `manifest=<path to manifest.jsonl>` so the iterator can look up `ground_truth_mm` for each row.

## Tests

```powershell
python -m pytest tests/test_langslice_traces_*.py tests/test_single_turn_terminal_states_synthetic.py tests/test_single_turn_section_state_randomized.py
```

~131 tests cover: schema/parser, all 4 renderers (incl. a byte-identity golden test against the SFT trainer's `render_example`), iterator dispatch + determinism, generator structural + realism (r > 0.85, GT-inside, GT-not-centered, sorted, grid-compliant), empirical distribution sums-to-1, and the Lane A / Lane B consumer wiring including the canonical-path resolution invariant.

## Deferred

These were specified in the design but explicitly scoped out:

- **`lane_c_intermediate` strategy + Lane C trainer.** Cut at any step (not just terminal). Built alongside the Lane C trainer when that lane comes online.
- **Lane B secondary local slate.** A tight non-GT-centered local slate near the SFT model's greedy estimate, appended to the broad slate. Adds production-style local refinement.
- **Legacy wiring cleanup.** Keep consumers on the shared `langslice_traces` renderers instead of private legacy helpers under `_local/`.
- **`ClaheMixToggle` augmentation.** Path-keyed atlas cache invalidation needs design work first — `iterator.py` raises `NotImplementedError` if the key is set.

## Why this exists

Before this package, trace plumbing was duplicated across the SFT trainer (`models/langslice-gemma-4/training/sft/render.py`), the RL prefix walker (`models/training-core/langslice_training/rl/single_turn/terminal_states.py`), and the synthetic distillation builder. Each had its own subtly different copy of the prefix-construction logic and the chat-template rendering. The factory consolidates that into one well-tested surface that consumers opt into.

The procedural generator extends the same surface to the much larger pool of RLVR-allocated slices that never had a Gemini teacher trace — bringing the realism work in `_empirical.py` along so synthesized data doesn't poison training with a learnable shortcut.
