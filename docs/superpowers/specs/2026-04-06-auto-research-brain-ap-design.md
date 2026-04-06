# Auto-Research: Brain AP Estimation Optimization

**Date:** 2026-04-06
**Status:** Design
**Goal:** Autonomously improve the whole-brain multi-slice AP estimation pipeline using hypothesis-driven iteration.

## Overview

Use [auto-agent](https://github.com/alfonsograziano/auto-agent) to run an overnight optimization loop that iteratively improves the `langslice/brain/` pipeline's accuracy on the M01 golden dataset. Claude Code is spawned fresh for each hypothesis — it analyzes failures, makes one scoped change, runs evals, and decides to keep or discard the change. Accumulated learnings persist across iterations via MEMORY.md.

### Two-Repo Architecture

```
C:\LabSoftware\
├── auto-agent\                         # Orchestrator (cloned from GitHub)
│   ├── src\                            # TypeScript orchestration loop
│   ├── templates\                      # JOB.md, REPORT.md, MEMORY.md templates
│   └── jobs\
│       └── brain-ap-m01\               # Job: optimize brain AP on M01
│           ├── JOB.md                  # Immutable config
│           ├── MEMORY.md               # Evolves each iteration
│           ├── out.log.txt             # Full stdout/stderr audit trail
│           └── hypotheses\
│               ├── 000-baseline\REPORT.md
│               ├── 001-a3f2b1\REPORT.md
│               └── ...
│
└── LangSlice\                          # Target repo
    ├── eval\
    │   └── eval_brain.py               # Eval harness (FORBIDDEN)
    ├── references\TestImages\M01\      # Golden data (FORBIDDEN)
    │   └── ground_truth.json           # 30 slices, AP 2.6–11.4mm
    └── langslice\                      # Pipeline code (MUTABLE)
        ├── brain\                      # Pipeline orchestration
        ├── ai\                         # VLM estimators and config
        ├── cli.py                      # CLI entry points
        └── image_prep.py               # Image preprocessing
```

**Why two repos:** The orchestrator manages git branches in LangSlice. If it lived in the same repo, branching the experiment would also branch the controller. Orchestrator state (MEMORY.md, reports, logs) stays stable across branch switches in the target repo.

## Eval Harness

### Location

`eval/eval_brain.py` — standalone script in LangSlice, imports `langslice.brain` directly. Listed as a forbidden file so the optimization agent cannot modify it.

### Interface

```bash
python eval/eval_brain.py \
  --images references/TestImages/M01 \
  --ground-truth references/TestImages/M01/ground_truth.json \
  --coarse-model gemma-4-31b-it \
  --fine-model gemma-4-31b-it \
  --json
```

**Arguments:**

| Flag | Required | Description |
|------|----------|-------------|
| `--images` | Yes | Directory containing slice TIFF images |
| `--ground-truth` | Yes | JSON file with `{filename: {ap_mm: float}}` entries |
| `--coarse-model` | Yes | Model for anchor estimation (Interactions API, tool-use) |
| `--fine-model` | Yes | Model for refinement passes (image-gen, visual comparison) |
| `--json` | No | Output structured JSON to stdout (default: human-readable) |

### Execution Flow

1. Parse arguments and load ground truth JSON.
2. Discover slice images in `--images` directory (reuses `langslice.brain.discovery.discover_slices`).
3. Match discovered filenames to ground truth entries. Error if any ground truth file is missing from the image directory.
4. Configure models: set `coarse_model` and `fine_model` via the dual-model config surface (see below).
5. Build `BrainEstimationConfig` with current pipeline defaults (the agent may have changed these).
6. Run `asyncio.run(run_brain_estimation(config))`.
7. Compare each slice's `position_mm` against `ground_truth[filename]["ap_mm"]`.
8. Compute summary metrics and classify pass/fail at 0.1mm threshold.
9. Print JSON to stdout.

### Output Format

```json
{
  "summary": {
    "mae_mm": 0.42,
    "median_error_mm": 0.31,
    "max_error_mm": 1.23,
    "pct_within_0.1mm": 0.23,
    "pct_within_0.25mm": 0.47,
    "pct_within_0.5mm": 0.73,
    "n_slices": 30,
    "n_failing": 23,
    "n_passing": 7,
    "failing_threshold_mm": 0.1,
    "accuracy": 0.23
  },
  "per_slice": [
    {
      "filename": "M01_001_001.tif",
      "estimated_mm": 2.71,
      "ground_truth_mm": 2.613,
      "error_mm": 0.097,
      "source": "anchor",
      "status": "pass"
    },
    {
      "filename": "M01_001_002.tif",
      "estimated_mm": 3.52,
      "ground_truth_mm": 2.913,
      "error_mm": 0.607,
      "source": "interpolated+refined",
      "status": "fail"
    }
  ],
  "config": {
    "coarse_model": "gemma-4-31b-it",
    "fine_model": "gemma-4-31b-it",
    "n_anchors": 4,
    "ordering": "strict",
    "refinement": true,
    "thickness_um": 50,
    "interval_um": 200
  }
}
```

The `summary.accuracy` field maps `pct_within_0.1mm` to the name auto-agent's `parseAccuracy` regex expects.

### Failure Classification

- **Pass:** `error_mm <= 0.1`
- **Fail:** `error_mm > 0.1`
- **Regression detection:** The optimization agent compares per-slice results against the previous accepted REPORT.md to identify slices that got worse. This is handled by the agent, not the eval harness.

## Dual-Model Configuration

### Problem

The brain pipeline currently reads a single global `MODEL_NAME` from `langslice/ai/config.py`. The two estimation stages have different requirements:

- **Coarse estimation** (`run_anchor_estimation`): Uses the Interactions API with tool-use. Needs a model that handles multi-turn reasoning well.
- **Fine refinement** (`run_refinement`): Uses `estimate_position_image_gen()` for visual comparison. Needs a model good at image understanding.

### Solution

Add a `coarse_model` / `fine_model` split to the config and thread it through the pipeline:

1. **`langslice/ai/config.py`**: Add `COARSE_MODEL_NAME` and `FINE_MODEL_NAME` globals with getters/setters. Default both to `MODEL_NAME` for backward compatibility.

2. **`langslice/brain/agents.py`**: `run_anchor_estimation()` and `run_refinement()` accept optional `model_name` parameter. If provided, temporarily override the config for that call.

3. **`langslice/brain/types.py`**: Add `coarse_model` and `fine_model` fields to `BrainEstimationConfig`.

4. **`langslice/brain/pipeline.py`**: Pass `config.coarse_model` to anchor estimation and `config.fine_model` to refinement.

5. **`eval/eval_brain.py`**: Maps `--coarse-model` / `--fine-model` CLI args to the config before running.

## auto-agent Job Configuration

### JOB.md

```markdown
## Objective

Minimize mean absolute error (MAE) of the whole-brain multi-slice AP estimation
pipeline on the M01 golden dataset (30 slices, Allen Mouse 25um atlas). Secondary
goal: maximize the percentage of slices within 0.1mm of ground truth.

## Target Repository

- **Path**: ../LangSlice
- **Branch**: auto-research/brain-ap

## Provider

- **Provider**: claude

## Metrics

- **Primary metric**: mae_mm (minimize)
- **Secondary constraints**:
  - pct_within_0.1mm: max 10% regression
  - n_failing: max 20% regression

## Scripts

| Script | Command | When it runs |
|--------|---------|--------------|
| Install dependencies | pip install -e . | Once at job start |
| Build | python -m ruff check langslice/ | After each hypothesis implementation |
| Run evals | python eval/eval_brain.py --images references/TestImages/M01 --ground-truth references/TestImages/M01/ground_truth.json --coarse-model gemma-4-31b-it --fine-model gemma-4-31b-it --json | After each successful build |
| Test | python -m pytest | After each successful build |

## Forbidden Files

- eval/**
- references/**
- tests/**
- langslice/atlas/**
- langslice/export.py
- langslice/registration/**
- tauri-gui/**
- pyproject.toml
- environment.yml

## Constraints

- Models are fixed via the eval command flags. Do not hardcode model names in
  config.py to circumvent this.
- Existing `python -m pytest` tests must pass.
- Do not add external dependencies beyond what is in pyproject.toml.
- Do not break the existing CLI interface (langslice estimate, langslice register,
  langslice estimate-brain must still work).
- The eval harness calls run_brain_estimation() with the config built from current
  defaults. If you change defaults (n_anchors, ordering, etc.), those changes are
  picked up automatically.

## Codebase Overview

LangSlice registers histology slice images to BrainGlobe atlases. The brain
pipeline estimates AP (anterior-posterior) positions for a folder of sequential
slice images.

**Pipeline entry point:** `langslice/brain/pipeline.py` → `run_brain_estimation(config)`

**Current 4-phase architecture:**
1. Parallel anchor estimation — select N anchor slices via center-out placement,
   run two-stage estimation (coarse tool-use + fine image-gen) on each.
2. Deterministic interpolation — distribute remaining slices evenly between anchors,
   extrapolate beyond anchors using interval spacing.
3. Wave-based refinement — refine non-anchor slices in waves radiating from anchors,
   using nano-banana image-gen with windowed search bounds from locked neighbors.
4. Constraint enforcement — enforce monotonic ordering and minimum spacing.

**Key modules:**
- `langslice/brain/pipeline.py` — orchestration, wave computation
- `langslice/brain/agents.py` — async wrappers for coarse and fine estimation
- `langslice/brain/anchor_selection.py` — center-out anchor placement
- `langslice/brain/interpolation.py` — even distribution, extrapolation
- `langslice/brain/constraints.py` — strict/loose/none ordering enforcement
- `langslice/brain/window.py` — refinement search window bounds
- `langslice/brain/checkpoint.py` — JSON save/load for resumability
- `langslice/brain/types.py` — BrainEstimationConfig, SlicePosition, results
- `langslice/brain/discovery.py` — natural-sort image file discovery
- `langslice/ai/estimator.py` — multi-turn Interactions API tool-use AP estimation
- `langslice/ai/estimator_image_gen.py` — image-gen visual comparison estimation
- `langslice/ai/estimator_tools.py` — tool definitions (fetch_atlas, submit_estimate)
- `langslice/ai/config.py` — model names, thinking level, temperature, feature flags
- `langslice/image_prep.py` — image normalization and VLM downsampling
- `langslice/cli.py` — argparse CLI (estimate, register, estimate-brain subcommands)

**Eval harness:** `eval/eval_brain.py` outputs JSON with summary metrics and
per-slice results. The `summary.accuracy` field = pct_within_0.1mm. Per-slice
entries include filename, estimated_mm, ground_truth_mm, error_mm, source, and
pass/fail status at 0.1mm threshold.

**AP coordinates:** Atlas-native millimeters from the anterior edge. 0.0mm =
extreme anterior (olfactory bulb), larger values = more posterior.

## What the Agent Can Do

- Rewrite langslice/brain/pipeline.py — change phase ordering, add/remove phases,
  restructure orchestration entirely
- Rewrite langslice/brain/anchor_selection.py — different placement strategies
- Rewrite langslice/brain/interpolation.py — different interpolation approaches
- Rewrite langslice/brain/constraints.py — different constraint modes/timing
- Rewrite langslice/brain/window.py — different refinement window strategies
- Modify langslice/brain/agents.py — change two-stage to single-stage, adjust
  parameters, add new estimation strategies
- Modify langslice/ai/estimator.py — change system prompts, tool definitions,
  search strategy, validation gates
- Modify langslice/ai/estimator_image_gen.py — change sweep strategy, image
  presentation, number of passes
- Modify langslice/ai/config.py — thinking levels, temperature, feature flags
- Modify langslice/image_prep.py — preprocessing changes
- Modify langslice/cli.py — add flags for new pipeline features
- Add new files in langslice/brain/ or langslice/ai/
- Remove the locking mechanism, eliminate phases, add global optimization passes,
  inject anatomy hints between slices, change wave ordering — any structural change

## Starting State

The pipeline is functional and produces estimates, but has not been optimized for
accuracy. It was designed for correctness and modularity. The current architecture
(4 phases, center-out anchors, wave refinement, strict constraints) reflects
initial design assumptions that have not been validated against ground truth.

No baseline MAE is known yet — the first auto-agent run will establish it.

## Golden Dataset Info

- **Location:** references/TestImages/M01/
- **Size:** 30 TIFF slice images from one mouse brain (M01)
- **Atlas:** Allen Mouse 25um (allen_mouse_25um)
- **AP range:** 2.613mm to 11.413mm
- **Ground truth source:** ABBA-registered positions with offset correction
- **Spacing:** Non-uniform (mostly ~0.3mm intervals, one 1.6mm gap between slices
  19-20 indicating tissue damage or intentional skip)
- **Format:** ground_truth.json with {filename: {ap_mm, atlas, slice_index}} entries

## Environment & Prerequisites

- Windows 11, Python 3.11 (conda env: langslice)
- GEMINI_API_KEY env var set (AI Studio backend)
- LANGSLICE_GENAI_BACKEND=ai_studio
- Node.js 22+ (for auto-agent orchestrator)
- Claude Code CLI installed and authenticated
- Git on PATH

## Priority Hints

- Start with pipeline-level structural changes — they are free (no extra API cost)
  and can have large impact (e.g., skipping interpolation, changing anchor count,
  different constraint timing).
- Prompt engineering changes to the VLM system instructions are high-leverage for
  the image-comparison passes.
- Test architectural alternatives (estimate every slice independently, remove
  locking, bidirectional refinement) before fine-tuning numeric parameters.
- The coarse estimation (Interactions API tool-use) is the most expensive stage.
  If accuracy can be achieved with image-gen only, that saves cost and simplifies
  the pipeline.
- Look at per-slice error patterns: are failures clustered in anterior/posterior
  regions? At extrapolated positions? At anchors vs refinements? The failure
  distribution reveals which pipeline phase needs the most work.
```

## Compatibility Notes

### auto-agent parseAccuracy

auto-agent extracts `| accuracy | <value> |` from REPORT.md for its summary table. The eval harness includes `summary.accuracy` mapped to `pct_within_0.1mm` so the agent can populate this field naturally.

### Windows Path Handling

JOB.md uses `**Path**: ../LangSlice` (forward slash, relative). auto-agent resolves this with `path.resolve(jobDir, targetRepoRelative)` which handles both separators on Windows via Node's `path` module.

### System Prompt Length

The hypothesis system prompt includes baseline report + MEMORY.md + JOB.md + prompt engineering skill. This can reach 10K+ characters. Node passes it via `execFileSync` arguments. Windows command-line limit is 32K characters — should be fine but worth monitoring if MEMORY.md grows large over many iterations.

### Node.js Version

auto-agent requires Node.js 22+. Verify with `node --version` before first run.

## Overnight Workflow

```bash
# 1. One-time setup
cd C:\LabSoftware
git clone https://github.com/alfonsograziano/auto-agent.git
cd auto-agent && npm install

# 2. Create the job
npm run create-job -- --id brain-ap-m01

# 3. Replace jobs/brain-ap-m01/JOB.md with the config above

# 4. In LangSlice: create base branch, add eval harness, commit
cd C:\LabSoftware\LangSlice
git stash  # stash uncommitted registration changes
git checkout -b auto-research/brain-ap
# ... eval harness and dual-model config already committed here ...
git push -u origin auto-research/brain-ap

# 5. Run overnight
cd C:\LabSoftware\auto-agent
npm run run-job -- --id brain-ap-m01 --max-iterations 10

# 6. Morning: review results
npm run generate-changelog -- --job brain-ap-m01
```

## What Gets Built (Implementation Scope)

### In LangSlice (on `auto-research/brain-ap` branch):

1. **`eval/eval_brain.py`** — eval harness script (~150 lines)
2. **Dual-model config** — small changes to `config.py`, `agents.py`, `types.py`, `pipeline.py` to support `coarse_model` / `fine_model`

### In auto-agent (job config only, no code changes):

3. **`jobs/brain-ap-m01/JOB.md`** — filled-in job configuration
4. **`jobs/brain-ap-m01/MEMORY.md`** — seeded with codebase notes from our research
