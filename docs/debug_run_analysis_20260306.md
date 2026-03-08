# Debug Run Analysis — 2026-03-06

Analysis of 15 AP estimation runs collected during initial testing of the agentic tool-use loop.
All runs used the `allen_mouse_25um` atlas.

## Run Summary

### Classified Runs (10 of 15)

| Run | Outcome | Image | Abs Error (mm) | Turns | Images Fetched | Wall Time (s) | Notes |
|---|---|---|---|---|---|---|---|
| 20260306_134858 | success | M26_R2C1.tif | 0.14 | — | — | — | |
| 20260306_140133 | success | M5_R2C5.tif | 0.00 | — | — | — | Clear anterior commissures |
| 20260306_143453 | success | M27_R1C1.tif | 0.00 | 7 | 16 | 60 | |
| 20260306_144618 | success | M28_R2C1.tif | 0.00 | 7 | 15 | 694 | Split hemispheres; agent took too long |
| 20260306_150749 | success | M25_R2C3.tif | 0.00 | 8 | 16 | 45 | |
| 20260306_162927 | success | M11_R2C5.tif | 0.00 | 7 | 15 | 613 | Accurate but unnecessarily long |
| 20260306_152328 | failure | M1_R1C1.tif | 0.30 | 11 | 34 | 59 | Very anterior; hard even for humans |
| 20260306_154626 | failure | M12_R1C1.tif | 1.40 | 7 | 15 | 347 | Large tear in one hemisphere |
| 20260306_161323 | failure | M18_R2C1.tif | 0.50 | 6 | 10 | 325 | Premature local commitment |
| 20260306_164706 | failure | M26_R1C3.tif | 0.25 | 7 | 16 | 30 | Very anterior; near-miss |

### Unclassified Runs (5 of 15)

| Run | Category | Has Telemetry |
|---|---|---|
| 20260305_180318 | success (no classification metadata) | no |
| 20260305_185405 | failure (no classification metadata) | no |
| 20260305_192527 | failure (no classification metadata) | no |
| 20260306_130655 | failure (no classification metadata) | no |
| 20260306_141306 | uncategorized | no |

The 5 unclassified runs predate the classification dialog and telemetry logging features.

## Accuracy Distribution

Among the 10 classified runs:

**Successes (6)**: median error 0.00 mm, max 0.14 mm
**Failures (4)**: errors 0.25, 0.30, 0.50, 1.40 mm

The failure threshold appears to be roughly 0.25 mm (the user labeled the 0.25 mm run as failure, though noting the agent was "quite close").

## Successful Run Patterns

Runs that achieved low error share a common structure:

1. **Coarse sweep first**: `get_atlas_info` → `fetch_multiple_atlas_slices` with 4-5 widely spaced positions
2. **Narrowing sweep**: a second `fetch_multiple_atlas_slices` call centered on the best coarse match, with tighter spacing (~0.5 mm)
3. **Optional verification**: occasionally `get_region_names` or `fetch_atlas_slice` for a final confirmation
4. **Confident submission**: `submit_estimate` after 2-3 visual comparison rounds

Typical successful runs use 7-8 turns and fetch 15-16 atlas images.

## Failure Modes

### 1. Over-Exploration Without Convergence (M1_R1C1, 0.30 mm error)

34 images fetched across 11 turns — the most of any run.
The agent kept sampling but never converged.
The slice was very anterior (~3.15 mm) with few distinguishing features.
**Root cause**: the agent lacked a clear decision rule for stopping when all nearby candidates look similar.

### 2. Wrong Early Gestalt Not Corrected (M12_R1C1, 1.40 mm error)

The worst error by far.
The slice had a large tissue tear that visually distorted the overall shape.
The agent committed to a wrong AP neighborhood early and did not recover.
Used `get_region_names` but the region list didn't help it recognize the mismatch.
**Root cause**: the agent's visual matching was thrown off by tissue damage, and symbolic tools (region names) were too noisy to correct course.

### 3. Premature Local Commitment (M18_R2C1, 0.50 mm error)

Only 6 turns and 10 images — the fewest of the classified telemetry runs.
The agent settled on a position without pressure-testing neighboring candidates.
**Root cause**: stopped too early without a verification step.

### 4. Near-Miss on Anterior Slices (M26_R1C3, 0.25 mm error)

7 turns, 16 images, 30s wall time — structurally a normal run.
The agent was close but lacked fine-grained discriminating features.
Anterior slices (~2.9-3.3 mm) have few large landmark structures.
**Root cause**: inherent difficulty of anterior registration.

## Latency Observations

Two successful runs took over 600 seconds despite being accurate:

- **M28_R2C1 (694s)**: split hemispheres caused extra deliberation
- **M11_R2C5 (613s)**: high-quality slice, but agent spent many tokens on reasoning

The latency pathology appears to be large thinking blocks.
When the model produces a thought-only turn (no tool call), `candidates_token_count` can reach ~65,000 tokens, which balloons the prompt on the next turn.
This is the primary latency driver, not tool count or image payload.

## Tool Effectiveness

### `fetch_multiple_atlas_slices`

The most important tool. Present in every successful run's coarse sweep phase.
Allows 4-5 images per call, enabling rapid neighborhood identification.

### `get_atlas_info`

Called at the start of nearly every run. Provides coordinate range context.
Low cost, consistently useful.

### `fetch_atlas_slice`

Occasionally used for single-slice confirmation after narrowing.
Largely superseded by `fetch_multiple_atlas_slices` for search.

### `get_region_names`

Appears in both successful and failed runs.
Returns up to 30 unsorted region names per position.
Too noisy to serve as a reliable verification signal.
Did not appear to be the deciding factor in any successful run.
**Candidate for replacement** with a more focused summary (e.g., top-K by area, hierarchy-aware).

## Model Selection Bug

All 10 classified runs report `model_name` from the GUI combo box in their `classification.json`.
However, the `reasoning.txt` and `telemetry.json` files always recorded `gemini-3-flash-preview` as the actual model.
The user confirmed via the Google API dashboard that only `gemini-3-flash-preview` was ever called.

**Root cause**: `estimator.py` imported `MODEL_NAME` at module load time, so `set_model_name()` in the GUI had no effect.
**Status**: fixed (2026-03-07). The estimator now reads `vlm_config.MODEL_NAME` at call time.

This means the 4 early successes that show `gemini-3.1-pro-preview` in their `classification.json` actually ran on `gemini-3-flash-preview`.
No valid model comparison data exists yet.

## Key Takeaways

1. The coarse-to-fine sweep strategy works well for slices with clear landmarks.
2. Anterior slices (~3.0-3.3 mm) are consistently the hardest to register.
3. Tissue damage (tears, split hemispheres) causes the largest errors.
4. `get_region_names` is not pulling its weight; a ranked area-based summary would be more useful.
5. Latency is dominated by large thinking blocks, not tool overhead.
6. The model selection bug means no `gemini-3.1-pro-preview` runs exist yet.

## Recommended Next Steps

1. Collect new runs with the model bug fixed, especially using `gemini-3.1-pro-preview`.
2. Replace `get_region_names` with a `get_dominant_regions` tool that returns top-K regions ranked by area.
3. Consider adding `fetch_composite_slice` and `fetch_boundary_slice` to help with tissue-damage cases.
4. Investigate adding a "difficulty assessment" preamble where the agent first evaluates target image quality.
5. Continue collecting labeled runs to build a larger baseline before adding more tools.

## References

- Tool implementation plan: `docs/ap_tooling_implementation_plan.md`
- Current workflow: `docs/current_workflow.md`
- Estimator code: `langslice/vlm/estimator.py`
