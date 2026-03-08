# AP Tooling Implementation Plan

## Objective

Improve anterior-posterior (AP) estimation by expanding the agent's toolset in a controlled, evidence-driven way.
The immediate priority is not adding tools blindly, but learning from existing `debug_runs/` traces which search strategies, tool sequences, and reasoning patterns already correlate with accurate estimates.

## Guiding Principle

Follow this order:

1. analyze current traces
2. identify repeated success and failure modes
3. add or revise tools to target those failure modes
4. compare against baseline behavior
5. prune tools that add latency or confusion without improving accuracy

This keeps the current "tool abundance first, prune later" strategy, but adds a decision gate before each expansion step.

## Current AP Toolset

The AP agent currently exposes these tools:

- `fetch_atlas_slice(position_mm)`
- `fetch_multiple_atlas_slices(positions_mm)`
- `get_atlas_info()`
- `get_region_names(position_mm)`
- `submit_estimate(position_mm, confidence, reasoning)`

These tools support broad visual search, but only limited anatomical verification.

## Available Evidence Sources

Current debug artifacts already provide useful material for analysis:

- `target.jpg`
- fetched atlas tool images
- `reasoning.txt`
- `telemetry.json`
- `classification.json` when a run has been labeled

Together, these should allow both qualitative review and structured comparison.

## Phase 0 - Analyze Existing Debug Runs First

### Goals

- establish a baseline before changing the toolset
- identify which current tool sequences are associated with low AP error
- identify which current tool sequences waste time or lead the agent astray
- determine whether additional tools are likely to help with search, verification, or both

### Data To Extract Per Run

- image name
- atlas name
- model name
- estimated AP position
- actual AP position, when available
- signed AP error
- absolute AP error
- run outcome (`success`, `failure`, unlabeled)
- tool sequence in order
- count of each tool used
- number of turns
- number of fetched atlas images
- per-turn wall time
- prompt token counts and thinking token counts, when present
- total image bytes sent across the run
- whether the run ended by `submit_estimate`, fallback, or error

### Questions To Answer

- Do accurate runs usually begin with a broad coarse sweep?
- Does `fetch_multiple_atlas_slices` help more than repeated single-slice fetches?
- Does `get_region_names` appear near successful final verification, or mostly add noise?
- Do poor runs over-sample one AP neighborhood without re-expanding the search?
- Do successful runs verify anatomy before submitting, or submit immediately after a visual match?
- Does more tool use improve accuracy, or mainly increase latency?
- Are some runs failing because the available tools do not support the model's likely next question?

### Failure Modes To Look For

- premature submission after a shallow sweep
- repeated narrow search in the wrong AP neighborhood
- over-reliance on grayscale similarity without structural verification
- unnecessary re-querying of atlas metadata
- too many image-heavy calls without narrowing uncertainty
- failure to recover after an incorrect early hypothesis

### Output Of Phase 0

Produce a short review document or table summarizing:

- common successful tool patterns
- common unsuccessful tool patterns
- tools that are clearly useful
- tools that are rarely used or not useful
- missing capabilities suggested by repeated failure cases

No tool additions should be prioritized until this review exists.

## Phase 1 - Low-Risk Tool Additions

These additions are backed by helpers already present in active LangSlice code and should be the first candidates after debug-run analysis.

### 1. `fetch_composite_slice(position_mm)`

Backed by `langslice.atlas.core.get_composite_slice`.

Why:

- overlays structure boundaries on the reference image
- likely improves landmark verification without requiring a separate mask tool
- may outperform plain grayscale slices for hard regions

### 2. `fetch_boundary_slice(position_mm)`

Backed by `langslice.atlas.core.get_boundary_slice`.

Why:

- isolates anatomy shape from intensity/stain mismatch
- useful when the model appears distracted by texture differences

### 3. `list_additional_references()`

Backed by `langslice.atlas.core.list_additional_references`.

Why:

- lets the model discover whether an atlas offers alternative reference modalities
- creates a path toward better stain matching without forcing extra images by default

### 4. `fetch_additional_reference_slice(position_mm, reference_name)`

Backed by `langslice.atlas.core.get_additional_reference_slice`.

Why:

- useful if a non-default atlas reference resembles the target stain more closely
- especially promising for future species and modality coverage

### 5. Replace or supplement `get_region_names(position_mm)` with a stronger summary tool

Current `get_region_names` is useful but noisy.

Potential replacement:

- `get_dominant_regions(position_mm, top_k, hierarchy_level, min_area_fraction)`

Why:

- a ranked area-based region summary is likely more actionable than an unsorted region list
- gives the model cleaner anatomical evidence during final verification

## Phase 2 - Structured Anatomy Query Tools

These are higher-value symbolic tools once Phase 0 suggests the model is trying to verify finer anatomical hypotheses.

### 6. `lookup_structure(query)`

Inspired by BrainGlobe `lookup_df`.

Why:

- lets the model search a structure by acronym or name
- enables downstream tools to work with explicit structure hypotheses

### 7. `get_structure_hierarchy(structure)`

Backed by `langslice.atlas.core.get_structure_hierarchy`.

Why:

- helps when the model recognizes a broad region but not the exact substructure
- supports coarse-to-fine symbolic reasoning, not just visual reasoning

### 8. `get_structure_mask_slice(position_mm, structure)`

Backed by `langslice.atlas.core.get_structure_mask_slice`.

Why:

- provides a binary mask for a hypothesized structure at a candidate AP
- useful for "does this structure plausibly match what I see?" checks

### 9. `get_region_at_point(position_mm, dv_index, ml_index, include_hierarchy=false)`

Backed by `langslice.atlas.core.get_region_at_position` and inspired by BrainGlobe coordinate-query methods.

Why:

- supports direct coordinate-to-structure checks
- becomes especially useful if the model uses code or geometric heuristics to inspect candidate landmarks

### 10. `get_structure_ap_extent(structure)`

Derived from atlas masks rather than currently exposed directly.

Why:

- lets the model ask where along AP a structure exists at all
- can quickly eliminate impossible AP neighborhoods

## Phase 3 - Tool Schema Cleanup

After enough runs are collected with the expanded toolset, simplify the interface.

### Candidate cleanup steps

- merge image-view tools behind a shared `view` parameter if that reduces schema clutter
- cap or discourage image-heavy tools that mainly increase latency
- keep text-first verification tools if they improve decision quality at lower cost
- remove tools that are rarely used or mostly appear in failed runs

## Decision Rules For Promotion Or Removal

Add or keep a tool only if at least one of the following becomes true in the trace data:

- lower median absolute AP error
- better recovery from early wrong hypotheses
- fewer tool turns to reach a correct estimate
- lower latency at equal accuracy
- clearer anatomical verification in successful runs

Deprioritize or remove a tool if it mainly causes:

- more image payload with no accuracy gain
- repeated use in failed runs
- redundant behavior already covered by another tool
- increased confusion about next steps

## Recommended Rollout Strategy

Do not add every candidate at once.

Instead:

1. finish Phase 0 trace review
2. add one small batch of low-risk tools
3. collect another set of labeled runs
4. compare against the current baseline
5. only then add more symbolic tools

Suggested first batch:

- `fetch_composite_slice`
- `fetch_boundary_slice`
- `list_additional_references`
- `fetch_additional_reference_slice`
- improved region-summary tool

Suggested second batch:

- `lookup_structure`
- `get_structure_hierarchy`
- `get_structure_mask_slice`
- `get_region_at_point`

## Non-Goals For This Plan

- full multi-species evaluation design
- prompt-only optimization without tool analysis
- immediate pruning before enough trace evidence exists
- replacing the AP agent with a non-agentic estimator

## Deliverable From This Planning Stage

Before implementing any new tool, the project should have:

- a short baseline analysis of current `debug_runs/`
- a ranked list of missing capabilities supported by trace evidence
- a first small batch of candidate tools chosen for implementation
- explicit criteria for deciding whether each added tool stays or goes
