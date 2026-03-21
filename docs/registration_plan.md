# Registration Runtime Status

This file keeps its legacy filename, but the content below is a description of the current registration implementation.

## Implemented Files

- `langslice/registration/core.py` - public wrapper `estimate_registration_runtime(...)`
- `langslice/registration/runtime.py` - runtime orchestration and debug-artifact writing
- `langslice/registration/agents.py` - shared utilities (retry, heartbeat, JSON extraction, coordinate conversion) and workflow router
- `langslice/registration/agents_image_gen.py` - two-shot workflow for image-generation models (CV-based landmark extraction)
- `langslice/registration/agents_tool_loop.py` - iterative tool-loop workflow for text-centric models (default)
- `langslice/registration/solver.py` - affine and TPS fitting helpers
- `langslice/registration/types.py` - result classes, annotation types, and affine helper functions

## Current Registration Pipeline

The live runtime currently does this:

1. receive the slice image, atlas name, and `position_mm`
2. load the atlas in `registration/runtime.py`
3. build either a composite atlas slice or a plain reference slice
4. ask Gemini for correspondence pairs through `estimate_registration_correspondences(...)`
5. require at least 3 pairs in `runtime.py`
6. fit an affine transform from atlas coordinates to slice coordinates
7. fit a TPS result from the same correspondence list
8. return a `RegistrationResult`

## Agent Stage

The correspondence agent system is split across `agents.py` (shared utilities and router) and two workflow modules.

### Workflow: image_gen_two_shot (agents_image_gen.py)

- Exclusively for Gemini image-generation models (e.g. gemini-3-pro-image-preview, gemini-3.1-flash-image-preview)
- Two passes: (1) atlas annotation, (2) slice transfer using annotated atlas as reference
- Atlas upscaled to ~1K before pass 1 to match model output resolution and avoid spatial copying
- Slice receives 1.5x exposure boost before pass 2 to improve anatomical visibility
- Prompt warns model about common microscopy artifacts (bubbles, tears, tissue damage)
- Generated images saved for visual inspection; marker extraction is TODO
- Must use `ai_studio` backend — Vertex serves degraded image-gen quality

### Workflow: multimodal_tool_loop (agents_tool_loop.py)

- Model iteratively proposes and refines landmarks across multiple turns using tool calls
- Configurable max steps via `REGISTRATION_TOOL_LOOP_MAX_STEPS`

### Shared utilities (agents.py)

- Retry with exponential backoff and heartbeat progress reporting
- JSON extraction from Gemini responses (structured output + text fallback)
- Coordinate conversion: normalised [y,x] to pixel [x,y], with `pixel_coordinates` flag for image-gen bypass
- Thinking level comes from `THINKING_LEVEL` in `langslice/ai/config.py`

## Deterministic Stage

The current deterministic helpers are:

- `fit_affine_from_correspondences(...)`
- `fit_tps_from_correspondences(...)`

`fit_affine_from_correspondences(...)`:

- solves a least-squares affine matrix
- validates the matrix with `is_valid_affine_matrix(...)`
- stores residual summary metrics in `AffineResult.provenance`

`fit_tps_from_correspondences(...)`:

- uses `scipy.interpolate.RBFInterpolator`
- uses the `thin_plate_spline` kernel
- computes residual metrics and Jacobian checks
- stores those values in `NonlinearResult.qc_metrics`

## Debug Artifacts

When a debug directory is available, `registration/runtime.py` writes:

- `slice.png`
- `atlas.png`
- `slice_landmarks.png`
- `atlas_landmarks.png`
- `registration.json`

If registration is launched from the AP agent pipeline, those files are written to a `registration/` subdirectory under the AP run directory.
If registration is launched from the manual workflow and `LANGSLICE_VLM_DEBUG_DIR` is set, the GUI creates a manual run directory first and registration writes into its `registration/` subdirectory.

## Important Literal Gaps In The Current Runtime

- `NonlinearResult` is computed and returned but the GUI export path does not serialize it.
- The `image_gen_two_shot` workflow generates annotated images but marker extraction from those images is not yet implemented.

## What The GUI Uses

The GUI currently uses registration results for:

- split-view correspondence markers
- overlay display setup
- affine summary text in the step indicator
- export of the affine path through `build_quint_export(...)`

The GUI does not currently expose a dedicated nonlinear review view or nonlinear export path.

## Summary

The registration subsystem is no longer just a plan.
It already contains:

- a model prompt for correspondences
- deterministic affine fitting
- deterministic TPS fitting
- GUI integration
- debug-artifact writing

The remaining gaps are mostly about completing marker extraction for the image-gen workflow and using the nonlinear result beyond internal storage and tests.
