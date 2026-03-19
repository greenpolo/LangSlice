# Registration Runtime Status

This file keeps its legacy filename, but the content below is a description of the current registration implementation.

## Implemented Files

- `langslice/registration/core.py` - public wrapper `estimate_registration_runtime(...)`
- `langslice/registration/runtime.py` - runtime orchestration and debug-artifact writing
- `langslice/registration/agents.py` - Gemini prompt and parsing for correspondence pairs
- `langslice/registration/solver.py` - affine and TPS fitting helpers
- `langslice/registration/types.py` - result classes and affine helper functions

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

The current correspondence agent lives in `langslice/registration/agents.py`.

Current facts from the code:

- it prepares one atlas image and one slice image
- it builds a single-pass schema for paired landmarks
- points are requested in normalized `[y, x]` coordinates on a `0..1000` scale
- those normalized values are converted back into pixel coordinates for both atlas and slice
- the registration thinking budget comes from `REGISTRATION_THINKING_BUDGET` in `langslice/vlm/config.py`

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

The codebase contains some helpers and concepts that are not fully wired into the live runtime yet.

- `vet_correspondences(...)` exists in `registration/solver.py`, but `registration/runtime.py` does not call it.
- `runtime.py` currently copies all returned correspondences into `accepted_correspondences`.
- `rejected_correspondences` is currently returned as an empty list.
- `qc_state` is currently set to `accepted` without additional branching.
- `pixel_size_um` is accepted by the runtime signature but ignored inside `registration/runtime.py`.
- `NonlinearResult` is computed and returned, but the GUI export path does not serialize it.

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

The remaining gaps are mostly about stricter vetting, richer QC states, and using the nonlinear result beyond internal storage and tests.
