# Registration Runtime Status

This file keeps its legacy filename, but the content below is a description of the current registration implementation.

## Implemented Files

- `langslice/registration/core.py` -- public wrapper `estimate_registration_runtime(...)`
- `langslice/registration/runtime.py` -- runtime orchestration and debug-artifact writing
- `langslice/registration/agents.py` -- shared utilities (retry, heartbeat, JSON extraction, coordinate conversion) and workflow router
- `langslice/registration/agents_colored_segmentation.py` -- colored segmentation workflow (default for image-gen models): model produces atlas-colored tissue segmentation, itk-elastix B-spline extracts deformation, VisuAlign markers from control points
- `langslice/registration/agents_image_gen.py` -- legacy two-shot workflow for image-generation models (superseded by colored segmentation)
- `langslice/registration/agents_tool_loop.py` -- iterative tool-loop workflow for text-centric models (experimental, on hold)
- `langslice/registration/solver.py` -- affine and TPS fitting helpers
- `langslice/registration/types.py` -- result classes, annotation types, and affine helper functions

## Current Registration Pipeline

### Colored segmentation workflow (default for image-gen models)

The primary registration workflow in `agents_colored_segmentation.py`:

1. Generate four atlas input images at the target AP position: colored region map, smoothed boundary lines, grayscale reference, and the histology slice.
2. Send all four images with prompt to Gemini image-gen. The model warps the colored atlas regions to match the histology anatomy.
3. Classify pixels in the model output back to atlas region IDs using nearest-color matching.
4. Extract smoothed borders from both the atlas and model-output classified maps.
5. Run itk-elastix B-spline registration on the border images to recover the dense deformation field.
6. Warp the atlas RGB through the recovered transform.
7. Extract VisuAlign-compatible `[ox, oy, nx, ny]` markers from B-spline control points.

### Legacy correspondence-based pipeline

Used by the `image_gen_two_shot` and `multimodal_tool_loop` workflows:

1. Receive the slice image, atlas name, and `position_mm`.
2. Load the atlas in `registration/runtime.py`.
3. Build either a composite atlas slice or a plain reference slice.
4. Ask Gemini for correspondence pairs through `estimate_registration_correspondences(...)`.
5. Require at least 3 pairs in `runtime.py`.
6. Fit an affine transform from atlas coordinates to slice coordinates.
7. Fit a TPS result from the same correspondence list.
8. Return a `RegistrationResult`.

## Agent Stage

The correspondence agent system is split across `agents.py` (shared utilities and router) and three workflow modules.

### Workflow: colored_segmentation (agents_colored_segmentation.py)

- Default for Gemini image-generation models.
- Uses itk-elastix B-spline registration instead of landmark correspondences.
- Generates four atlas input images: colored region map, smoothed boundaries, grayscale reference, histology slice.
- Model produces a colored segmentation of the tissue anatomy.
- Pixels classified to atlas region IDs via nearest-color matching.
- Border images extracted from both atlas and model-output classification.
- Elastix B-spline registration recovers the dense deformation field between border images.
- Atlas RGB warped through the recovered transform.
- VisuAlign markers extracted from B-spline control points as `[ox, oy, nx, ny]` pairs.
- Must use `ai_studio` backend -- Vertex serves degraded image-gen quality.

### Workflow: image_gen_two_shot (agents_image_gen.py)

- Legacy workflow for Gemini image-generation models, superseded by colored segmentation.
- Two passes: (1) atlas annotation, (2) slice transfer using annotated atlas as reference.
- Atlas upscaled to ~1K before pass 1 to match model output resolution and avoid spatial copying.
- Slice receives 1.5x exposure boost before pass 2 to improve anatomical visibility.

### Workflow: multimodal_tool_loop (agents_tool_loop.py)

- Experimental, on hold.
- Model iteratively proposes and refines landmarks across multiple turns using tool calls.
- Configurable max steps via `REGISTRATION_TOOL_LOOP_MAX_STEPS`.

### Shared utilities (agents.py)

- Retry with exponential backoff and heartbeat progress reporting
- JSON extraction from Gemini responses (structured output + text fallback)
- Coordinate conversion: normalised [y,x] to pixel [x,y], with `pixel_coordinates` flag for image-gen bypass
- Thinking level comes from `THINKING_LEVEL` in `langslice/ai/config.py`

## Deterministic Stage

The current deterministic helpers are:

- `fit_affine_from_correspondences(...)`
- `fit_tps_from_correspondences(...)`

These are used by the legacy correspondence-based workflows. The colored segmentation workflow uses itk-elastix B-spline registration instead.

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

The colored segmentation workflow additionally writes atlas input images, model output, classified maps, border images, warped atlas, and marker data.

## Important Literal Gaps In The Current Runtime

- `NonlinearResult` from the legacy workflows is computed and returned but the export path does not serialize it.

## Summary

The registration subsystem contains three workflows:

- colored segmentation (default for image-gen models): dense deformation via Elastix B-spline with VisuAlign markers
- image_gen_two_shot (legacy): landmark correspondences from image-gen models
- multimodal_tool_loop (experimental, on hold): iterative landmark refinement

The colored segmentation workflow is the primary path. It produces a dense deformation field and VisuAlign markers without relying on sparse landmark extraction.
