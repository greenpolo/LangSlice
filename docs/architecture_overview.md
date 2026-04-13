# Architecture Overview

## Purpose

LangSlice is a CLI tool and Tauri desktop application for aligning a histology slice image to a BrainGlobe atlas.
The active implementation combines model-assisted image understanding with itk-elastix B-spline registration for dense deformation recovery.

## Package Boundaries

### `langslice.atlas`

`langslice.atlas.core` wraps BrainGlobe access behind plain Python helpers.
Current responsibilities:

- atlas-name normalization in `canonicalize_atlas_name(...)`
- cached atlas loading in `load_atlas(...)`
- AP mm/index conversion through `position_mm_to_index(...)`, `index_to_position_mm(...)`, and `get_position_range_mm(...)`
- reference, boundary, composite, and additional-reference slice extraction
- colored region slice extraction in `get_colored_region_slice(...)` -- returns annotation slice as RGB with atlas-defined colors
- smoothed boundary contour extraction in `get_smoothed_boundary_slice(...)` -- returns annotation boundaries with smoothed vector contours
- structure, hierarchy, mask, and visible-region helpers
- local and remote atlas listing helpers

`langslice.atlas.space` is the only place that interprets atlas orientation metadata.
It builds an `AtlasSpaceContext` from `brainglobe-space`, requires AP to increase from anterior to posterior, and then requires coronal layout with AP/DV/ML on axes `0/1/2`.

### `langslice.vlm_config`

`langslice.vlm_config` owns Gemini configuration and backend selection, shared by both estimation and registration.
Current backends:

- `ai_studio`
- `vertex_api_key`
- `vertex_adc`

It also exposes:

- `AVAILABLE_MODELS`
- `AVAILABLE_THINKING_LEVELS` (MINIMAL / LOW / MEDIUM / HIGH)
- `CODE_EXECUTION_ENABLED` and `supports_code_execution()`
- Registration workflow constants and `default_registration_workflow()`
- AP rollout flags for token counting, File API, and context cache
- a cached shared `google.genai.Client`

### `langslice.estimation`

Single-slice AP estimation. `estimate_position(...)` runs a multi-turn tool loop with these tool names:

- `fetch_atlas`
- `get_atlas_info`
- `get_region_names`
- `submit_estimate`

The estimator is split across four files:

- `google/ap_tool_use.py` -- the single-slice tool loop, retry/backoff, optional File API/cache, trace emission, debug-artifact writing, and `estimate_ap(...)` alias
- `google/ap_multi_slice.py` -- multi-slice group tool-use AP estimation for 2-8 consecutive slices (`estimate_group(...)`)
- `google/tool_definitions.py` -- tool definitions and tool-response construction helpers
- `debug.py` -- shared debug-artifact writing helpers

`google/ap_image_gen.py` implements nano-banana multi-pass zoom AP estimation using image-gen models.

`google/batch_eval.py` is an offline helper for one-shot AP Batch API experiments. It is not part of the live pipeline.

All estimation uses `generate_content`; the Interactions API was removed from the estimation module.

### `langslice.registration`

The registration subsystem is split across eight files:

- `types.py` -- affine helpers plus `AffineResult`, `RegistrationCorrespondence`, `NonlinearResult`, `RegistrationResult`, `LandmarkAnnotation`, `RegistrationAnnotationSession`
- `common.py` -- shared utilities (retry, heartbeat, JSON extraction, coordinate conversion) and workflow router
- `google/warping_image_gen.py` -- warping workflow via colored segmentation (default for image-gen models): model produces atlas-colored tissue segmentation, Elastix B-spline extracts deformation, VisuAlign markers from control points
- `google/landmarks_image_gen.py` -- legacy two-shot landmark workflow for image-gen models (superseded by warping workflow)
- `google/landmarks_tool_use.py` -- iterative landmark tool-loop workflow for text-centric models (experimental, on hold)
- `solver.py` -- deterministic affine least-squares fit and TPS fit
- `runtime.py` -- registration orchestration and debug artifact writing
- `core.py` -- public wrapper `estimate_registration_runtime(...)`

Three registration workflows are available, selected by model capabilities or user override:

- **colored_segmentation** (default for image-gen models) -- The model produces a colored segmentation of the tissue guided by four atlas input images (colored region map, smoothed boundaries, grayscale reference, histology slice). Pixels are classified back to atlas region IDs via nearest-color matching. Smoothed borders are extracted from both atlas and model-output classified maps. itk-elastix B-spline registration recovers the dense deformation field. The atlas RGB is warped through the recovered transform. VisuAlign-compatible `[ox, oy, nx, ny]` markers are extracted from B-spline control points.
- **image_gen_two_shot** (legacy) -- Two passes: (1) the model draws numbered landmark annotations on the atlas, (2) a second call transfers matching landmarks onto the histology slice. Superseded by colored segmentation.
- **multimodal_tool_loop** (experimental, on hold) -- The model iteratively proposes and refines landmarks across multiple turns using tool calls.

Current runtime behavior in `runtime.py` for legacy workflows:

1. Load the atlas.
2. Build either a composite or reference atlas slice.
3. Ask Gemini for correspondence pairs via the selected workflow.
4. Require at least 3 pairs.
5. Fit one affine transform from atlas coordinates to slice coordinates.
6. Fit one TPS result from the same pairs.
7. Return both results and optionally save debug artifacts.

Current runtime behavior for the colored segmentation workflow:

1. Generate four atlas input images at the target AP position.
2. Send all four images with prompt to Gemini image-gen.
3. Classify model output pixels to atlas region IDs.
4. Extract smoothed borders from both atlas and model-output classified maps.
5. Run itk-elastix B-spline registration on the border images.
6. Warp atlas RGB through the recovered transform.
7. Extract VisuAlign markers from B-spline control points.
8. Return results and optionally save debug artifacts.

### `langslice.whole_brain`

Whole-brain multi-slice AP estimation. Scales the single-slice estimator to 20-60 slices with a four-phase pipeline:

1. **Phase 1 — Anchor estimation:** Select anchor slices via center-out placement, run two-stage estimation on each sequentially (cap=1 semaphore): Stage A is image-gen 2-pass (broad + neighborhood) for coarse positioning, Stage B is nano-banana fine pass within ±0.5mm of the coarse result. 3-tier fallback: Stage A failure → atlas midpoint, Stage B failure → return coarse. Anchor results are soft estimates, not locked truth.
2. **Phase 2 — Interpolation:** Deterministic interval-based interpolation between anchors, extrapolation beyond outermost anchors, clamped to atlas bounds. Produces center positions for Phase 3.
3. **Phase 3 — Parallel slice estimation:** Estimate all non-anchor slices in parallel using 2-pass nano-banana (0.10mm fine resolution) within ±2mm windows centered on interpolated positions, followed by a confirmation pass (9 slices, ±0.25mm, ~0.06mm spacing) to push near-threshold estimates below 0.1mm error.
4. **Phase 4 — Isotonic fitting:** Fit a monotone-increasing curve through all estimates (anchors and non-anchors alike) using Huber-loss constrained optimization with local-interval spacing priors and hard minimum-thickness constraints.

The module is split into focused files:

- `types.py` — `BrainEstimationConfig`, `SlicePosition`, `BrainEstimationSummary`, `BrainEstimationResult`
- `discovery.py` — image folder discovery with natural sort
- `anchor_selection.py` — center-out anchor index selection
- `interpolation.py` — interval-based interpolation and extrapolation
- `window.py` — nano-banana search window bounds and dynamic image count
- `checkpoint.py` — incremental JSON checkpoint for resumability
- `estimation_agents.py` — async wrappers: `run_anchor_estimation()` (image-gen 2-pass coarse + nano-banana fine, 3-tier fallback) and `run_slice_estimation()` (2-pass windowed + confirmation pass), with CLAHE adaptive preprocessing
- `pipeline.py` — Huber-loss isotonic fitting and main `run_brain_estimation()` async entry point

Concurrency uses plain asyncio (`asyncio.gather()` + `asyncio.Semaphore`), not an agent framework. Anchor estimation runs sequentially (semaphore cap=1) to avoid API rate limits. `estimate_position_image_gen()` is the sole VLM entry point; `estimate_position()` (tool-use) is no longer used by the brain pipeline.

### `langslice.image_prep`

This module handles image ingest and the image that is actually shown to Gemini.
Current responsibilities:

- normalize arbitrary PIL modes to 8-bit RGB
- infer channel labels
- detect pixel size from OME metadata or TIFF resolution tags
- downsample the image for VLM use to the configured pixel and long-edge limits
- return a `LoadedImageState` with canonical and VLM-ready images

### `langslice.export`

This module converts the current AP choice and affine result into QUINT/ABBA-compatible JSON.
Key points:

- BrainGlobe AP/DV/ML is mapped to QuickNII ML/AP/DV axis order
- anchoring is computed in atlas voxel space
- `compute_coronal_frame_geometry(...)` is shared with the overlay preview contract
- `build_quint_export(...)` builds one-slice exports only
- `SliceExport.markers` supports VisuAlign `[ox, oy, nx, ny]` pairs from Elastix B-spline control points
- `save_quint_json(...)` writes the JSON file

### `tauri-gui/`

The Tauri desktop app lives in `tauri-gui/`. Rust backend (`src-tauri/`) handles atlas loading, reslicing, and mesh serving. React + Three.js frontend (`src/`) provides 3D atlas visualization, dashboard, split/overlay views, and settings management.

Launched via `cd tauri-gui && pnpm tauri dev`.

## End-To-End Control Flow

### CLI: `langslice estimate`

1. Load and normalize the image with `load_image_state(...)`.
2. Optionally apply adaptive preprocessing (CLAHE + brightness normalization).
3. Downscale to VLM resolution.
4. Run `estimate_position(...)` or `estimate_position_image_gen(...)` depending on model type.
5. Print position and reasoning; optionally write debug artifacts.

### CLI: `langslice estimate-group`

1. Load and normalize each image (2-8 slices in anterior-to-posterior order).
2. Optionally apply adaptive preprocessing (CLAHE + brightness normalization).
3. Downscale each to VLM resolution.
4. Run `estimate_group(...)` with the configured slice interval and thickness.
5. Print per-slice positions and group reasoning; optionally write debug artifacts.

### CLI: `langslice register`

1. Load, normalize, and downscale the image.
2. Call `estimate_registration_runtime(...)` at the specified AP position with the selected workflow.
3. Print registration summary (correspondences, rotation, translation, scale, residuals).
4. Write debug artifacts to output directory.

### CLI: `langslice estimate-brain`

1. Discover and naturally sort all slice images in the folder.
2. Select anchor slices via center-out placement.
3. Run two-stage anchor estimation sequentially (image-gen 2-pass coarse + nano-banana fine pass).
4. Interpolate center positions for all non-anchor slices.
5. Estimate all non-anchor slices in parallel with 2-pass nano-banana (±2mm windows).
6. Fit Huber-loss constrained monotonic curve through all estimates.
7. Write final positions to JSON. Checkpoint after each phase for resumability.

### Tauri GUI

The Tauri GUI communicates with the Python pipeline via a sidecar process. The Rust backend handles atlas loading and 3D mesh serving; the Python sidecar runs AP estimation and registration. Results are displayed in the React frontend with split/overlay views.

## Trace And Debug Artifacts

Trace events are assembled with helpers in `langslice.agent_trace`.

When `LANGSLICE_VLM_DEBUG_DIR` is set:

- AP estimation creates a run directory with the prepared target image, `reasoning.txt`, and `telemetry.json`
- Registration writes a `registration/` subdirectory with atlas images, model output, border images, warped atlas, and `registration.json`

## Current Gaps

- Only coronal-layout atlases are supported by the active helpers and export math.
- The legacy workflows compute a TPS result, but export still uses the affine result only for those paths.
