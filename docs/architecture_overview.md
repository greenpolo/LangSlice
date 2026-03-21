# Architecture Overview

## Purpose

LangSlice is a desktop application for aligning a histology slice image to a BrainGlobe atlas.
The active implementation combines model-assisted image understanding with deterministic local geometry code.

## Package Boundaries

### `langslice.atlas`

`langslice.atlas.core` wraps BrainGlobe access behind plain Python helpers.
Current responsibilities:

- atlas-name normalization in `canonicalize_atlas_name(...)`
- cached atlas loading in `load_atlas(...)`
- AP mm/index conversion through `position_mm_to_index(...)`, `index_to_position_mm(...)`, and `get_position_range_mm(...)`
- reference, boundary, composite, and additional-reference slice extraction
- structure, hierarchy, mask, and visible-region helpers
- local and remote atlas listing helpers

`langslice.atlas.space` is the only place that interprets atlas orientation metadata.
It builds an `AtlasSpaceContext` from `brainglobe-space`, requires AP to increase from anterior to posterior, and then requires coronal layout with AP/DV/ML on axes `0/1/2`.

### `langslice.vlm`

`langslice.vlm.config` owns Gemini configuration and backend selection.
Current backends:

- `ai_studio`
- `vertex_api_key`
- `vertex_adc`

It also exposes:

- `AVAILABLE_MODELS`
- `AVAILABLE_THINKING_LEVELS` (OFF / LOW / MEDIUM / HIGH)
- `CODE_EXECUTION_ENABLED` and `supports_code_execution()`
- AP rollout flags for token counting, File API, context cache, and Interactions API
- a cached shared `google.genai.Client`

`langslice.vlm.estimator` implements AP estimation.
`estimate_position(...)` runs a multi-turn tool loop with these tool names:

- `fetch_atlas_slice`
- `fetch_multiple_atlas_slices`
- `get_atlas_info`
- `get_region_names`
- `submit_estimate`

The file also contains:

- retry/backoff around `generate_content`
- optional File API transport
- optional cached-content use
- optional Interactions API pilot path
- trace emission and debug-artifact writing
- `estimate_ap(...)` as a thin alias to `estimate_position(...)`

`langslice.vlm.batch_eval` is an offline helper for one-shot AP Batch API experiments. It is not part of the live GUI workflow.

### `langslice.registration`

The registration subsystem is split across eight files:

- `types.py` - affine helpers plus `AffineResult`, `RegistrationCorrespondence`, `NonlinearResult`, `RegistrationResult`, `LandmarkAnnotation`, `RegistrationAnnotationSession`
- `agents.py` - shared utilities (retry, heartbeat, JSON extraction, coordinate conversion) and workflow router
- `agents_single_pass.py` - single-pass structured JSON workflow for text-centric models (e.g. gemini-3-flash-preview, gemini-3.1-pro-preview)
- `agents_image_gen.py` - two-shot image-generation workflow for Gemini image-gen models (e.g. gemini-3-pro-image-preview)
- `agents_tool_loop.py` - iterative tool-loop workflow for text-centric models
- `solver.py` - deterministic affine least-squares fit and TPS fit
- `runtime.py` - registration orchestration and debug artifact writing
- `core.py` - public wrapper `estimate_registration_runtime(...)`

Three registration workflows are available, selected by model capabilities or user override:

- **single_pass** — the model receives both images in one turn and returns all paired correspondences as structured JSON.  Points use a flexible coordinate system (pixel, normalized, etc.) declared by the model.
- **multimodal_tool_loop** — the model iteratively proposes and refines landmarks across multiple turns using tool calls.
- **image_gen_two_shot** — exclusively for image-generation models.  Two passes: (1) the model draws numbered landmark annotations on the atlas, (2) a second call transfers matching landmarks onto the histology slice.  The atlas is upscaled to ~1K before pass 1 and the slice receives an exposure boost before pass 2.  Generated images are saved for inspection; marker extraction is not yet implemented.

Current runtime behavior in `runtime.py`:

1. Load the atlas.
2. Build either a composite or reference atlas slice.
3. Ask Gemini for correspondence pairs via the selected workflow.
4. Require at least 3 pairs.
5. Fit one affine transform from atlas coordinates to slice coordinates.
6. Fit one TPS result from the same pairs.
7. Return both results and optionally save debug artifacts.

Important current facts:

- `pixel_size_um` is accepted by the runtime but ignored there.
- `affine_result.provenance["transform_direction"]` is set to `atlas_to_slice`.
- `rejected_correspondences` is currently returned as an empty list by the runtime.
- `qc_state` is currently hardcoded to `accepted` in the runtime.
- `vet_correspondences(...)` exists in `solver.py` and is tested, but `runtime.py` does not call it.

### `langslice.image_prep`

This module handles image ingest and the image that is actually shown to Gemini.
Current responsibilities:

- normalize arbitrary PIL modes to 8-bit RGB
- infer channel labels for the GUI
- render the operator-selected channel/exposure/brightness/contrast settings into a new image
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
- `save_quint_json(...)` writes the JSON file

### `langslice.gui`

The GUI is centered on `MainWindow` in `langslice.gui.main_window`.
Other GUI files are support modules:

- `main_window_components.py` - reusable widgets and display helpers
- `atlas_viewer.py` - threaded split-view atlas loader
- `overlay_viewer.py` - shared-scene overlay viewer using export-style coronal geometry
- `settings_dialog.py` - backend and credential editor for `.env`
- `trace_inspector.py` - in-app viewer for structured AP and registration trace events
- `run_metadata_dialog.py` - dialog used when classifying a saved trace as success or failure

## End-To-End Control Flow

### 1. Image load

When the user loads a file, `MainWindow._load_image(...)` calls `load_image_state(...)`.
That sets:

- `source_image` to the canonical normalized image
- `pixel_size_um` and pixel-size source metadata
- `pil_image` and `agent_vlm_image` after slice-adjustment rendering
- the GUI into manual-position mode with the AP slider initialized to the current slider range

### 2. Agent input adjustments

The main window keeps a second layer of operator-controlled image settings:

- channel enable/disable
- exposure
- brightness
- contrast
- atlas border visibility

`_apply_slice_adjustments()` renders a new display image with `render_slice_agent_image(...)`, prepares a VLM-ready copy with `prepare_image_for_vlm(...)`, and clears old AP/registration outputs when inputs changed.

### 3. AP estimation path

If the user clicks `Run Agent`, `AgentWorker` runs in a `QThread` and calls:

1. `estimate_position(...)`
2. `estimate_registration_runtime(...)` using the returned `position_mm`

### 4. Manual-position path

If the user clicks `Run Registration at Manual Position`, `ManualRegistrationWorker` runs in a `QThread` and:

1. emits an `APResult` built from the slider-selected position
2. calls `estimate_registration_runtime(...)` directly

### 5. Registration solve

`estimate_registration_runtime(...)` delegates to `registration.runtime.estimate_registration(...)`, which produces:

- `RegistrationResult.correspondences`
- `RegistrationResult.affine_result`
- `RegistrationResult.nonlinear_result`
- optional debug artifacts

### 6. Preview and export

The GUI then updates:

- single view: transformed slice image
- split view: transformed slice plus async atlas view with optional landmark markers
- overlay view: slice and atlas in a shared scene using `compute_coronal_frame_geometry(...)`

Export is enabled only after both AP and affine steps complete.
The export path uses `build_quint_export(...)` plus `save_quint_json(...)`.

## Trace And Debug Artifacts

Trace events are assembled with helpers in `langslice.agent_trace` and shown in the GUI trace inspector.

When `LANGSLICE_VLM_DEBUG_DIR` is set:

- AP estimation creates a run directory with the prepared target image, `reasoning.txt`, and `telemetry.json`
- registration writes a `registration/` subdirectory when it receives the AP run directory, or a standalone registration directory when run manually
- the GUI can move the run directory into `success/` or `failure/` and add `classification.json`

## Current Gaps

- Only coronal-layout atlases are supported by the active helpers and export math.
- Registration computes a TPS result, but GUI export still uses the affine result only.
- The runtime does not currently use `vet_correspondences(...)`, even though that helper exists and is tested.
- The overlay viewer accepts pixel-size input for compatibility but does not calibrate placement from it.
