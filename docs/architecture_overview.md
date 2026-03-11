# Architecture Overview

## Purpose

LangSlice is a Python desktop application for VLM-assisted registration of histological brain slice images to BrainGlobe atlases.
The active runtime is organized around four package areas:

- `langslice.atlas` - BrainGlobe atlas loading, AP indexing, and slice extraction
- `langslice.vlm` - Gemini configuration, AP estimation, and Gemini affine fallback
- `langslice.registration` - matrix-first affine registration types and backend orchestration
- `langslice.gui` - PySide6 desktop workflow and threaded execution
- `langslice.export` - QUINT/ABBA-compatible JSON export

## Package Boundaries

### `langslice.atlas`

`langslice.atlas.core` wraps BrainGlobe access behind plain Python helpers.
The active responsibilities are:

- atlas name canonicalization
- atlas loading and caching
- conversion between `position_mm` and AP index
- extraction of reference, boundary, and composite slices
- structure and region lookup helpers
- atlas metadata and atlas listing helpers

The atlas layer is the source of truth for atlas-native AP coordinates.

### `langslice.vlm`

`langslice.vlm.config` builds the Gemini client from environment configuration.
The supported backends are:

- `ai_studio`
- `vertex_api_key`
- `vertex_adc`

`langslice.vlm.estimator` contains the active model-facing workflows:

- AP estimation via manual function calling
- Gemini fallback affine estimation from the target image, optionally with atlas context

### `langslice.registration`

`langslice.registration` is the primary affine layer.
It owns:

- the matrix-first `AffineResult` model
- the nonlinear `NonlinearResult` model
- affine matrix helpers and decomposition helpers
- registration-agent orchestration
- deterministic landmark vetting, affine fitting, and TPS fitting

`langslice.image_prep` handles ingest normalization, pixel-size metadata detection, and VLM-target downsampling policy.

### `langslice.gui`

The GUI is centered on `MainWindow` in `langslice.gui.main_window`.
It owns:

- image loading
- atlas and model selection
- threaded AP and affine estimation
- single, split, and overlay previews
- export
- debug trace curation buttons

Async atlas preview loading is handled separately in:

- `langslice.gui.atlas_viewer`
- `langslice.gui.overlay_viewer`

### `langslice.export`

`langslice.export` converts the estimated AP position and affine transform into a QUINT export payload.
The output is designed to be imported by QUINT-family tooling and ABBA-compatible workflows, but LangSlice itself does not depend on ABBA at runtime.

## Runtime Data Flow

### 1. GUI input

The user loads a slice image and selects an atlas in the GUI.
The image ingest layer normalizes to 8-bit RGB, tries to detect pixel size from TIFF metadata, and prepares a VLM derivative image for Gemini calls.
The canonical normalized image remains the source for preview/export/registration.

### 2. AP estimation

`AgentWorker` runs `estimate_position(...)` in a worker thread.
That function:

- loads the selected atlas
- computes the valid AP range in millimeters
- sends the target image to Gemini
- uses manual function calling so tool responses can include atlas images
- returns `APResult(position_mm, reasoning, debug_dir)`

### 3. Affine estimation

Once AP estimation completes, the same worker runs the LangSlice registration runtime.
That runtime asks a dedicated registration agent for paired anatomical correspondences, then derives affine and TPS outputs deterministically from the same vetted landmark set.
The affine result remains the current preview/export contract.

### 4. Preview

The GUI updates:

- a single transformed slice view
- a split view with the target and atlas
- an overlay view that layers the atlas composite over the slice

Atlas preview loading is asynchronous so atlas I/O does not block the UI thread.

### 5. Export

After AP and affine estimation complete, the GUI calls `build_quint_export(...)` and writes JSON through `save_quint_json(...)`.

The export uses:

- `position_mm`
- atlas shape and resolution
- the full 3x3 affine matrix
- the affine result output frame size

## AP Estimation Tool Loop

The AP agent currently exposes these tools to Gemini:

- `fetch_atlas_slice`
- `fetch_multiple_atlas_slices`
- `get_atlas_info`
- `get_region_names`
- `submit_estimate`

The model is expected to:

1. sweep broadly across AP space
2. narrow around promising positions
3. verify landmarks or region identities
4. submit a final estimate

Automatic function calling is disabled.
The implementation handles the loop manually so atlas images can be injected directly into tool responses.

## Worker-Thread Model

The GUI does not run AP or affine work on the main thread.
`AgentWorker` executes AP estimation and affine estimation inside a `QThread`, then emits progress and result signals back to the main window.

Separate background loader workers are used for atlas preview widgets so atlas slice rendering stays responsive while the user changes atlas or AP position.

## Debug Traces

If `LANGSLICE_VLM_DEBUG_DIR` is set, AP estimation writes per-run artifacts such as:

- the normalized target image
- fetched atlas slice images
- a reasoning log
- the recorded tool conversation

The GUI can then move the trace folder into `success/` or `failure/` subdirectories using the post-run feedback buttons.

## Non-Goals In The Current Codebase

The following are not current runtime goals:

- ABBA, Fiji, or JVM integration at runtime
- Bregma-referenced internal coordinates
- fully physically calibrated viewer overlays
- complete pixel-size-aware affine orchestration across all backends
- quantitative affine quality scoring in the GUI
- automatic migration of atlas coordinates into a skull landmark space

LangSlice is atlas-native, BrainGlobe-based, and export-oriented.
