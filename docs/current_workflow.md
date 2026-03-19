# Current Workflow

This file describes what the GUI currently does in code.

## Loading A Slice

When the user loads an image, `MainWindow._load_image(...)`:

1. calls `load_image_state(...)`
2. normalizes the source image to 8-bit RGB
3. detects pixel size from TIFF or OME metadata when available
4. stores a canonical image and a VLM-ready downsampled image
5. resets prior AP and registration results
6. switches the step label to `Manual Position`
7. initializes the AP slider and manual-position UI

The GUI supports these file types through the file picker and drag-and-drop path:

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`

## Agent Input Adjustments

Before running either workflow, the operator can change the exact image shown to the model.
The current controls are:

- channel toggles
- exposure slider
- brightness slider
- contrast slider
- atlas border toggle

Current implementation details:

- these settings update the displayed slice image
- a VLM-sized derivative image is rebuilt after adjustments
- changing them clears prior AP and registration outputs
- atlas border visibility also updates the split and overlay atlas viewers

## Automatic Workflow: `Run Agent`

The automatic button starts `AgentWorker` on a `QThread`.
That worker does two steps in order:

1. `estimate_position(...)` in `langslice.vlm.estimator`
2. `estimate_registration_runtime(...)` in `langslice.registration.core`

The AP step returns an `APResult(position_mm, reasoning, debug_dir)`.
The registration step returns a `RegistrationResult` containing correspondences, affine output, nonlinear output, QC state, and optional debug-dir metadata.

## Manual Workflow: `Run Registration at Manual Position`

The manual button starts `ManualRegistrationWorker` on a `QThread`.
That worker does not call the AP estimator.
Instead it:

1. emits an `APResult` built directly from the current slider value
2. runs `estimate_registration_runtime(...)` at that exact `position_mm`

The GUI still shows two steps, but the first step is labeled `Manual Position` in this mode.

## Registration Runtime Behavior

The active registration runtime currently does the following:

1. load the selected atlas
2. build either a composite atlas slice or a plain reference slice
3. ask Gemini for correspondence pairs in `registration/agents.py`
4. require at least 3 pairs
5. fit one affine transform from atlas coordinates to slice coordinates
6. fit one TPS result from the same pairs
7. return both results to the GUI

Important current limitations of the live runtime:

- `vet_correspondences(...)` exists in `registration/solver.py` but is not called by `registration/runtime.py`
- `qc_state` is currently always set to `accepted`
- `rejected_correspondences` is currently empty in runtime output

## View Modes

### Single

Shows the current slice image.
If the affine result is defined in the slice-image frame, the GUI applies that transform to a transparent output canvas before display.
If the affine result is marked as `atlas_to_slice`, the base slice image is shown without applying the transform.

### Split

Shows:

- the current slice image on the left
- an asynchronously loaded atlas slice on the right

If preview or accepted correspondences are available, both panes draw labeled marker overlays.

### Overlay

Shows the slice and atlas in a shared `QGraphicsScene`.
Atlas placement uses `compute_coronal_frame_geometry(...)`, which is the same coronal frame contract used by export anchoring.
The opacity slider only affects the atlas layer.

## AP Slider Behavior

The AP slider is always expressed in hundredths of a millimeter.
When atlas loading succeeds, the GUI updates the slider range from `get_position_range_mm(...)`.

Changing the slider:

- updates `current_pos`
- updates the manual-position label
- updates the split atlas viewer
- updates the overlay viewer

## Export Behavior

Export is enabled only after both steps have completed successfully.
The current export path:

- loads atlas shape and resolution from BrainGlobe when possible
- falls back to Allen Mouse 25 um defaults if atlas loading fails at export time
- calls `build_quint_export(...)`
- writes JSON with `save_quint_json(...)`

The export currently uses the affine result only.
The nonlinear TPS result is not written into the JSON output.

## Trace Viewer And Run Classification

The right panel includes a `TraceInspector`.
It displays:

- stage events
- runtime status events
- model events
- tool-call events
- tool-result events

If a completed run has a debug directory, the GUI shows `Mark Success` and `Mark Failure` buttons.
Using either button:

- opens `RunMetadataDialog`
- moves the run directory into a sibling `success/` or `failure/` folder
- writes `classification.json` with current run context and optional ground-truth AP

## Pixel Size Notes

Pixel size is tracked through the GUI and image-prep pipeline.
Current behavior is literal:

- metadata-derived pixel size is applied immediately when available
- manual pixel-size edits update GUI state and the VLM-prepared image
- the registration runtime currently accepts `pixel_size_um` but does not use it internally
- `OverlayGraphicsView.set_pixel_size(...)` is currently a no-op kept for API compatibility
