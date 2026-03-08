# Current Workflow

## What The GUI Does Today

The current desktop workflow is:

1. load a histology image
2. select a BrainGlobe atlas
3. optionally choose the Gemini model
4. run AP estimation
5. run affine estimation
6. inspect the result in the preview panes
7. export QUINT/ABBA-compatible JSON

The GUI entrypoint is `langslice gui`.

## AP Estimation

AP estimation is agentic.
The model receives the target image plus a small atlas toolset that lets it request atlas slices, atlas metadata, and visible region names before submitting a final AP estimate.

The output shown in the GUI is:

- estimated `position_mm`
- model reasoning text
- optional debug trace directory when `LANGSLICE_VLM_DEBUG_DIR` is set

## Affine Estimation

Affine estimation runs after AP estimation.
The current affine result is matrix-first.
The primary backend is ANTsPyX registration against the atlas reference slice at the estimated AP position.
If ANTsPyX is unavailable or fails sanity checks, the code falls back to Gemini-based visual affine estimation.

The GUI shows derived values such as:

- `backend`
- `rotation_deg`
- translation in pixels
- scale
- shear

The affine result also carries the full 3x3 transform matrix and an explicit output frame size.
Those values are used for:

- transformed preview rendering in the GUI
- QUINT export anchoring

## Preview Modes

### Single view

Shows the current slice image rendered into the affine result's output frame after the estimated in-plane affine transform is applied.

### Split view

Shows the transformed histology slice beside an asynchronously loaded atlas reference slice for the current AP position.

### Overlay view

Shows the transformed slice and atlas composite in a shared `QGraphicsView`.
The atlas layer is scaled to be visually comparable to the slice.
This is useful for quick inspection, but it is not a physically calibrated ABBA-style viewer.
It is also a visual inspection tool only; the GUI does not yet compute quantitative registration quality metrics.

## Pixel Size Input

The GUI currently collects pixel size from the user.
At present, that value is retained in GUI state for workflow compatibility, but it does not yet drive physical atlas scaling in the overlay and it does not currently change the ANTsPyX affine coordinate system.

In other words:

- pixel size is part of the current UI
- pixel size is not yet used to physically calibrate preview scaling

## Debug Trace Curation

If `LANGSLICE_VLM_DEBUG_DIR` is configured, successful AP runs produce a trace directory on disk.
After the pipeline finishes, the GUI exposes:

- `Mark Success`
- `Mark Failure`

These buttons move the run directory into corresponding subfolders so traces can be curated for later inspection.
During classification, the GUI can also capture optional run metadata such as:

- ground-truth `position_mm`
- signed and absolute AP estimation error
- freeform notes about the run

That metadata is written to `classification.json` inside the curated trace folder.

## Export Behavior

The export button writes QUINT/ABBA-compatible JSON.
The current export path is intended for downstream import into QUINT-family tooling such as QuickNII, VisuAlign, and ABBA-compatible workflows.

What export does today:

- builds a single-slice QUINT payload
- encodes the atlas plane using an anchoring vector
- uses atlas-native AP coordinates derived from `position_mm`
- derives anchoring from transformed image corners using the full affine matrix

What export does not do:

- invoke ABBA directly
- require Fiji or a JVM
- guarantee full physical-space registration semantics in the preview widgets

## Current Limitations

- Preview scaling is visual rather than full physical-space calibration.
- Pixel size is not yet used to physically scale atlas overlays.
- The GUI currently validates affine results visually; it does not report quantitative alignment metrics.
- The app exports compatible JSON but does not embed itself into ABBA runtime workflows.
