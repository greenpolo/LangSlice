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

The GUI also provides a standalone affine debug lane.
`Run ANTs Affine` skips AP estimation, uses the current manual AP slider position, and runs ANTs-only affine registration for isolated troubleshooting.

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
The atlas layer is placed with the same coronal frame geometry contract used by export anchoring, so preview placement aligns with exported geometry semantics.
The overlay remains a visual inspection tool, not a full physically calibrated ABBA-style viewer, and the GUI does not yet compute quantitative registration quality metrics.

## Pixel Size Input

Image ingest now attempts pixel-size auto-detection from TIFF metadata.
If metadata is found, pixel size is auto-applied immediately; otherwise the existing manual value is retained.
The canonical normalized image is kept for preview/export/registration, while a VLM-ready derivative is generated for Gemini calls using aspect-ratio-preserving resize constraints.

In other words:

- pixel size can be metadata-derived or manually overridden
- VLM/AP calls can use a downsampled derivative with dynamically adjusted effective um/px
- full physical pixel-size calibration in registration/orchestration is still incomplete

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

- Overlay geometry now mirrors export geometry, but full physical-space calibration from per-image pixel size is not complete yet.
- Pixel-size auto-detection currently focuses on TIFF metadata.
- The GUI currently validates affine results visually; it does not report quantitative alignment metrics.
- The app exports compatible JSON but does not embed itself into ABBA runtime workflows.
