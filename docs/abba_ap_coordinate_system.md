# AP Coordinate System

## Atlas-Native, Not Bregma

LangSlice uses atlas-native coordinates throughout the active codebase.
The AP value carried through atlas helpers, VLM prompts, GUI state, and export is `position_mm`: distance in millimeters from the anterior edge of the atlas volume.

Bregma is a skull landmark, not an intrinsic coordinate in extracted-brain atlases.
BrainGlobe atlases such as Allen and Kim do not define a universal Bregma origin, so LangSlice does not try to derive or store one.

## What `position_mm` Means

For a BrainGlobe atlas with AP resolution `res_ap_um`:

`position_mm = ap_index * (res_ap_um / 1000.0)`

The active helper functions are:

- `position_mm_to_index(...)` in `langslice_harness.atlas.core`
- `index_to_position_mm(...)` in `langslice_harness.atlas.core`
- `get_position_range_mm(...)` in `langslice_harness.atlas.core`

These helpers are now backed by `langslice_harness.atlas.space`, which builds a `brainglobe-space` `AnatomicalSpace` context from atlas orientation/shape/resolution.

These functions treat:

- `0.0 mm` as the most anterior coronal slice available in the volume
- increasing values as more posterior positions
- the valid range as `0.0` to `(n_ap - 1) * res_ap_mm`

Current guardrails intentionally require coronal-layout atlases (`AP/DV/ML -> axes 0/1/2`) and AP increasing from anterior to posterior.

## Where It Is Used

LangSlice uses `position_mm` consistently in the current implementation:

- atlas slice loading in `langslice_harness.atlas.core`
- AP estimation prompts and tool calls in `langslice_harness.estimation.ap_tool_use`
- CLI and Tauri GUI state management
- QUINT export anchoring in `langslice_harness.export`

The AP estimator asks Gemini to search the atlas in millimeters from the anterior edge, not in Bregma-relative units.

## QUINT / QuickNII Export Expectations

The QUINT anchoring vector stores the slice plane in atlas voxel space, not in Bregma space.
For LangSlice's current coronal export path:

- `position_mm` is converted back to an AP voxel coordinate
- that AP voxel value becomes the origin's atlas AP component in the anchoring vector
- the full in-plane affine matrix maps source-image pixels into a coronal output frame
- the anchoring vector is derived from transformed source-image corners in that frame

This is the expectation implemented by `compute_anchoring(...)` in `langslice_harness.export`.

## BrainGlobe Fields LangSlice Depends On

The current atlas helpers rely on:

- `reference.shape` for `[n_ap, n_dv, n_ml]`
- `resolution` for voxel size in micrometers
- `orientation` and metadata for display and diagnostics

That is enough for LangSlice's current atlas-native AP handling.
The code does not depend on a Bregma offset or on any extra transform metadata.
Broader orientation support is planned but not active yet.
