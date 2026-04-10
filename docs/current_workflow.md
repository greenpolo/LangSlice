# Current Workflow

This file describes the CLI and Tauri GUI workflows as currently implemented.

## CLI: `langslice estimate`

Run AP estimation from the command line:

```
langslice estimate <image> [--atlas ...] [--model ...] [--workflow ...]
```

1. Load and normalize the image to 8-bit RGB.
2. Detect pixel size from TIFF or OME metadata when available.
3. Downscale to VLM resolution (default 2048px long edge).
4. Optionally apply adaptive preprocessing (`--preprocess auto`): CLAHE + brightness normalization.
5. Run AP estimation via Gemini tool-use or image-gen workflow.
6. Print the estimated AP position and reasoning.
7. Optionally write debug artifacts (`--out`).

Supported file types: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.

## CLI: `langslice register`

Run registration at a known AP position:

```
langslice register <image> --position <mm> [--workflow colored_segmentation] [--model ...] [--out ...]
```

1. Load, normalize, and downscale the image.
2. Call `estimate_registration_runtime(...)` at the specified position with the selected workflow.
3. Print registration summary: accepted pairs, rotation, translation, scale, shear, residuals.
4. Write debug artifacts to the output directory.

Workflow selection (`--workflow`):
- `colored_segmentation` (default for image-gen models)
- `image_gen_two_shot` (legacy)
- `multimodal_tool_loop` (experimental, on hold)

If `--workflow` is not specified, the default is auto-selected based on the model via `default_registration_workflow()`.

## Tauri GUI

The desktop application lives in `tauri-gui/` and is launched via `cd tauri-gui && pnpm tauri dev`.

The GUI provides:
- 3D atlas viewer with region mesh rendering
- Pipeline sidecar for running AP estimation and registration
- Settings management (auth backends, model selection)
- Split and overlay views for reviewing registration results
- Dashboard for managing runs

The Python pipeline runs as a sidecar process; the Rust backend handles atlas loading, reslicing, and mesh serving.

## Registration Runtime Behavior

### Colored segmentation workflow (default for image-gen models)

The colored segmentation workflow in `warping_image_gen.py`:

1. Generate four atlas input images at the target AP position: colored region map, smoothed boundary lines, grayscale reference, and the histology slice.
2. Send all four images with prompt to Gemini image-gen. The model warps the colored atlas regions to match the histology anatomy.
3. Classify pixels in the model output back to atlas region IDs using nearest-color matching.
4. Extract smoothed borders from both the atlas and model-output classified maps.
5. Run itk-elastix B-spline registration on the border images to recover the dense deformation field.
6. Warp the atlas RGB through the recovered transform.
7. Extract VisuAlign-compatible `[ox, oy, nx, ny]` markers from B-spline control points.
8. Return results including the warped atlas, border images, and markers.

### Legacy workflows

The legacy workflows (`image_gen_two_shot`, `multimodal_tool_loop`) follow the correspondence-based pipeline:

1. Load the atlas and build an atlas slice.
2. Ask Gemini for correspondence pairs via the selected workflow.
3. Require at least 3 pairs.
4. Fit one affine transform from atlas coordinates to slice coordinates.
5. Fit one TPS result from the same pairs.
6. Return both results.

## Export Behavior

The export path uses `build_quint_export(...)` plus `save_quint_json(...)`.

- Loads atlas shape and resolution from BrainGlobe.
- `SliceExport.markers` can be populated with VisuAlign `[ox, oy, nx, ny]` pairs from Elastix B-spline control points (colored segmentation workflow).
- The legacy workflows use the affine result only; the nonlinear TPS result is not exported.
