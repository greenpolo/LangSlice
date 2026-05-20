# CLI usage

This page describes the active CLI and Tauri GUI workflows.

## Position Estimation

```bash
langslice estimate <image> [--atlas ...] [--model ...] [--workflow ...]
langslice estimate-group <img1> <img2> ... [--interval 200] [--atlas ...]
langslice estimate-brain <image_folder> [--atlas ...] [--anchors ...]
```

Single-slice and group position estimation run through the ADK harness. The agent
surface is intentionally small: `fetch_atlas`, `submit_estimate`, and
`submit_group_estimate`.

Whole-brain estimation discovers a folder of slices, estimates anchor slices,
interpolates center positions, runs windowed image-gen estimation for the
remaining slices, and fits a constrained monotonic position curve.

## Image-Gen Registration

```bash
langslice register <image> --position <mm> [--registration-mode direct|agentic] [--image-model ...] [--review-model ...] [--max-candidates 3] [--out ...]
```

Registration has one active method: image-gen registration.

1. Load, normalize, and downsample the histology slice.
2. Generate atlas inputs at the requested atlas position.
3. Ask the image model to generate an atlas-colored target aligned to the histology.
4. Register the generated target to the atlas color map with itk-elastix.
5. Warp the atlas through the recovered transform.
6. Return the model-generated atlas target, Elastix-warped atlas, warped-border overlay, and VisuAlign markers.

Modes:

- `direct` generates one candidate and returns it.
- `agentic` lets an ADK review agent inspect up to three candidates before confirming one.

Provider routing:

- Google image models use the Google provider adapter.
- `gpt-image-2` uses the OpenAI Images API by default.
- Flux models use the OpenAI-compatible Images API path.

## Tauri GUI

The desktop app lives in `tauri-gui/` and runs the Python harness as a sidecar.
The Rust backend handles atlas loading, reslicing, and mesh serving. The React
frontend provides the 3D atlas view, settings, dashboard, and registration review
views.

## Debug And Request Capture

Set `LANGSLICE_VLM_DEBUG_DIR` to save run artifacts. For ADK estimation request auditing,
set `LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` to write redacted JSONL request captures.
