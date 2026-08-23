# CLI usage

This page describes the active CLI workflows.

The CLI is grouped by method:

```bash
langslice linear    {estimate, estimate-group, estimate-brain, quick-affine}
langslice nonlinear {register}
langslice           {version, serve, collect-traces}
```

`linear` and `nonlinear` are independent. `nonlinear register` takes a slice
position as an argument and does not care where it came from, so it can follow
`langslice linear estimate` or a placement made in another tool.

## Linear: Position Estimation

```bash
langslice linear estimate <image> [--atlas ...] [--model ...] [--plane ...]
langslice linear estimate-group <img1> <img2> ... [--interval 200] [--atlas ...]
langslice linear estimate-brain <image_folder> [--atlas ...] [--anchors ...] [--model ...]
```

Single-slice and group position estimation run through the ADK harness -- the
only estimation path. The agent surface is intentionally small: `fetch_atlas`,
`submit_estimate`, and `submit_group_estimate`.

Whole-brain estimation discovers a folder of slices, estimates anchor slices
with the tool-use estimator, interpolates center positions, estimates the
remaining slices independently with the same tool-use estimator over the full
atlas range, and fits a constrained monotonic position curve. A single
`--model` flag configures the model used for both anchor and non-anchor
estimation.

## Linear: Quick Affine

```bash
langslice linear quick-affine <image> --position <mm> [--atlas ...] [--plane ...] [--out ...]
```

An affine-only preview that aligns the tissue silhouette to the atlas silhouette
at a known position. No image generation, no B-spline.

## Nonlinear: Image-Gen Registration

```bash
langslice nonlinear register <image> --position <mm> [--registration-mode direct|agentic] [--image-model ...] [--review-model ...] [--max-candidates 3] [--out ...]
```

Registration has one active method: image-gen registration. In QUINT/ABBA-style
workflows, linear placement happens in the host tool and this step stands in for
the manual spline/BigWarp deformation.

1. Load, normalize, and downsample the histology slice.
2. Generate atlas inputs at the requested atlas position.
3. Ask the image model to generate an atlas-colored target aligned to the histology.
4. Register the generated target to the atlas color map with itk-elastix.
5. Warp the atlas through the recovered transform.
6. Return the model-generated atlas target, Elastix-warped atlas, warped-border overlay, and VisuAlign markers.

Modes:

- `direct` generates one candidate and returns it.
- `agentic` lets an ADK review agent inspect up to three candidates before confirming one.

Provider routing is explicit, not inferred from the model name:

- `--provider google` (default) uses the Google/Gemini image adapter.
- `--provider openai` uses the OpenAI-compatible path. `--openai-image-route`
  picks the Images API (`images`, default) or the Responses API (`responses`),
  and `--endpoint` points it at a non-OpenAI base URL. The default
  OpenAI-compatible image model is `gpt-image-2`.

## Engine Service

Non-Python clients drive the same pipeline over a newline-delimited JSON
protocol:

```bash
langslice serve --stdio
```

The service accepts `version`, `estimate.run`, `register.run`,
`quick_affine.run`, and `export.run` request envelopes, streams progress/log
events, and returns typed JSON result or error envelopes. The contract is
defined by the Pydantic models in `src/langslice/api/models.py`.

## Debug And Request Capture

Set `LANGSLICE_VLM_DEBUG_DIR` to save run artifacts. For ADK estimation request auditing,
set `LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` to write redacted JSONL request captures.
