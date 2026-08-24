# CLI usage

This page describes the active CLI workflows.

The CLI is grouped by method:

```bash
langslice linear    {estimate, estimate-brain, quick-affine}
langslice nonlinear {register}
langslice           {version, login, serve, collect-traces}
```

`linear` and `nonlinear` are independent. `nonlinear register` takes a slice
position as an argument and does not care where it came from, so it can follow
`langslice linear estimate` or a placement made in another tool.

## Linear: Position Estimation

```bash
langslice linear estimate <image> [--atlas ...] [--model ...] [--plane ...]
langslice linear estimate-brain <image_folder> [--atlas ...] [--plane ...] [--interval 200] [--thickness 50] [--keep-order|--no-keep-order] [--model ...] [--out ...] [--resume|--fresh] [--stop-after NODE]
```

Single-slice estimation runs through the ADK harness. The agent surface is
intentionally small: `fetch_atlas` and `submit_estimate`.

Whole-brain estimation runs the node engine in
`src/langslice/linear/whole_brain/`. The graph is
`ingest → survey → fix → seed → position → transforms → review → emit`;
`fix` can route back to `survey` and `review` back to `position`, with
per-node cycle limits.

- `ingest` discovers the folder (natural sort), loads the atlas, builds the
  stack state, and writes a labelled contact sheet next to the checkpoint.
- `survey` is one agent pass over the whole stack: it is shown the contact
  sheet plus a manifest and triages damage, hemisphere flips, section order
  and interval breaks in a single pass. Its tools (`view_slices`,
  `fetch_atlas`, `flip_slices`, `reorder_slices`, `mark_damaged`,
  `submit_survey`) record corrections as data on the stack state. Under
  `--keep-order` (the default) `reorder_slices` refuses and the agent reports
  ordering problems in its notes instead.
- `fix` re-renders the contact sheet from the corrected stack and sends it
  back to `survey` for one verification pass. A clean survey skips `fix`.
- `emit` writes the results JSON (default `<image_folder>/brain_results.json`,
  or `--out`).
- `seed`, `position`, `transforms` and `review` are still stubs that pass
  through.

`--stop-after NODE` runs the graph up to and including that node, writes the
checkpoint and stops; re-running continues from there. It is how a single step
is measured or tuned in isolation.

Everything the engine produces -- corrected order, flips, positions, oblique
angles, per-slice transforms -- is a proposal recorded as data. The user's
image files are never modified.

State is checkpointed to `<image_folder>/brain_estimate.json` after every
node, in the same shape as the results file. `--resume` (default) skips nodes
the checkpoint lists as complete; `--fresh` re-runs the whole graph.

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
- `--provider chatgpt` uses a ChatGPT subscription instead of an API key: it
  sends `gpt-image-2` requests through the Codex Responses backend with the
  token stored by `langslice login`. Reference images are the colored region
  map, the atlas reference slice, and the histology slice; the output size is
  the `gpt-image-2` aspect ratio closest to the slice.

## Sign In With ChatGPT

```bash
langslice login
```

Runs the OAuth (PKCE) "Sign in with ChatGPT" flow in a browser, with a callback
on `localhost:1455`, and writes the token to `~/.langslice/openai_auth.json`
(mode 600). Access tokens are refreshed automatically; an existing Codex CLI
login (`~/.codex/auth.json` or its keyring entry) is used as a fallback.

Once signed in, no API key is needed for either model surface:

- chat/vision/tool-use agents accept `chatgpt/<model>` model strings, e.g.
  `--model chatgpt/gpt-5.6-luna`, served by the ADK backend in
  `src/langslice/providers/chatgpt.py`;
- image-gen registration accepts `--provider chatgpt`.

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
