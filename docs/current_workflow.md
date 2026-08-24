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
langslice linear estimate-brain <image_folder> [--atlas ...] [--plane ...] [--interval 200] [--thickness 50] [--keep-order|--no-keep-order] [--model ...] [--preprocess auto|none] [--out ...] [--resume|--fresh|--rerun-from NODE] [--stop-after NODE]
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
  The sheet is for the user: no model is shown it. Every agent step gets the
  stack as a labelled sequence of per-section images instead (a thumbnail grid
  splits one vision-encoder patch budget across every section at once, and
  individual images -- even small ones -- read better).
- `survey` is one agent pass over the whole stack: it is shown every section
  as its own image in corrected order, each preceded by an
  `<index>: <filename>` label, plus the same stack as a text manifest, and
  triages damage, hemisphere flips, section order and interval breaks in a
  single pass. Its tools (`view_slices`,
  `fetch_atlas`, `flip_slices`, `reorder_slices`, `mark_damaged`,
  `submit_survey`) record corrections as data on the stack state. Under
  `--keep-order` (the default) `reorder_slices` refuses and the agent reports
  ordering problems in its notes instead.
- `fix` rebuilds the contact sheet from the corrected stack and sends the
  stack back to `survey` for one verification pass, re-rendered as corrected.
  A clean survey skips `fix`.
- `seed` runs whatever automatic seeder is available — today, none. DeepSlice
  would place a whole coronal mouse stack in one shot, but it is an optional
  extra that is not installed, so the node writes a note and passes the stack
  through unplaced. Nothing prescribes key sections here on purpose: picking
  good ones needs intimate atlas knowledge, and a badly chosen key section
  drags every position interpolated from it.
- `position` is the second agent pass and it owns the placement strategy. The
  whole stack is in context, usually with no positions on it. Its prompt is a
  MENU, not a prescription: key sections plus interpolation, estimating every
  section, or a mix — the agent picks. The prompt is deliberately
  atlas-agnostic (no region names, no landmarks, no absolute positions) so the
  same text works for every BrainGlobe atlas, species and plane, and it carries
  the failure modes that bite whichever strategy is chosen: re-verify
  disagreeing estimates instead of averaging them or sliding a self-consistent
  ladder to match a minority reading; anchor only where the atlas level is
  identifiable at a glance; check both ends of the stack before submitting,
  because a plausible ladder hung at the wrong absolute position looks
  consistent from the inside; and a constant slicing interval does not mean no
  sections were lost. Tools: `view_slices`, `fetch_atlas`, `estimate_slices`
  (up to 8 named sections per call, each a full single-slice sweep, run one
  after another and reported but not written), `interpolate_between` (fixed
  points in, one suggestion per section out, not written), `set_positions`
  (batch write, clamped to the atlas range, returns the resulting
  neighbour-interval table), `get_advisories` (interval table plus a
  monotone-fit suggestion, both explicitly advisory) and `submit_positions`,
  which is rejected unless every section has a position. Oblique-angle
  estimation is not part of this step yet.
- `transforms` proposes one in-plane alignment per section, on two routes.
  Intact sections take the plain-code route: the shared silhouette affine
  (`src/langslice/affine.py`) against the atlas section their position names,
  recorded as six normalized numbers on `slice.affine`. Damaged sections take
  an interactive agent loop, one session per section: `preview_transform`
  renders the section under a candidate rotation/scale/translation over its
  atlas section (side by side plus a magenta/green overlay), the agent looks,
  adjusts and repeats, then `submit_transform` records the five parameters on
  `slice.interactive_transform`. A failed fit is a caveat, never a failed run.
- `review` is the last agent pass: the whole finished stack, its manifest, and
  the advisory spacing signals. It can attach caveats (`flag_slice`) and, once,
  send the stack back to `position` with notes the next pass reads. A review
  that runs out of turns approves with a note -- it can never block hand-back.
- `emit` writes the results JSON (default `<image_folder>/brain_results.json`,
  or `--out`).

`--stop-after NODE` runs the graph up to and including that node, writes the
checkpoint and stops; re-running continues from there. It is how a single step
is measured or tuned in isolation.

Everything the engine produces -- corrected order, flips, positions, oblique
angles, per-slice transforms -- is a proposal recorded as data. The user's
image files are never modified.

`--preprocess auto` (the default) runs adaptive CLAHE plus a DAPI-weighted
grayscale blend on every section the engine renders -- the per-section stack
images, the `view_slices` images, the contact sheet, the transform previews
and the silhouette fit -- so dim
fluorescence reads like the atlas instead of like a black field.
`--preprocess none` shows the raw sections. Either way this is display only:
the enhanced pixels are never written back to the user's files, and the
single-slice estimation worker applies the same setting once, internally.

State is checkpointed to `<image_folder>/brain_estimate.json` after every
node, in the same shape as the results file. `--resume` (default) skips nodes
the checkpoint lists as complete; `--fresh` re-runs the whole graph.

`--rerun-from {position,transforms,review}` rewinds an existing checkpoint to
just before that node and resumes from there, so a single step can be
re-benchmarked without re-running (and re-paying for) the agent steps ahead of
it. It clears that node's own output plus everything derived from it --
`position` also clears `transforms`, since the transform step reads
positions -- but leaves `survey`'s findings (order, flips, damage) untouched:
those are not cheaply reproducible, so rewinding `survey`/`fix`/`seed` is not
supported. Requires a checkpoint to already exist; mutually exclusive with
`--fresh`; composes with `--stop-after` to run exactly one rewound node and
stop.

## Linear: Quick Affine

```bash
langslice linear quick-affine <image> --position <mm> [--atlas ...] [--plane ...] [--out ...]
```

An affine-only preview that aligns the tissue silhouette to the atlas silhouette
at a known position, using the shared affine core (`src/langslice/affine.py`)
that the whole-brain transform step also runs on. No image generation, no
B-spline.

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
