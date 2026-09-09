# CLI usage

This page describes the active CLI workflows.

The CLI is grouped by method:

```bash
langslice linear    {run, quick-affine}
langslice nonlinear {register}
langslice           {version, login, serve, abba}
```

`linear` and `nonlinear` are independent. `nonlinear register` takes a slice
position as an argument and does not care where it came from, so it can follow
`langslice linear run` or a placement made in another tool.

## Linear: Order, Position, Transform

```bash
langslice linear run FOLDER [--tasks reorder,position,transform]
    [--atlas ...] [--plane ...] [--model ...] [--preprocess auto|none]
    [--reasoning low|medium|high|xhigh|max] [--pixel-size-um UM]
    [--pitch DEG] [--yaw DEG]
    [--no-flip] [--hemisphere-cue TEXT]
    [--thickness UM] [--interval UM] [--strict-interval] [--deepslice] [--bayesian]
    [--angles] [--elastix]
    [--fact TEXT ...] [--positions JSON] [--order JSON]
    [--out PATH] [--fresh] [--trace-dir PATH]
```

One agent environment over a whole folder of sections -- one state, one
toolbox, one job statement, one session. A single section is a stack of one.
The design is `docs/linear_design.md`; this page is the CLI surface.

`--tasks` picks which of the three jobs are on (default: all three). A task
that is OFF contributes no tools and takes its answer from the host instead:
`--order` (a JSON list of filenames) and `--positions` (a JSON mapping
filename to millimetres) are read as a file path or as inline JSON.

The toolbox is built from the spec, so the agent only ever sees the tools its
run can use:

| tool | on when | does |
| --- | --- | --- |
| `status` | always | one row per section in corrected order: index, id, `position_mm`, `delta_to_next_mm` (signed), flip, rotation, damaged (+note), transform kind, `transform_iou`, `transform_mirrored`, caveats; plus the stack's cutting angles and interval breaks. Writes answer with only the rows they changed (`changed` + `n_sections`); this is the whole table |
| `validate` | always | runs the submit checks without submitting; writes nothing |
| `view_slices` | always | up to 4 sections at higher resolution, rendered as corrected, each captioned with its index and filename |
| `fetch_atlas` | always | up to 4 atlas sections, rendered at the stack's current cutting angles, each captioned with its position |
| `note`, `undo`, `redo` | always | run notes; snapshot undo where one tool call undoes as one step |
| `mark_damaged` / `unmark_damaged` | always | agent-internal classification: an outline an affine cannot bite on |
| `orient_slices` | `reorder` | flip and quarter-turn per section (`--no-flip` refuses the flip half); returns the changed sections as they now stand |
| `reorder_slices` / `move_slice` | `reorder` | full permutation (by filename) or one incremental move; corrected indices only, positions and transforms are kept |
| `set_positions` | `position` | batch write, clamped to the atlas range; returns each newly placed or moved section over the atlas at its new position |
| `compare_placement` | `position` | sections against the atlas at the candidate positions named for each (or its current one), up to 4 pairs, one image per pair, on one physical-scale canvas; `mode`, `zoom`, `template_opacity`, `outlines` as on `adjust_transform`; writes nothing |
| `view_stack` | `position` | one contact sheet of every section in the order of its written position over the atlas at that position, captioned with position and the distance to the next, plus a position-vs-index plot (two images); writes nothing |
| `run_deepslice` | `--deepslice` | reports `UNAVAILABLE` until the optional extra lands |
| `fit_position` | `--bayesian` | `oblique.fit_oblique` around a section's current position; writes nothing |
| `set_cutting_angles` | `--angles` | stack-wide pitch/yaw; later fetches and previews follow |
| `fit_affine` | `transform` | silhouette affine per section, written as its transform, with the overlap, the transform as the five physical parameters (`rotation_deg`, `scale_x`, `scale_y`, `translate_x_mm`, `translate_y_mm`, plus `shear`) about the canvas centre, and a physical-scale overlay (up to 16); `roi` ([x0, y0, x1, y1] of the canvas) fits only the tissue and atlas outline inside that box, which is how a damaged section is fitted — damage is refused without one; `--elastix`'s method is not wired yet |
| `adjust_transform` | `transform` | writes one positioned section's in-plane transform (rotation / per-axis scales / millimetre shifts, plus a note) and returns the section drawn under it with the atlas outlines at true physical scale; every call writes and the last one stays, the same parameters again only re-draw; `mode` (overlay, side_by_side, checkerboard, outlines, section, template, ab = new beside what it carried before), `zoom`, `template_opacity`, `pivot` (canvas, tissue, or [fx, fy] of the canvas) and `outlines` (all, outer, none) are the view and the centre it turns about; it does not change the section's flip or rotation |
| `landmarks` | `transform` | point pairs as fractions of the canvas: the residual in millimetres per pair, their RMS, the transform fitted to them (similarity from 2 pairs, affine from 3), and the pairs drawn on the view; writes nothing |
| `submit` | always | ends the run; gated |

Sections and fetched atlas sections are framed the same way (foreground plus a
6% margin) so apparent scale is not a cue. Every image a tool returns has its
label burned into the pixels (section id, atlas position), because tool images
arrive as bare attachments with no text beside them. The seed message is every section as
its own labelled image in corrected order plus the status table -- not a
thumbnail grid, which splits one vision-encoder patch budget across the whole
stack at once.

The job statement carries the job, the run's facts (`--fact`,
`--hemisphere-cue`), one factual line per tool that exists, the hard
constraints and, when positioning is on, a short `Method` section: place each
section on its own evidence and compare candidate positions before writing,
work in batches, review the whole stack afterwards, re-check both sides of a gap before
reporting a break, validate, submit. No rules of thumb, no failure-mode
warnings, no region names, and no tool payload carries an opinion.

`submit` is refused, with the numbers that refused it, when:

- `position` is on and any section has no position (`MISSING_POSITIONS`);
- the positions do not run one way along the corrected order
  (`ORDER_POSITION_MISMATCH`, naming the offending neighbour pairs);
- `--strict-interval` is on and a consecutive spacing is more than 10% off the
  nominal interval, or any interval break is reported (`STRICT_INTERVAL`);
- a reported interval break is not in the positions that were written -- the
  written interval there must exceed 1.5x the stack's median written spacing
  (`INTERVAL_BREAKS_UNSUPPORTED`).

Reordering changes only the corrected index -- positions and transforms are
kept, and the rows show which sections were renumbered. The
`ORDER_POSITION_MISMATCH` gate above is what keeps order and position honest.

Alignment renders are in PHYSICAL space. The section's micrometres per pixel
come from the image file (TIFF `XResolution` + `ResolutionUnit`, or the OME-XML
`PhysicalSizeX`), or from `--pixel-size-um`, which overrides it; the atlas is
placed at `atlas um/px / canvas um/px` with its anatomy centred, never fitted
to the canvas. With no pixel size anywhere the run estimates one from the
tissue's width against the atlas anatomy's and records
`calibration.source = "estimated"` on the transform. The interactive preview,
`landmarks` and `fit_affine` all draw the same picture: the transformed section under
the atlas's family-level region outlines, each in its own color, with a 1 mm
scale bar. The alignment parameters (`rotation_deg`, `scale_x`, `scale_y`,
`translate_x_mm`, `translate_y_mm`) are stored alongside the host-facing six
normalized numbers -- on every transform, silhouette fits included, so a fit
and a hand alignment are the same five numbers.

`--gates` refuses `set_positions` for a section not compared since its last
write, and `submit` until `view_stack` has run after the last write;
`--playbook` replaces the Method section with GPT-6 Astra's own method
(hypothesise order and every position from the opening images, confirm each
section at that position four per call, write, re-check, review). Both are for
the cheaper models and off by default.

`--reasoning` sets the reasoning effort on models that expose one (the
`openai-oauth/*` backend); unset leaves the provider's own default.

`--preprocess auto` (the default) runs adaptive CLAHE plus a DAPI-weighted
grayscale blend on every section the run renders -- the seed images, the
`view_slices` images, the preview panels and the silhouette fits -- so dim
fluorescence reads like the atlas instead of like a black field. Display only:
the enhanced pixels are never written back to the user's files.

Every write checkpoints the whole state to `<image_folder>/linear_state.json`,
and the results file (default `<image_folder>/linear_results.json`, or `--out`)
has the same shape. A run that dies resumes from that checkpoint with the state
it had -- the agent is re-seeded, not replayed. `--fresh` ignores the
checkpoint and starts over. `--trace-dir PATH` writes a full-content JSONL
trace of every agent session.

Every model call prints a `[tokens]` line (input, cached, output, run input
so far) and the run ends with a total. `--max-quota-percent N` (default 25) ends the session when this run's share
of the provider's usage window reaches N (the OAuth lane's quota headers;
cached input is ~0.13x there, so the window, not the raw count, is the
cost); `--max-input-tokens N` (default
`JobSpec.max_input_tokens`, 2M) ends the session when the run's summed
input passes N: the OAuth lane resends the whole history every call, so a
long run grows quadratically and would otherwise be ended by the account's
usage window instead of by the job. Writes made before the stop are kept.

## Linear: Quick Affine

```bash
langslice linear quick-affine <image> --position <mm> [--atlas ...] [--plane ...] [--out ...]
```

An affine-only preview that aligns the tissue silhouette to the atlas silhouette
at a known position, using the shared affine core (`src/langslice/affine.py`)
that the linear `fit_affine` tool also runs on. No image generation, no
B-spline.

## Nonlinear: Image-Gen Registration

```bash
langslice nonlinear register <image> --position <mm> [--registration-mode direct|agentic] [--image-model ...] [--review-model ...] [--max-candidates 3] [--palette family|leaf-borders] [--out ...]
```

Registration has one active method: image-gen registration. In QUINT/ABBA-style
workflows, linear placement happens in the host tool and this step stands in for
the manual spline/BigWarp deformation.

1. Load, normalize, and downsample the histology slice.
2. Generate atlas inputs at the requested atlas position. The colored region
   map the model sees is drawn from smoothed region contours at canvas
   resolution (flat, exact palette colors — no voxel staircase); the render
   Elastix registers against stays pixel-exact.
3. Ask the image model to generate an atlas-colored target aligned to the histology.
4. Register the generated target to the atlas color map with itk-elastix.
5. Warp the atlas through the recovered transform.
6. Return the model-generated atlas target, Elastix-warped atlas, warped-border overlay, and VisuAlign markers.

Modes:

- `direct` generates one candidate and returns it.
- `agentic` runs a hosted-router conversation (openai-oauth only): the prompt
  and images go to the hosted GPT model with the image_generation tool, Elastix
  reports come back as follow-up messages, and the loop accepts the first
  clean-report candidate (cap `--max-candidates`, default 4).

Palette:

- `--palette family` (default) draws flat regions, one color per registration
  unit.
- `--palette leaf-borders` draws the same colors plus Allen-Reference-Atlas
  plate delineation: a hairline at every leaf boundary in a darker shade of
  that region's own color (2px at a 2048 canvas, family boundaries heavier),
  so the model is shown the full parcellation without any color moving. Only
  the model-facing render changes — the Elastix-side render is identical —
  and the classifier accepts the hairline color as its own region, so a model
  that paints the lines back does not cut background through its regions. It
  is a process-wide setting (`LANGSLICE_ATLAS_PALETTE`).

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

The service accepts `version`, `register.run`, `quick_affine.run`, and
`export.run` request envelopes, streams progress/log
events, and returns typed JSON result or error envelopes. The contract is
defined by the Pydantic models in `src/langslice/api/models.py`.

## Debug And Request Capture

Set `LANGSLICE_VLM_DEBUG_DIR` to save run artifacts. For ADK estimation request auditing,
set `LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` to write redacted JSONL request captures.

### Linear agent traces

`langslice linear run --trace-dir PATH` (or `LANGSLICE_TRACE_DIR`; the flag
wins) writes one JSONL file per agent session —
`<trace_dir>/<run_label>_<8 hex>.jsonl`, e.g. `linear_stack_1a2b3c4d.jsonl` for
the main session, which is the only session a run has — with one record per
event:

| `kind` | contents |
| --- | --- |
| `session` | run label, agent name, model name, full system instruction (first line) |
| `seed` | the session's seed message: text verbatim, images as descriptors labelled with the text above them |
| `model` | one model turn: its text, any thought summary, every function call with full JSON arguments, and its `usage` (input, cached, output tokens) |
| `tool_result` | the complete tool payload the model reads, plus descriptors for media riding on the response |
| `nudge` | a nudge the driver actually sent |
| `summary` | tool-call count, turn count, whether the session submitted, run token totals, and `stopped` when a budget ended it (last line) |

Unlike the request captures above, this records values, not shapes: full text,
full tool arguments, full tool responses. Images are always descriptors (mime
type, byte count, pixel size) — never bytes — so a trace stays small. Nothing
but content, the prompt and a model name is read, so no credentials are
written. Unset, the recorder is never constructed.
