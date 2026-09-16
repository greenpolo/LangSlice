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
    [--reasoning none|minimal|low|medium|high] [--pixel-size-um UM]
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
| `view_slices` | always | up to 8 sections at higher resolution, rendered as corrected, each captioned with its index and filename |
| `fetch_atlas` | always | up to 8 atlas sections, rendered at the stack's current cutting angles, each captioned with its position |
| `note`, `undo`, `redo` | always | run notes; snapshot undo where one tool call undoes as one step |
| `mark_damaged` / `unmark_damaged` | always | agent-internal classification: an outline an affine cannot bite on |
| `orient_slices` | `reorder` | flip and quarter-turn per section (`--no-flip` refuses the flip half) |
| `reorder_slices` / `move_slice` | `reorder` | full permutation or one incremental move; corrected indices only, positions and transforms are kept |
| `set_positions` | `position` | batch write, clamped to the atlas range |
| `distribute_spacing` | `position` | interpolates from the points you fix, `keep` holds sections in place, `apply=false` computes without writing |
| `run_deepslice` | `--deepslice` | reports `UNAVAILABLE` until the optional extra lands |
| `fit_position` | `--bayesian` | `oblique.fit_oblique` around a section's current position; writes nothing |
| `set_cutting_angles` | `--angles` | stack-wide pitch/yaw; later fetches and previews follow |
| `fit_affine` | `transform` | silhouette affine per section, with the overlap, the transform as the five physical parameters (`rotation_deg`, `scale_x`, `scale_y`, `translate_x_mm`, `translate_y_mm`, plus `shear`) about the canvas centre, and a physical-scale overlay (up to 16); `roi` ([x0, y0, x1, y1] of the canvas) fits only the tissue and atlas outline inside that box, which is how a damaged section is fitted — damage is refused without one; `--elastix`'s method is not wired yet |
| `preview_transform` | `transform` | one positioned section under a candidate rotation / per-axis scales / millimetre shifts, with the atlas outlines at true physical scale; `mode` (overlay, side_by_side, checkerboard, outlines, section, template, ab), `zoom`, `template_opacity`, `pivot` (canvas, tissue, or [fx, fy] of the canvas) and `outlines` (all, outer, none) are the view and the centre it turns about; writes nothing |
| `landmarks` | `transform` | point pairs as fractions of the canvas: the residual in millimetres per pair, their RMS, the transform fitted to them (similarity from 2 pairs, affine from 3), and the pairs drawn on the view; writes nothing |
| `set_transform` | `transform` | records the in-plane transform of one section (parameters in millimetres, plus a note); it does not change the section's flip or rotation |
| `copy_transform` | `transform` | copies one section's transform onto others |
| `submit` | always | ends the run; gated |

Sections and fetched atlas sections are framed the same way (foreground plus a
6% margin) so apparent scale is not a cue. Every image a tool returns has its
label burned into the pixels (section id, atlas position), because tool images
arrive as bare attachments with no text beside them. The seed message is every section as
its own labelled image in corrected order plus the status table -- not a
thumbnail grid, which splits one vision-encoder patch budget across the whole
stack at once.

The job statement carries the job, the run's facts (`--fact`,
`--hemisphere-cue`), one factual line per tool that exists, and the hard
constraints. Nothing else: no strategy, no rules of thumb, no failure-mode
warnings, and no tool payload carries an opinion. Every major benchmark failure
worth tracing came back to advice the harness injected.

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

## Linear: Quick Affine

```bash
langslice linear quick-affine <image> --position <mm> [--atlas ...] [--plane ...] [--out ...]
```

An affine-only preview that aligns the tissue silhouette to the atlas silhouette
at a known position, using the shared affine core (`src/langslice/affine.py`)
that the linear `fit_affine` tool also runs on. No image generation, no
B-spline.

## Nonlinear: Border Refinement

```bash
# Standalone: initial color registration, then border correction (two model calls)
langslice nonlinear register slice.png --position 3.9

# Supplied placement: border correction directly (one model call)
langslice nonlinear register slice.png --position 3.9 --initial-alignment placement.json
```

The preferred input is the rough alignment already established by the linear
agent or a host tool. The image model receives the placed yellow atlas borders
over histology, followed by the same photograph without lines, and adjusts the
boundaries to match the visible tissue. It is not asked for another color map.

When no placement is supplied, the standalone route first uses the existing
three-image color-map request (color atlas, grayscale atlas, histology), registers
that reply, then sends the registered borders and clean histology for correction.
These are two prompts and two image-generation calls. Supplied placement bypasses
the color-map stage; it is never replaced by automatic silhouette alignment.

After correction, yellow lines are extracted and displayed on the original
photograph. A residual Elastix fit transfers that correction to the atlas labels.
Exports compose the complete initial placement and residual deformation.
Keep the raw reply, corrected lines on original tissue, and fitted atlas overlay
distinct when reviewing results.

`placement.json` contains a 3×3 affine mapping oriented native atlas pixel
centers to pixels in the input image. Position, cutting angles and atlas axes
must agree with that placement. `--mirror-atlas-lr` applies an explicit atlas
reflection; no reflection is inferred from the tissue. The top-level
`registration_handoff` bridge prepares this contract from linear section state.
ABBA uses its existing host alignment directly.

`--preprocess auto` remains the default shared tissue-visibility enhancement;
`none` disables it. In either case, both correction attachments use the identical
prepared photograph. `--canvas-pad`, `--pitch-deg`, `--yaw-deg` and
`--deformation bspline|affine` remain available. The border route requires
`--draws 1` (one draw per stage), not color-map voting.

`--provider none` is an explicit model-free diagnostic: retain supplied placement
or fit a silhouette placement, then return its borders and composed coordinates.
It does not run image generation or residual Elastix fitting.

Provider selection remains explicit: `google`/`gemini-api` for Gemini,
`openai`/`openai-api` for API access, or `chatgpt`/`openai-oauth` for
subscription access. API requests retain `--openai-image-route` and
`--endpoint`. Both stages use the selected provider.

See [the nonlinear design](nonlinear_design.md) for coordinate contracts,
artifact meanings, supported handoffs and review limitations.

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
| `model` | one model turn: its text, any thought summary, every function call with full JSON arguments |
| `tool_result` | the complete tool payload the model reads, plus descriptors for media riding on the response |
| `nudge` | a nudge the driver actually sent |
| `summary` | tool-call count, turn count, whether the session submitted (last line) |

Unlike the request captures above, this records values, not shapes: full text,
full tool arguments, full tool responses. Images are always descriptors (mime
type, byte count, pixel size) — never bytes — so a trace stays small. Nothing
but content, the prompt and a model name is read, so no credentials are
written. Unset, the recorder is never constructed.
