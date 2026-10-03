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
langslice linear run FOLDER [--tasks reorder,position,transform[,nonlinear]]
    [--atlas ...] [--plane ...] [--model ...] [--preprocess auto|none]
    [--image-provider NAME] [--image-model NAME]
    [--reasoning low|medium|high|xhigh|max] [--pixel-size-um UM]
    [--pitch DEG] [--yaw DEG]
    [--no-flip] [--hemisphere-cue TEXT]
    [--thickness UM] [--interval UM] [--strict-interval] [--deepslice] [--bayesian]
    [--angles]
    [--fact TEXT ...] [--positions JSON] [--order JSON] [--transforms JSON]
    [--out PATH] [--fresh] [--trace-dir PATH]
    [--max-quota-percent N] [--max-input-tokens N] [--gates] [--playbook]
    [--image-retention legacy] [--no-debrief]
```

One agent environment over a whole folder of sections -- one state, one
toolbox, one job statement, one session. A single section is a stack of one.
The design is `docs/linear_design.md`; this page is the CLI surface.

`--tasks` picks which jobs are on (default: `reorder,position,transform`;
`nonlinear` is opt-in and needs a linear placement for every section; its job
is a deformation per section, fitted with `fit_deformable`, and
`--image-provider none` runs it without the image model, so without
`trace_borders`). A task
that is OFF contributes no tools and takes its answer from the host instead:
`--order` (a JSON list of filenames), `--positions` (a JSON mapping filename to
millimetres) and `--transforms` (filename to a transform record in the
checkpoint format) are read as a file path or as inline JSON. Hosts present
`reorder` + `position` as one Positioning task and `transform` as Linear; see
[the interface design](interface_design.md). The ABBA dialog's extra controls
(image resolution low/medium/high/auto, per-task notes, the per-call section
cap, user damage marks, locked sections, the agent's damage tool) are `JobSpec`
fields, not CLI flags yet; see `docs/linear_design.md`. Image resolution sets
the long edge of each opening-strip tile and of every later picture the agent sees
(low 256/512 px, medium 384/768, high 512/1024; auto 256 and then the agent's
own `view.resolution` per call, up to 1536).

The toolbox is built from the spec, so the agent only ever sees the tools its
run can use:

| tool | on when | does |
| --- | --- | --- |
| `status` | always | one row per section in corrected order: index, id, `position_mm`, `delta_to_next_mm` (signed), flip, rotation, damaged (+note), transform kind, `transform_iou`, `transform_mirrored`, caveats; plus the stack's cutting angles and interval breaks. Ordinary writes answer with only the rows they changed; transform writes use their physical result instead. This is the whole table |
| `view_slices` | always | up to 4 sections at higher resolution, rendered as corrected, each captioned with its index and filename; `view` mode `channels` shows each section's raw channels side by side, unmodified and labelled |
| `view_atlas` | always | up to 4 atlas sections, rendered at the stack's current cutting angles, each captioned with its position |
| `note`, `undo`, `redo` | always | run notes; snapshot undo where one tool call undoes as one step |
| `mark_damaged` | `agent_damage` (on in the CLI) | set or clear damage per entry with `damaged` (default True); clearing also removes the note; a damage flag the host set cannot be cleared |
| `preprocess` | `--agent-preprocessing` (`agent_preprocessing`) | the section appearance (channel weights, CLAHE, ANTs N4/denoise) for target `view`, `fit` or both, stack-wide or per section; undoable; returns each pictured section (up to 4) BEFORE and AFTER the call, labelled |
| `orient_slices` | `transform` | flip and quarter-turn per section (`--no-flip` refuses the flip half); returns the changed sections as they now stand |
| `reorder_slices(slices, after="start")` | reorder | move the listed filenames as a block, in the listed order, after a named section or at the start. One filename moves one slice; the full list sets the whole order. Unlisted sections keep their relative order. Corrected indices only; positions and transforms are kept. One undo step. |
| `set_positions` | `position` | batch write, clamped to the atlas range; returns a placement picture (any `view_placement` mode, default `stacked`) unless the model has already seen that exact section, position, orientation and cutting-angle combination in a full-canvas atlas-bearing view. Section-only and zoomed comparisons do not suppress the full placement. A compare and write requested together still return the write image because neither sibling result was visible when they were planned |
| `view_placement` | `position`, `transform` or `nonlinear` | the complete current registration (the section under its stored transform and, in the modes that draw the section under its placement, its applied deformation unless `view.deformation` is `none`) or up to 4 candidate pairs; `stacked` and `side_by_side` are tissue-framed and full-view only (`side_by_side` up to 8 separate images); other modes return one physical-canvas image per pair; writes nothing |
| `view_stack` | `position` | one contact sheet of every section in the order of its written position over the atlas at that position, captioned with position and the distance to the next, plus a position-vs-index plot (two images); writes nothing |
| `run_deepslice` | `--deepslice` | reports `UNAVAILABLE` until the optional extra lands |
| `search_position` | `--bayesian` | `oblique.fit_oblique` around a section's current position; writes nothing |
| `set_cutting_angles` | `--angles` | stack-wide pitch/yaw; later fetches and previews follow |
| `fit_affine` | `transform` | in-plane affine per section, written as its transform: by default the Elastix intensity affine refining the section's current transform (stain and edges against `fit_atlas`: the ARA template by default, ABBA's Nissl where installed; never a search from scratch), or `method="silhouette"` (outline moments fit from scratch); with the overlap, the transform as the five physical parameters (`rotation_deg`, `scale_x`, `scale_y`, `translate_x_mm`, `translate_y_mm`, plus `shear`) about the canvas centre, and a physical-scale overlay for every successful fit; `include`/`exclude` regions (as in `fit_deformable`) fit only the kept atlas regions against the tissue the current placement lays there, with a `regions` report; damaged sections are refused unless regions are given |
| `adjust_transforms(entries, view)` | transform.interactive | set one to four independent sections, each with rotation, per-axis scales and millimetre shifts. Per-entry pivot and note; one `view` draws every entry; `ab` and `side_by_side` return two images, other modes one. Each result maps its images with `image_indexes`. One undo step; repeat unchanged parameters to redraw. Replaces the complete transform, including spline or shear. Inspect before a dependent correction in a later call. |
| `trace_borders(id, prompt)` | `nonlinear` unless `--image-provider none` | sends the section's placed atlas borders and the clean section to the image model with the agent's edited copy of the base correction prompt; runs in the background and returns at once; keeps the first reply with the extracted borders on the original, and `submit` waits for running calls. No deformation is fitted and no transform changes. See [the image-tool contract](nonlinear_image_tool.md) |
| `grep_atlas(query, id)` | `nonlinear` | looks regions up in the atlas hierarchy (acronym, name substring or id; 40 rows max, with a count of the rest): acronym, id, name, ancestry as acronyms, descendant count, and with a positioned section `id` an `in_section` flag for the region or any descendant in the atlas plane at that placement. Text only, writes nothing |
| `fit_deformable` | `nonlinear` | library deformable fit (ANTs SyN or Elastix B-spline) of the placed atlas onto sections, on top of their linear placement: include/exclude regions (an entry may name one side of the section, `"CTX:left"`), start linear or current (composed steps), `fit_section` (the fit appearance, or with an image model the section's `trace_borders` result, waited for up to 300 s while it runs and drawn on the section in the reply), `fit_atlas` (ara or nissl on ABBA hosts for the fit appearance, borders for traced fit sections), stiffness (soft, medium, firm), and `engine` when `nonlinear.engine` is `either`; detail and line softening are fixed. 2–4 candidates preview and write nothing; one setting applies it (undoable, cached results reused); `keep_linear="reason"` records instead that a section's linear placement stands. `submit` requires a deformation or a `keep_linear` reason for every section (`MISSING_DEFORMATIONS`). Returns the final borders on the section per result plus displacement, folds and flags (including `DISPLACEMENT_OUTSIZED`). Any later linear change to a section clears its deformation (`deformation_cleared`); `view_placement` draws the applied warp. Records go to `<results dir>/deformable/` |
| `submit` | always | ends the run; gated |

Sections and fetched atlas sections are framed the same way (foreground plus a
6% margin) so apparent scale is not a cue. Every image a tool returns has its
label burned into the pixels (section id, atlas position), because tool images
arrive as bare attachments with no text beside them. The seed message shows the
stack as ABBA-style strips (`linear/opening.py`, since 2026-10-03): rows of
labelled sections in corrected order with the atlas at each section's current
position beneath it, each strip as long as the model's largest image (2048 px
on the OpenAI lanes) and inside its patch budget, so nothing is shrunk by the
encoder; without positions the strips are section-only and atlas reference
strips (evenly spaced positions, labelled in mm) follow. Then the status
table. Until 2026-10-03 every section and atlas section was its own image
(about 80 for a 40-section stack).

The job statement carries the job, the run's facts (`--fact`,
`--hemisphere-cue`), one factual line per tool that exists, the hard
constraints and, when positioning is on, a short `Method` section: place each
section on its own evidence and compare candidate positions before writing,
review the whole stack afterwards, re-check both sides of a gap before
reporting a break, submit. No rules of thumb, no failure-mode
warnings, no region names, and no tool payload carries an opinion. The default
Method does not prescribe batching; the model chooses how to group its work.

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
`calibration.source = "estimated"` on the transform. The interactive preview
and `fit_affine` draw the same picture: the transformed section under
the atlas's family-level region outlines in yellow (1 px by default), with a 1 mm
scale bar. The alignment parameters (`rotation_deg`, `scale_x`, `scale_y`,
`translate_x_mm`, `translate_y_mm`) are stored alongside the host-facing six
normalized numbers -- on every transform, silhouette fits included, so a fit
and a hand alignment are the same five numbers. Tool feedback returns those
physical controls and the image; the normalized matrix, derived forms and full
adjustment history remain in the checkpoint/local toolbox rather than being
repeated into each later model turn.

`--gates` refuses `set_positions` for a section not compared since its last
write, and `submit` until `view_stack` has run after the last write;
`--playbook` replaces the Method section with GPT-6 Astra's own method
(hypothesise order and every position from the opening images, confirm each
section at that position four per call, write, re-check, review). Both are for
the cheaper models and off by default.

`--reasoning` sets the reasoning effort on models that expose one (the
`openai-oauth/*` backend); unset leaves the provider's own default.

`--preprocess auto` (the default) runs adaptive preprocessing on every section
the run renders -- the seed images, the `view_slices` images, the preview panels
and the silhouette fits -- so dim fluorescence reads like the atlas instead of
like a black field. Fluorescence gets per-channel CLAHE and a grayscale blend
weighted toward the channels that cover the most tissue (usually DAPI);
brightfield stains, detected by a bright slide border, get CLAHE on optical
density. Display only: the enhanced pixels are never written back to the user's
files.

Every write checkpoints the whole state to `<image_folder>/linear_state.json`,
and the results file (default `<image_folder>/linear_results.json`, or `--out`)
has the same shape. A run that dies resumes from that checkpoint with the state
it had -- the agent is re-seeded, not replayed. `--fresh` ignores the
checkpoint and starts over. `--trace-dir PATH` writes a full-content JSONL
trace of every agent session.

With tracing enabled, the subscription provider also records content-free usage
diagnostics: optional cache-write counts, backend item/content counters, hashed
request slots, and reconciliation residuals. It changes no request content,
cache keys, image delivery, or model tools. Image bytes, raw request payloads
and encrypted reasoning are excluded from these diagnostics. Backend-generated
item IDs are not request indices: only explicit ID matches are attributed to
request/output items; ambiguous matches remain unmapped. Missing counters are
unknown, not zero. Parent item and child content counters are alternative
breakdowns, not additive charges. Usage-only responses are recorded as well.

The public [Responses usage schema](https://developers.openai.com/api/reference/python/resources/responses/methods/retrieve)
documents aggregate cached and cache-write counts. The subscription backend's
additional per-item attribution is optional observed data, not a guaranteed
public API contract. These measurements do not equate subscription quota with
API dollars or establish a causal cost for pruning.

Every model call prints a `[tokens]` line separating request input, cached
input, output, cumulative input and peak request input. Cumulative input is
repeated processing for cost accounting, not context size; cached input still
occupies context. The run ends with totals and its peak request input.
`--max-quota-percent N` (default 25) ends the session when this run's share
of the provider's usage window reaches N (the OAuth lane's quota headers;
cached input is ~0.13x there, so the window, not the raw count, is the
cost); `--max-input-tokens N` (default disabled;
`JobSpec.max_input_tokens=None`) stops after a single request reports more
than N input tokens, including cached input. It is checked after the response,
not a preflight guarantee. Both stops allow one grace call to submit; writes
made before the stop are kept. No cumulative input-token stop is applied.

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
# No placement: route "atlas" draws boundaries against the outlined atlas (one model call)
langslice nonlinear register slice.png --position 3.9

# Supplied placement: route "supplied" corrects the placed borders directly (one model call)
langslice nonlinear register slice.png --position 3.9 --initial-alignment placement.json
```

There are exactly two border-based routes, chosen automatically by whether a
placement is supplied. There is no colormap route left: an image model is
never shown a colored atlas region map on either route.

With a supplied placement (route "supplied"), the image model receives the
placed yellow atlas borders over histology, followed by the same photograph
without lines, and adjusts the boundaries to match the visible tissue.

Without a placement (route "atlas"), a local silhouette-moments fit stands in
for the rough placement, and the model instead draws boundaries from nothing
onto the clean tissue, using an outlined grayscale atlas template (the
reference plate with its own thin yellow boundaries) as its only atlas
reference. Add `--passes 2` for an optional second call that audits and
corrects the first pass's lines against the same template; pass 1 alone was
measured sufficient on undamaged coronal sections, so `--passes 1` (the
default) is fine unless the first pass looks incomplete.

After either route's model call(s), yellow lines are extracted and displayed
on the original photograph. By default (`--deformation none`) no fit runs: the
residual is identity and the exported placement is the rough one, while the
deformation algorithm is still being designed. `--deformation bspline|affine`
runs a residual Elastix fit that transfers the correction to the atlas labels.
Exports compose the complete initial placement and residual deformation. Keep the raw reply, corrected lines on original tissue,
and fitted atlas overlay distinct when reviewing results.

`placement.json` contains a 3×3 affine mapping oriented native atlas pixel
centers to pixels in the input image. Position, cutting angles and atlas axes
must agree with that placement. `--mirror-atlas-lr` applies an explicit atlas
reflection on either route; no reflection is inferred from the tissue. The
top-level `registration_handoff` bridge prepares this contract from linear
section state. ABBA uses its existing host alignment directly.

`--preprocess auto` remains the default shared tissue-visibility enhancement;
`none` disables it. In either case, both correction attachments use the
identical prepared photograph. `--canvas-pad`, `--pitch-deg` and `--yaw-deg`
remain available. One draw per model call is
supported; there is no multi-draw voting.

`--provider none` is an explicit model-free diagnostic: retain supplied placement
or fit a silhouette placement, then return its borders and composed coordinates.
It does not run image generation or residual Elastix fitting.

Provider selection remains explicit: `gemini-api` for Gemini, `openai-api`
for API access, or `openai-oauth` for subscription access (the legacy
spellings `google`, `openai` and `chatgpt` still resolve). API requests retain `--openai-image-route` and
`--endpoint`. Both stages use the selected provider.

See [the nonlinear design](nonlinear_design.md) for coordinate contracts,
artifact meanings, supported handoffs and review limitations.

## Sign In With ChatGPT

```bash
langslice login
```

Runs the OAuth (PKCE) "Sign in with ChatGPT" flow in a browser, with a callback
on `localhost:1455`, and writes the token to `~/.langslice/openai_auth.json`
(mode 600). Access tokens are refreshed automatically. This file is the only
one read: a Codex CLI or Codex app login is never used, so the two keep separate
accounts.
To use a second account for one process, set `LANGSLICE_OPENAI_AUTH` to another
file for both `langslice login` and the run; only that file is then read.

Once signed in, no API key is needed for either model surface:

- chat/vision/tool-use agents accept `openai-oauth/<model>` model strings,
  e.g. `--model openai-oauth/gpt-5.6-luna` (legacy `chatgpt/<model>` is
  accepted), served by the ADK backend in `src/langslice/providers/openai_oauth.py`;
- image-gen registration accepts `--provider openai-oauth`.

## Engine Service

Non-Python clients drive the same pipeline over a newline-delimited JSON
protocol:

```bash
langslice serve --stdio
```

The service accepts `version`, `register.run`, `quick_affine.run`, and
`export.run` request envelopes, plus the Fiji connector's `setup.status`,
`setup.login`, `setup.api_key`, `linear.run`, `linear.estimate`,
`preprocess.preview` and `nonlinear.abba` (see
[the connector design](abba_plugin_design.md)). It streams progress/log
events and returns typed JSON result or error envelopes. The contract is
defined by the Pydantic models in `src/langslice/api/models.py`.

## Debug And Request Capture

Set `LANGSLICE_VLM_DEBUG_DIR` to save run artifacts. For ADK estimation request auditing,
set `LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` to write redacted JSONL request captures.

### Linear agent traces

`langslice linear run --trace-dir PATH` (or `LANGSLICE_TRACE_DIR`; the flag
wins; the worker's `trace_dir` and the ABBA dialog's **Save traces to** set it
for one run) writes one JSONL file per agent session —
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

Every tool that returns a picture takes all its picture options in one
argument, `view` (`mode`, `channels`, `atlas_channels`, `atlas_opacity`,
`regions`, `outlines`, `border_color`, `border_thickness`, `zoom`,
`deformation`, and `resolution` at image resolution `auto`;
`linear/display.py`), one per call on `adjust_transforms`; they apply to that
call only. `channels` is raw channels (several overlaid in colours) or the
`view`/`fit` version; `atlas_channels` is any of `ara`, `nissl` (only where
ABBA's cached Allen atlas is installed) and `borders` (the lines). By default
the tissue-framed pictures (`view_atlas`, `stacked`, `side_by_side`) and the
clean `section` / `template` modes stay free of outlines; borders are yellow,
1 px. Fractional widths are antialiased. A key that means nothing for the tool
or mode, and any unknown argument or key, is refused with the reason. Native
ABBA display colors are unchanged.

### Linear and nonlinear scope

The linear agent handles section order, atlas position and affine alignment.
It has no paired-landmark tools. With `--tasks ...,nonlinear` its job also
includes a deformation per section (`fit_deformable`, or a `keep_linear`
reason), fitted to the stain and, unless `--image-provider none`, to the
borders the image-model tool described above traces. Export adapters for the
stored deformations are not built yet.

Applied spline transforms in historical checkpoints still load, render and
export with their saved mapping. A new affine fit or adjustment replaces a saved
spline. Editable-point drafts from the removed tools are ignored when loading.
