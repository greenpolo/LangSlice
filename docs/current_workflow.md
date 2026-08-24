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
langslice linear estimate-brain <image_folder> [--atlas ...] [--plane ...] [--interval 200] [--thickness 50] [--keep-order|--no-keep-order] [--model ...] [--preprocess auto|none] [--landmark-tools|--no-landmark-tools] [--out ...] [--resume|--fresh|--rerun-from NODE] [--stop-after NODE]
```

Single-slice estimation runs through the ADK harness. The agent surface is
intentionally small: `fetch_atlas` and `submit_estimate`. `fetch_atlas` returns
at most 8 sections per call; ask for more and the extras come back listed under
`dropped_positions_mm` with `truncated: true`, never silently missing. Atlas
images from older tool calls are trimmed out of the model's context as it goes
(`langslice.adk.plugins.trim_stale_tool_images`): the JSON naming every fetched
position stays, and the pixels of the newest calls stay — three full sweeps'
worth, and always at least the two newest calls whatever their size, so two
sweeps can always be compared against each other.

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
  In the whole-brain visual path only, sections and fetched atlas sections are
  both cropped to their foreground plus a 6% margin before resizing, so the two
  fill their frames about equally: histology arrives filling most of its scan
  while a fixed-canvas atlas render leaves an anterior brain small and centred,
  and that difference in apparent scale is itself a cue. Histology foreground
  is "far from the border's background level", cropped to the LARGEST connected
  blob so a neighbouring fragment or a speck of debris on the slide cannot drag
  the frame open around both and shrink the section to a corner of it; the atlas
  silhouette comes from the annotation (a reference volume's faint background
  noise would defeat a non-zero test). The geometry path is deliberately
  excluded: the `transforms`
  step fits its affine on the uncropped render, since the six numbers it hands
  back are normalized against that frame.
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
  good ones needs intimate atlas knowledge, and that is the positioning
  agent's decision.
- `position` is the second agent pass. The whole stack is in context, usually
  with no positions on it, and the agent reasons its own way to a placement.
  The step is deliberately lean: the prompt states the job (give every section
  a position in mm along the slicing axis, report the interval breaks it
  concludes are real, damaged sections included), the run's facts (section
  count, plane, atlas and species, valid axis range, the cutting protocol's
  nominal interval and thickness, whether the order is fixed, what is already
  placed), one factual line per tool, and the hard constraints — nothing else.
  No strategy menu, no rules of thumb, no warnings about failure modes: full
  traces showed every major benchmark failure tracking back to advice the
  harness injected, so tools and prompts report data and the model does the
  judging. The prompt stays atlas-agnostic (no region names, no hard-coded
  landmarks, no absolute positions) so the same text works for every
  BrainGlobe atlas, species and plane.
  Tools: `view_slices`, `fetch_atlas`, `atlas_structures_at` (what the atlas
  annotation carries at up to 8 levels, by in-plane area share),
  `structure_range` (the slicing-axis span over which up to 10 named structures
  exist, descendants included; each hit reports the `name` the query actually
  resolved to plus up to three `also_matches` acronyms it could have meant
  instead, so a short acronym cannot resolve to the wrong structure unnoticed),
  `stack_positions` (one row per section in corrected order — index, filename,
  `position_mm` or null, and `spacing_to_next_mm`, the distance to the next
  placed section), `interpolate_between` (fixed points in, one position per
  section out, not written; beyond the outermost fixed points it steps at the
  interval those points imply, and it refuses a single fixed point),
  `set_positions` (batch write, clamped to the atlas range, returns the same
  rows `stack_positions` gives) and `submit_positions`. There is no per-slice
  estimation worker in this step: full-trace forensics found its estimates
  carried essentially no signal on real data while consuming most of the
  step's wall-clock. `submit_positions` is rejected unless every section has a
  position, its reported `interval_breaks` are visible in the positions it
  wrote, AND its two `end_anchors` hold:
  - `interval_breaks` — each reported index names the section AFTER a gap, and
    the written interval at that neighbour pair must be more than 1.5x the
    stack's own median written spacing. An index whose own numbers show
    ordinary spacing is refused as `INTERVAL_BREAKS_UNSUPPORTED`, naming the
    interval actually written there, so a break cannot travel downstream as a
    finding the placement does not contain.
  - `end_anchors` — one entry per END of the corrected order naming a structure
    visible in that section, whose atlas existence range
    (`langslice.atlas.landmarks.axis_range_of`) must contain that section's
    submitted position, within one slice thickness, AND whose own span covers no
    more than `MAX_ANCHOR_SPAN_FRACTION` (8%) of the atlas's full slicing-axis
    extent — a structure present almost everywhere (cortex, say) "proves" any
    placement and is refused as `STRUCTURE_TOO_BROAD` before its span is even
    checked against the position. The cap is deliberately tight: at a quarter of
    the axis a mouse-atlas anchor could still be 3 mm wide, wider than the
    errors the gate exists to catch (310 of the 839 annotated Allen structures
    still qualify). A refused anchor comes back with the structure's actual
    span (and, for a too-broad one, `span_fraction`) against the proposed
    position and does not escalate. Refusal messages state the facts that
    caused them ("X spans a-b mm, you placed S at c mm") and stop there — they
    do not tell the agent what to do about it. The accepted anchors are
    recorded in the run notes, along with the accepted interval breaks.

  Oblique-angle estimation is not part of this step yet.

  All of the above — the two landmark tools and the end-anchor gate — are
  gated by `BrainConfig.landmark_tools` (default on; CLI
  `--landmark-tools`/`--no-landmark-tools`), an ablation switch for
  experiments, not a normal deployment knob. Off, `atlas_structures_at` and
  `structure_range` are not registered, `submit_positions` takes no
  `end_anchors` argument (the interval-break check still runs), and the prompt
  drops the two tool lines and the end-anchor constraint. Everything else in
  the prompt is identical.
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
  the same positions/spacing rows the positioning step reads. Its prompt gets
  the same lean treatment — job, run facts, tools, no checklist. It can attach
  caveats (`flag_slice`) and, once, send the stack back to `position` with
  notes the next pass reads. A review that runs out of turns approves with a
  note -- it can never block hand-back.
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
the enhanced pixels are never written back to the user's files.

State is checkpointed to `<image_folder>/brain_estimate.json` after every
node, in the same shape as the results file. `--resume` (default) skips nodes
the checkpoint lists as complete; `--fresh` re-runs the whole graph.

`--rerun-from {position,transforms,review}` rewinds an existing checkpoint to
just before that node and resumes from there, so a single step can be
re-benchmarked without re-running (and re-paying for) the agent steps ahead of
it. It clears that node's own output plus everything derived from it --
`position` also clears `transforms`, since the transform step reads
positions -- including the run notes those steps wrote (every note is prefixed
with the node that wrote it), so a rewound pass does not start by reading the
ladder it is meant to redo. It leaves `survey`'s findings (order, flips,
damage) and the earlier steps' notes untouched:
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

### Whole-brain agent traces

`langslice linear estimate-brain --trace-dir PATH` (or `LANGSLICE_TRACE_DIR`;
the flag wins) writes one JSONL file per agent session —
`<trace_dir>/<run_label>_<8 hex>.jsonl`, e.g. `whole_brain_survey_1a2b3c4d.jsonl`,
`whole_brain_position_*`, `whole_brain_review_*`,
`whole_brain_transform_003_*` — with one record per event:

| `kind` | contents |
| --- | --- |
| `session` | run label, agent name, model name, full system instruction (first line) |
| `seed` | the step's seed message: text verbatim, images as descriptors labelled with the text above them |
| `model` | one model turn: its text, any thought summary, every function call with full JSON arguments |
| `tool_result` | the complete tool payload the model reads, plus descriptors for media riding on the response |
| `nudge` | a nudge the driver actually sent |
| `summary` | tool-call count, turn count, whether the step submitted (last line) |

Unlike the request captures above, this records values, not shapes: full text,
full tool arguments, full tool responses. Images are always descriptors (mime
type, byte count, pixel size) — never bytes — so a trace stays small. Nothing
but content, the prompt and a model name is read, so no credentials are
written. Unset, the recorder is never constructed.
