# LangSlice linear — unified environment design (2026-09-05)

`langslice linear` places a stack of histology sections on a BrainGlobe atlas:
order, position along the slicing axis, and one in-plane transform per section.
It is ONE agent environment — one state, one toolbox, one job statement — that
a host switches on task by task. Running everything is the default.

## Why one environment

Looking at the whole stack once yields the information for every task: which
sections are mirrored, which are out of order, where sections are missing, which
are damaged, where each one sits. Splitting those into separate agent sessions
throws that shared reading away and re-pays for it. The one exception is the
per-section interactive alignment, which is local to a section and runs as a
bounded sub-session the main agent spawns and reviews.

## The job spec

A host (CLI, ABBA plugin, engine service) fills one spec. Every checkbox a user
sees maps to a field here; nothing else is user-facing.

```
JobSpec
  image_folder, atlas, plane, model, out, preprocess (auto|none)
  tasks: subset of {reorder, position, transform}     # default: all three
  reorder:
    flip: bool = True             # may flip sections across the midline
    hemisphere_cue: str = ""      # user text: what marks a hemisphere (notch, injection...)
  position:
    thickness_um, interval_um     # cutting protocol; passed as facts
    strict_interval: bool = False # sections must sit exactly one interval apart
    deepslice: bool = False       # run_deepslice tool available (coronal mouse/rat only)
    bayesian: bool = False        # fit_position tool available (oblique.py fitter)
  transform:
    angles: bool = False          # may set the stack-wide cutting angles
    elastix: bool = False         # fit_affine may use Elastix intensity affine
    subagents: bool = True        # align_slice runs as a tool the main agent calls;
                                  # False = engine runs the sessions after submit
  facts: free-form user facts, one line each, passed verbatim
  inputs: order/positions/angles supplied by the host for tasks that are OFF
```

Task OFF means its outputs are inputs: reorder off → discovery order is fixed;
position off → positions come from the host and are facts; transform off →
no transform tools, no alignment.

## State

`StackState` is the checkpoint, the result, and the thing every tool writes.
Same JSON shape for all three.

```
StackState
  image_folder, atlas, plane, interval_mm, thickness_mm
  spec: the JobSpec as run
  cutting_angles_deg: {pitch, yaw}      # stack-wide; 0/0 = flat
  interval_breaks: [corrected indices]  # section AFTER a gap the agent concluded is real
  notes: [str]                          # agent notes, run log
  slices: [SliceState]
  submitted: bool
SliceState
  id, index_original, index_corrected
  flip: bool, rotation_deg: 0|90|180|270   # rotate first, then flip left-right
  damaged: bool, damage_note: str          # agent-internal: excludes from DeepSlice
                                           # and automatic affine; never a user option
  position_mm | null, confidence
  transform | null: {kind: silhouette|elastix|interactive, params|matrix, iou?,
                     mirrored?, note?}
  caveats: [str]
```

Order and position are separate fields that must agree at submit. Reordering
touches ONLY `index_corrected` — positions and transforms are kept — and
`submit` refuses positions that are not monotone along the corrected order.
The gate is the whole of the rule: clearing positions on a reorder cost the
agent its work and forced it to re-enter numbers it still believed.
Standalone reorder therefore outputs a permutation; unified runs output
positions and the order comes for free.

The user's image files are never modified. Everything is a proposal as data.

## Toolbox

Conventions, applied to every tool: sections are addressed by filename or
corrected index; every write returns the same status rows; every write is
undoable; every write checkpoints; fits have a preview form that computes
without writing. Tools report data. No advice, no interpretation, no strategy
in any payload or prompt (see `lean-harness` history in `linear/CLAUDE.md`).

| tool | task gate | does |
| --- | --- | --- |
| `status` | always | one row per section in corrected order: index, id, position_mm, delta_to_next_mm (signed), flip, rotation_deg, damaged(+note), transform kind, transform_iou, transform_mirrored, confidence, caveats; plus cutting angles and interval breaks. The `ls` of the environment. |
| `validate(interval_breaks)` | always | runs exactly the submit gates and returns the refusal or `{"status": "ok", "would_submit": true}`; writes nothing. |
| `view_slices(ids)` | always | up to 8 sections at higher resolution, rendered as corrected, each captioned with its index and filename. |
| `fetch_atlas(positions_mm)` | always | up to 8 atlas sections, rendered at the current cutting angles, each captioned with its position (and the angles when oblique). |
| `note(text)` | always | append to the run notes. |
| `undo()` / `redo()` | always | snapshot stack; a batch call undoes as one. |
| `orient_slices([{id, flip?, rotate_deg?}])` | reorder.flip (flip) / reorder (rotate) | toggle flip, add rotation. |
| `reorder_slices(new_order)` | reorder | full permutation; changes corrected indices only, positions and transforms are kept. |
| `move_slice(id, after)` | reorder | incremental move; same rule. |
| `mark_damaged([{id, note}])` / unmark | always | agent-internal classification. |
| `set_positions([{id, position_mm, confidence?}])` | position | batch write, clamped to the atlas range. |
| `distribute_spacing(fixed=[{id, position_mm}], keep=[ids])` | position | linear interpolation between fixed points, extrapolating at the implied interval; `keep` sections are not moved; returns rows, writes nothing (`apply=True` writes). |
| `run_deepslice(ids?, allow_angle_change, keep=[ids])` | position.deepslice | positions (+ angles) for undamaged sections; UNAVAILABLE unless installed and plane/atlas supported. |
| `fit_position(id, window_mm, angles?)` | position.bayesian | `oblique.fit_oblique` at the section's current position: best position (and angles) with score; writes nothing. |
| `set_cutting_angles(pitch_deg, yaw_deg)` | transform.angles | stack-wide; subsequent atlas fetches and fits use them. |
| `fit_affine(ids, method=silhouette\|elastix, apply=True)` | transform | per-section in-plane affine against its atlas section; returns iou, the transform decomposed (rotation, scales, shear, translation, `mirrored`) and a captioned overlay panel per section (≤16); refuses damaged sections. |
| `align_slice(id, notes)` | transform (subagents) | runs the bounded interactive sub-session for ONE section; records an in-plane transform only (flip/rotation flags are untouched); returns params, their decomposition and the final overlay. |
| `copy_transform(from_id, to_ids)` | transform | copy one section's transform to others. |
| `submit(summary, notes, interval_breaks)` | always | ends the run; gated (below). |

`fetch_atlas` and `view_slices` frame tissue the same way so apparent scale is
not a cue. Every image a tool returns carries its label burned into the pixels
(`render.caption`): tool images reach the model as bare attachments, so the
text that ties an image to a section or a position has to ride in the image.
The image a fit MEASURES is never captioned — only what is shown. Seed
message: every section as its own labelled image in corrected order plus the
status table.

## Submit gates (constraints, not coaching)

- position on: every section has a position; positions monotone along the
  corrected order (`ORDER_POSITION_MISMATCH`, naming the pairs). `validate`
  runs the same gates without submitting.
- strict_interval: every consecutive spacing within 10% of the interval;
  `interval_breaks` must be empty.
- not strict: each reported break index must sit where the written interval
  exceeds 1.5x the stack's median written spacing (`INTERVAL_BREAKS_UNSUPPORTED`).
- transform on, subagents off: nothing to gate; the engine runs the sessions
  after submit (silhouette affine for intact, interactive for damaged).

Refusals state the numbers and stop.

## Session

One ADK `LlmAgent`, tools built from the spec, the job statement built from the
spec and state, the seed message from the state. The loop nudges when the model
answers in prose and ends at submit or the turn budget. Every write tool
checkpoints; a run that dies mid-way resumes from the checkpoint with the
state it had (the agent is re-seeded, not replayed). `LANGSLICE_TRACE_DIR`
records the full trajectory (`trace.py`, unchanged).

`align_slice` is an `AgentTool`-style sub-session: its own small toolbox
(`preview_transform`, `submit_transform`), its own turn cap, seeded with the
section, its atlas section, and the notes the main agent passed.

## CLI

```
langslice linear run FOLDER [--tasks reorder,position,transform]
    [--atlas ..] [--plane ..] [--model ..] [--preprocess auto|none]
    [--reasoning none|minimal|low|medium|high] [--pixel-size-um UM]
    [--no-flip] [--hemisphere-cue TEXT]
    [--thickness UM] [--interval UM] [--strict-interval] [--deepslice] [--bayesian]
    [--angles] [--elastix] [--no-subagents]
    [--fact TEXT ...] [--positions JSON] [--order JSON]
    [--out PATH] [--fresh] [--trace-dir PATH]
langslice linear quick-affine ...   (unchanged)
```

`estimate`, `estimate-brain`, `--stop-after`, `--rerun-from` and
`collect-traces` are removed. A single section is a stack of one.

## What leaves the repo

- The single-slice agent (`runner.py`, `single_slice.py`, `prompts.py`,
  `tools.py`, `validators.py`, `session.py`, `_types.py`) and its tests.
- The node graph (`whole_brain/nodes.py`, `engine.py` routing, `survey.py`,
  `position.py`, `review.py` as separate agents) — replaced by the one toolbox.
- `trace_collection.py` and `collect-traces` → `../LangSlice-Training`.
- The engine service's `estimate.run` (hosts use the plugin or `linear run`).

## Open / deferred

- DeepSlice integration behind `run_deepslice` (optional extra; stub reports
  UNAVAILABLE).
- Elastix affine method (behind `--elastix`), first version may land after the
  silhouette method.
- Host adapters (ABBA hand-back of order/positions/transforms).

## Interactive transform: physical space and atlas outlines (2026-09-05, Nash)

The per-section alignment loop is a vision-action loop: tool → section with
the atlas overlaid in physical space → tool → repeat. If the render lies, the
loop is compromised, so this is the part of linear that adopts standard
registration-software practice (ABBA) exactly.

**Physical calibration.** Both images are placed in millimetres.
- Section pixel size (µm/px) comes from the image file (TIFF XResolution +
  ResolutionUnit, or OME-XML PhysicalSizeX), overridable by the host
  (`--pixel-size-um`, `JobSpec.inputs["pixel_size_um"]`). The working canvas
  is the downsampled section, so canvas µm/px = file µm/px × downsample.
- Atlas µm/px is the atlas voxel size (25 µm for allen_mouse_25um); an atlas
  render is placed on the canvas at scale = atlas µm/px ÷ canvas µm/px, its
  anatomy centred on the canvas centre. Never fit-to-canvas: that is not a
  calibration (measured 0.69× vs true 1.05× on M01).
- No pixel size anywhere → the run records `calibration: "estimated"` from
  the silhouette fit's scale and says so in the tool payload; it never
  silently pretends.
- Transform parameters are physical, ABBA's: rotation (deg, about the canvas
  centre), scale_x / scale_y (unitless), translate_x_mm / translate_y_mm.
  The stored transform stays the host-facing normalized six numbers, plus the
  physical params and the calibration used.

**Atlas outlines, ABBA's way.** ABBA's border channel is the 1-voxel edge of
the label volume shown as one neutral grey hairline; region colors belong to
the filled map, not to lines over tissue. Ours: one smoothed contour per
FAMILY-level region (the organized `color_lut`, families merged at
`MERGE_EPS` — leaf boundaries are visual noise here), drawn as a 1 px
anti-aliased line in light grey on dark fluorescence (dark grey on
brightfield), no rim, no per-region color, AFTER the resize to the output
size so a hairline stays a hairline on the screen the model sees. Coloured
2 px lines with a dark rim were tried first and rejected (Nash: "too thick,
colors are weird"). The contour
code (`region_contours`, `_smooth_closed`, family mapping, annotation slice
at cutting angles) moves from `nonlinear/` to `atlas/render.py` so both
methods draw the same lines from the same source.

**The screen.** `preview_transform` returns the transformed section
(display-preprocessed grayscale) with the family outlines on top at true
scale, a 1 mm scale bar, and a caption (section id, position, angles, the
params). `fit_affine`'s panels use the same renderer
(`render.physical_overlay`, the `overlay` view of `render.physical_views`), so
the main agent and the sub-session see the same picture.

**View controls (2026-09-06).** Four sessions of gpt-5.6-luna aligning damaged
M04 sections converged on the same three complaints — one fixed small image,
no way to see the atlas apart from the section, and no magnification — so
`preview_transform` takes three optional controls. Defaults reproduce the
older single overlay exactly.
- `zoom = [x0, y0, x1, y1]`, fractions of the CANVAS (not of the section).
  The crop happens BEFORE the resize to the output long edge, so it is real
  magnification rather than an upscale; outlines and bar are drawn after the
  crop at output resolution, and the bar is redrawn for the magnified
  µm/px, so it is still exactly 1 mm. The caption states the box.
- `mode` — `overlay` (default), `side_by_side` (TWO captioned images, the
  warped section and the atlas template at the same µm/px and the same crop,
  outlines on both), `checkerboard` (the two in alternating tiles, 8 across),
  `outlines` (the atlas lines plus the section's own silhouette contour in a
  second grey, on black — no pixels).
- `template_opacity` (0..1, default 0.0) replaces the old `show_template`
  bool and dials the template blended under the outlines in `overlay`.

Region acronyms on the outlines were asked for and deliberately NOT built
(Nash: "it has no use for this"), and neither was damage masking.

**What the payload carries.** Beside the params, their decomposition and the
calibration: `translate_px` (the
entered millimetres as canvas pixels, plus `px_per_mm`, so mm↔px is explicit),
`history` (every parameter set previewed in this session, oldest first) and
`view` (the mode and zoom box the image was drawn with). `decompose_affine`
names its shifts `translate_x_frac` / `translate_y_frac`: they are fractions of
width and height, and one session read 0.0075 as millimetres because the old
name did not say so.

**Rock-solid means tested.** Geometry tests pin: a 1 mm translation moves the
section by exactly 1000/µm-per-px pixels; an atlas of known physical width
renders at that width in pixels; the scale bar is 1000/µm-per-px pixels long
before AND after a zoom crop; outline pixels lie on family-color boundaries of
the filled render; `side_by_side` returns two frames of one size;
`checkerboard` carries both sources in pixels; `outlines` is black off the
lines; a file with no pixel size
yields `calibration: "estimated"`, never a crash.
