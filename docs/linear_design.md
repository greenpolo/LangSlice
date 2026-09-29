# LangSlice linear — unified environment design (2026-09-05)

`langslice linear` places a stack of histology sections on a BrainGlobe atlas:
order, position along the slicing axis, and one in-plane affine per section.
It is ONE agent environment — one state, one toolbox, one job statement — that
a host switches on task by task. The three linear tasks run by default; an
optional image-model border-correction task follows an existing linear placement.

## Why one environment

Looking at the whole stack once yields the information for every task: which
sections are mirrored, which are out of order, where sections are missing, which
are damaged, where each one sits. Splitting those into separate agent sessions
throws that shared reading away and re-pays for it. The per-section interactive
alignment was the one exception until 2026-09-06 — it ran as a bounded
sub-session — and is not one any more: `adjust_transforms` is a
main-session tool the agent reaches for whenever it wants (GPT-6 Astra moved
every parameter at once and finished a section in four previews, so the
fan-out bought nothing). Preview and set are one tool, the computer-use
pattern: every action writes and returns the picture of what it did.

## The job spec

A host (CLI, ABBA plugin, engine service) fills one spec. Every checkbox a user
sees maps to a field here; nothing else is user-facing. Users see the tasks
grouped as Positioning (`reorder` + `position`), Linear (`transform`) and
Nonlinear; [the interface design](interface_design.md) holds that grouping and
the ABBA dialog's mapping.

```
JobSpec
  image_folder, atlas, plane, model, out, preprocess (auto|none)
  image_resolution: low|medium|high = low # size of every picture the agent sees;
                                # display only, fits and stored transforms unchanged
  agent_damage: bool = True     # build mark_damaged; host flags can never be cleared
  tasks: subset of {reorder, position, transform, nonlinear} # default: first three
  reorder:
    flip: bool = True             # may flip sections across the midline
    hemisphere_cue: str = ""      # user text: what marks a hemisphere (notch, injection...)
  position:
    thickness_um, interval_um     # cutting protocol; passed as facts
    strict_interval: bool = False # sections must sit exactly one interval apart
    deepslice: bool = False       # run_deepslice tool available (coronal mouse/rat only)
    bayesian: bool = False        # fit_position tool available (oblique.py fitter)
    notes: str = ""               # user notes, shown under this task in the job statement
  transform:
    interactive: bool = True      # direct visual affine adjustments
    automatic: bool = True        # automatic affine fitting tool
    angles: bool = False          # may set the stack-wide cutting angles
    elastix: bool = False         # fit_affine may use Elastix intensity affine
    max_parallel: int = 4         # 1..4; below 4, fit_affine and adjust_transforms
                                  # refuse calls naming more sections (TOO_MANY_SECTIONS)
    notes: str = ""               # user notes, shown under this task
  facts: free-form user facts, one line each, passed verbatim
  nonlinear: {provider: openai-oauth, image_model: null, notes: ""}
  inputs: order/positions/angles/transforms supplied by the host for tasks that are OFF,
          plus pixel_size_um, damaged {id: note} (flags the agent cannot clear) and
          locked [ids] (flip, rotation and transform the agent cannot change)
```

Task notes are rendered right after the job line, each under its task's name,
only when that task is on and the note is non-empty; `facts` stay in the run
facts. A locked section carries a `"host"` identity transform (its snapshot is
already aligned) unless `inputs.transforms` supplies one: it counts at submit,
is exempt from the damaged-section transform gate, and its position still
moves. `orient_slices`, `fit_affine` and `adjust_transforms` refuse it per
section with `LOCKED`; `mark_damaged` refuses to clear a host flag with
`DAMAGE_SET_BY_USER`. `image_resolution` multiplies the pictures only: seed,
view, atlas, placement and contact-sheet images, fit panels and interactive
overlays are drawn from a larger render of the same section, while the
`PREVIEW_LONG_EDGE` working frame, calibration, fits and the six stored
numbers are the same at every setting.

Task OFF means its outputs are inputs: reorder off → discovery order is fixed;
position off → positions come from the host and are facts; transform off →
no transform tools, with existing transforms supplied by the host or checkpoint.

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
  damaged: bool, damage_note: str          # excludes from DeepSlice and automatic
                                           # affine; set by the agent or by the user
                                           # through the host (inputs.damaged)
  position_mm | null
  transform | null: {kind: silhouette|elastix|interactive|host, params (six
                     normalized numbers), physical (rotation_deg, scale_x,
                     scale_y, shear, translate_x_mm, translate_y_mm, pivot),
                     calibration, iou?, mirrored?, note?}
  caveats: [str]
  image_correction | null: first image-model reply, notes, geometry and artifact paths
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
corrected index; every write returns the rows it changed (`changed` plus
`n_sections`; `status` is the whole table, and `undo`/`redo`/`submit` answer
with it too), except transform writes whose physical result already identifies
what changed; every write is undoable; every write checkpoints; fits write
their result. Tools report data. No advice, no interpretation, no strategy
in any payload or prompt (see `lean-harness` history in `linear/CLAUDE.md`).

| tool | task gate | does |
| --- | --- | --- |
| `status` | always | one row per section in corrected order: index, id, position_mm, delta_to_next_mm (signed), flip, rotation_deg, damaged(+note), transform kind, transform_iou, transform_mirrored, caveats; plus cutting angles and interval breaks. The `ls` of the environment. |
| `view_slices(ids)` | always | up to 4 sections at higher resolution, rendered as corrected, each captioned with its index and filename. |
| `fetch_atlas(positions_mm)` | always | up to 4 atlas sections, rendered at the current cutting angles, each captioned with its position (and the angles when oblique). |
| `note(text)` | always | append to the run notes. |
| `undo()` / `redo()` | always | snapshot stack; a batch call undoes as one. |
| `orient_slices([{id, flip?, rotate_deg?}])` | reorder.flip (flip) / reorder (rotate) | toggle flip, add rotation; returns each changed section rendered as it now stands (≤8). |
| `reorder_slices(new_order, after="start")` | reorder | move the listed filenames as a block, in the listed order, after a named section or at the start. One filename moves one slice; the full list sets the whole order. Unlisted sections keep their relative order. Corrected indices only; positions and transforms are kept. One undo step. |
| `mark_damaged([{id, damaged?, note?}])` | agent_damage (default on) | set damage (default True), or clear with damaged=False; clearing also removes the note. A flag the host set (`inputs.damaged`) is never cleared (`DAMAGE_SET_BY_USER`). |
| `set_positions([{id, position_mm}])` | position | batch write, clamped to the atlas range; returns a placement image only when that exact section, position, orientation and cutting-angle combination has not already reached the model in a full-canvas atlas-bearing view. A compare and write planned in the same model round both return their images. |
| `compare_placement([{id, positions_mm?}], mode, zoom, template_opacity, outlines)` | position | ≤4 candidate pairs; no positions = current position. `side_by_side` returns separate original section and atlas reference images (one section per distinct id, one atlas per pair; ≤8 images), independently tissue-framed, full-view only. Other modes return one physical-canvas image per pair. Writes nothing. |
| `view_stack()` | position | one contact sheet of every section in the order of its written position, each over the atlas at its position and captioned with index, filename, position and the distance to the next, plus a plot of position against corrected index (damaged in red): two images. Writes nothing. |
| `run_deepslice(ids?, allow_angle_change, keep=[ids])` | position.deepslice | positions (+ angles) for undamaged sections; UNAVAILABLE unless installed and plane/atlas supported. |
| `fit_position(id, window_mm, angles?)` | position.bayesian | `oblique.fit_oblique` at the section's current position: best position (and angles) with score; writes nothing. |
| `set_cutting_angles(pitch_deg, yaw_deg)` | transform.angles | stack-wide; subsequent atlas fetches and fits use them. |
| `fit_affine(ids, method=silhouette\|elastix)` | transform | per-section in-plane affine against its atlas section, written as the section's transform; returns iou, the transform as the same five `physical` knobs `adjust_transforms` takes (plus `shear`, about the canvas centre) and a captioned overlay panel for every successful fit. Damaged sections are refused. |
| `adjust_transforms(entries)` | transform.interactive | set one to four independent sections, each with rotation, per-axis scales and millimetre shifts. Per-entry mode, zoom, opacity, pivot and border controls; `ab` and `side_by_side` return two images, other modes one. Each result maps its images with `image_indexes`. One undo step; repeat unchanged parameters to redraw. Replaces the complete transform, including spline or shear. Inspect before a dependent correction in a later call. |
| `correct_slice_borders(id, additional_notes)` | nonlinear | one image-model correction of the section's placed atlas borders, first reply kept; no fit, no transform change. See [the image-tool contract](nonlinear_image_tool.md). |
| `submit(summary, notes, interval_breaks)` | always | ends the run; gated (below). |

`fetch_atlas` and `view_slices` frame tissue the same way so apparent scale is
not a cue. Every image a tool returns carries its label burned into the pixels
(`render.caption`): tool images reach the model as bare attachments, so the
text that ties an image to a section or a position has to ride in the image.
The image a fit MEASURES is never captioned — only what is shown. Seed
message: every section as its own labelled image in corrected order plus the
status table.

The default full-task toolbox contains 15 tools (9 for interactive-only
transform refinement).

## Submit gates (constraints, not coaching)

- position on: every section has a position; positions monotone along the
  corrected order (`ORDER_POSITION_MISMATCH`, naming the pairs). Failed
  submission returns the missing requirement without writing or ending the run.
- strict_interval: every consecutive spacing within 10% of the interval;
  `interval_breaks` must be empty.
- not strict: each reported break index must sit where the written interval
  exceeds 1.5x the stack's median written spacing (`INTERVAL_BREAKS_UNSUPPORTED`).
- transform on: every section carries a transform (`MISSING_TRANSFORMS`,
  naming the sections without one). Damaged sections require a non-identity
  interactive transform (`DAMAGED_REQUIRES_MANUAL_TRANSFORM`, naming each section
  and the missing, automatic, identity or invalid transform). `submit` returns an instruction to align the surviving anatomy with the
  interactive tools and inspect the overlays. If those tools are disabled, the
  refusal reports that the host must enable them. This checks an applied manual
  correction, not anatomical alignment quality; intact identity transforms remain
  valid. Position-only runs are unaffected. Locked sections (host `"host"`
  transform) are exempt: their alignment is the user's and cannot change here.

Refusals state the numbers and stop.

When interactive transforms are enabled, the prompt asks for inspection of each
fit or adjustment against surviving internal anatomy, then refinement of each
slice until no further improvement is possible with the available transforms.
Changes should be kept only if they improve alignment. This adds no mandatory
adjustment count or final-review hook.

### Linear and nonlinear scope

The linear agent supplies order, position and affine alignment. With the optional
`nonlinear` task it also exposes `correct_slice_borders(id, additional_notes="")`.
The top-level registration bridge sends placed borders plus the clean photograph
to the image model using the fixed correction prompt and optional per-slice notes.
The first result is retained without agent selection or rejection. Submit requires
a completed correction at each section's current placement. No deformation is
fitted and no transform is changed by this tool. See
[the image-tool contract](nonlinear_image_tool.md).

Historical checkpoints with applied spline transforms remain readable and render
their complete saved mapping. Affine fitting or adjustment replaces that mapping;
A/B comparison can still show it. Obsolete editable-point fields are ignored when
loading. Native ABBA export retains calibrated spline support for these saved
results.

## Session

One ADK `LlmAgent`, tools built from the spec, the job statement built from the
spec and state, the seed message from the state. The loop nudges when the model
answers in prose and ends at submit or the turn budget. Every write tool
checkpoints; a run that dies mid-way resumes from the checkpoint with the
state it had (the agent is re-seeded, not replayed). `LANGSLICE_TRACE_DIR`
records the full trajectory (`trace.py`, unchanged).

Context is bounded, because the whole history is resent on every call: tool
images live in a working set (`WorkingSetImages`, 256 high / 128 low, trimmed
in batches so the cached prefix stays stable; the seed strip is never
trimmed), every call's token usage is printed and traced, and
`JobSpec.max_quota_percent` ends a run whose share of the provider's usage
window reaches it. Optional `JobSpec.max_input_tokens` (default `None`)
checks a single request's reported input, including cached tokens, after the
response and allows one grace submission call if exceeded. It does not limit
cumulative usage. Logs report cumulative input for cost and peak request input
for context pressure separately.

The established working-set policy is the only active policy: its first
transform-media stage cut and 256-to-128 image-count cuts are unchanged.
There is no image-acceptance tool, additional inspection gate, or predictive
cost trigger. `--image-retention legacy` remains a compatibility option;
the removed `completion` value is rejected rather than silently remapped.
The [visual context design record](visual_context_design.md) preserves the
shelved experiment and the measurement-first direction: improve image
delivery and observe actual cache costs before adding retention machinery.

There is ONE session. The alignment tools live in it like every other tool, so
a section's preview history, its atlas fetches and the stack reading that
produced its position are all in one context.

Successful placement renders are associated with their function-call result
and become "seen" only when their media survives filtering into a later model
request. This keeps same-round parallel compare/write calls honest, does not
count failed renders, and works across providers that omit generated call ids.

## CLI

```
langslice linear run FOLDER [--tasks reorder,position,transform[,nonlinear]] ...
langslice linear quick-affine ...   (unchanged)
```

The full flag list is in `docs/current_workflow.md`. `image_resolution`,
`agent_damage`, the per-task `notes`, `transform.max_parallel` and
`inputs.damaged` / `inputs.locked` are spec fields without CLI flags so far.

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

## The write-observer hook and the ABBA host adapter (2026-09-10)

Host adapters were open until 2026-09-10: `checkpoint.py` now exposes
`observe_checkpoints(fn)`, a context manager that registers `fn` to be called
with the state right after every `save_checkpoint` write — the one point
every tool's write already funnels through (`toolbox.py`'s writers,
`engine.run`'s own checkpoint after ingest). `engine.run(spec, ...,
on_write=fn)` wraps the whole session in it and calls `fn` once more up
front, on the freshly-ingested state, so a host sees the stack before the
agent has touched it. The agent itself never knows a host is watching:
nothing about the toolbox, the job statement, or the render path changes.

`langslice.integrations.abba_linear.AbbaStackMirror` is the first such host:
`langslice abba --linear FOLDER` runs the ordinary headless agent — it still
renders its own BrainGlobe pictures — inside a live ABBA session, and on
every `on_write` call diffs the new state against the last one it saw and
pushes only what changed into ABBA (order/position as `moveSlice`, flip and
the quarter-turn as the slice's pre-transform, the in-plane affine as a
registration step, cutting angles onto the resliced atlas), so a person
watches the stack move in BigDataViewer as the agent works. ABBA is display
plus the final home of the result; see
`src/langslice/integrations/CLAUDE.md` for what is and is not verified about
its sign/axis conventions.

## Interactive transform: physical space and atlas outlines (2026-09-05, Nash)

In the main trajectory since 2026-09-06 — Astra one-shots it. The alignment is
a vision-action loop: tool → section with
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
  physical params and the calibration used. ONE representation: silhouette
  fits carry `physical` too (`transform.physical_params` reads their canvas
  2x3 back into the knobs about the canvas centre), which is what lets the
  A/B view show a fit as its B side and lets a preview start from one.

**Atlas outlines.** One smoothed contour per FAMILY-level region (the
organized `color_lut`, families merged at `MERGE_EPS`; leaf boundaries are
omitted). Borders default to yellow, 0.5 anti-aliased output pixels, without a
rim. Agent overlay tools expose `border_color` (named color or `#RRGGBB`) and
`border_thickness` (0.25–8 output pixels, including fractional widths) alongside `template_opacity`.
Lines are drawn after crop/resize so zoom does not change their pixel thickness.
These controls affect display only; changing style with the same transform
redraws without an undo step. Automatic fit feedback uses the yellow/0.5 px default.
The separate native ABBA viewer retains ABBA's own display settings.
The contour
code (`region_contours`, `_smooth_closed`, family mapping, annotation slice
at cutting angles) moves from `nonlinear/` to `atlas/render.py` so both
methods draw the same lines from the same source.

**The screen.** `adjust_transforms` returns the transformed section
(display-preprocessed grayscale) with the family outlines on top at true
scale, a 1 mm scale bar, and a caption (section id, position, angles, the
params). `fit_affine`'s panels use the same renderer
(`render.physical_overlay`, the `overlay` view of `render.physical_views`), so
every look at a section is the same picture.

**View controls (2026-09-06).** Four sessions of gpt-5.6-luna aligning damaged
M04 sections converged on the same three complaints — one fixed small image,
no way to see the atlas apart from the section, and no magnification — so
`adjust_transforms` takes three optional controls. Defaults reproduce the
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

**The agents' wishlist, built 2026-09-06.** If Astra wants it, it gets it.
- `pivot` on `adjust_transforms`: `"canvas"`
  (the canvas centre, the old behaviour), `"tissue"` (the section's own tissue
  centroid, from `image_prep.foreground_mask`) or `[fx, fy]` fractions of the
  canvas. Rotation and the scales turn about it; the translation does not care.
  The arithmetic is `affine.physical_affine_matrix(pivot=...)`, and
  `normalized_physical_affine` takes the pivot on the SECTION's frame, so the
  six stored numbers still describe the map the canvas showed.
- `mode="ab"` renders TWO overlays at one crop: the parameters passed, then the
  transform the section carried before the call, whatever made it (identity
  only when it had none; `ab_reference` says which). The before/after toggle.

**Astra's requests, built 2026-09-06.** GPT-6 Astra (medium) aligned damaged
M05 sections in the main trajectory at human level and was debriefed; three of
its five asks are in:
- An ROI-restricted fit was built and removed the same day (2026-09-06): a
  moments fit inside a box has no anatomy to lock rotation to and inflated
  remnants; damaged sections are aligned by hand.
- one transform representation: `physical` on every stored transform and every
  fit payload, so a fit and a hand alignment are the same five numbers ("its
  reported matrix/decomposition was not directly interchangeable with the
  manual controls"). The fraction-based decomposition left the fit payload.
- `outlines` on `adjust_transforms`: `all` (default), `outer` (the atlas
  outline alone) or `none` — "all family outlines together were visually
  busy". The caption names the layer when it is not `all`.
- concise writes: a write answers with the rows it changed, not the whole
  table ("on a large stack, concise change reports would be easier to review").
- concise transform feedback: the full adjustment history, normalized matrix
  and derived representations stay in local state; each result sends the
  physical controls and image once. A silhouette fit likewise does not repeat
  its result in a generic changed row.
- confidence is GONE, on Nash's call: nothing downstream reads it, and the
  reasoning lives in `adjust_transforms`'s note.

Region acronyms on the outlines were asked for and deliberately NOT built
(Nash: "it has no use for this"), and neither was damage masking — after the
overlap number left the payload there is no metric a damage mask would clean.

**What the payload carries.** The physical controls, whether they changed
state, the requested view and the labelled image. The checkpoint still carries
the normalized matrix, calibration and note, and the toolbox keeps the full
per-section adjustment history, but those derived and accumulated forms are
not repeated into every later model turn.

**Rock-solid means tested.** Geometry tests pin: a 1 mm translation moves the
section by exactly 1000/µm-per-px pixels; an atlas of known physical width
renders at that width in pixels; the scale bar is 1000/µm-per-px pixels long
before AND after a zoom crop; outline pixels lie on family-color boundaries of
the filled render; `side_by_side` returns two frames of one size;
`checkerboard` carries both sources in pixels; `outlines` is black off the
lines; a file with no pixel size
yields `calibration: "estimated"`, never a crash.
