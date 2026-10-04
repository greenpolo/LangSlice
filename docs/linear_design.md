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
  host_preprocessing: dict|None = None # a host's channel blend (preprocess.preview
                                # settings) for snapshots exported one page per
                                # channel: the DEFAULT appearance; pages stay raw channels
  agent_preprocessing: bool = False # build `preprocess` (agent-set appearance)
  image_resolution: low|medium|high|auto = low # long edge of each opening-strip
                                # tile and of every later picture (render.PICTURE_EDGES);
                                # auto: the agent's `view.resolution` per call; display
                                # only, fits and stored transforms unchanged
  agent_damage: bool = True     # build mark_damaged; host flags can never be cleared
  tasks: subset of {reorder, position, transform, nonlinear} # default: first three
  reorder:                        # the order only; reorder.flip / reorder.hemisphere_cue
                                  # are accepted as aliases of the transform fields
  position:
    thickness_um, interval_um     # cutting protocol; passed as facts
    strict_interval: bool = False # sections must sit exactly one interval apart
    deepslice: bool = False       # run_deepslice tool available (coronal mouse/rat only)
    bayesian: bool = False        # search_position tool available (oblique.py fitter)
    notes: str = ""               # user notes, shown under this task in the job statement
  transform:
    flip: bool = True             # may mirror sections left-right (orient_slices)
    hemisphere_cue: str = ""      # user text: what marks a hemisphere (notch, injection...)
    interactive: bool = True      # direct visual affine adjustments
    automatic: bool = True        # automatic affine fitting tool
    angles: bool = False          # may set the stack-wide cutting angles
    max_parallel: int = 4         # 1..4; below 4, fit_affine and adjust_transforms
                                  # refuse calls naming more sections (TOO_MANY_SECTIONS)
    notes: str = ""               # user notes, shown under this task
  facts: free-form user facts, one line each, passed verbatim
  nonlinear: {provider: openai-oauth, image_model: null, engine: either, notes: ""}
                                # provider "none": no image model (no trace_borders)
  inputs: order/positions/angles/transforms supplied by the host for tasks that are OFF,
          plus pixel_size_um, damaged {id: note} (flags the agent cannot clear),
          locked [ids] (flip, rotation and transform the agent cannot change) and
          channel_names [one per exported page] (names of the raw channels)
```

**A mirror is a linear transform (2026-09-29).** A left-right mirror is the
sign of the in-plane affine (a negative determinant): a host's own alignment
(ABBA) carries it inside the transform, and a left-right symmetric brain
gives positioning nothing to see it by. Flip and quarter-turn therefore
belong to the `transform` task: `orient_slices` is built only when
`transform` is on, and a positioning-only run (`reorder` + `position`) has
no flip tool and its job statement never mentions mirroring or the
hemisphere cue. A host transform in `inputs.transforms` may be mirrored; it
is kept exactly as supplied (the section's own `flip` stays false), counts
at submit, and reaches the nonlinear handoff unchanged. The silhouette fit
never reflects, so an agent that refits such a section expresses the mirror
with `orient_slices` instead.

Task notes are rendered right after the job line, each under its task's name,
only when that task is on and the note is non-empty; `facts` stay in the run
facts. A locked section carries a `"host"` identity transform (its snapshot is
already aligned) unless `inputs.transforms` supplies one: it counts at submit,
is exempt from the damaged-section transform gate, and its position still
moves. `orient_slices`, `fit_affine` and `adjust_transforms` refuse it per
section with `LOCKED`; `mark_damaged` refuses to clear a host flag with
`DAMAGE_SET_BY_USER`. `image_resolution` sizes the pictures only
(`render.PICTURE_EDGES`, long edges):

| level | opening strip tiles (seed message) | every later picture |
| --- | --- | --- |
| low (default) | 256 px | 512 px |
| medium | 384 px | 768 px |
| high | 512 px | 1024 px |
| auto | 256 px | the agent's `view.resolution` per call, 128 px up to the driver model's largest image (2048 px on the OpenAI lanes, 2000 px for a Claude host); 512 px when it gives none |

Opening = each tile of the opening strips. The seed message shows the stack
the way ABBA's slice strip does (`linear/opening.py`, 2026-10-03): horizontal
strips in corrected order, the sections on top and, directly beneath each,
the atlas at that section's current position and the stack's cutting angles
(drawn to the section's size). Every tile is labelled in its pixels
(`<index>: <filename>`, `atlas <mm> mm`, `no position`), columns are split by
a thin line, and a text part before each strip lists its sections. A strip's
long edge is the model lane's largest image, 2048 px on the OpenAI lanes and
1568 px for Claude (`show_stack`), and stays inside the lane's patch budget
(2500 32-px patches on OpenAI, ~1.2 MP for Claude), so a strip holds 8 / 5 / 4
tiles at low / medium / high on OpenAI (6 / 4 / 3 for Claude) and a larger
level means more strips: 5 / 8 / 10 for a 40-section stack on OpenAI. When no
section has a position the strips are section-only and the atlas reference
(evenly spaced positions, labelled in mm) follows as atlas-only strips; when
every section has one the reference is not sent; a partly placed stack gets
both. Later =
every picture a tool returns, each panel of a multi-panel picture
(`view_slices`, `view_atlas`, placement pictures, fit panels,
`adjust_transforms`, `fit_deformable`). Larger pictures are drawn from a
larger render of the same section; nothing is upsampled past its source (a
section's working copy, the atlas plane's voxels). The `PREVIEW_LONG_EDGE`
working frame, calibration, fits, the six stored numbers, the deformable fit
grid and the image model's inputs are the same at every setting. Until
2026-10-01 every picture was instead sized to the atlas voxel (25 um on the
Allen mouse), capped at 512 px, so a 6 mm section showed at ~220-360 px.

Task OFF means its outputs are inputs: reorder off → discovery order is fixed;
position off → positions come from the host and are facts; transform off →
no transform or orientation tools, with existing orientations and transforms
supplied by the host or checkpoint.

## State

`StackState` is the checkpoint, the result, and the thing every tool writes.
Same JSON shape for all three; the checkpoint adds `format_version` (1 since
2026-10-03; an unversioned checkpoint is read as it is, and a newer format is
refused).

The job layer (`linear/job.py`, 2026-10-03) owns it: one `Job` holds the
state and the spec, the host's locked and damaged sections, undo/redo, the
checkpoint and its observers, the submit gates, the stale-deformation rule,
the deformation records and the background image corrections. `Job.open`
resumes the checkpoint (`spec.resume`) or ingests the folder and applies the
host's inputs. The ADK tools and the MCP server both sit on it. A write is
one pattern: `before = job.snapshot()`, write, `job.commit(before)` (one undo
step, then the checkpoint). The undo history (depth 50) is the job
folder's `history/`: `index.json` lists the undo and redo steps, oldest
first, and each step is one whole-state file written once, so undo reaches
back across a resume; a fresh run empties it. Before every tool call the job
reloads a state file changed on disk since it last read or wrote it (a
script, or a second job on the same folder), so a script's edit is picked up
mid-run instead of overwritten; the edit becomes one undo step (a second
job's own history is read back instead).

Everything a job writes lives in its job folder next to the images,
`<images>/langslice/` (2026-10-03, `src/langslice/job/CLAUDE.md`):
`job.json` (the spec and the folder's format version), `state.json` (the
checkpoint, state format 2), `history/`, `sections/<stem>/` (the section's
deformation records, image corrections and the pictures of it the model
was shown), `views/` (pictures of several sections), `views.jsonl` (every
saved picture), `exports/` (`linear_results.json`) and `logs/`. Every path
the state stores is relative to the job folder. Every picture the model is
shown is saved there as the JPEG it received; a placement picture adds its
atlas labels (uint32 TIFF), its border mask (PNG) and its frame
(`view.json`: plane, µm per pixel, placement and the pixel-to-atlas matrix
in BrainGlobe micrometres), from which `langslice.core.layers.coordinate_map`
computes each pixel's atlas position. An old layout (the files beside the
images, or a saved Claude job under `~/.langslice/jobs/<id>/`) is moved in on
open; a newer one is refused. `JobSpec.job_dir` (`--job-dir`) puts the job
folder elsewhere, refusing one that holds another image folder's job; a
read-only image folder falls back to `~/.langslice/jobs/<id>/`, said once in
the run log and recorded in the id's index entry.

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
Direction is never the agent's concern: a stack cut posterior-first yields
the same sections as one cut anterior-first, so at submit the code reverses a
descending stack (`normalize_to_atlas_order`) and the emitted corrected order
always runs the atlas way, positions increasing from the atlas origin end.
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

**Code layout (layered refactor, 2026-10-03).** A tool is a door: argument
checking, the look-before-commit gates, one operation and the wording. The
writes are operations (`src/langslice/ops/`: positions, order, orientation,
damage, appearance, notes, transforms, the deformable fit and `keep_linear`),
the job (`linear/job.py`) owns state, undo, checkpoint and the submit gates,
and the pictures are built by the core (`src/langslice/core/`: `pictures.py`
for the viewing tools, `placement.py` for every placement picture, whose
`draw_canvas` also returns the `CanvasFrame` the picture was drawn in). The
tools return plain PIL pictures; the ADK driver packages them as JPEG message
parts (`adk/media.py` `packaged`) and the MCP server as image blocks, so the
toolbox imports no model SDK.

| tool | task gate | does |
| --- | --- | --- |
| `status` | always | one row per section in corrected order: index, id, position_mm, delta_to_next_mm (signed), flip, rotation_deg, damaged(+note), transform kind, transform_iou, transform_mirrored, caveats; plus cutting angles and interval breaks. The `ls` of the environment. |
| `view_slices(slices, view)` | always | up to 4 sections at the later-picture size, rendered as corrected, each captioned with its index and filename. Mode `section` (the section in `view.channels`; `zoom` crops a larger render, magnifying up to the section's working copy) or `channels` (per section one strip of small tiles, one per raw channel, unmodified and labelled; `channels` in the reply lists them). |
| `view_atlas(positions_mm, view)` | always | up to 4 atlas sections, rendered at the current cutting angles, each captioned with its position (and the angles when oblique), framed to the anatomy. Mode `template`; atlas channels default `[ara]` (add `borders` for lines); `regions` draws those regions' borders; `regions_not_in_plane` names any absent at a position. |
| `note(text)` | always | append to the run notes. |
| `undo()` / `redo()` | always | the job's snapshot history, saved beside the checkpoint and kept across a resume; a batch call undoes as one. |
| `orient_slices([{id, flip?, rotate_deg?}], view)` | transform (rotate) / transform.flip (flip) | set flip and rotation; an orientation change clears the section's transform; returns each changed section rendered as it now stands (≤4). |
| `reorder_slices(slices, after="start")` | reorder | move the listed filenames as a block, in the listed order, after a named section or at the start. One filename moves one slice; the full list sets the whole order. Unlisted sections keep their relative order. Corrected indices only; positions and transforms are kept. One undo step. |
| `mark_damaged([{id, damaged?, note?}])` | agent_damage (default on) | set damage (default True), or clear with damaged=False; clearing also removes the note. A flag the host set (`inputs.damaged`) is never cleared (`DAMAGE_SET_BY_USER`). |
| `preprocess(target="both", slices=[], channel_weights=[], clahe_clip=4, clahe_tiles=8, n4=False, denoise=False, reset=False, view)` | agent_preprocessing (default off) | sets the appearance for target `view` (every picture shown), `fit` (what a deformable fit reads, `appearance.fit_image`) or `both`, stack-wide or per section (overrides); from the raw channels: optional ANTs N4 and denoising per weighted channel, CLAHE (clip 0 = off), weighted blend (empty weights = tissue-coverage auto). `reset` returns to the default (a section to the stack's). Refuses a weight count that does not match a section's channels (`CHANNEL_COUNT_MISMATCH`) and N4/denoise without antspyx (`UNAVAILABLE`). One undo step, checkpointed (`StackState.appearance`); returns the settings in force, the channels and, for up to 4 sections, two labelled pictures each: BEFORE (the target's appearance before the call) and AFTER (as it now is), mapped by `image_indexes`. |
| `set_positions([{id, position_mm}], view)` | position | batch write, clamped to the atlas range; returns a placement picture (any `view_placement` mode, default `stacked`; mapped by `image_indexes`) only when that exact section, position, orientation and cutting-angle combination has not already reached the model in a full-canvas atlas-bearing view. A compare and write planned in the same model round both return their images. |
| `view_placement([{id, positions_mm?}], view)` | position, transform or nonlinear | the complete current registration, or ≤4 candidate pairs; no positions = current position. Physical modes (`template` default, `overlay`, `checkerboard`, `outlines`, `section`) draw the section under its STORED in-plane transform (the six numbers, spline included; identity without one; `transform` in each row) on the millimetre canvas. `stacked` is one picture, the framed section over the framed atlas; `side_by_side` returns separate original section and atlas reference images (one section per distinct id, one atlas per pair; ≤8 images). Both framed modes are full-view only. Writes nothing. A section's applied deformation is drawn too (the section resampled by the warp; `deformation_drawn` in the row) at the position it was fitted at, in every mode that draws the section under its placement (`overlay`, `checkerboard`, `outlines`, `section`), unless `view.deformation` is `none`. |
| `view_stack(view)` | position | one contact sheet of every section in the order of its written position, each over the atlas at its position and captioned with index, filename, position and the distance to the next, plus a plot of position against corrected index (damaged in red): two images. Mode `stacked`; no zoom. Writes nothing. |
| `run_deepslice(slices, allow_angle_change, keep=[ids])` | position.deepslice | positions (+ angles) for undamaged sections; UNAVAILABLE unless installed and plane/atlas supported. |
| `search_position(id, window_mm, angles?)` | position.bayesian | `oblique.fit_oblique` at the section's current position: best position (and angles) with score; writes nothing. |
| `set_cutting_angles(pitch_deg, yaw_deg)` | transform.angles | stack-wide; subsequent atlas fetches and fits use them. |
| `fit_affine(slices, method=elastix\|silhouette, fit_atlas="", include=[], exclude=[], view)` | transform | per-section in-plane affine against its atlas section, written as the section's transform (`kind` = the method). `elastix` (default, 2026-10-03) is a local refinement of the section's CURRENT transform (identity without one), never a search from scratch: the deformable package's stain-fit inputs (the `fit` appearance against `fit_atlas`, the ARA template by default or ABBA's Nissl where installed, tissue and atlas masks, the edge channel, excluded regions blanked; an intact section gets no torn-edge band) and an Elastix `AffineTransform` (mutual information + edges, fixed seed and threads, so identical inputs give identical numbers; `transform.fit_elastix`, `deformable.engines.run_elastix_affine`). A full affine: the six stored numbers carry its shear exactly, and `physical` reports it as `shear`. `silhouette` is the moments fit of the outlines from scratch. Returns iou (for `elastix`: tissue against the kept atlas footprint under the fitted placement), the transform as the same five `physical` knobs `adjust_transforms` takes (plus `shear`, about the canvas centre) and a captioned panel (default `overlay`; any physical mode) for every successful fit, mapped by `image_indexes`. Damaged sections are refused unless regions are given. `include`/`exclude` (as in `fit_deformable`, one-sided entries such as `"CTX:left"` included) restrict the fit to the kept atlas regions and the tissue the current placement lays on them — silhouette on the true-scale canvas (`transform.region_silhouette_fit`), elastix through the same masks `fit_deformable` builds — with a `regions` report; without them the fit is unchanged. The silhouette fit reads the default appearance, the Elastix fit the `fit` appearance. |
| `adjust_transforms(entries, view)` | transform.interactive | set one to four independent sections, each with rotation, per-axis scales, millimetre shifts and an optional `shear` (2026-10-03; `affine.decompose_affine`'s convention: linear part `R(rotation) . [[scale_x, shear*scale_x], [0, scale_y]]`, a unitless slant before the rotation in units of `scale_x`, the number `fit_affine` reports; left out, the section's current shear is kept, 0 when it has none; an explicit value, 0 included, sets it). Per-entry pivot and note; one `view` draws every entry (modes as the physical views plus `ab`); `ab` and `side_by_side` return two images, other modes one. Each result maps its images with `image_indexes`. One undo step; repeat unchanged parameters to redraw. Replaces the complete transform, including any spline; a shear left out is kept. Inspect before a dependent correction in a later call. |
| `trace_borders(id, prompt)` | nonlinear, unless provider `none` | one image-model correction (agent edits the base prompt per section) of the section's placed atlas borders, run in the background (submit waits), first reply kept; no fit, no transform change. See [the image-tool contract](nonlinear_image_tool.md). |
| `grep_atlas(query, id)` | nonlinear | text lookup of atlas regions by acronym, name substring or id (40 rows max): acronym, id, name, ancestry as acronyms, descendant count; with a positioned section `id`, whether each region or a descendant is in the atlas plane at its placement. Writes nothing; for choosing regions to exclude from a later fit. |
| `fit_deformable(slices, include=[], exclude=[], start="linear", fit_section="fit", fit_atlas="", engine="", stiffness="medium", candidates=[{stiffness, fit_section, fit_atlas, engine}], keep_linear="", view)` — `engine` only when `nonlinear.engine` is `either`; modes: borders, ab | nonlinear | a library deformable fit (ANTs SyN or Elastix B-spline, `src/langslice/deformable/`) on top of each section's linear placement. `include` restricts the fit to those regions plus a margin, `exclude` removes regions from the atlas side; either may name one side of the section, `"CTX:left"` / `"CTX:right"`; `start="current"` composes onto the applied deformation. `fit_section`: the `fit` appearance (one raw channel is a fit appearance `preprocess` sets) or (image model only) the section's `trace_borders` result at this placement (`traced_borders` = named regions, ANTs; `traced_lines` = lines vs borders); `fit_atlas` `ara`/`nissl` for the fit appearance, `borders` for traced fit sections only; stiffness soft/medium/firm; detail and line softening fixed. Defaults: fit appearance, ara, ANTs, medium; with a completed trace the description recommends traced_borders + ANTs + medium; a call waits up to 300 s for a trace still running and adds the trace drawn on the section (`traces`). `keep_linear="reason"` fits nothing and records that the named sections' linear placement stands. 2–4 `candidates` preview and write nothing; one setting applies (one undo step), reusing an identical cached result. Returns per result the final borders drawn on the section image (included strong, excluded pink; `view.atlas_channels` with `ara`/`nissl` blends that atlas image, warped, under the lines), displacement, fold fraction, flags and engine numbers. A later change to position, orientation, cutting angles or transform clears the deformation (`deformation_cleared` in that reply). |
| `submit(summary, notes, interval_breaks)` | always | ends the run; gated (below). |

**`view`: the picture options** (`linear/view_options.py`: `parse_view`
validates one call's `view` against the tool's `Profile` into a
`linear/display.py` `DisplayOptions`, which the renderers draw; the same typed
object, `arguments.View`, on every picture tool; `ViewAuto` adds `resolution`
at image resolution `auto` via `view_options.view_schema`):

| key | values | applies to |
| --- | --- | --- |
| `mode` | per tool (above); the first is its default | which composition |
| `channels` | raw channel names (each stretched by percentile; one gray, several added in distinct colours, `view.channel_colors`), or ONE version, `view` (default) / `fit` | every section picture; mixing → `BAD_CHANNELS`; unknown → `UNKNOWN_CHANNEL`; refused where no section is drawn |
| `atlas_channels` | any of `ara`, `nissl` (ABBA's cached Allen Nissl; otherwise `ATLAS_CHANNEL_UNAVAILABLE`), `borders` (the family boundaries as lines) | images under the section at `atlas_opacity` (overlay, ab, outlines, borders) or as the atlas picture; two images in two colours (`view.atlas_colors`); no `borders` = no lines. Defaults: `[borders]` overlay/ab/outlines/borders, `[ara]` template/stacked/framed side_by_side, `[ara, borders]` checkerboard/physical side_by_side |
| `atlas_opacity` | 0..1; default 0.5 when an image is listed | the blending modes only, with an image listed |
| `regions` | acronyms or ids, descendants included; `"CTX:left"` / `"CTX:right"` for one side of the section as shown | those regions' borders at full strength; `borders` lines then at 0.35 strength; `UNKNOWN_REGIONS` (also a bad side); `NO_SIDES` on a sagittal stack; `regions_not_in_plane` names absent ones |
| `outlines` | `all` (default), `outer` | which lines `borders` draws; refused without `borders`; `none` refused with the fix |
| `border_color`, `border_thickness` | named or `#RRGGBB` (yellow); 0.25..8 output px (default 1) | every drawn line; refused when no line is drawn |
| `zoom` | `[x0, y0, x1, y1]` fractions; empty = all | crop before sizing (physical canvas, framed sections, framed atlas); refused on `stacked`/`side_by_side`/`view_stack` |
| `deformation` | `applied` (default), `none` | `view_placement`/`set_positions` modes that draw the section under its placement |
| `resolution` (auto only) | long edge in px, 128 up to the driver model's largest image (the door passes it: 2048 on the OpenAI lanes, 2000 for a Claude host, 2026-10-03; was a fixed 1536); 0 = 512; clamped with `view.resolution_note` | each picture of the call (each tile of `view_stack`); at every other level not in the schema and refused with the reason |

A key that means nothing for the tool or the mode answers `VIEW_KEY_UNUSED`
with the reason per key; an unknown key (in `view`, at the top level, or in
any `entries`/`candidates` dict) answers `UNKNOWN_ARGUMENTS`
(`arguments.argument_refusal`, applied by the toolbox wrapper, the ADK
`StrictArgumentsPlugin` and the MCP server). Options apply to their call
only: nothing in `display.py` or `view_options.py` writes state. The job statement describes `view`
once, with this run's raw channel names and the atlas channels this host has
(`prompt.display_facts`, `prompt.display_lines`); tool descriptions list only
their modes.

**Raw channels and appearance** (`linear/appearance.py`). A section's DEFAULT
appearance is today's: `preprocess` auto (`adaptive_preprocess`) for a plain
file, or the host's blend (`host_preprocess`) when `host_preprocessing` is set
— the ABBA worker no longer stages blended copies; it passes its settings and
the run reads the snapshots themselves. The raw channels stay readable
(`Workspace.section_channels`: red/green/blue, or `gray`, for one page;
one plane per page, named by `inputs.channel_names` or ch1.., for several),
read at working size (`image_prep.read_working_pages`: pyramid level or
per-page downsample, never a whole-slide decode). `render_slice(look=...)`
draws any other look from those channels over the same frame, crop and size,
so geometry never depends on appearance. Pictures use the view look; the
silhouette fit, calibration, the tissue pivot, `search_position` and
`trace_borders`' input (`registration_handoff`) always read the default.

`view_atlas` and `view_slices` frame tissue the same way so apparent scale is
not a cue. Every image a tool returns carries its label burned into the pixels
(`render.caption`): tool images reach the model as bare attachments, so the
text that ties an image to a section or a position has to ride in the image.
The image a fit MEASURES is never captioned — only what is shown. Seed
message: every section as its own labelled image in corrected order plus the
status table.

The default full-task toolbox contains 15 tools (11 for interactive-only
transform refinement: `orient_slices` and, since 2026-10-03, the read-only
`view_placement` included); `agent_preprocessing` adds `preprocess`.

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
- nonlinear on: every section carries a deformation applied at its current
  linear placement, or a `keep_linear` reason given through `fit_deformable`
  (`MISSING_DEFORMATIONS`, naming each section and why). With an image model,
  a completed image correction at the current placement is required first
  (`MISSING_IMAGE_CORRECTIONS`; submit waits for running calls); with provider
  `none` traces are neither built nor required.

Refusals state the numbers and stop.

When interactive transforms are enabled, the prompt asks for inspection of each
fit or adjustment against surviving internal anatomy, then refinement of each
slice, damaged slices included, until no further improvement is possible with the available transforms.
Changes should be kept only if they improve alignment. This adds no mandatory
adjustment count or final-review hook. With a deformable fit, Method asks that
every region a section still has drive its fit (exclude the lost regions rather
than include a few survivors) and that candidates be compared before applying
(2026-10-03: on M11_B_08/C_08 Astra included only MB/HPF and applied one
setting unseen, where on M04_C_08 it excluded the lost cortex and compared;
Nash: "Astra knows, it was just lazy").

### Linear and nonlinear scope

The linear agent supplies order, position and affine alignment. The optional
`nonlinear` task's job (2026-10-01) is a deformation per section on top of its
linear placement, with the stain (and, with the image model, the traced
borders) as evidence: it builds `grep_atlas` and `fit_deformable`, which fits
and stores a deformation per section (`SliceState.deformation`, the record
under `<results dir>/deformable/`; see the tool table), or records with
`keep_linear` that a section's linear placement stands. Submit requires one or
the other for every section. Export adapters are to read that record; none
exist yet.

With an image model (any `nonlinear.provider` but `none`) the task also
exposes `trace_borders(id, prompt="")`. The top-level registration bridge
sends placed borders plus the clean photograph to the image model with the
agent's per-slice edited copy of the base prompt. The first result is retained
without agent selection or rejection. Submit also requires a completed
correction at each section's current placement. No deformation is fitted and
no transform is changed by this tool; `fit_deformable` can read its lines. See
[the image-tool contract](nonlinear_image_tool.md). With provider `none`
(CLI `--image-provider none`) there is no `trace_borders`, no traced section
image and no trace gate, and the job statement never mentions them.

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

The established working-set policy is the only active policy: one
500-to-250 image-count backstop (the transform-media stage cut was removed
2026-10-03, matching Codex CLI and Claude Code, which keep images until a
hard request limit).
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

## The write-observer hook and the ABBA host adapter (2026-09-10)

Host adapters were open until 2026-09-10: `checkpoint.py` exposes
`observe_checkpoints(fn)`, a context manager that registers `fn` to be called
with the state right after every checkpoint write — the one point every
write funnels through (`Job.checkpoint`, since 2026-10-03; the job also
calls them after reloading a state file changed on disk). A job has its own
observers too (`Job.observe(fn)`). `engine.run(spec, ..., on_write=fn)`
registers `fn` on the run's job for the whole session and calls it once more
up front, on the opened state, so a host sees the stack before the agent has
touched it. The agent itself never knows a host is watching:
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
omitted). Borders default to yellow, 1 anti-aliased output pixel (0.5 until
2026-10-03, when 0.5 px was seen to fade into bright tissue), without a rim.
Every picture tool takes `view.border_color` (named color or `#RRGGBB`) and
`view.border_thickness` (0.25–8 output pixels, including fractional widths)
alongside `view.atlas_opacity`. Lines are drawn after crop/resize so zoom does
not change their pixel thickness. These controls affect display only;
changing style with the same transform redraws without an undo step.
Automatic fit feedback uses the same default.
The separate native ABBA viewer retains ABBA's own display settings.
The contour
code (`region_contours`, `_smooth_closed`, family mapping, annotation slice
at cutting angles) moves from `nonlinear/` to `atlas/render.py` so both
methods draw the same lines from the same source.

**The screen.** `adjust_transforms` returns the transformed section
(display-preprocessed grayscale) with the family outlines on top at true
scale, a 1 mm scale bar, and a caption (section id, position, angles, the
params). `fit_affine`'s panels use the same renderer
(`core.placement.draw_canvas` over `render.physical_views`; was
`render.physical_overlay`, its `overlay` view), so every look at a section
is the same picture.

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
- `atlas_opacity` (0..1; `template_opacity` until 2026-10-01,
  `show_template` before that) dials the atlas image blended under the
  outlines in `overlay`. Since 2026-10-03 all three are keys of `view`, and
  the atlas image is blended only when `view.atlas_channels` lists one
  (`ara`/`nissl`; opacity then defaults to 0.5).

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
