# Interface design: layers, tasks and checkboxes

Target design, stated by the user on 2026-09-26. The "on main" notes say what
the code does today; `docs/linear_design.md` stays literal to that code.
Three things are separate and stay separate:

1. **Operations** — pure functions with no agent and no policy (silhouette
   affine, atlas section at a position and angle, border correction given an
   image plus a placement, deformation fit, export). This is the Python API and
   what a batch pipeline, a volume host or a future GUI calls directly.
2. **Agent environment** — the stack agent and its toolbox. Tools wrap
   operations; every policy (undo, submit gates, damage marks, "linear before
   nonlinear") lives here as a default.
3. **Hosts** — CLI, ABBA plugin, napari plugin, JSON-lines worker. A host
   fills a job spec, supplies what the user already did, and renders results.
   Hosts hold no registration logic.

Every checkbox below is one field of the job spec; a task that is OFF builds no
tools and takes its answer from the host. Nothing else is user-facing.

## Top level

| Control | Meaning | On main |
| --- | --- | --- |
| Provider | ChatGPT (default) or Claude | ChatGPT runs through `openai-oauth`; Claude changes Run to Copy prompt and uses the LangSlice MCP connector in Claude Desktop or Claude Code |
| Agent model, reasoning level | ADK model string and effort | `model`, reasoning |
| Image model / provider | image transport and model for nonlinear | `nonlinear.provider`, `nonlinear.image_model` |
| Image resolution: low / medium / high / auto | how large the pictures the agent sees are: a long edge for the opening images and one for every later picture (table below); auto lets the agent choose each later picture's size. Does not touch the image model's inputs | `image_resolution` low/medium/high/auto (`render.PICTURE_EDGES`); display only, fits and stored transforms unchanged |
| Estimated cost | shown at the bottom once every box is chosen | worker `linear.estimate` (`linear/cost.py`): percent of the usage window from measured runs; refused at medium/high/auto resolution, where nothing is measured (the low runs were measured before the 2026-10-01 sizes) |
| View agent log | the agent's activity in a window during the run | Fiji connector: a text log window (or a compact status window when off). The Python-started ABBA launcher has a richer browser log (`integrations/abba_chat.py`) |
| Save traces | full record of what the agent was shown, said and did, saved to a chosen folder | worker `trace_dir` (`LANGSLICE_TRACE_DIR` for one run); ABBA dialog checkbox + folder |
| Enable agent viewer | an ABBA-style brain display of the agent's work as it happens | built for the Python-started ABBA launcher (`integrations/abba_compare.py`, `abba_overview.py`, `abba_follow.py`); the Fiji connector shows the checkbox disabled until it is ported |
| Atlas, plane, preprocess | as today | `atlas`, `plane`, `preprocess` |

## 1. Positioning (absorbs reorder)

Ordering and positioning are one act once positions exist; reorder stays an
internal capability (the order only), not a user-facing task. Flipping moved
to Linear on 2026-09-29 (below).

| Control | On main |
| --- | --- |
| Slice thickness (may be provided by the host) | `position.thickness_um` |
| Slicing interval (ABBA often provides it) | `position.interval_um` |
| Enable DeepSlice tool (TODO; coronal mouse/rat only) | `position.deepslice` |
| Enable Bayesian optimizer tool (not implemented; needs design) | `position.bayesian` builds `search_position` |
| Extra notes for the agent, attached to this task | `position.notes`, shown under the task in the job statement (global `facts` remain) |

Positioning ON means the agent moves positions freely; there is no separate
"allow moving slices" option. The overwrite option below concerns transforms.

Dropped from the old spec: strict interval as a checkbox (the agent receives
interval and thickness as facts and may place breaks; strict stays a submit
gate a host can request), `gated` and `playbook` (coaching for cheaper models;
internal, never user-facing).

## 2. Linear (was transform)

| Control | On main |
| --- | --- |
| Enable hemisphere flipping, with optional text describing the hemisphere cue (moved here from Positioning, 2026-09-29) | `transform.flip`, `transform.hemisphere_cue` (`reorder.flip` / `reorder.hemisphere_cue` still accepted as aliases); `orient_slices` is built with the transform task |
| Enable affine tool. Off: every transform is X/Y movement and scaling by the agent | `transform.automatic` (+ `transform.elastix` for the intensity affine) |
| Max parallel slice transforms, 1 to 4. 1 = one section per call | `transform.max_parallel`; below 4, `fit_affine` and `adjust_transforms` refuse larger calls (at 4, `fit_affine` stays uncapped) |
| Enable slice angle estimation (yes/no); later tools: DeepSlice angle, Bayesian optimizer | `transform.angles` builds `set_cutting_angles` (manual only); the Fiji connector still refuses angle changes and shows the box disabled |
| Extra notes for the agent, attached to this task | `transform.notes` |

The interactive tool is always on. Damaged sections refuse the automatic
affine and require an applied interactive transform (unchanged).

**A mirror flip is a linear transform** (user decision, 2026-09-29, replacing
the 2026-09-26 placement under Positioning). A mirror is the sign of the
in-plane affine (negative determinant); a host's own alignment (ABBA) carries
it inside the transform; and a left-right symmetric mouse brain gives
positioning no way to see it. So flip and rotation (`orient_slices`) are
built with Linear, a positioning-only run never mentions mirroring, and a
host-supplied transform (`inputs.transforms`) may carry a mirror, which the
harness keeps as supplied. Flips are scored in the Linear step, not in
Positioning (SliceBench `mirror_accuracy`). Cutting
angles are stack-wide and belong here because they change the atlas section
every transform is fit against.

## 3. Nonlinear

Precondition: a linear placement for every section, either from the host
(`inputs.transforms`, the user's own ABBA alignment) or from task 2 in the same
run. Nonlinear ON with sections lacking a placement is a job-spec validation
error naming those sections, before any model call. A host turns it into a
dialog: "Slices a-z have no transform. Nonlinear requires a linear step first.
Allow the agent to align them?" Yes = Linear ON for those sections only. On
main this check is not in the spec yet: `trace_borders` refuses a
section without a usable placement one call at a time
(`INVALID_LINEAR_PLACEMENT`).

The deliverable is a registration, not drawings. Order of work: design the
deformation algorithm, then settle the border output format it consumes, then
test. Drawings remain the review artifact.

| Control | On main |
| --- | --- |
| Enable image-gen tool | task `nonlinear` builds `trace_borders` |
| Use agent (GUI). The agent writes per-slice notes for the image model. The agent-free path (fixed prompt as a plain operation over supplied placements) stays in the API only; it is the 3D-volume path, where notes have no purpose | notes are a tool argument; the `nonlinear` CLI is the agent-free operation |
| Deformable-fit engine: ANTs, Elastix or either | `nonlinear.engine` (`ants`, `elastix`, `either` = default, the agent picks per call); the `fit_deformable` tool is built with task `nonlinear` |
| Further tools: open | (none) |

Candidates for the open slot, none committed: a deformation-fit choice for
the output stage (none, B-spline, correspondence/TPS that preserves the model's
edits); a bounded redo with new notes per section.

## Cross-cutting rules (settled 2026-09-26)

- **Placement object.** Atlas + calibration, position, cutting angles, in-plane
  transform, and later the deformation. The pieces exist (slice record, stack
  state, transform record, the ABBA offset fitted at startup) but are assembled
  ad hoc by the nonlinear handoff. Name it once, validate it once; it is the
  contract every host produces and the deformation algorithm is designed against.
- **Host-supplied answers.** By default positions and transforms a host supplies
  are facts the agent cannot overwrite; the agent fills only what is missing. One
  option, "allow the agent to overwrite existing positions and transforms",
  lifts that for both together (never one without the other). When on, the host
  keeps the user's originals so the run is undoable. On main, transforms only:
  `inputs.locked` sections keep their flip, rotation and transform (a `"host"`
  identity, counted at submit) while their positions still move.
- **Angles.** A yes/no option in Linear: "Enable slice angle estimation". On
  means the agent may change the stack-wide cutting angles. Changing them
  invalidates every in-plane transform fit against the old atlas section, so
  the agent refits the transforms it owns. DeepSlice returns positions only
  unless its angle box is ticked.
- **Output format follows the fit (agreed).** The border output is settled only
  once the deformation algorithm's input is chosen: raster lines need extraction and
  matching, which discarded the model's placement in every Elastix fit, while
  the correspondence fit that preserved edits consumed vector paths with region
  identity. Decide what the fit consumes, then settle the format, then test.
- **Slice list (top level).** One row per section: damaged checkbox, host
  exclusion, has-transform. User damage checks win; the agent may add flags when
  "let the agent flag damage" is on, never remove a user's. On main:
  `inputs.damaged` (clearing refused), `agent_damage` builds `mark_damaged`.
- **One agent model, one image model,** top level. Per-task models rejected
  (cache complexity, no evidence cheap models hold on Linear).
- **Max parallel transforms** ships as a cap; the accuracy claim is unverified.

## Preprocessing (tab)

Tunes what the agent sees for the selected slices: per-channel CLAHE weights,
CLAHE off, and other preprocessing steps. Tip shown: "Try to maximize contrast
between different regions." On main: `image_prep.host_preprocess` blends a
snapshot's pages (one per channel) into the section's DEFAULT appearance:
auto, or custom per-channel weights, CLAHE on/off and strength. `linear.run`
takes it as `preprocessing` (the run's `host_preprocessing`); `preprocess.preview`
writes the same image for the dialog. The blend is an appearance, not a lossy
step: the snapshots are read as they are (nothing is staged), and their pages
stay available to the agent and to fitting as raw channels, named by the
optional `channel_names` (one per exported page) or ch1, ch2... The CLI keeps
`preprocess` auto/none.

| Control | On main |
| --- | --- |
| Let the agent drive preprocessing (checkbox, default off) | `agent_preprocessing` (worker `spec`, CLI `--agent-preprocessing`) builds the `preprocess` tool; off, what the agent views and what a fit reads both stay the default appearance above |

TODO (Fiji connector): the checkbox is not in the Java dialog yet. Adding it
means a `RegistrationSettings` field saved with the others, `spec.agent_preprocessing`,
and exporting EVERY channel (weight 0 included) when it is on, since custom
mode exports only weighted channels today; sending `channel_names` from the
dialog's channel list is the matching one-liner.

## Agent tool surface (locked 2026-10-01)

The linear stack agent's tools, as built (`linear/toolbox.py`). Every write is
undoable and checkpointed; every tool that returns a picture takes the shared
display options below.

| Tool | Built when | Signature |
| --- | --- | --- |
| `status` | always | `status()` |
| `view_slices` | always | `view_slices(slice_ids, +display)` — modes: section |
| `view_atlas` | always | `view_atlas(positions_mm, +display)` — modes: template (was `fetch_atlas`) |
| `note`, `undo`, `redo` | always | `note(text)`, `undo()`, `redo()` |
| `mark_damaged` | `agent_damage` | `mark_damaged(entries)` |
| `preprocess` | `agent_preprocessing` | `preprocess(target="both", sections=[], channel_weights=[], clahe_clip=4, clahe_tiles=8, n4=False, denoise=False, reset=False, +display)` |
| `reorder_slices` | reorder | `reorder_slices(new_order, after="start")` |
| `set_positions` | position | `set_positions(entries, +display)` — modes as `view_placement`, default stacked |
| `view_placement` | position | `view_placement(entries, +display)` — modes: template (default), stacked, side_by_side, overlay, checkerboard, outlines, section (was `compare_placement`) |
| `view_stack` | position | `view_stack(+display)` — modes: stacked |
| `run_deepslice` | position.deepslice | `run_deepslice(slice_ids, allow_angle_change, keep)` |
| `search_position` | position.bayesian | `search_position(slice_id, window_mm, angles)` (was `fit_position`) |
| `orient_slices` | transform | `orient_slices(entries, +display)` — modes: section |
| `fit_affine` | transform.automatic | `fit_affine(slice_ids, method, +display)` — modes: overlay (default), side_by_side, checkerboard, outlines, section, template |
| `adjust_transforms` | transform.interactive | `adjust_transforms(entries)`; each entry: id, rotation_deg, scale_x, scale_y, translate_x_mm, translate_y_mm, pivot, note, +display — modes as `fit_affine` plus ab |
| `set_cutting_angles` | transform.angles | `set_cutting_angles(pitch_deg, yaw_deg)` |
| `trace_borders`, `grep_atlas` | nonlinear | `trace_borders(id, prompt="")`, `grep_atlas(query, section="")` |
| `fit_deformable` | nonlinear | `fit_deformable(sections, include=[], exclude=[], start="linear", section_image="fit", atlas_image="", engine="", stiffness="medium", detail="standard", candidates=[], mode="borders", zoom, atlas_opacity, regions, outlines, border_color, border_thickness=1.0)` — `engine` only when `nonlinear.engine` is `either`; modes: borders, ab |
| `submit` | always | `submit(summary, notes, interval_breaks)` |

**`fit_deformable`** (`linear/deformation.py` over `src/langslice/deformable/`).
A library deformable fit on top of a section's linear placement (position and
transform required). Fit inputs: `sections` (≤4, ≤8 fits per call),
`include` (fit only these regions plus a 300 µm margin) and `exclude`
(removed from the atlas side), descendants included; `start` `linear` or
`current` (compose onto the applied deformation, region by region);
`section_image` `fit` (the fit appearance, `appearance.fit_image`), a raw
channel, `traced_borders` (the completed `trace_borders` result at this
placement as named regions, ANTs label-map mode) or `traced_lines` (those
lines against atlas borders); `atlas_image` `ara`, `borders`, `nissl` (same
host rules as the display options; empty = borders for traced images, else
ara); `engine` (only when the user left it open; missing ANTs is said plainly),
`stiffness` soft/medium/firm/stiff, `detail` coarse/standard/fine. 2–4
`candidates` (setting variants) run concurrently and write nothing; exactly one
setting applies it (one undo step, checkpointed), reusing an identical earlier
result. Pictures: one per result, the final borders drawn smoothly on the
section image the fit read, at the call's picture size, included (or
`regions`) borders strong, excluded regions pink; `ab` adds what the fit
started from. Text: settings and engine numbers, displacement max/median,
fold fraction, plausibility flags. The deformation is stored per section
(`SliceState.deformation` plus a record directory under the results folder);
any change to that section's position, orientation, cutting angles or
transform clears it and the tool's reply says `deformation_cleared` (undo
restores both). `view_placement` and `set_positions` draw the current warp in
their physical modes. Export adapters (ABBA, VisuAlign, BrainGlobe) will read
the saved record; none is built.

**Shared display options** (`+display`; one parser, `linear/display.py`; the
same names on every picture tool and on each `adjust_transforms` entry; a
call's options never change any stored setting):

- `mode` — the tool's compositions (above).
- `zoom` — `[x0, y0, x1, y1]` fractions; cropped before sizing, so it
  magnifies up to the section's own pixels (its working copy). Refused on
  tissue-framed pair
  pictures (`stacked`, `side_by_side`) and `view_stack`.
- `section_image` — `current` (the section's view appearance, default) or one
  raw channel by name, unenhanced.
- `atlas_image` — `ara` (BrainGlobe reference, default), `borders` (the atlas
  family boundaries as lines) or `nissl` (ABBA's cached Allen Nissl). Hosts
  with ABBA's cached atlas (`~/cached_atlas`, matching the run's atlas) get
  nissl; others get ara and borders, and asking for nissl answers
  `ATLAS_IMAGE_UNAVAILABLE` with the available list.
- `atlas_opacity` — 0..1, the atlas image blended under the lines over the
  section (`overlay`); was `template_opacity`.
- `regions` — atlas acronyms or ids, descendants included: only these
  regions' borders are drawn at full strength, the `outlines` layer faint
  behind them for context; regions missing from a plane are named in
  `regions_not_in_plane`.
- `outlines` — `all`, `outer` or `none`; empty is the mode's default (all on
  the physical canvas, none on tissue-framed pictures).
- `border_color`, `border_thickness` — named or `#RRGGBB`; 0.25..8 output px.
- `resolution` — only when the user chose image resolution `auto`: the long
  edge in pixels of each picture this call returns (of each section tile in
  `view_stack`), 128..1536, 0 = 512. Out of range is clamped and the reply's
  `view.resolution_note` says so. At every other level the argument does not
  exist and picture size never appears in model-facing text.

**Picture sizes** (`image_resolution`, `render.PICTURE_EDGES`). Each size is
the long edge of one picture, or of each panel of a multi-panel picture:

| level | opening images (seed message) | every later picture |
| --- | --- | --- |
| low (default) | 256 px | 512 px |
| medium | 384 px | 768 px |
| high | 512 px | 1024 px |
| auto | 256 px | the agent's `resolution` per call, 128..1536 px; 512 px when it gives none |

Nothing is drawn larger than its source: a section never past its working
copy (a small snapshot stays small), an atlas image never past the plane's
own voxels except where it sits under a section in one picture (`stacked`,
`view_stack`), where it is drawn to the section's size. The `view_stack`
contact sheet draws each tile at the opening size and shrinks the tiles until
the sheet is at most 2048 px (a 40-section stack lands near 250 px tiles at
every level). Fits and stored numbers are computed on fixed working frames
(512 px for affine fits, 1536 px for deformable fits), never on a picture.
Captions wrap onto as many lines as a small picture needs.

**Appearance (`preprocess`).** Two targets, set independently: `view` (every
picture the agent is shown: seed on resume, views, write pictures) and `fit`
(what a deformable fit reads). Each holds a stack-wide setting and per-section
overrides; with no call both are the default appearance (`preprocess` auto
CLAHE, or the host's blend), so nothing changes. A setting is built from the
raw channels: per weighted channel optional ANTs N4 bias-field correction and
denoising (antspyx, the optional `registration` extra, imported only when
asked; without it the tool answers `UNAVAILABLE`), CLAHE (clip 0..40, tiles
1..32), then the weighted blend (no weights = automatic by tissue coverage).
The tool returns the affected sections as that target now sees them. The
image model's input (`trace_borders`), the silhouette fit, calibration and
`search_position` always use the default appearance.

**Raw channels from host to views and fit.** Host snapshot pages (or a plain
file's red/green/blue) are read once at working size
(`image_prep.read_working_pages`: a pyramid level or a per-page downsample,
never a full-size whole-slide decode) and kept as named 8-bit planes
(`EngineContext.section_channels`). The default appearance is drawn from the
same working copy, so every look shares one frame, crop and size; a look only
changes intensities.

## ABBA host (settled 2026-09-28)

- Menu: ABBA's Register menu, like DeepSlice: **Register > LangSlice > LangSlice
  Registration…**. ABBA appends external entries at the end of that menu, so it
  cannot sit directly beside DeepSlice.
- Selected slices are the slices sent to LangSlice.
- Existing transforms: any ABBA registration step counts as linear; spline
  steps (BigWarp, Elastix spline) are the only nonlinear ones.
- In ChatGPT mode, results land in the user's ABBA when the agent is done, as one undoable step;
  a stopped run can apply its last checkpoint. The agent viewer, once ported to
  the connector, shows the work along the way.
- Nonlinear is a facade in ABBA for now: shown, does nothing.
- DeepSlice and Bayesian checkboxes are added later.
- No over-saturation warning.

### Claude mode

Choose **Claude**, configure Positioning / Linear, and click **Copy prompt**.
The dialog exports the same calibrated snapshots and preprocessing settings as
Run, saves a local job, and copies a Python-generated prompt. Paste it into
Claude Desktop or Claude Code with only the LangSlice connector enabled.
No Claude credentials are collected by LangSlice and no ChatGPT sign-in is needed.
Agent model, reasoning and image model controls are disabled; Nonlinear remains
unavailable. Image resolution and preprocessing still control LangSlice's pictures
(auto gives the `resolution` argument to Claude's tools).
Usage belongs to Claude, so there is no LangSlice cost estimate.

Keep the progress window open: section changes appear live in ABBA through its
native actions. Close or Disconnect ends only the live connection, not Claude's
work; checkpoints and completed results remain in the saved job directory.
Each live checkpoint is an undoable ABBA step, unlike ChatGPT's single final apply.
Avoid editing the selected slices until the connection is finished.
The log shows LangSlice activity, not Claude's conversation. Saved MCP traces
likewise record tool calls and pictures shown, not Claude's intervening words.

## Presets a host may expose

Named combinations of tasks-on and inputs-supplied. Four cover today's users:

- **Full auto** — positioning, linear, nonlinear.
- **Correct my placement** — nonlinear only; host supplies positions and transforms.
- **Position only** — the ABBA workflow; user aligns and registers in ABBA.
- **Borders only, no agent** — the nonlinear operation over host placements.

## Left open on purpose

3D light-sheet volumes: a volume host samples cross sections, the border
correction operation runs on each, and a fitting stage to be designed turns the
outputs into a volume registration. Same pipeline shape as sections; only the
sampling and the fit differ. Nothing in the layers above prevents it.
