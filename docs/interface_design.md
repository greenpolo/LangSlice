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
| Image model / provider | image transport and model for nonlinear, or none | `nonlinear.provider`, `nonlinear.image_model`; provider `none` (CLI `--image-provider none`) runs the Nonlinear task without the image model: no `trace_borders`, deformations fitted to the stain alone |
| Image resolution: low / medium / high / auto | how large the pictures the agent sees are: a long edge for each section in the opening strips (more strips at a higher level) and one for every later picture (table below); auto lets the agent choose each later picture's size. Does not touch the image model's inputs | `image_resolution` low/medium/high/auto (`core.sizes.PICTURE_EDGES`); display only, fits and stored transforms unchanged; CLI `--image-resolution` |
| Estimated cost | shown at the bottom once every box is chosen | worker `linear.estimate` (`agent/cost.py`): percent of the usage window from measured runs; refused at medium/high/auto resolution, where nothing is measured (the low runs were measured before the 2026-10-01 sizes) |
| View agent log | the agent's activity in a window during the run | Fiji connector: a text log window (or a compact status window when off). The Python-started ABBA launcher has a richer browser log (`hosts/integrations/abba_chat.py`) |
| Save traces | full record of what the agent was shown, said and did, saved to a chosen folder | worker `trace_dir` (`LANGSLICE_TRACE_DIR` for one run); ABBA dialog checkbox + folder |
| Enable agent viewer | an ABBA-style brain display of the agent's work as it happens | built for the Python-started ABBA launcher (`hosts/integrations/abba_compare.py`, `abba_overview.py`, `abba_follow.py`); the Fiji connector shows the checkbox disabled until it is ported |
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
| Enable affine tool. Off: every transform is X/Y movement and scaling by the agent | `transform.automatic` (`fit_affine`: Elastix intensity affine by default, silhouette as an option; the former `transform.elastix` switch is gone) |
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
main this check is not in the spec yet: `trace_borders` and `fit_deformable`
refuse a section without a usable placement one call at a time
(`INVALID_LINEAR_PLACEMENT`).

The agent's job (2026-10-01): give every section a deformation onto the atlas
on top of its linear placement, with the stain (and, with the image model, its
traced borders) as the evidence. `submit` refuses (`MISSING_DEFORMATIONS`)
until every section carries a deformation applied at its current placement or
a `keep_linear` reason (`fit_deformable(sections, keep_linear="why")`) saying
its linear placement stands. With the image model, traces are still required
first (`MISSING_IMAGE_CORRECTIONS`).

The deliverable is a registration, not drawings. Order of work: design the
deformation algorithm, then settle the border output format it consumes, then
test. Drawings remain the review artifact.

| Control | On main |
| --- | --- |
| Enable image-gen tool | task `nonlinear` builds `grep_atlas` and `fit_deformable`, and `trace_borders` unless `nonlinear.provider` is `none` |
| Use agent (GUI). The agent writes per-slice notes for the image model. The agent-free path (fixed prompt as a plain operation over supplied placements) stays in the API only; it is the 3D-volume path, where notes have no purpose | notes are a tool argument; the `nonlinear` CLI is the agent-free operation |
| Deformable-fit engine: ANTs, Elastix or either | `nonlinear.engine` (`ants`, `elastix`, `either` = default, the agent picks per call; CLI `--engine`); the `fit_deformable` tool is built with task `nonlinear` |
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

## Surfaces: one vocabulary, four doors (phase 5, 2026-10-04)

Every verb below has ONE name, one argument list and one description,
declared once (`doors/declarations.py`) and listed once with its operation,
read or write, task group and the specs that have it (`ops/registry.py`,
`VERBS`, `enabled(spec)`). Every door is generated from those two:

| Door | Driver | What it gets |
| --- | --- | --- |
| Agent tools (ADK) | LangSlice's own agent | the verbs `enabled(spec)` names, declared for the run (`build_tools`); pictures as message images; the look-before-commit gates |
| MCP (`langslice mcp`) | Claude Desktop, Claude Code locked to the server | the same tools (plus the door's own `start_job`, `show_stack`); pictures as image blocks; `readOnlyHint` on the read verbs; with Nonlinear, `trace_borders` only when the job's image model is connected (its key or login present), else not listed and the statement says why |
| Agent CLI (`langslice job FOLDER VERB`, `ops`, `schema`) | a coding agent (Claude Code, Codex) | the same tools on a job folder, one JSON envelope per call, pictures as saved file paths, no gates, any picture size (`docs/agent_cli.md`) |
| Library (`import langslice`) | a script, a scripted pipeline | `langslice.open_job(folder)` / `create_job(images, ...)`: the same verbs as methods; `register_section` / `register_job`: the image model's trace and the deformable fit without the agent, with a model profile (`image_model`) and a lean job folder (`docs/library.md`); `coordinate_map`, `load_atlas` |

A verb is never renamed once shipped. Every job folder carries a reference
card for coding agents (`AGENTS.md` = `CLAUDE.md`, generated from the
registry, `doors/card.py`). Host connectors (ABBA, napari later) stay
separate: a person drives them.

## Agent tool surface (locked 2026-10-01; one shape per tool 2026-10-03)

The linear stack agent's tools, as declared (`doors/declarations.py`; the
bodies are `doors/tools/toolbox.py`). Every write is
undoable and checkpointed. Operation arguments are top-level; every tool that
returns a picture takes ALL its picture options in one argument, `view`
(below). The slice list is `slices` on every tool, a single section is `id`
(as in every `entries` dict), and arguments, `view` keys and entry keys are
typed (`doors/tools/arguments.py`), so the tool schema the model is sent names
every key and its type.

| Tool | Built when | Signature (top level) | `view` modes |
| --- | --- | --- | --- |
| `status` | always | `status()` | — |
| `view_slices` | always | `view_slices(slices, view)` | section (default), channels |
| `view_atlas` | always | `view_atlas(positions_mm, view)` (was `fetch_atlas`) | template |
| `note`, `undo`, `redo` | always | `note(text)`, `undo()`, `redo()` | — |
| `mark_damaged` | `agent_damage` | `mark_damaged(entries)`; entry: id, damaged, note | — |
| `preprocess` | `agent_preprocessing` | `preprocess(target="both", slices=[], channel_weights=[], clahe_clip=4, clahe_tiles=8, n4=False, denoise=False, reset=False, view)` | section |
| `reorder_slices` | reorder | `reorder_slices(slices, after="start")` | — |
| `set_positions` | position | `set_positions(entries, view)`; entry: id, position_mm | as `view_placement`, default stacked |
| `view_placement` | position, transform or nonlinear | `view_placement(entries, view)`; entry: id, positions_mm (was `compare_placement`) | template (default), stacked, side_by_side, overlay, checkerboard, outlines, section |
| `view_stack` | position | `view_stack(view)` | stacked |
| `run_deepslice` | position.deepslice | `run_deepslice(slices, allow_angle_change, keep)` | — |
| `search_position` | position.bayesian | `search_position(id, window_mm, angles)` (was `fit_position`) | — |
| `orient_slices` | transform | `orient_slices(entries, view)`; entry: id, flip, rotate_deg | section |
| `fit_affine` | transform.automatic | `fit_affine(slices, method="elastix", fit_atlas="", include=[], exclude=[], view)` | overlay (default), side_by_side, checkerboard, outlines, section, template |
| `adjust_transforms` | transform.interactive | `adjust_transforms(entries, view)`; entry: id, rotation_deg, scale_x, scale_y, translate_x_mm, translate_y_mm, shear (optional: the unitless slant `fit_affine` reports; left out, the section's current shear is kept, 0 sets none), pivot, note; one `view` draws every entry | as `fit_affine`, plus ab |
| `set_cutting_angles` | transform.angles | `set_cutting_angles(pitch_deg, yaw_deg)` | — |
| `trace_borders`, `grep_atlas` | nonlinear (`trace_borders` not with provider `none`) | `trace_borders(id, prompt="")`, `grep_atlas(query, section="")` | — |
| `fit_deformable` | nonlinear | `fit_deformable(slices, include=[], exclude=[], start="linear", fit_section="fit", fit_atlas="", engine="", stiffness="medium", candidates=[], keep_linear="", view)`; candidate: stiffness, fit_section, fit_atlas, engine; `engine` only when `nonlinear.engine` is `either` | borders (default), ab |
| `submit` | always | `submit(summary, notes, interval_breaks)` | — |

**Strict arguments.** An unknown or misplaced argument, an unknown `view`
key, or an unknown key in any `entries` / `candidates` dict is refused
(`UNKNOWN_ARGUMENTS`) with the key named, where it belongs when it belongs
elsewhere (`` `mode` belongs inside `view` ``), and the accepted keys listed;
nothing runs. One rule (`arguments.argument_refusal`) is applied at every
door: the toolbox's own wrapper, the ADK plugin
(`agent.plugins.StrictArgumentsPlugin`; ADK drops unknown top-level arguments
before a tool runs) and the MCP server (`doors.mcp.server.strict_arguments`;
FastMCP drops them too).

**`fit_deformable`** (`core/deformation.py` over `src/langslice/core/deformable/`).
A library deformable fit on top of a section's linear placement (position and
transform required). Fit inputs: `slices` (≤4, ≤8 fits per call),
`include` (fit only these regions plus a 300 µm margin) and `exclude`
(removed from the atlas side), descendants included; an entry may name one
side, `"CTX:left"` / `"CTX:right"`, left and right of the section as the
tool's pictures show it (the oriented section, rotation and flip applied),
for damage on one side only; `start` `linear` or
`current` (compose onto the applied deformation, region by region);
`fit_section` (what of the section the fit reads) `fit` (the fit
appearance, `appearance.fit_image`; one raw channel is a fit appearance set
with `preprocess`), `traced_borders` (the completed `trace_borders` result
at this placement as named regions, ANTs label-map mode) or `traced_lines`
(those lines against atlas borders; both only with an image model, and a
call waits up to `TRACE_WAIT_S` 300 s for that section's trace still
running, answering `TRACE_TIMEOUT` / `TRACE_FAILED` otherwise, and adds one
picture per traced section of the trace's lines on the section, mapped by
`traces`); `fit_atlas` (what of the atlas the fit reads) `ara` or `nissl`
for the fit appearance (a Nissl-stained reference; same host rules as the
atlas channels), `borders` for traced fit sections only (the fit appearance
against borders is refused); empty = borders for traced fit sections, else
ara; `engine` (only when the user left it open; missing ANTs is said
plainly), `stiffness` soft/medium/firm. Defaults without a trace: fit
appearance, ara, ANTs, medium. A fit of the fit appearance compares the
section and the atlas image by local correlation (small 80 µm windows;
Elastix, which has none, uses mutual information) plus their edges, and with
ANTs also matches the tissue outline and empty ventricles found in the
section; these are fixed, not arguments (2026-10-02 stain ceiling test). The
same inputs always give the same warp; with a completed trace the agent
chooses, and the tool description states that traced_borders with ANTs at
medium is the recommended pairing. Detail (standard) and line softening
(60 µm) are fixed, not arguments. On the fluorescent LSD_910 sections of the
2026-10-01 ceiling test, the Nissl reference's outer edge sat 40-80 µm
inside the tissue's bright surface rim where ara followed the edge. 2–4
`candidates` (setting variants) run concurrently and write nothing; exactly
one setting applies it (one undo step, checkpointed), reusing an identical
earlier result. Pictures: one per result, the final borders drawn smoothly
on the section image the fit read, at the call's picture size, included (or
`view.regions`) borders strong, excluded regions pink; `view.atlas_channels`
adding `ara` or `nissl` blends that atlas image, pulled through the warp,
under the lines at `atlas_opacity`; `ab` adds what the fit started from.
Text: settings and engine numbers, displacement max/median, fold fraction,
plausibility flags (regions compressed, expanded, vanished or folded;
`DISPLACEMENT_OUTSIZED` when the largest displacement passes a tenth of the
tissue's extent or the median passes 0.6 mm). The deformation is stored per
section (`SliceState.deformation` plus a record directory in the section's
folder of the job folder, `<images>/langslice/sections/<stem>/deformable/`);
any change to that section's position, orientation, cutting angles
or transform clears it and the tool's reply says `deformation_cleared` (undo
restores both). `view_placement` and `set_positions` draw the applied warp
in every mode that draws the section under its placement (`overlay`,
`checkerboard`, `outlines`, `section`; `view.deformation` `none` shows the
linear placement alone). The job folder's maps and VisuAlign markers read
the saved record (`docs/file_formats.md`); ABBA and BrainGlobe adapters are
not built. `keep_linear` (a reason) fits nothing
and draws nothing: each named section records that its linear placement
stands (one undo step; cleared like a deformation when the placement
changes; satisfies `submit`); `view` or `candidates` beside it are refused.

**`fit_affine`** method `elastix` (default) refines the current placement by
an intensity affine against `fit_atlas` (`ara` default, `nissl` where ABBA's
atlas is installed; a non-default choice is stored on the transform as
`fit_atlas`); `silhouette` fits outlines and refuses `fit_atlas`.
**Regions** (2026-10-01). `include` / `exclude` mean what they
mean in `fit_deformable` and resolve through the same code (acronyms or ids,
descendants included, optionally one side: `"CTX:left"`). Without them the fit is the whole-outline moments fit,
unchanged. With them (`transform.region_silhouette_fit`) the atlas side is
the kept footprint (minus excluded regions; with `include`, only the part
within 300 µm of them) and the section side is the tissue the section's
CURRENT placement lays there, one pass, on the true-scale canvas. The reply
adds a `regions` report: kept atlas and used tissue fractions, the two axis
ratios, `outline_share` for `include`, and a note when the fit turns the
section more than 45° or an outline is nearly round (a moments fit turns the
tissue's long axis onto the atlas's). An included zone that never reaches the
atlas outline is refused (`REGIONS_INSIDE_OUTLINE`). With regions given,
damaged sections are fitted (the submit gate on damaged sections still wants
an interactive transform). Included regions are highlighted in the pictures
unless `view.regions` names others.

**`view`: the picture options** (`doors/tools/view_options.py`: `parse_view`
validates one call's `view` against the tool's `Profile` into
`core/display.py`'s `DisplayOptions`; the job statement describes
it once and each tool's description lists only its modes; a call's options
never change any stored setting). Its keys:

- `mode` — the tool's compositions (table above); the first is the default.
- `channels` — what of the SECTION is shown: one or more raw channel names
  (each stretched by percentile, 1st to 99.5th, on the whole working plane;
  one is shown in gray, several are added in distinct colours like ABBA's
  multichannel display, a channel named after a colour keeping it, and the
  reply's `view.channel_colors` says which is which), or ONE version:
  `view` (the agent's appearance, the default) or `fit` (the appearance
  registration reads). Raw channels and a version cannot be mixed
  (`BAD_CHANNELS`); an unknown name answers `UNKNOWN_CHANNEL` with each
  section's channels.
- `atlas_channels` — what of the ATLAS is shown: any of `ara` (the
  reference template), `nissl` (ABBA's cached Allen Nissl; hosts with ABBA's
  cached atlas at `~/cached_atlas` matching the run's atlas, otherwise
  `ATLAS_CHANNEL_UNAVAILABLE` with the available list) and `borders` (the
  family boundaries as lines). Images are blended under the section at
  `atlas_opacity` in overlay, ab, outlines and borders modes, and are the
  atlas picture in the others; two images are added in two colours (ara
  green, nissl magenta, `view.atlas_colors`). No `borders` means no lines.
  Defaults reproduce each mode's earlier picture: `[borders]` in overlay, ab,
  outlines and borders; `[ara]` in template, stacked and the tissue-framed
  side_by_side of `view_placement`/`set_positions`; `[ara, borders]` in
  checkerboard and the physical side_by_side.
- `atlas_opacity` — 0..1, default 0.5 when an atlas image is listed in a
  mode that blends it.
- `regions` — atlas acronyms or ids, descendants included: these regions'
  borders at full strength, any `borders` lines faint behind them; regions
  missing from a plane are named in `regions_not_in_plane`. `"CTX:left"` /
  `"CTX:right"` draws one side: the section's side as `view_slices` shows it
  (on `view_placement`'s canvas a mirrored placement shows it on the other
  side, said in the caption); on a picture of the atlas alone, the picture's
  own side.
- `outlines` — `all` (default) or `outer`: which lines `borders` draws.
- `border_color`, `border_thickness` — named or `#RRGGBB` (default yellow);
  0.25..8 output px, default 1 everywhere (picked by eye on M11_B_03,
  2026-10-03: 0.5 px faded into bright tissue, 1.5 px began to cover
  ventricle edges).
- `zoom` — `[x0, y0, x1, y1]` fractions; cropped before sizing, so it
  magnifies up to the section's own pixels (its working copy). Refused on
  tissue-framed pair pictures (`stacked`, `side_by_side`) and `view_stack`.
- `deformation` — `view_placement` and `set_positions` only: `applied`
  (default) draws the section's applied deformation where the section is
  drawn under its placement; `none` the linear placement alone.
- `resolution` — only when the user chose image resolution `auto`: the long
  edge in pixels of each picture this call returns (of each section tile in
  `view_stack`), 128 up to the driver model's largest image (2048 px on the
  OpenAI lanes, 2000 px for a Claude host: a request holding more than 20
  images takes none larger), 0 = 512. Out of range is clamped and the reply's
  `view.resolution_note` says so. At every other level the key is not in the
  schema and is refused with the reason (the user fixed the picture size).

A key that means nothing for the tool or the mode is refused
(`VIEW_KEY_UNUSED`, each key with its reason): atlas keys on a tool or mode
that draws no atlas, `channels` where no section is drawn (or on
`preprocess` and `fit_deformable`, whose pictures are a target's appearance
and the image the fit read), `outlines` without `borders`, `atlas_opacity`
without an atlas image or in a mode that does not blend it, border style
with no lines drawn, `deformation` off the placement modes. `outlines:
"none"` is refused with the fix (leave `borders` out). Every reply echoes
what the call drew as `view`.

**Picture feedback.** `view_slices` mode `channels` returns, per section, one
strip of small tiles, one per raw channel, unmodified (no stretch) and
labelled with its name, so the agent can tell which channel is the stain.
`preprocess` returns BEFORE (the target's appearance before the call) then
AFTER for each pictured section (up to 4), labelled, mapped by
`image_indexes`.

**Picture sizes** (`image_resolution`, `core.sizes.PICTURE_EDGES`). Each size is
the long edge of one picture, or of each panel of a multi-panel picture:

| level | each tile of the opening strips (seed message) | every later picture |
| --- | --- | --- |
| low (default) | 256 px | 512 px |
| medium | 384 px | 768 px |
| high | 512 px | 1024 px |
| auto | 256 px | the agent's `view.resolution` per call, 128 px up to the driver model's largest image (2048 px on the OpenAI lanes, 2000 px for a Claude host); 512 px when it gives none |

The opening shows the stack as ABBA-style strips, sections over the atlas at
their current positions (atlas reference strips instead when nothing is
placed); each strip is as long as the model's largest image (2048 px on the
OpenAI lanes, 1568 px for Claude), so a 40-section stack is 5 / 8 / 10 strips
at low / medium / high on the OpenAI lanes.

Nothing is drawn larger than its source: a section never past its working
copy (a small snapshot stays small), an atlas image never past the plane's
own voxels except where it sits under a section in one picture (`stacked`,
`view_stack`, the opening strips), where it is drawn to the section's size. The `view_stack`
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
(`Workspace.section_channels`). The default appearance is drawn from the
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
(auto gives Claude's tools the `view.resolution` key).
Usage belongs to Claude, so there is no LangSlice cost estimate.

Keep the progress window open: section changes appear live in ABBA through its
native actions. Close or Disconnect ends only the live connection, not Claude's
work; checkpoints and completed results remain in the job folder next to the
exported snapshots (`~/.langslice/snapshots/claude-*/langslice/`).
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
