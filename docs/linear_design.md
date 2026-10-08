# The linear job: order, position, in-plane transform

A job places a stack of histology sections on a BrainGlobe atlas: their order,
each section's position along the slicing axis (and cutting angles), and one
in-plane affine per section; with the opt-in `nonlinear` task also a
deformation per section ([nonlinear_design.md](nonlinear_design.md)). It is
one environment: one state (`StackState`), one toolbox, one job statement,
one session. Looking at the whole stack once informs every task (mirrored,
misordered, missing or damaged sections, where each sits), so the tasks are
not separate sessions.

## The job spec

A host (CLI, Fiji connector, MCP, library) fills one `JobSpec`
(`src/langslice/core/spec.py`; stored in `job.json`). Main fields:

| Field | Meaning |
|---|---|
| `image_folder`, `atlas`, `plane`, `job_dir`, `output_level` | what is registered and where the job lives (`full` or `lean`, see [library.md](library.md)) |
| `tasks` | subset of `reorder`, `position`, `transform`, `nonlinear`; default the first three. Hosts present `reorder` + `position` as Positioning, `transform` as Linear, `nonlinear` as Nonlinear |
| `model`, `reasoning`, `image_resolution`, `preprocess`, `agent_preprocessing`, `agent_damage` | agent model and what it sees |
| `position` | `thickness_um`, `interval_um`, `strict_interval`, `bayesian` (offers `search_position`), `gated`, `playbook`, `notes` |
| `transform` | `flip`, `hemisphere_cue`, `interactive`, `automatic`, `angles` (agent may set cutting angles), `max_parallel` (1 to 4), `notes` |
| `nonlinear` | `provider` (default `openai-oauth`; `none` = no image model), `image_model`, `engine` (`ants`, `elastix`, `either`), `notes` |
| `facts` | free-form lines passed to the agent verbatim |
| `inputs` | what the host supplies for tasks that are off |
| `max_quota_percent`, `max_input_tokens` | run budgets |

`inputs` keys (`core.spec.INPUT_KEYS`; any other key is refused): `order`,
`positions`, `angles` (`{pitch, yaw}` for the stack, or per section),
`transforms`, `orientation` (`{id: {flip, rotation_deg}}`), `pixel_size_um`,
`damaged` (`{id: note}`: a mark that names no regions, which the agent may add
regions to but cannot remove), `locked` (ids
whose flip, rotation and transform the agent cannot change), `keep_warp`
(ids carrying the user's own deformation; `fit_deformable` refuses them),
`nonlinear_skip` (ids left out of Nonlinear) and `channel_names`. A
registration made elsewhere becomes these inputs
([file_formats.md](file_formats.md)).

A task that is off builds none of its tools. A mirror is a property of the
in-plane affine (a negative determinant), so flip and quarter turn belong to
`transform`: a positioning-only job has no flip tool and never mentions
mirroring. A supplied mirrored transform is kept as supplied.

## State

`StackState` is the checkpoint, the result and what every tool writes (one
JSON shape; `state.json` in the job folder, format version 3):

- stack: atlas, plane, interval and thickness, the spec, `interval_breaks`
  (indices after a real gap), run `notes`, `submitted`;
- per section: original and corrected index; `flip` and `rotation_deg`
  (rotate first, then flip left-right); damage: `damaged_regions` (the atlas
  regions the section is missing or has badly displaced, acronyms or ids,
  `"CTX:left"` for one side), `damage_marked` (a mark that names no regions)
  and `damage_note`, the section being damaged when it has either (a state
  saved with the older `damaged` flag reads as `damage_marked`); `position_mm`;
  `transform` (`kind`: `silhouette`, `elastix`, `interactive`, `host`,
  `imported`; six normalized numbers `params`; `physical`: rotation, scales,
  shear, translations in mm; `calibration`); `cutting_angles_deg` (pitch and
  yaw, per section); `image_correction`; `deformation`; `caveats`.

Order and position are separate fields that must agree at submit: reordering
changes only the corrected index. Direction is not the agent's concern:
at submit a stack running the wrong way is reversed so the corrected order
always runs anterior to posterior along the atlas axis. Positions are atlas
millimetres from the anterior edge of the volume.

The `Job` (`job/job.py`) owns the state: a write is `snapshot`, change,
`commit` (one undo step, then the checkpoint). History (depth 50) is the job
folder's `history/`, so undo works across a resume. Before every call the job
reloads a `state.json` changed on disk, so a script's edit is picked up and
becomes one undo step.

## Pictures

In-plane alignment renders are in physical space: the section's micrometres
per pixel come from the file's TIFF or OME metadata or `--pixel-size-um`
(else estimated from the tissue width, recorded as `calibration.source =
"estimated"`); the atlas is drawn at true scale. Where a section and the
atlas are separate panels (`stacked`, `side_by_side`, `view_stack`'s sheet,
the opening strips), each is framed to its own anatomy (foreground plus a
margin) and both are drawn at one micrometres per pixel (`core/scale.py`):
the section at the scale its overlay draws it (calibration times its stored
transform's scale), the larger panel at the picture's long edge, so a
section reads at its true size against the atlas. `side_by_side` sizes by
the largest atlas plane, so one section picture serves every position of a
call. Every picture carries its label in its pixels, naming each section by
its filename. The opening message shows
the stack as strips (`core/opening.py`): labelled sections in corrected order
with the atlas at each section's current position beneath, then the status
table. `image_resolution` sets the long edge of each strip tile and of every
later picture (`core.sizes.PICTURE_EDGES`); it never changes fits or stored
transforms. Each picture a tool returns takes its options in one `view`
argument (`core/display.py`; `langslice-job schema VERB` lists them). Every
picture is saved in the job folder as the JPEG the model got.

## Tools

The verbs are the agent's tools, under the same names in every door;
`langslice-job ops` lists them and `langslice-job schema VERB` gives the exact
arguments:

- Common: `status`, `view_slices`, `view_atlas`, `note`, `undo`, `redo`,
  `mark_damaged`, `preprocess`, `submit`.
- Positioning: `reorder_slices`, `set_positions`, `view_placement`,
  `view_stack`, `search_position` (with `bayesian`).
- Linear: `orient_slices`, `fit_affine`, `adjust_transforms`,
  `set_cutting_angles` (with `angles`). `fit_affine` is by default an Elastix
  intensity affine refining the current placement, or `method="silhouette"`
  (outline moments); `include` / `exclude` regions fit only the kept regions.
  Its overlap compares shapes, not inner anatomy, so a fit that turns a
  section more than 45 degrees from its previous transform carries a
  `warning` in its row. `set_cutting_angles` tilts the plane every section is
  cut at; its description gives, per plane, which picture edge a positive
  pitch or yaw moves to a larger position (`tests/test_core_layers.py`).
  `adjust_transforms` sets rotation, scales, shear and shifts by hand.
  `grep_atlas` looks region names up for the `include` / `exclude`
  arguments (also with Nonlinear alone).
- Nonlinear: `trace_borders` (image model; not built when the provider is
  `none`), `fit_deformable` ([nonlinear_design.md](nonlinear_design.md)).
- Scripting only (CLI and library, never a model's tool): `export_maps`.

A section is named by its image filename everywhere: every argument that
names one (`submit`'s `interval_breaks` and `left_linear` included) takes the
filename, or the filename without its extension when no other section's
filename has that stem; a number names no section and is refused
(`UNKNOWN_SLICE_IDS`) with the valid filenames listed. Picture labels,
captions, the status table and the job statement name sections by filename
alone; the corrected index is the state's own bookkeeping.

Every write returns the picture of what it did; there is no preview-then-apply,
every write is undoable. `fit_affine`, `fit_deformable`, `trace_borders` and
`export_maps` compute outside the job lock and apply at the end; a section
whose inputs changed meanwhile is refused as `STALE_INPUT`.

## Submit gates

`submit` refuses, with the numbers that refused it, and writes nothing:

- `position` on: a section without a position of its own (`MISSING_POSITIONS`;
  `at_default` names the sections still at the starting position the job gave
  them, below); with `strict_interval`, a spacing more than 10% off the interval
  (`STRICT_INTERVAL`); a reported interval break where the written interval is
  not above 1.5 times the stack's median spacing (`INTERVAL_BREAKS_UNSUPPORTED`).
- `transform` on: a section without a transform (`MISSING_TRANSFORMS`); a
  damaged section whose transform is missing, identity, invalid or a
  whole-section automatic fit (`DAMAGED_REQUIRES_MANUAL_TRANSFORM`; a manual
  adjustment or a `fit_affine` fit that left regions out satisfies it; a
  section's marked regions are left out of every fit on their own).
  Locked sections are exempt.
- `nonlinear` on: a section with neither a deformation at its current
  placement nor a `keep_linear` reason (`MISSING_DEFORMATIONS`). The
  operation `ops.submit.submit` also takes `left_linear` (`[{id, reason}]`):
  the sections left without a deformation, each with the reason its linear
  placement stands, counted as covered; a host that requires a deformation
  on every section (`NonlinearSpec.require_deformation`) refuses it
  (`DEFORMATION_REQUIRED`). No section needs an image-model trace; submit
  waits for traces and background work still running. Sections the host kept
  out of Nonlinear need neither.

With `position` on, a new job gives every section without a supplied
position a starting one (`SliceState.position_source` `"default"`): evenly
spaced in the order the images were found, at the stack's interval and
centred in the atlas's range (evenly over the whole range when the stack
does not fit), or between and beyond the supplied positions. Status rows say
`position_source: "default"` while a section is there; any write of its
position makes the position its own.

`--gates` additionally refuses `set_positions` for a section not compared since
its last write, and `submit` until `view_stack` has run after the last write.

## The built-in agent session

One ADK `LlmAgent` (`src/langslice/agent/`): tools built from the spec, the
job statement from spec and state, the seed message from the state. The
statement carries the job, the run facts, one factual line per tool that
exists, the constraints and, with positioning on, a short method section.
The loop nudges when the model answers in prose and ends at `submit` or a
budget; one grace call to submit is allowed after a budget stop. Tool images
are kept until a 500-image backstop, then trimmed to 250. Token usage is
printed and traced every call. Traces: [cli.md](cli.md).
