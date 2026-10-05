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
`damaged` (`{id: note}`; the agent cannot clear these flags), `locked` (ids
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
  (rotate first, then flip left-right); `damaged` and note; `position_mm`;
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
"estimated"`); the atlas is drawn at true scale. Sections and atlas sections
are framed alike (foreground plus a margin) so apparent scale is not a cue,
and every picture carries its label in its pixels. The opening message shows
the stack as strips (`core/opening.py`): labelled sections in corrected order
with the atlas at each section's current position beneath, then the status
table. `image_resolution` sets the long edge of each strip tile and of every
later picture (`core.sizes.PICTURE_EDGES`); it never changes fits or stored
transforms. Each picture a tool returns takes its options in one `view`
argument (`core/display.py`; `langslice schema VERB` lists them). Every
picture is saved in the job folder as the JPEG the model got.

## Tools

The verbs are the agent's tools, under the same names in every door;
`langslice ops` lists them and `langslice schema VERB` gives the exact
arguments:

- Common: `status`, `view_slices`, `view_atlas`, `note`, `undo`, `redo`,
  `mark_damaged`, `preprocess`, `submit`.
- Positioning: `reorder_slices`, `set_positions`, `view_placement`,
  `view_stack`, `search_position` (with `bayesian`).
- Linear: `orient_slices`, `fit_affine`, `adjust_transforms`,
  `set_cutting_angles` (with `angles`). `fit_affine` is by default an Elastix
  intensity affine refining the current placement, or `method="silhouette"`
  (outline moments); `include` / `exclude` regions fit only the kept regions.
  `adjust_transforms` sets rotation, scales, shear and shifts by hand.
- Nonlinear: `trace_borders` (image model; not built when the provider is
  `none`), `grep_atlas`, `fit_deformable` ([nonlinear_design.md](nonlinear_design.md)).
- Scripting only (CLI and library, never a model's tool): `export_maps`.

Every write returns the picture of what it did; there is no preview-then-apply,
every write is undoable. `fit_affine`, `fit_deformable`, `trace_borders` and
`export_maps` compute outside the job lock and apply at the end; a section
whose inputs changed meanwhile is refused as `STALE_INPUT`.

## Submit gates

`submit` refuses, with the numbers that refused it, and writes nothing:

- `position` on: a section without a position (`MISSING_POSITIONS`); positions
  not monotone along the corrected order (`ORDER_POSITION_MISMATCH`, naming the
  pairs); with `strict_interval`, a spacing more than 10% off the interval
  (`STRICT_INTERVAL`); a reported interval break where the written interval is
  not above 1.5 times the stack's median spacing (`INTERVAL_BREAKS_UNSUPPORTED`).
- `transform` on: a section without a transform (`MISSING_TRANSFORMS`); a
  damaged section whose transform is missing, identity, invalid or a
  whole-section automatic fit (`DAMAGED_REQUIRES_MANUAL_TRANSFORM`; a manual
  adjustment or `fit_affine` with `include` / `exclude` regions satisfies it).
  Locked sections are exempt.
- `nonlinear` on: a section with neither a deformation at its current
  placement nor a `keep_linear` reason (`MISSING_DEFORMATIONS`); with an image
  model, a section without a completed trace at its current placement
  (`MISSING_IMAGE_CORRECTIONS`; submit waits for running traces). Sections the
  host kept out of Nonlinear need neither.

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
