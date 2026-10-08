# The job: positions, in-plane transforms, the toolbox

A job places a stack of histology sections on a BrainGlobe atlas: each
section's position along the slicing axis (and the stack's cutting angles),
one in-plane affine per section and, with the `nonlinear` task, a
deformation per section ([nonlinear_design.md](nonlinear_design.md)). It is
one environment: one state (`StackState`), one toolbox, one job statement,
one session. Looking at the whole stack once informs every decision
(mirrored, misordered, missing or damaged sections, where each sits), so the
work is not split into separate sessions.

## The job spec

A host (CLI, Fiji connector, MCP, library) fills one `JobSpec`
(`src/langslice/core/spec.py`; stored in `job.json`). Main fields:

| Field | Meaning |
|---|---|
| `image_folder`, `atlas`, `plane`, `job_dir` | what is registered and where the job lives |
| `tasks` | subset of `reorder`, `position`, `transform`, `nonlinear`; default the first three. Hosts present `reorder` + `position` as Positioning, `transform` as Linear, `nonlinear` as Nonlinear |
| `model`, `reasoning`, `image_resolution`, `preprocess`, `agent_preprocessing`, `agent_damage` | agent model and what it sees |
| `position` | `thickness_um`, `interval_um`, `strict_interval`, `gated`, `playbook`, `notes` |
| `transform` | `flip`, `hemisphere_cue`, `interactive` (offer `interactive_transform`), `automatic` (offer `elastix_affine`), `angles` (the agent may set the cutting angles), `max_parallel` (1 to 4), `notes` |
| `nonlinear` | `provider` (default `openai-oauth`; `none` = no image model), `image_model`, `require_deformation`, `notes` |
| `facts` | free-form lines passed to the agent verbatim |
| `inputs` | what the host supplies for tasks that are off |
| `max_quota_percent`, `max_input_tokens` | run budgets |

`inputs` keys (`core.spec.INPUT_KEYS`; any other key is refused): `order`,
`positions`, `angles` (`{pitch, yaw}` for the stack, or per section),
`transforms`, `orientation` (`{filename: {flip, rotation_deg}}`),
`pixel_size_um`, `damaged` (`{filename: note}`: a note, which marks no
region), `locked` (filenames whose flip, rotation and transform the agent
cannot change), `keep_warp` (filenames carrying the user's own deformation;
`ants_syn` and `trace_borders` refuse them), `nonlinear_skip` (filenames left
out of Nonlinear) and `channel_names`. A registration made elsewhere becomes
these inputs ([file_formats.md](file_formats.md)).

## Tasks switch whole tools on and off

The registry (`ops/registry.py`) gives each tool a condition; a tool whose
condition is false is not built, and the job takes that answer from `inputs`.

- Always: `look`, `zoom`, `set_channel_properties`,
  `set_preprocessed_channel_properties`, `grep_atlas`, `grep_atlas_view`,
  `status`, `list_files`, `search_files`, `read_file`, `note`, `undo`, `redo`,
  `submit`.
- `position_sections`: the `position` task, or `transform.angles` (the
  stack's cutting angles left to the agent).
- `interactive_transform`: `transform` and `transform.interactive`.
- `elastix_affine`: `transform` and `transform.automatic`.
- `mark_damage`: `agent_damage` (on by default).
- `ants_syn`: `nonlinear`. `trace_borders`: `nonlinear` and an image model
  (provider not `none`; a door that cannot reach it leaves it out).

The `reorder` task builds no tool of its own: the stack's order follows the
positions.

A mirror is part of the in-plane affine (a negative determinant), so flip and
quarter turn belong to `interactive_transform`: a job without it has no flip
tool. A supplied mirrored transform is kept as supplied.

## State

`StackState` is the checkpoint, the result and what every tool writes (one
JSON shape; `state.json` in the job folder, format version 3):

- stack: atlas, plane, interval and thickness, the spec, `interval_breaks`,
  run `notes`, the `appearance` settings, `submitted`;
- per section: `flip` and `rotation_deg` (rotate first, then flip
  left-right); `damaged_regions` (the atlas regions the section has lost,
  acronyms or ids, `"CTX:left"` for one side) and `damage_note`: a section is
  damaged exactly when it has marked regions; `position_mm`;
  `transform` (`kind`: `elastix`, `interactive`, `host`, `imported`; six
  normalized numbers `params`; `physical`: rotation, scales, shear,
  translations in mm; `calibration`); `cutting_angles_deg` (pitch and yaw);
  `image_correction` (the image model's trace); `deformation`; `caveats`.

Appearance settings are display and fit inputs held beside the sections: the
display properties of each raw channel and the recipe of the preprocessed
channel (below). They are undone and checkpointed with everything else.

Positions are atlas millimetres from the anterior edge of the volume, and the
stack's order is derived from them: `position_sections` renumbers the
sections. At submit a stack running the wrong way is reversed so the corrected
order always runs anterior to posterior.

The `Job` (`job/job.py`) owns the state: a write is `snapshot`, change,
`commit` (one undo step, then the checkpoint). History (depth 50) is the job
folder's `history/`, so undo works across a resume. Before every call the job
reloads a `state.json` changed on disk, so a script's edit is picked up and
becomes one undo step.

## Pictures

Every picture a tool draws is saved in the job folder as the JPEG the model
got, with a number and a caption (positions, cutting angles, scale and channel
settings), indexed in `views.jsonl`; `zoom` takes the number, and
`read_file` on a picture file answers with its index entry. Alignment pictures
are in physical space: the section's micrometres per pixel come from the
file's TIFF or OME metadata or the host (else estimated from the tissue width,
`calibration.source = "estimated"`), and the atlas is drawn at true scale.
Where a section and the atlas are separate panels (the positioning picture,
the opening strips), each is framed to its own anatomy and both are drawn at
one micrometres per pixel (`core/scale.py`). Every picture carries its label
in its pixels, naming each section by its filename. The opening message shows
the stack as strips (`core/opening.py`): labelled sections in the stack's
order with the atlas at each section's current position beneath, then the
status table. `image_resolution` sets the long edge of each strip tile and of
every later picture (`core.sizes.PICTURE_EDGES`); it never changes fits or
stored transforms.

## Tools

The verbs are the agent's tools, under the same names in every door;
`langslice-job ops` lists them and `langslice-job schema VERB` gives the exact
arguments. The design is shared tools for looking and for the channels, and
one distinct tool per registration method.

Looking, the same in every run:

- `look` draws pictures in four modes: `section` (each section alone),
  `atlas` (the atlas plane at given positions), `overlay` (each section under
  its registration with the atlas borders on it, the applied deformation
  unless `warp` is `none`) and `positioning` (the sections and the atlas
  along the slicing axis, as ABBA lays them out). It never depends on the
  stack's state for its defaults; at most four pictures per call, the rest
  named.
- `zoom` redraws a box of an earlier picture by its number. A section or
  overlay picture is read again from the original image file at the file's own
  resolution; a positioning picture is redrawn from the working copies.
- `grep_atlas` looks regions up by acronym, name or id; `grep_atlas_view`
  draws the atlas with chosen regions' borders highlighted.
- `status` is the stack as it stands and what the run lets the agent change.
- `list_files`, `search_files`, `read_file` read the job folder.

Channels. Raw channels are never edited.
`set_channel_properties` sets how a raw channel is displayed (contrast
limits, gamma, colormap) in every section that has it: display only, kept
until changed, stated in every caption. `set_preprocessed_channel_properties`
sets the recipe of the one preprocessed channel per section that the fits and
the image model read (channel weights, CLAHE, N4, denoising; default weights
by tissue coverage and CLAHE), stack-wide or per section, and returns
before/after pictures. `look` draws it with `channels: ["preprocessed"]`.

Changing:

- `position_sections`: positions and the stack's cutting angles in one write.
- `interactive_transform`: orientation (flip, quarter turn) and the in-plane
  knobs (rotation, scales, shear, shifts) by hand, for one to four sections;
  values are absolute and a value left out keeps the section's current one.
- `mark_damage`: the atlas regions a section has lost. Marked regions are left
  out of `elastix_affine`, `ants_syn` and the image model's trace
  automatically; there is no whole-section damage flag, and small tears,
  bubbles and folds are not damage.

Fitting, one tool per method: `elastix_affine` refines each section's
placement against an atlas image (`template` or `nissl`), optionally by
`restrict_to` regions; `ants_syn` and `trace_borders` are in
[nonlinear_design.md](nonlinear_design.md).

Bookkeeping: `note`, `undo`, `redo`, `submit`. The MCP door adds `start_job`
and `show_stack`; the scripting doors add `export_maps`
([file_formats.md](file_formats.md)). A replaced tool's old name answers with
`RETIRED_TOOL` and the tool to use.

A section is named by its image filename everywhere: every argument that names
one takes the filename, or the filename without its extension when no other
section's filename has that stem; a number names no section and is refused
(`UNKNOWN_SLICE_IDS`) with the valid filenames listed.

Every write is one undo step and returns the picture of what it did, unless
its `view` argument is false. `elastix_affine`, `ants_syn`, `trace_borders`
and `export_maps` compute outside the job lock and apply at the end; a section
whose inputs changed meanwhile is refused as `STALE_INPUT`. `trace_borders`
runs in the background: it returns at once and the next tool reply starts with
its notice.

## Submit gates

`submit` refuses, with the numbers that refused it, and writes nothing:

- `position` on: a section without a position of its own (`MISSING_POSITIONS`;
  `at_default` names the sections still at the starting position the job gave
  them, below); with `strict_interval`, a spacing more than 10% off the
  interval (`STRICT_INTERVAL`); a reported interval break where the written
  interval is not above 1.5 times the stack's median spacing
  (`INTERVAL_BREAKS_UNSUPPORTED`).
- `transform` on: a section without a transform (`MISSING_TRANSFORMS`),
  damaged sections included.
- `nonlinear` on: a section with neither a deformation at its current
  placement nor an entry in `left_linear` (`[{id, reason}]`: the reason its
  linear placement stands) (`MISSING_DEFORMATIONS`). A host that sets
  `nonlinear.require_deformation` refuses `left_linear`
  (`DEFORMATION_REQUIRED`). No section needs an image-model trace; submit waits
  for traces and background work still running. Sections the host kept out of
  Nonlinear need neither.

With `position` on, a new job gives every section without a supplied position
a starting one (`position_source` `"default"`): evenly spaced in the order the
images were found, at the stack's interval and centred in the atlas's range
(evenly over the whole range when the stack does not fit), or between and
beyond the supplied positions. Status rows say `position_source: "default"`
while a section is there; any write of its position makes it the section's own.

`--gates` (`position.gated`) additionally refuses `position_sections` for a
section not looked at in mode `overlay` or `positioning` since its last write,
and `submit` until `look` in mode `positioning` has shown every section after
the last write. Only the agent and MCP tools are gated; a script never is.

## The built-in agent session

One ADK `LlmAgent` (`src/langslice/agent/`): tools built from the spec, the
job statement from spec and state, the seed message from the state. The
statement carries the job, the run facts, the tools by name (each tool's own
description says what it does), the constraints and a short method. The loop
nudges when the model answers in prose and ends at `submit` or a budget; one
grace call to submit is allowed after a budget stop. Tool images are kept
until a 500-image backstop, then trimmed to 250. Token usage is printed and
traced every call. Traces: [cli.md](cli.md).
