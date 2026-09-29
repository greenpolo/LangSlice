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
| Provider | ChatGPT account only for now | `openai-oauth`; `setup.status` lists its `agent_models` and `image_models` with defaults |
| Agent model, reasoning level | ADK model string and effort | `model`, reasoning |
| Image model / provider | image transport and model for nonlinear | `nonlinear.provider`, `nonlinear.image_model` |
| Image resolution: low / medium / high | scales every picture the agent sees by an opaque multiple of the calibrated sizes; low = today's calibration (judged good enough for the cost). Does not touch the image model's inputs | `image_resolution` low/medium/high; display only, fits and stored transforms unchanged |
| Estimated cost | shown at the bottom once every box is chosen | worker `linear.estimate` (`linear/cost.py`): percent of the usage window from measured runs; refused at medium/high resolution, where nothing is measured |
| View agent log | the agent's activity in a window during the run | Fiji connector: a text log window (or a compact status window when off). The Python-started ABBA launcher has a richer browser log (`integrations/abba_chat.py`) |
| Save traces | full record of what the agent was shown, said and did, saved to a chosen folder | worker `trace_dir` (`LANGSLICE_TRACE_DIR` for one run); ABBA dialog checkbox + folder |
| Enable agent viewer | an ABBA-style brain display of the agent's work as it happens | built for the Python-started ABBA launcher (`integrations/abba_compare.py`, `abba_overview.py`, `abba_follow.py`); the Fiji connector shows the checkbox disabled until it is ported |
| Atlas, plane, preprocess | as today | `atlas`, `plane`, `preprocess` |

## 1. Positioning (absorbs reorder)

Ordering and positioning are one act once positions exist; reorder stays an
internal capability (order plus flip), not a user-facing task.

| Control | On main |
| --- | --- |
| Enable hemisphere flipping, with optional text describing the hemisphere cue | `reorder.flip`, `reorder.hemisphere_cue` |
| Slice thickness (may be provided by the host) | `position.thickness_um` |
| Slicing interval (ABBA often provides it) | `position.interval_um` |
| Enable DeepSlice tool (TODO; coronal mouse/rat only) | `position.deepslice` |
| Enable Bayesian optimizer tool (not implemented; needs design) | `position.bayesian` builds `fit_position` |
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
| Enable affine tool. Off: every transform is X/Y movement and scaling by the agent | `transform.automatic` (+ `transform.elastix` for the intensity affine) |
| Max parallel slice transforms, 1 to 4. 1 = one section per call | `transform.max_parallel`; below 4, `fit_affine` and `adjust_transforms` refuse larger calls (at 4, `fit_affine` stays uncapped) |
| Enable slice angle estimation (yes/no); later tools: DeepSlice angle, Bayesian optimizer | `transform.angles` builds `set_cutting_angles` (manual only); the Fiji connector still refuses angle changes and shows the box disabled |
| Extra notes for the agent, attached to this task | `transform.notes` |

The interactive tool is always on. Damaged sections refuse the automatic
affine and require an applied interactive transform (unchanged). Cutting
angles are stack-wide and belong here because they change the atlas section
every transform is fit against.

## 3. Nonlinear

Precondition: a linear placement for every section, either from the host
(`inputs.transforms`, the user's own ABBA alignment) or from task 2 in the same
run. Nonlinear ON with sections lacking a placement is a job-spec validation
error naming those sections, before any model call. A host turns it into a
dialog: "Slices a-z have no transform. Nonlinear requires a linear step first.
Allow the agent to align them?" Yes = Linear ON for those sections only. On
main this check is not in the spec yet: `correct_slice_borders` refuses a
section without a usable placement one call at a time
(`INVALID_LINEAR_PLACEMENT`).

The deliverable is a registration, not drawings. Order of work: design the
deformation algorithm, then settle the border output format it consumes, then
test. Drawings remain the review artifact.

| Control | On main |
| --- | --- |
| Enable image-gen tool | task `nonlinear` builds `correct_slice_borders` |
| Use agent (GUI). The agent writes per-slice notes for the image model. The agent-free path (fixed prompt as a plain operation over supplied placements) stays in the API only; it is the 3D-volume path, where notes have no purpose | notes are a tool argument; the `nonlinear` CLI is the agent-free operation |
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
snapshot's pages (one per channel) into the image the agent sees: auto, or
custom per-channel weights, CLAHE on/off and strength. `linear.run` takes it as
`preprocessing`; `preprocess.preview` writes the same image for the dialog. The
CLI keeps `preprocess` auto/none.

## ABBA host (settled 2026-09-28)

- Menu: ABBA's Register menu, like DeepSlice: **Register > LangSlice > LangSlice
  Registration…**. ABBA appends external entries at the end of that menu, so it
  cannot sit directly beside DeepSlice.
- Selected slices are the slices sent to LangSlice.
- Existing transforms: any ABBA registration step counts as linear; spline
  steps (BigWarp, Elastix spline) are the only nonlinear ones.
- Results land in the user's ABBA when the agent is done, as one undoable step;
  a stopped run can apply its last checkpoint. The agent viewer, once ported to
  the connector, shows the work along the way.
- Nonlinear is a facade in ABBA for now: shown, does nothing.
- DeepSlice and Bayesian checkboxes are added later.
- No over-saturation warning.

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
