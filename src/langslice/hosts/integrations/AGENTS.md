# LangSlice `hosts/integrations/` — ABBA (and the QUINT writer in `job/`)

Package guide for `src/langslice/hosts/integrations/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

This was the top-level `integrations/` package until the folder move
(2026-10-04). Its ABBA modules moved here, into the hosts layer;
`quint.py` moved down to the job layer as `src/langslice/job/quint.py`,
because an operation (`ops.exports`) writes it and an operation may not
import a host. The QUINT entry below describes that file.

- `hosts/integrations/` — one module per external registration ecosystem.
  `job/quint.py`: QUINT/QuickNII/VisuAlign-compatible JSON export (anchoring
  vectors; file-based, formerly top-level `export.py`). A job's export
  (formats phase, 2026-10-04): `job_export(sections, atlas)`,
  `anchoring_from_pixel_map` (the anchoring from a section's exact file
  pixel -> BrainGlobe µm matrix, any plane and cutting angle) and
  `atlas_um_to_quicknii_points` (SliceBench's conversion: `brainglobe_space`
  to the `lpi` QuickNII axes, voxel centres to edges); written by
  `ops.exports.export_maps` as `exports/quicknii.json` and
  `exports/visualign.json` (VisuAlign markers of each applied deformation,
  `core.maps.residual_markers`). Their inverses, for reading a
  registration back (`job/imports.py`): `quicknii_points_to_atlas_um`,
  `from_target_grid`, `quicknii_target`. The older coronal-frame
  `compute_anchoring` path is unchanged.
  `abba.py`: LangSlice as an abba-python registration plugin
  (`enable_langslice_registration`, `register_selected_slices`). Implements
  ABBA's `SimpleRegistrationPlugin` socket from Python via JPype: the fixed
  image requested from ABBA is its atlas *coordinate* channels (per-pixel
  AP/DV/ML mm), so the adapter samples the BrainGlobe volumes at exactly
  those coordinates — no offset constants or axis assumptions; the existing
  host placement is retained. Those labels supply yellow borders over the
  histology; the shared nonlinear boundary-refinement core sends that overlay
  plus the clean histology for ONE image-model correction call. There is no
  standalone color-map initialization or silhouette refit in this adapter.
  The fit of the corrected lines is the deformable package's
  (`core/nonlinear/border_fit.fit_border_lines`, Elastix lines against the family
  borders; ABBA's labels are given as the native grid, placed by an identity
  at `voxel_size_um`; since 2026-10-04, replacing the Elastix residual fit),
  and the provider is resolved here (`providers.registry.resolve_image_model`).
  Provider `none` calls no model and fits nothing: the landmarks are the
  identity, ABBA's placement stands (as `nonlinear register`'s model-free
  route).
  The fit maps output histology coordinates to the rough atlas frame; ABBA
  receives the paired coordinates in the opposite direction, not a negated
  displacement field. The measured ABBA↔brainglobe AP offset is ~0.99, not
  1.0; never hardcode it. The
  nonlinear result returns as a serializable invertible thin-plate-spline
  (`InvertibleWrapped2DTransformAs3D` — the plain wrapper is not invertible)
  sampled from the fitted deformation field, and lands on the slice's
  registration stack like a native step (undo, state save/reload included).
  Reopening a saved state requires the plugin registered first, else ABBA
  drops the step. `install_gui` adds a `Register > LangSlice` menu entry via
  ABBA's registration-plugin UI registry (PyCommandBuilder dialog; needs the
  `pyimagej-scijava-command` artifact, test-scope in ABBA, added as a scyjava
  endpoint before JVM start — `run_gui_session` handles it), and
  `langslice abba` is the CLI launcher: full ABBA GUI with LangSlice in the
  Register menu, one command. Ships in the recommended
  `langslice` conda env — environment.yml carries openjdk 11 + maven, and
  `pip install -e ".[abba]"` adds abba-python (a separate env from the user's
  `abba`/`deepslice` envs). Live-session probe scripts are kept outside the repo.
  `abba_linear.py`: a live mirror for the LINEAR agent (the nonlinear plugin
  above is a different door). `langslice abba --linear FOLDER [linear
  flags]` (`run_linear_in_abba`) launches the ABBA GUI with the nonlinear
  plugin installed too, imports the folder's images in `core/discovery.py`
  order, and runs the linear agent with `AbbaStackMirror.on_write` attached
  to `engine.run`'s `on_write` hook (fired by `checkpoint.observe_checkpoints`
  after every tool write). The mirror diffs each `StackState` against the
  last one and pushes only what changed: order/position → `moveSlice` (one
  write = one ABBA undo step via `MarkActionSequenceBatchAction`);
  flip/quarter-turn → the slice pre-transform, rebuilt from a captured
  import-time base so it never accumulates; the in-plane affine → an
  `AffineRegistration` step, replaced (not stacked) when revised; cutting
  angles → `ReslicedAtlas.setRotateX/Y`. `damaged` has no ABBA equivalent.
  ABBA has ONE atlas angle for the whole stack, so a job whose sections
  carry different cutting angles (a registration made elsewhere, supplied
  per section) is refused, with
  `doors.api.abba_worker.ABBA_MIXED_ANGLES` (the job was made elsewhere
  with an angle per section, which ABBA cannot show): `run_linear_in_abba`
  and `run_existing_in_abba` refuse such supplied angles before ABBA is
  touched (`abba_worker.refuse_mixed_job`; the launcher also refuses a
  resumed checkpoint that has them), and the mirror pushes nothing of such
  a state (no moves, no angles), logs the message once and keeps it in
  `sync_errors` under `MIXED_ANGLES_KEY`. Jobs from an ABBA session are
  single-angle (its angles are their stack-wide input) and mirror as
  before.
  The agent still renders its own BrainGlobe pictures; ABBA is display plus
  the final home (`finish` writes an `.abba` state file).
  Coordinates: `measure_axis_offset` fits ABBA slicing-axis mm to BrainGlobe
  AP mm at startup instead of hardcoding (measured `z_abba = ap_mm + 0.985`
  on LSD_910/M01). It works by moving one slice to two z's and running a
  throwaway probe registration at each, reading the fixed image's AP channel
  — the resliced atlas's own channel sources are a static cross-section
  view, not a 3D sampler, so they cannot be read directly. Falls back to
  `(1.0, 1.0)` loudly.
  Signs: the in-plane constants (`FLIP_ROTATION_AXIS`, `QUARTER_TURN_SIGN`,
  `INPLANE_ROTATION_SIGN`, translation/scale axes) were MEASURED 2026-09-10
  with the same probe trick (scripts kept outside the repo): ABBA's ML coordinate DECREASES with screen x,
  and ImgLib2 `rotate` is clockwise on ABBA's y-down screen, so both
  rotation signs are −1 and the translation signs are +1. COMPOSITION ORDER
  matters as much as sign: every ImgLib2 `scale`/`rotate`/`translate` acts
  after the transform built so far, LangSlice's `affine_matrix` is translate
  ∘ rotate ∘ scale and `core/sections.py` turns before it flips — so the
  mirror calls scale, rotate, translate (affine) and quarter-turn, then flip
  (pre-transform). The first live M01 run (2026-09-10) caught the affine
  order: with unequal scales rotate-then-scale differs, and single-knob
  probes never see it. A probe script (kept outside the repo) reads
  the ImgLib2 matrices straight back for combined knobs and matches
  LangSlice's blocks exactly; another checks a saved
  `.abba` against the agent's checkpoint (`langslice/state.json`) slice by slice. The
  pitch/yaw ↔ `setRotateX/Y` mapping is pitch_deg = −deg(rotateX), yaw_deg
  = −deg(rotateY), pinned 2026-09-29 to stage 0 of ABBA's own exported
  transform chains (`tests/test_integrations_abba_math.py`). The 2026-09-22
  probe had yaw = +deg(rotateY) because it read ABBA's ML coordinate channel
  as BrainGlobe ML; that channel runs against screen x (see above), and the
  symmetric Allen labels cannot reveal the mirror, so never settle an ML
  sign by label agreement. Do not rely on screenshots
  to check these on this Wayland box (Java Robot returns black; offscreen
  painting shows only overlays) — use the probe.
  Imports JPype/scyjava lazily like `abba.py`, so its diff logic is
  unit-tested in the plain `.venv` against a fake ABBA facade
  (`tests/test_integrations_abba_linear.py`); the live/JVM smoke test is
  kept outside the repo.

## Linear menu and existing sessions (2026-09-13)

`abba_gui.py::install_menu` runs after `show_bdv_ui` and adds the top-level
LangSlice menu. Session-local settings expose model/reasoning, interval and
thickness, ordering/flip/cue, position/strict-spacing, and independent
interactive/automatic/Elastix transform choices. Unset interval is the median
positive ABBA slice spacing; unset thickness is the median `getThicknessInMm`
value, not an independent biological metadata field. User overrides persist
for the session. Swing updates run on the EDT; the agent runs in a guarded
background thread. Agent-viewer and agent-log opening are independently optional. The installer also
initializes ABBA's theme if the Python launcher left its strokes null.

`abba_linear.py::run_existing_in_abba` snapshots selected slices (all if none
selected) with `SliceToImagePlus.export` onto a centred calibrated frame. It
attaches an explicit filename-to-slice mapping, seeds current positions/order,
and skips the initial mirror write so ingestion cannot reset the user's work.
Snapshots, checkpoint and `abba_run.json` mapping remain in the reported run
folder. Existing registrations stay underneath new corrections; disabled task
properties are preserved. Source orientation on already registered sections is
refused. This entry point currently requires flat coronal Allen mouse sessions;
GUI cutting-angle control is not exposed yet.
Human edits during a run are not read back. Save results through ABBA's normal
state-save UI.

`core/abba_affine.py` (moved to the core 2026-10-04: the door-level snapshot
worker emits its rows) converts the stored normalized six-number affine to ABBA world
millimetres for these snapshots, preserving pivot displacement and shear. The
normalization refers to the oriented section frame, not the padded atlas
canvas. Use original snapshot physical dimensions, swapping axes after a
quarter-turn; do not use the preview's resized calibration. The legacy CLI
fresh-import path still reconstructs its affine from physical knobs.

## Live agent companion window and native following

`abba_chat.py::create_activity_window` prefers a local Chrome/Chromium app
window beside ABBA, falling back to `abba_activity.py::ActivityWindow` if browser
startup fails. The browser transcript renders escaped Markdown, streamed visible
assistant text and provider-exposed reasoning summaries in a neutral monospace
log. Compact expandable tool rows show exact tool names, targets and status,
and image-count links. A smaller image panel beneath it supports history and
full-image inspection; it collapses to a header when no images are available. Summary wording comes directly from the provider; there
is no summarizing model or first-person rewrite. Encrypted reasoning and
signatures are excluded. Exact incoming image bytes are served locally; browser
scaling changes only their display. The loopback GET server uses a random path
token and serves bundled assets, sanitized events and images, with no registration
or command endpoints. Recent history is bounded to 1,500 events / 2 MiB event
JSON and 64 MiB images. Console and optional JSONL diagnostics remain separate.
Close does not cancel the agent; **Show agent log** reopens the viewer.
New runs dispose the previous viewer and its server/private browser profile.
The Swing fallback retains its adjustable vertical split, bounded image history,
and off-EDT image decoding.

`abba_follow.py::AbbaFollower` consumes seed and actual `tool_start`/`tool_end`
events. `doors.tools.toolbox._serialized` emits execution events inside its lock,
resolving corrected-index references to stable filenames before each operation.
Model `tool_call` announcements may be batched ahead of execution and must not
drive navigation. The existing-session adapter owns the explicit mapping and
fans events out to the follower and sidebar; the CLI import path does likewise.
With a comparison factory configured, the follower routes every supported
positioning, inspection and transform event, including single targets and seed,
to `abba_compare.py::ComparisonWindow`. It leaves the main ABBA camera, selection
and display modes untouched. Viewer failures are isolated and do not fall back
to moving the main view. Matching post-write end events update focus after the
checkpoint mirror issues registration changes. Callers without a viewer factory
retain the legacy selection/camera follower and its selection restoration.

The native agent viewer stays visible as focus changes between single and
multiple targets. `abba_overview.py::NativeOverview` borrows ABBA's actual
`extendedSlicedSources` positioning mosaic at the existing `getStep`, preserving
its native active atlas channels and copied converter settings. All registered
sections retain native size. When the host is in positioning mode, their centers
come directly from `getDisplayedCenter`, including its overlap/stair placement.
If the host is in review mode, the separate overview uses ABBA's voxel-snapped
positioning formula and normal below-atlas placement; it never switches the host
mode. The camera fits the stack, retains in-range framing, and reframes if the
host changes its display interval. No artificial atlas interval, thumbnail lanes
or custom position markers are introduced.

Selection is drawn by ABBA's actual `CircleGraphicalHandle` and
`SquareGraphicalHandle` components, with local agent-target suppliers and no
editing behaviors. Dashed guides and connecting lines use `ABBABdvViewPrefs`.
The lower independent BigDataViewer focus panels page targets in execution order,
four per page, using native registered section and atlas sources. Local source
wrappers, converters and cameras belong to this viewer,
with no changes to registration state or the global source registry. A 250 ms
poll tracks source replacement, native affine changes and positions as ABBA's
asynchronous actions finish. Both overview and focus cameras maintain their
calibrated fit, correcting late BDV initialization without rewriting an unchanged
camera. Steady polling must not restart progressive rendering.
Both areas show committed ABBA registrations. Candidate atlas positions are
neither written nor marked here; exact tool images in the log show speculative
comparisons.

`MenuSettings.open_agent_viewer` and `open_agent_log` independently control
**Open agent viewer in ABBA** and **Open agent log**, for GUI and CLI runs.
`MenuController` owns the lazily created native viewer beyond run completion;
**Show agent viewer** reopens its last page. The next run disposes the old
companions. Closing or hiding the native viewer stops refresh without cancelling
registration; disabling its opening option prevents subsequent automatic updates.
The retained `follow_agent` settings field is for compatibility and no longer
controls these window options. Native ABBA APIs verified against installed
0.11.0 and upstream sources.

## Saved spline compatibility

The linear agent no longer exposes paired-landmark tools. Historical applied
spline checkpoints retain paired source/target points normalized to the oriented
section frame; Elastix payloads save authoritative parameters and an explicit
affine. `core/abba_spline.py` samples their complete target-to-source
pullback into a dense TPS carrier, refining from 9² to at most 33² grid points
and requiring at most 5 µm error at independent probes, with sampled fold
checks. Failure refuses export before replacing the old registration. This
grid is not the agent's anatomical landmark set. Legacy TPS checkpoints retain
their original exact TPS. Both paths convert to centred world millimetres using
the original calibrated snapshot. The spline includes
its affine component; baseline affine parameters are not applied again.
A completed native `SacBigWarp2DRegistration` carries this serialized transform
through `RegisterSliceAction` without opening BigWarp. Save/reload uses ABBA's
normal native registration adapters. Revisions replace the mirror's owned step;
undoing to an affine replaces the spline with the exact earlier affine.
Geometry and native serialization are validated before deleting the old step.
The fresh-import CLI path has no snapshot calibration and refuses spline
mirroring with an actionable message directing users to the ABBA menu; it must
never silently substitute an affine. The follower has no routing for the
removed landmark tools. Applied saved spline mappings are still synchronized
through checkpoint writes.
Registration failures remain available in `mirror.sync_errors` and retry on the
next checkpoint even if the transform is unchanged. Native JVM validation on
ABBA 0.11.0 confirmed the TPS against an independent fit at landmarks and other
points, one owned step through revisions, and pixel-identical affine restoration
and spline state reload.

An isolated native Elastix-export serialization smoke additionally checked
1,081 probes: Java agreed with the sampled Python TPS to numerical precision,
remained within 1.6 µm of the exact synthetic Elastix map, and retained the map
after native transform serialization/reload. This does not replace a full
ABBA project save/reload test for the new backend.

## Independent Fiji connector (2026-09-16)

`connectors/fiji/` is a separate Java/SciJava plugin for an existing ABBA session.
It starts a selected Python environment via `langslice serve --stdio`; it does
not use `abba_python` or PyCommandBuilder. Setup loads without Python and owns
environment selection; authentication runs in the worker. The existing modules
above remain the Python-started ABBA route.

`doors/api/abba_worker.py` reuses the linear engine, and
`hosts/api/nonlinear_worker.py` the nonlinear
`compute_registration_landmarks`, without importing Java. Linear snapshots are
centred/calibrated; host AP mapping is measured, ingestion emits no mutations,
and streamed updates express complete replacement corrections in world mm.
Every checkpoint also carries `updates_since_start` (ingested state to that
checkpoint) and the result `final_updates` (ingested to final), which the
connector applies once at the end. `locked` snapshots (existing registrations
the user did not let the agent overwrite) never produce orientation or
transform rows; `damaged` becomes `inputs.damaged`. With a `preprocessing`
setting or multi-page snapshots (one page per ABBA channel), the settings
become `JobSpec.host_preprocessing`: the engine reads the snapshots themselves
(nothing is staged) and shows each as `image_prep.host_preprocess` blends it,
exactly what `preprocess.preview` shows, as the DEFAULT appearance; the pages
stay raw channels (named by the optional `channel_names` param) for the
agent's `view.channels` and `preprocess`. Without either, the engine's own
`auto` path runs on the snapshots as before. A `trace_dir` param points the session trace (`LANGSLICE_TRACE_DIR`) at that
folder for the one run and returns the new files as `trace_files`.
`linear.estimate` prices a spec from
`agent/cost.py` (percent of the usage window per section, measured runs only;
refused at medium/high/auto resolution) without importing the engine. The Java host owns native actions and persistence. Read
`docs/abba_plugin_design.md` and `docs/abba_installation.md` for the protocol,
current source-preview installation, and publication requirements.

### Claude mode in the independent connector

The Fiji dialog's Claude choice saves a job via `claude.prepare` and copies
a prompt for Claude Desktop/Code. `doors.api.abba_worker.prepare_linear` and
`checkpoint_callback` are shared by ADK and MCP; never duplicate their
calibration or checkpoint-to-native geometry translation. An authenticated
loopback listener passes MCP checkpoints to the same `AbbaHostSession.apply`
native action path. Claude changes are live, while ChatGPT retains its existing
final-only apply. Closing the Claude window disconnects the host, not the MCP
job; saved results remain in the job folder next to the exported snapshots
(`~/.langslice/snapshots/claude-*/langslice/`). No Fiji scripting tools
are exposed. See `docs/abba_plugin_design.md` for the wire/file contracts.
