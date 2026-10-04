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


## One ABBA integration (2026-10-04)

ABBA 0.24.x only. The integration is the Java Fiji connector
(`connectors/fiji/`: the dialog under ABBA's Register menu, Setup, the
native actions) plus LangSlice as its own Python worker process
(`langslice serve --stdio`: `hosts/api/service.py` and
`doors/api/abba_worker.py`). The abba-python registration plugin
(`abba.py`: `enable_langslice_registration`, `register_selected_slices`,
`compute_registration_landmarks`, `install_gui`, `run_gui_session`), the
launcher's settings menu (`abba_gui.py`) and the Python live mirror
(`abba_linear.py`, `langslice abba --linear` / `--save-state`) were
deleted that day. Their measured sign table lives on in
`core/abba_angles.py` (`PITCH_TO_ROTATE_X_SIGN`, `YAW_TO_ROTATE_Y_SIGN`,
both -1: pitch_deg = -deg(rotateX), yaw_deg = -deg(rotateY), pinned to
stage 0 of ABBA's own exported transform chains,
`tests/test_abba_angles.py`; never settle an ML or yaw sign by label
agreement on the symmetric Allen atlas); `langslice.integrations.abba_linear`
re-exports just those two constants for SliceBench. ABBA's import, export
and save-state commands are not wrapped by LangSlice.

`abba_launch.py` — `langslice abba` (`hosts/cli.py`): ABBA started from
Python (pyimagej + abba-python's `Abba` class, `ij=` our own) with
`ABBA_JAVA_DEPENDENCIES` (ImageToAtlasRegister 0.24.1 and the versions its
pom pins; abba-python 0.11.0's own list is for ABBA 0.11) on Java 21
(`scyjava.config.set_java_constraints(fetch="auto", version="21")`), the
connector jar on the classpath (`connector_jar`: `--connector-jar`, else
`LANGSLICE_CONNECTOR_JAR`, else the newest `connectors/fiji/target/*.jar`
of this checkout; a clear `mvn package` message when there is none),
ABBA's theme initialized (`init_theme`: abba-python builds the BDV view
without the Java launcher that sets it, and null strokes crash the overlay
painter), and the connector pointed at this Python environment
(`point_connector_at`: the `langslice.environment` system property, and
`EnvironmentDiscovery.save` when the user has saved none). `RunListener`
is registered with `org.langslice.fiji.LangSliceEvents.addListener` (a
`java.util.function.Consumer<String>`): the connector forwards every worker
message of a run (`checkpoint`, `agent_event`, `log`) and its own
(`run_started` with `viewer` and `sections`, `job`, `applied`,
`run_finished`) as JSON text. The Java thread only queues; one Python
thread handles them in order: a new section map (`LangSliceEvents.slices()`)
or `run_started` is a new run (the old follower finishes), the follower
(`AbbaFollower` + `ComparisonWindow`) opens only when the run asked for the
viewer, and the log (`create_activity_window`) gets every agent event with
its pictures read from the event's `views` (saved `view.jpg` paths in the
job folder, waiting up to `VIEW_WAIT_S` for the background writer; events
never carry image bytes). Both are PASSIVE: the connector applies every
change. JPype/scyjava/pyimagej/abba-python are imported lazily; the
plumbing is tested with fakes (`tests/test_integrations_abba_launch.py`).
A headless smoke on a real ABBA 0.24.1 JVM with the connector jar
(2026-10-04) registered the listener and delivered a Java message into it;
the GUI launch itself needs a display session.

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
Close does not cancel the agent. In `langslice abba` one log window serves
the session (reopened when closed); its pictures are the saved views the
events name.
The Swing fallback retains its adjustable vertical split, bounded image history,
and off-EDT image decoding.

`abba_follow.py::AbbaFollower` consumes seed and actual `tool_start`/`tool_end`
events. `doors.tools.toolbox._serialized` emits execution events inside its lock,
resolving corrected-index references to stable filenames before each operation.
Model `tool_call` announcements may be batched ahead of execution and must not
drive navigation. `abba_launch.RunListener` owns the connector's explicit
filename-to-slice mapping and fans events out to the follower and the log.
Followed tools: in review mode (`_TRANSFORMS`) `adjust_transforms`,
`fit_affine`, `fit_deformable`, `trace_borders`; in positioning mode
(`_POSITIONING`) the viewing and positioning tools plus `mark_damaged` and
`preprocess`; every write but `preprocess` refreshes on its end event.
With a comparison factory configured, the follower routes every supported
positioning, inspection and transform event, including single targets and seed,
to `abba_compare.py::ComparisonWindow`. It leaves the main ABBA camera, selection
and display modes untouched. Viewer failures are isolated and do not fall back
to moving the main view. Matching post-write end events update focus after the
connector has applied the checkpoint's rows. Callers without a viewer factory
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

## Saved spline compatibility

The linear agent no longer exposes paired-landmark tools. Historical applied
spline checkpoints still produce legacy `spline_source_mm` /
`spline_target_mm` rows (`core/abba_spline.py`: Elastix payloads sampled
into a 9x9..33x33 TPS within 5 µm, legacy TPS pairs as they are); new jobs
never emit them.

## The worker side of the connector (`doors/api/abba_worker.py`)

The Java host owns snapshots, native actions and persistence; the worker
reuses the linear engine without importing Java. Snapshots are centred and
calibrated; `positions_mm` are BrainGlobe AP mm (the connector converts
with ABBA's `toAtlasZ` and sends `z_offset_mm`, which the worker keeps in
`job.json` under `host.abba` and returns as the result's `abba`).
`angles_deg` (`{"pitch_deg", "yaw_deg"}`, `core/abba_angles.py` signs)
becomes the job's stack-wide `inputs.angles`; tilted sessions and the
angle task are accepted, sections with different angles are refused
(`ABBA_MIXED_ANGLES`). Ingestion emits no mutations; later checkpoints carry
`host_updates` (replacement corrections from the previous checkpoint),
`updates_since_start`, and `host_angles` when the stack-wide angles
changed; the connector applies every checkpoint live, in both modes, as one
ABBA undo step, and the run ends with one more checkpoint of the final
state (`final_updates` is informational). In a job with the `nonlinear`
task each row may carry `warp` (`HostCheckpoints._warps`): an applied
deformation as `core/abba_warp.py` landmark pairs on top of the affine
step (`source_mm` / `target_mm` as `[[x...], [y...]]`, the record, the
measured error, the point count), or `null` when the deformation is
cleared, undone, kept linear, cannot be expressed (with a `log` event) or
its placement changed. Locked sections get warp rows but never linear
rows. `existing_warp` sections that are locked (the dialog's "Allow the
agent to overwrite existing transforms" reaches the worker as `locked`)
become `inputs.keep_warp`, and `nonlinear_skip` sections
`inputs.nonlinear_skip`: `fit_deformable` and `trace_borders` refuse them
(`KEEPS_HOST_WARP`, `NONLINEAR_SKIPPED`, `Job.nonlinear_refusal`).
`damaged` becomes `inputs.damaged`; `preprocessing` settings or multi-page
snapshots become `JobSpec.host_preprocessing` (the default appearance; the
pages stay raw channels named by `channel_names`). A `trace_dir` param
points the session trace at that folder for the run. The tracker is
attached to the opened job and workspace (`engine.run(on_open=...)`; the
MCP door after opening) so it can read deformation records and section
frames. `agent_event`s are forwarded without bytes: `tool_end` carries
`views` (saved picture paths), the seed the opening's. `setup.status`
lists `image_models` (ChatGPT image lane, Gemini API, OpenAI API, None, each
`connected` from offline checks); `linear.estimate` (`agent/cost.py`)
answers an unmeasured setting or a Nonlinear-only run with `available:
false` and the reason. Read `docs/abba_plugin_design.md` and
`docs/abba_installation.md` for the protocol and installation.

### Claude mode

The dialog's Claude choice saves a job via `claude.prepare` and copies a
prompt for Claude Desktop/Code. `doors.api.abba_worker.prepare_linear` and
`checkpoint_callback` are shared by ADK and MCP; never duplicate their
calibration or checkpoint-to-native geometry translation. An authenticated
loopback channel (`doors/mcp/host_channel.py`) carries the MCP checkpoints
and, through the session's `EventRelay`, the tool events (`agent_event`,
plus one `seed` per `show_stack` page with its saved views) to the same
native apply path. Closing the Claude window disconnects the host, not the
MCP job; saved results remain in the job folder next to the exported
snapshots. No Fiji scripting tools are exposed.
