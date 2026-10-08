# Fiji connector and LangSlice worker

A Java/SciJava plugin (`connectors/fiji/`) in the user's existing Fiji/ABBA
talks to a LangSlice worker in a separate, user-managed conda environment.
Setup and sign-in happen inside ABBA.

## Responsibilities

- `connectors/fiji/`: environment discovery, the setup and account dialogs, the
  Registration dialog, starting and stopping worker processes, snapshotting
  the selected sections with calibration, and applying the worker's results
  with ABBA's native undoable actions. It targets ABBA 0.24.x on Java 21.
- `src/langslice/doors/api/setup.py`: offline installation and credential
  status, saved API keys, the browser OAuth login.
- `src/langslice/doors/api/abba_worker.py`: JVM-free linear registration of the
  snapshots; the scientific code stays in `core/`, `job/`, `ops/`.
- `src/langslice/hosts/api/service.py`: the JSON-lines transport.

The worker starts no JVM and imports no `abba_python`.

## Protocol version 1

`python -m langslice serve --stdio` in the selected environment. A request is
`{id, method, params}`; a response carries the same `id` and `type` of
`event`, `result` or `error`. Standard output carries protocol messages only;
diagnostics go to stderr. Each connector request owns a worker process;
credentials travel on stdin, never in arguments; cancelling terminates the
worker and its children.

| Method | Purpose |
|---|---|
| `version` | protocol and package version |
| `setup.status` | environment location and offline credential presence |
| `setup.login` | browser OAuth; emits a `login_url` event |
| `setup.api_key` | save an OpenAI or Gemini key (never echoed) |
| `mcp.prepare` | validate the host snapshots and save a job for an MCP host such as Claude; returns `job_id`, `job_dir` and the prompt to copy |
| `linear.run` | run calibrated snapshots through the agent |
| `linear.estimate` | estimated cost of a `linear.run` spec |
| `preprocess.preview` | the grayscale image the agent would see for one snapshot and preprocessing choice |

Host-specific payloads use `event.kind = data` and `event.payload.kind`
(checkpoints, agent activity, login URLs).

A `linear.run` carries an image folder of snapshots (TIFFs, one page per
exported ABBA channel), the pixel size, filename-to-AP positions, the job
settings and the filenames that already carry registrations. Optional:
`preprocessing` (`auto`, or `custom` with `clahe`, `clahe_strength` and one
`channel_weights` entry per page), `locked`, `damaged` (filename to note),
`trace_dir` (the result lists `trace_files`), `z_offset_mm` (ABBA's
`ReslicedAtlas.getZOffset()`; positions use ABBA's own `toAtlasZ` /
`fromAtlasZ`), `angles_deg` (stack-wide cutting angles: pitch =
-degrees(rotateX), yaw = -degrees(rotateY)), `existing_warp` (sections carrying
a spline or BigWarp step that is not LangSlice's), `channel_names` and
`nonlinear_skip`. Only coronal Allen Mouse V3 / V3p1 sessions are accepted.

The first checkpoint describes the snapshots and carries no changes. Later
checkpoints carry replacement corrections in ABBA world millimetres relative to
the previous checkpoint (`host_updates`, and `host_angles` when the angles
changed) and `updates_since_start`; the result carries `final_updates`.

The connector applies each checkpoint as one ABBA undo step, in ChatGPT and
Claude mode alike. Per section it owns at most a LangSlice affine step
(`AffineRegistration`) and, above it, a LangSlice warp step
(`BigWarpSource2DRegistration`), on top of the registrations the section had
when the run started. A row's `warp` is `{source_mm, target_mm, record,
max_error_mm, p99_error_mm, points}` in the centred ABBA world frame; the
thin-plate spline maps `target_mm` onto `source_mm`, after the affine step. The
two errors say how far ABBA's spline is from LangSlice's deformation; they never
withhold the warp. `warp: null` removes the step. Order: delete the warp,
delete the affine when the placement changes, append the new affine, append the
new warp. A step is replaced only while LangSlice's steps are the section's
newest registrations; a section changed outside the run is refused and named.
`host_angles` goes through an undoable action of the connector's own. A failed
row is reported, kept, merged into the next checkpoint and retried; the run
window offers **Retry failed updates** at the end. Saved projects hold only
ABBA's own registration types: they reopen without LangSlice.

## Registration dialog

**Register > LangSlice Registration...** opens one non-modal dialog for the
slices selected when it opens (or all). Controls map to the job spec:

| Control | Request |
|---|---|
| Provider ChatGPT / Claude; agent model; reasoning | `spec.model` (`openai-oauth/...`), `spec.reasoning`; Claude saves a job via `mcp.prepare` |
| Image model (always with "None (fit to the stain only)") | `spec.nonlinear.provider`, `spec.nonlinear.image_model` |
| Image resolution | `spec.image_resolution` |
| Show agent log; Open agent viewer (only when ABBA was started with `langslice abba`) | log window or compact status window; `viewer` in the `run_started` event |
| Save traces to FOLDER | `trace_dir` |
| Positioning | task `position`; `position.thickness_um`, `interval_um` (from ABBA), `notes` |
| Linear | task `transform`; `transform.flip`, `hemisphere_cue`, `angles`, `notes` |
| Nonlinear | task `nonlinear`; `nonlinear.notes`. With Linear off and unregistered slices listed, Run asks whether the agent should align them first (No leaves them out of Nonlinear) |
| Slices tab: note on damaged tissue; agent may mark damaged regions | `damaged` (filename to note; a note, not a damage mark); `spec.agent_damage` |
| Allow the agent to overwrite existing transforms (off) | off: every listed slice with registrations is in `locked` |
| Preprocessing tab: Auto / Custom, weights, CLAHE; agent-driven preprocessing | `preprocessing`; `spec.agent_preprocessing` |

At least one of the three tasks must be on. The cost line calls
`linear.estimate` with `{spec, n_slices, locked}`.

## Claude mode

`mcp.prepare` takes the exact `linear.run` parameters plus `notes` and an
optional `host_channel`. It shares `abba_worker.prepare_linear` and the
checkpoint translator with the ADK path, loads no credentials and calls no model.
The snapshots stay under `~/.langslice/snapshots/mcp-*` (no automatic
cleanup). The job lives in the job folder beside them
(`.../langslice/`): `job.json` holds the settings, the notes and the original
request; `prompt.txt` the copy prompt; `~/.langslice/jobs/<id>.json` leads
there and records the host channel.

`langslice mcp` starts without a folder. `start_job(job_id=...)` loads the job
and returns the same job statement LangSlice's own agent gets, with the status
table; `show_stack(page)` serves the opening strips (at Claude's 1568 px long
edge), 1-based pages bounded to 680,000 serialized bytes. Every tool reply has
the same budget; past it its pictures are shrunk together. Writes are refused
(`OPENING_NOT_READ`) until every page has been read. Reopening a job resumes
from its checkpoint; keep one session per job.

The Java listener binds `127.0.0.1` on an OS-chosen port; its first JSON line
must carry the 256-bit token, and the first accepted connection consumes the
listener. Later lines are the worker's event/result envelopes with the job id
as `id`; checkpoints reach `AbbaHostSession.applyCheckpoint`, the same
implementation ChatGPT mode uses. Closing the progress window closes the
listener; a refused or disconnected host does not stop the MCP tools, and
results stay in the job folder (`state.json`, `exports/`). The channel accepts
only LangSlice result events, never Fiji scripts or arbitrary commands.

## Credentials

Setup works with no Python selected; the plugin loads without starting Python
and remembers a verified environment path (the only thing in Fiji's
preferences). OAuth uses `langslice login`'s token file. API keys are stored
atomically in `~/.langslice/provider_credentials.json` (owner-only on POSIX);
explicit environment variables take precedence. Offline status does not
refresh tokens or validate provider access.
