# Independent Fiji connector and LangSlice worker

Accepted product design: a user-managed LangSlice conda environment plus an
independently distributed Java/SciJava plugin in the user's existing Fiji/ABBA.
No combined ABBA installer. Setup and authentication happen inside ABBA, and
either component may be installed first. Future inclusion of the connector in
ABBA's distribution is optional.

## Responsibilities

- `fiji-plugin/`: discover/select the environment, setup/account dialogs, the
  Registration dialog, start and stop worker processes, snapshot selected sections
  with calibration, and apply returned geometry using ABBA's native actions.
- `src/langslice/api/setup.py`: offline installation/credential status, saved
  API keys, and the existing browser OAuth login with a structured URL callback.
- `src/langslice/api/abba_worker.py`: JVM-free linear and nonlinear registration
  requests. The scientific engines remain in their existing packages.
- `src/langslice/api/service.py`: JSON-lines transport. Standard output carries
  protocol messages only; Python and native diagnostic output goes to stderr.

The existing `abba_python` integration remains a separate launcher and reference
implementation. The new worker does not start a JVM or import `abba_python`.

## Protocol version 1

Start `python -m langslice serve --stdio` in the selected environment. Requests
have `{id, method, params}`. Responses have the matching `id` and `type` of
`event`, `result` or `error`. Each connector request owns a worker process;
credentials travel on stdin, never in process arguments. Cancellation terminates
that worker and its child processes rather than queuing a cancel request behind
the synchronous operation.

New methods:

| Method | Purpose |
| --- | --- |
| `setup.status` | Protocol/package version, environment location and offline credential presence |
| `setup.login` | Browser OAuth; emits a `login_url` event and stores tokens in LangSlice |
| `setup.api_key` | Save an OpenAI or Gemini key; no secret in the response |
| `linear.run` | Run calibrated host snapshots through the existing linear engine |
| `linear.estimate` | Estimated cost of a `linear.run` spec; the connector shows "estimate unavailable" when a worker rejects the method |
| `preprocess.preview` | Write the grayscale image the agent would see for one snapshot and preprocessing choice |
| `nonlinear.abba` | Refine the host's current placement and return atlas-to-tissue pairs (kept for later use; the dialog does not call it) |

The older `version`, `register.run`, `quick_affine.run` and `export.run` methods
remain available. Host-specific payloads use `event.kind = data` and
`event.payload.kind` to distinguish checkpoints, agent activity and login URLs.

Linear inputs are an image folder, pixel size, filename-to-BrainGlobe-AP positions,
the job settings, and filenames with existing registrations. Snapshots are TIFFs
with one page per exported ABBA channel. Optional inputs: `preprocessing`
(`mode` auto or custom; custom adds `clahe`, `clahe_strength` and one
`channel_weights` entry per page, in page order), `locked` (snapshot filenames
whose in-plane geometry the agent may not change), `damaged` (filename to the
user's note) and `trace_dir` (save the run's full agent trace in that folder; the
result then lists the new files as `trace_files`). The host measures the ABBA/AP mapping from atlas coordinate
channels; it never hardcodes an offset. The initial checkpoint describes the
snapshots and carries no host mutations. Later checkpoints carry replacement
corrections in ABBA world millimetres relative to the previous checkpoint, plus
`updates_since_start` relative to the initial state; the result carries
`final_updates`, initial to final. The connector applies `final_updates` once when
the run ends, and never changes the user's ABBA during a run. After a stop it
offers to apply the last checkpoint's `updates_since_start`. The host keeps the
original baseline beneath its owned native registration step and refuses a slice
whose registration count changed outside the run.

## ABBA Registration dialog

ABBA's **Register → LangSlice → LangSlice Registration…** (one entry; Setup is in
Fiji's **Plugins → LangSlice** menu and behind the dialog's **Setup…** button) opens
one non-modal dialog for the slices selected when it opens, or all slices. Its
controls map to the job spec as follows:

| Control | Request |
| --- | --- |
| Provider: ChatGPT only; agent model; reasoning | `spec.model` (`openai-oauth/…`), `spec.reasoning` (omitted for "default") |
| Image model | `spec.nonlinear.image_model` |
| Image resolution Low/Medium/High | `spec.image_resolution` |
| Show agent log | log window, or a compact status window with Stop and the final message |
| Open agent viewer | disabled ("Coming soon") |
| Save traces to FOLDER (default `~/LangSlice/traces`) | `trace_dir`; the final message names the saved trace |
| Positioning | `spec.tasks` += `reorder`, `position`; `reorder.flip`, `reorder.hemisphere_cue`, `position.thickness_um`, `position.interval_um` (prefilled from ABBA), `position.notes`; DeepSlice and Bayesian shown disabled |
| Linear | `spec.tasks` += `transform`; `transform.automatic` = affine tool, `interactive` true, `elastix` false, `angles` false (shown disabled), `max_parallel` 1–4, `transform.notes` |
| Nonlinear | shown disabled: "Not yet available in ABBA" |
| Slices tab: Damaged + note | `damaged` |
| Let the agent flag damaged slices | `spec.agent_damage` |
| Allow the agent to overwrite existing transforms (off) | off: every listed slice with registrations is in `locked` |
| Preprocessing tab: Auto/Custom, channel weights, CLAHE, strength | `preprocessing`; the exported pages follow it |
| Snapshot pixel size (µm) | `pixel_size_um` |

At least one of Positioning and Linear must be on. Every choice except the per-slice
damage checks is saved between runs. The estimated cost line calls `linear.estimate`
with `{spec, n_slices, locked}`.

Nonlinear inputs are a common-grid grayscale section and three AP/DV/ML coordinate
channels plus registration settings. Returned point pairs map fixed atlas pixels
to moving tissue pixels. The Java host owns the pixel-to-world conversion and
native registration serialization.

## Credentials and setup

Setup works with no Python installation selected. The plugin loads without
starting Python. It remembers a verified environment path and checks protocol
compatibility before running work. Credentials are owned by the Python worker;
Fiji preferences contain only the environment path.

OAuth uses the existing LangSlice login and token file. API keys are atomically
stored in `~/.langslice/provider_credentials.json` with owner-only POSIX file
permissions. Explicit environment keys take precedence. Saved OpenAI keys use
OpenAI's endpoint unless a base URL was explicitly configured. Offline status
does not refresh tokens or validate provider access. Raw validation inputs and
authentication exception details are excluded from protocol errors.

## Distribution and acceptance

The source environment uses pip inside conda. A true conda package depends on
packaging the missing `itk-elastix` dependency first; see `packaging/README.md`.
The plugin is a separate JAR, with ABBA dependencies provided by the host. A
public Fiji update site must be published before promising checkbox installation.

Acceptance includes both installation orders, absent/moved environments,
credential-free startup, login cancellation, protocol mismatch, paths containing
spaces, selected/all sections, calibrated geometry, native undo and save/reload.
Offline tests must not call model providers or change a user's credentials.
Platform CI and live registration checks are distinct from successful compilation.

The upstream discussion should ask about long-term support for the existing
registration-menu hooks and extension lifecycle. No core dependency change or
`PyCommandBuilder` inclusion is required by this architecture.
