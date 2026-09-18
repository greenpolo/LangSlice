# Independent Fiji connector and LangSlice worker

Accepted product design: a user-managed LangSlice conda environment plus an
independently distributed Java/SciJava plugin in the user's existing Fiji/ABBA.
No combined ABBA installer. Setup and authentication happen inside ABBA, and
either component may be installed first. Future inclusion of the connector in
ABBA's distribution is optional.

## Responsibilities

- `fiji-plugin/`: discover/select the environment, setup/account dialogs, start
  and stop worker processes, snapshot selected sections with calibration, and
  apply returned geometry using ABBA's native actions.
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
| `nonlinear.abba` | Refine the host's current placement and return atlas-to-tissue pairs |

The older `version`, `register.run`, `quick_affine.run` and `export.run` methods
remain available. Host-specific payloads use `event.kind = data` and
`event.payload.kind` to distinguish checkpoints, agent activity and login URLs.

Linear inputs are an image folder, pixel size, filename-to-BrainGlobe-AP positions,
the job settings, and filenames with existing registrations. The host measures
the ABBA/AP mapping from atlas coordinate channels; it never hardcodes an offset.
The initial checkpoint describes the snapshots and carries no host mutations.
Later checkpoints carry replacement corrections in ABBA world millimetres.
The host keeps the original baseline beneath its owned native registration step.

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
