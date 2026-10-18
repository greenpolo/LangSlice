# Architecture

LangSlice is one Python package in `src/langslice/`. Its interfaces share a
job and the same operations. Package `CLAUDE.md` files (with identical
`AGENTS.md` twins) map the implementation in detail.

## Layers

Dependencies flow downward: `hosts` → `doors` / `agent` → `ops` → `job` →
`core`. Import-linter enforces the boundaries without exceptions.

| Package | Responsibility |
|---|---|
| `core` | State and job spec, calibrated geometry, atlas access, pictures, registration engines, image-model border processing |
| `job` | Job folder, checkpoint, undo history, submit checks and exports |
| `ops` | Operations on a job; `registry.py` lists tools, task conditions and call limits |
| `doors` | Shared declarations, native agent tools, MCP, CLI, Python library and worker contract |
| `agent` | Built-in ADK agent session |
| `hosts` | JSON-lines service, ABBA launcher, viewer and log |
| `providers` | Model access, beside the layers; receives no task logic |

`core`, `job` and `ops` import no provider or model SDK. Image models are
passed in as arguments. `connectors/` contains code installed in another
program: Fiji's Java plugin and the Claude Desktop, Claude Code, Antigravity
and Codex setups.

## One job through every interface

An interface creates a `JobSpec` or opens saved settings, then calls
operations on a `Job`. `core/toolset.py` lists the tools, their options and
what each needs, and `ops/registry.py` binds them to the operations and
decides tool availability; `doors/declarations.py` supplies the common
argument declarations. Tasks remove whole registration tools, a job's
`disabled` list removes single tools and options, and viewing and channel
tools remain available otherwise. Model-facing text that names a tool or an
option is written with what it needs (`core/needs.py`) and rendered for the
job. See [registration](registration.md).

A job can load [tool plugins](plugins.md): `JobSpec.plugins` names modules
whose tools join the toolbox. `core/plugins.py` holds the plugin types and the
registry of loaded plugins, which `core/toolset.py` (`tools_for(spec)`) and
`ops/registry.py` (`verbs_for(spec)`) merge with the native tables for that
spec; `doors/plugins/` loads them, gives each handler a context that writes
through the same operations, and builds their declarations, provenance and the
MCP adapter. The native tables are unchanged, and a job without plugins has
none of this in any declaration, reply or picture.

`state.json` is the checkpoint; every write creates an undo step and rewrites
the derived `registration.json` once, as the write's hold of the job folder's
lock ends. Source images remain immutable. Processes
share a file lock and reload changed state before calls. The agent CLI's calls
on one job folder are answered by that folder's warm process
(`doors/cli/warm.py`, [details](agents.md#the-warm-process)): it loads the
libraries and the atlas once and forks a child per call, which runs the cold
call's code as a process of its own. Long operations
compute outside the lock, then validate their inputs before applying;
conflicting section results are refused as `STALE_INPUT`.

`submit` validates the enabled tasks and writes maps and exports. CLI and
Python also expose `export_maps` without submission. The [file contract](file_formats.md)
is independent of which interface produced the registration.

## Fiji connector and worker

The Java plugin in `connectors/fiji/` targets ABBA 0.24 on Java 21. It exports
calibrated section snapshots and launches `langslice serve --stdio` in the
selected Python environment. The worker starts no JVM. Installation and
user workflow are in [ABBA](abba.md).

Protocol version 1 uses one JSON object per line. Requests have `id`,
`method`, `params`; responses carry the same `id` and `type` of `event`,
`result` or `error`. Stdout is protocol-only; diagnostics go to stderr.
The contract is `doors/api/models.py`, preparation and translation are in
`doors/api/abba_worker.py`, and transport is in `hosts/api/service.py`.

| Method | Purpose |
|---|---|
| `setup.status`, `setup.login`, `setup.api_key` | Offline setup status (including `tools`: what a job can switch off), OAuth and saved API keys |
| `linear.run`, `linear.estimate` | Run the agent on snapshots; estimate cost |
| `preprocess.preview` | Preview snapshot preprocessing |
| `mcp.prepare` | Save a job and copy prompt for an MCP host |

The request includes snapshots, pixel calibration, positions, cutting angles,
job settings and existing-registration constraints. Only coronal Allen Mouse
V3/V3p1 ABBA sessions are accepted. Position and angle conversion uses ABBA's
atlas offset and centred world coordinates.

The first checkpoint describes the snapshots. Later checkpoints carry
replacement `host_updates` and changed `host_angles`; a warp that cannot be
converted is sent as `{"error": …}` and kept as a failed update. Each
checkpoint becomes an undoable ABBA step. The connector
owns at most one affine and one BigWarp registration per section, above its
pre-existing registrations. Deformations become thin-plate splines sampled
on a 9×9 to 33×33 grid, refined toward a 5 µm error target. Reported spline
errors do not withhold a warp; a spline folding at every grid is refused.

Updates replace LangSlice's steps only while they remain the newest
registrations. External edits cause that section's update to be refused;
failed updates remain available for retry. Saved projects contain ABBA's
native registration types and reopen without LangSlice.

### MCP from ABBA

`mcp.prepare` uses the same preparation and checkpoint translator as the
built-in agent, without calling a model. Snapshots and their job remain
under `~/.langslice/snapshots/mcp-*`; a saved job id points to them.
`start_job(job_id=...)` opens that job. MCP opening pages must be read before
writes are accepted.

The connector listens on a loopback port with a single-use 256-bit token.
It accepts LangSlice result events, not arbitrary Fiji commands. Closing
the progress window disconnects live updates but leaves the MCP job usable
and its results on disk. Each direct worker request owns its process;
cancelling it terminates that worker and its children.

## Atlas cache

BrainGlobe keeps the atlases it downloads in `~/.brainglobe`. LangSlice keeps
two volumes it derives from an atlas in `~/.langslice/atlas-cache/`, one
folder per atlas name and version: the template's cubic spline coefficients,
from which every tilted template plane is sampled, and the annotation as
indices into its sorted region ids, from which every tilted label plane is
sampled. The first run on an atlas builds them (about 1.6 GB for
`whs_sd_rat_39um`, 0.4 GB for `allen_mouse_25um`); later processes open them
memory-mapped, which takes about 5 s off the start of a run on an angled rat
stack and off every `langslice-job` call that draws a tilted plane. Planes
sampled from the cache are identical to planes sampled without it.

`LANGSLICE_ATLAS_CACHE` names another folder, or `off` turns the cache off.
Atlases of more than 500 million voxels, such as `allen_mouse_10um`, are not
cached. A cache folder that cannot be written, or a damaged file, is never an
error: the volume is computed in memory, and a damaged file is rewritten. A
new BrainGlobe version of an atlas gets a new folder; old folders can be
deleted at any time.

## Verification and diagnostics

Tests cover layer contracts, geometry and the shared interfaces, and run in
parallel. A text snapshot pins what the agent is told (tool declarations, job
statements, refusals); an intended change is regenerated with
`LANGSLICE_UPDATE_SNAPSHOT=1` and its diff reviewed. Picture changes are
reviewed by looking at them.
Image correctness requires inspecting raw output and fitted anatomy, not
only metrics.

[Agent traces](agents.md#logs-and-credentials) record calls and results.
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` additionally writes redacted ADK request
captures. Credentials belong to user settings, not job or ABBA project files.
