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
program: Fiji's Java plugin and the Claude Desktop, Claude Code and Codex
setups.

## One job through every interface

An interface creates a `JobSpec` or opens saved settings, then calls
operations on a `Job`. `ops/registry.py` controls tool availability;
`doors/declarations.py` supplies the common argument declarations.
Tasks remove whole registration tools, while viewing and channel tools
remain available. See [registration](registration.md).

`state.json` is the checkpoint; every write creates an undo step and rewrites
the derived `registration.json`. Source images remain immutable. Processes
share a file lock and reload changed state before calls. Long operations
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
| `version` | Package and protocol versions |
| `setup.status`, `setup.login`, `setup.api_key` | Offline setup status, OAuth and saved API keys |
| `linear.run`, `linear.estimate` | Run the agent on snapshots; estimate cost |
| `preprocess.preview` | Preview snapshot preprocessing |
| `mcp.prepare` | Save a job and copy prompt for an MCP host |

The request includes snapshots, pixel calibration, positions, cutting angles,
job settings and existing-registration constraints. Only coronal Allen Mouse
V3/V3p1 ABBA sessions are accepted. Position and angle conversion uses ABBA's
atlas offset and centred world coordinates.

The first checkpoint describes the snapshots. Later checkpoints carry
replacement `host_updates` and changed `host_angles`; the final result has
`final_updates`. Each checkpoint becomes an undoable ABBA step. The connector
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

## Verification and diagnostics

Tests cover layer contracts, geometry and the shared interfaces. Golden
fixtures pin model-facing declarations, replies and pictures; intentional
changes are recorded with `LANGSLICE_UPDATE_GOLDEN=1` and visually reviewed.
Image correctness requires inspecting raw output and fitted anatomy, not
only metrics.

[Agent traces](agents.md#logs-and-credentials) record calls and results.
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` additionally writes redacted ADK request
captures. Credentials belong to user settings, not job or ABBA project files.
