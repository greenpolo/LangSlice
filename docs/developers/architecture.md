# Architecture

LangSlice is one Python package in `src/langslice/`. Its interfaces share a
job and the same operations. Package `CLAUDE.md` files (with identical
`AGENTS.md` twins) map the implementation in detail.

## Layers

Dependencies flow downward: `hosts` → `doors` → `tools` → `job` → `core`.
`providers` sits beside the layers and imports `core` only; `engines` (the
in-tree plugins) sits beside `hosts` and imports the plugin API only.
Import-linter enforces the boundaries without exceptions.

| Package | Responsibility |
|---|---|
| `core` | The science and the stack's shapes: the stack and section state (`stack/`), geometry, atlas access, the section files and images (`sections/`), pictures, the registration methods and engines (`fits/`), maps; of a job's settings it reads only the few the workspace carries (`ImageSettings`) |
| `job` | Job folder, checkpoint, undo history, inputs, submit gates and exports; the job's settings (`settings/`: the form `JobSpec`, the table of every setting, what is frozen, the tool table, the needs marks) and the registry of loaded plugins |
| `tools` | Operations on a job; `catalog.py` lists tools, task conditions and call limits; the shared declarations and the job statement (`descriptions/`) |
| `doors` | The ways in: native agent tools, the built-in ADK agent (`agent/`), MCP, the agent CLI, the `langslice` command, the Python library and the worker contract |
| `hosts` | JSON-lines service, ABBA launcher, viewer and log, the LangSlice window (`langslice gui`) |
| `providers` | Model access, beside the layers; receives no task logic |

`core`, `job` and `tools` import no provider or model SDK. Image models are
passed in as arguments.

A call that cannot be done raises one exception, `Refused` (`core/refusal.py`),
with a code from the `RefusalCode` enum (every code the agent can meet, each
with its meaning), a message and the facts behind it; nothing is written for
it. The core, the job and the tools raise it, and each interface turns it into
its own answer once: the tool reply `{"status": "error", "error": code, ...}`,
the agent CLI's envelope with its exit code and fix, the MCP text block, the
library's reply. A plugin raises the same refusal as `ToolError`. `connectors/` contains code installed in another
program: Fiji's Java plugin and the Claude Desktop, Claude Code, Antigravity
and Codex setups.

## One job through every interface

An interface creates a `JobSpec` or opens saved settings, then calls
operations on a `Job`. Every setting is one row of
`job/settings/schema.py`: its name, type, default, command-line flags,
library keyword, `job.json` key and the doors that may set it. The commands'
flags are added from it (`doors/commands/flags.py` holds their help text),
the agent CLI's `schema init` lists those flags, the `job.json` reader
checks keys against it and the LangSlice window's prompt generator spells
its flags from it. The job hands the core only the settings it reads, as
an `ImageSettings` value on the workspace (atlas, plane, preprocessing, the
host's pixel sizes and channel names, the picture level).

`job/settings/tool_table.py` lists the tools, their options and what each
needs, and `tools/catalog.py` binds them to the operations and decides tool
availability; each tool's description is data in `tools/descriptions/`
(one module per tool family), which `render.py` lays out for every door. Tasks remove whole registration tools, a job's `disabled`
list removes single tools and options, and viewing and image tools remain
available otherwise. Model-facing text that names a tool or an option is
written with what it needs (`job/settings/needs.py`) and rendered for the
job. See [tools and options](../reference/tools.md).

A job can load [tool plugins](plugins.md): `JobSpec.plugins` names modules
whose tools join the toolbox. `core/plugins.py` holds the plugin types and
`job/plugins.py` the declaration checks and the registry of loaded plugins,
which `job/settings/tool_table.py` (`tools_for(spec)`) and
`tools/catalog.py` (`verbs_for(spec)`) merge with the native tables for that
spec; `doors/plugins/` loads them, gives each handler a context that writes
through the same operations, and builds their declarations, provenance and the
MCP adapter. The native tables are unchanged, and a job without plugins has
none of this in any declaration, reply or picture.

`state.json` is the checkpoint; every write appends a line to the history of
each section it changed (`sections/<name>/history.jsonl`, `job/history.py`),
records one undo step in `history/undo.json` (the line each section held
before it), and rewrites the derived `registration.json` once, as the write's
hold of the job folder's lock ends. Undo, redo and `restore_registration` put
lines of a section's history back on it; a deformation record no line names
is deleted. Source images remain immutable. Processes
share a file lock and reload changed state before calls; a read never waits
for the lock and reads the state as last saved. Each agent CLI
call is a process of its own. Long operations
compute outside the lock, then validate their inputs before applying;
conflicting section results are refused as `STALE_INPUT`.

`submit` validates the enabled tasks and writes the QuickNII and VisuAlign
exports; the per-section maps are written only by `export_maps` (CLI and
Python), with or without submission. The [file contract](../reference/file_formats.md)
is independent of which interface produced the registration.

## Fiji connector and worker

The Java plugin in `connectors/fiji/` targets ABBA 0.25 on Java 21. It exports
calibrated section snapshots and launches `langslice serve --stdio` in the
selected Python environment. The worker starts no JVM. Installation and
user workflow are in [ABBA](../abba.md).

Protocol version 3 uses one JSON object per line. Requests have `id`,
`method`, `params`; responses carry the same `id` and `type` of `event`,
`result` or `error`. Stdout is protocol-only; diagnostics go to stderr.
The contract is `doors/abba/contract.py`, the preparation
`doors/abba/prepare.py`, the follower `doors/abba/follow.py`, the
checkpoint rows `doors/abba/checkpoints.py`, and transport is in
`hosts/service.py`.

| Method | Purpose |
|---|---|
| `setup.status` | The installation: version, protocol, Python and its environment (offline) |
| `abba.prepare` | Make a prepared job from an ABBA session's snapshots |
| `abba.follow` | Stream a job's checkpoints as ABBA rows until the request's input closes |

`abba.prepare` takes the snapshots, their pixel size, positions, ABBA's
stack-wide cutting angles and slicing offset, the slices the user registered
or warped in ABBA, the page names and the session's facts (project key,
atlas, slice names, ABBA's median thickness and interval). It writes the job
folder beside the snapshots with `job.json` only: ABBA's inputs, the user's
ABBA registrations frozen, the cutting-angle estimate on, and the session's
facts under `host`. It writes no checkpoint, and refuses once the job has
one. Whichever way the job is then opened (`langslice-job FOLDER init`,
`linear run`, MCP, the library), the saved settings are the defaults of
that opening (`job.host_inputs.opening_spec`). Only coronal Allen Mouse
V3/V3p1 ABBA sessions are accepted. Position and angle conversion uses ABBA's
atlas offset and centred world coordinates.

`abba.follow` reads the job folder only. Every 0.5 s it looks at
`state.json`; when it changed, it sends one checkpoint, so several writes
between two looks are one ABBA step. The first checkpoint is the job's whole
state measured against ABBA with this job's own steps removed; later
checkpoints carry replacement `host_updates` and changed `host_angles`; a
warp that cannot be converted is sent as `{"error": …}` and kept as a failed
update. A job that has no checkpoint yet gets one `waiting` event. Each
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

A saved ABBA job (a job folder whose `job.json` carries a `job_id`, found
through `~/.langslice/jobs/<id>.json`) opens with
`start_job(job_id=...)`. MCP opening pages must be read before writes are
accepted. Its live updates go to a loopback port with a single-use 256-bit
token, which accepts LangSlice result events, not arbitrary Fiji commands.

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

A golden test (`tests/test_golden_outputs.py`) pins everything else a run
produces, so a change meant to alter no output can show that it alters
none. Scripts in `tests/golden/` register three real sections to
`allen_mouse_50um` with no model, once through each door, each door a test
of its own so a failure names the door that changed:

- `library`: the public library calls every tool that needs no image
  model (positions, hand transforms, `elastix_affine`, section images,
  masks, a damage mark, `fireants_syn` coarse on the CPU, landmarks, undo
  and redo, `restore_registration`, `export_maps`, the atlas lookups and
  job-folder reads, `note`, refusals and `submit`), with every `look` mode,
  a zoom and `status` after each write and `look` with its other options;
- `cli`: the installed `langslice-job` command, one process per call: the
  catalog, `init`, reads, writes, a dry run, a fit, refusals and a launch
  prompt, each call's exit code and printed envelope;
- `mcp`: the installed `langslice mcp` server over stdio with the `mcp`
  library's client: the greeting, the tool lists (names, descriptions,
  input schemas) before and after `start_job`, every `show_stack` page and
  reads, writes and refusals, each result's text blocks whole and image
  blocks as decoded-pixel digests.

The declarations LangSlice's own agent sends its model have no public
entry point without a model call; the text snapshot pins them. Each script
runs in a child interpreter on the checkout's code, through public entry
points only, with a fresh home folder and only the caller's `PATH` and
locale, so no setting, credential or saved job of the user's reaches it;
BrainGlobe's configuration there points at the user's atlas folder. The
test compares, with no tolerance, every reply, every JSON and text file of
the job folder (the event log, the agent CLI's call log and the engine log
included; paths, clock times, durations and call ids replaced by tokens),
and the decoded pixels or arrays of every picture, map, mask, section
image and deformation record with the goldens. The pictures the writing
tools return are kept there whole. The sections are unpublished data, so
they and the goldens live outside the repository, in `tests/golden/data/`
(git-ignored) or the folder `LANGSLICE_GOLDEN_DATA` names, and the test is
skipped where they are missing. It needs the atlas in the local BrainGlobe
cache, records this machine's pixels and takes about 30 s with the doors in
parallel, so the default run leaves it out: run it with
`python -m pytest -m golden`. A failure writes the run's outputs, every
picture among them, to `actual/<door>/` beside the goldens and lists each
changed key (a changed text around its first difference), array and
picture (the differing pixels and the largest difference), headed by the
door and the commit the goldens were recorded at. An intended change is
recorded on `main` with `LANGSLICE_UPDATE_GOLDEN=1 python -m pytest -m
golden`, which rewrites the goldens, records the commit, branch, date and
uncommitted files in each door's `provenance.json` and fails once, printing
the diff, so that it is reviewed. Because the goldens are not under version
control, a branch that regenerated its own would pass on its own changes:
the update refuses to run on any branch but `main` unless
`LANGSLICE_GOLDEN_FORCE=1` is set as well.

[Agent traces](../agents/builtin.md#logs-and-credentials) record calls and results.
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` additionally writes redacted ADK request
captures. Credentials belong to user settings, not job or ABBA project files.
