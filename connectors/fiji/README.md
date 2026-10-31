# LangSlice Fiji connector

This independently distributed Java plugin opens LangSlice from an existing ABBA
session and mirrors the LangSlice job into ABBA while ABBA is open. It does not run
the registration: the user's agent app (Codex, Claude Code, Antigravity) does, from
the message the LangSlice window's prompt generator writes. Python is not needed to
load the plugin; the connector starts a separately installed LangSlice environment.

The connector targets ABBA **0.25.x** (built and tested against 0.25.0) and **Java 21**,
the Java level ABBA 0.25 itself requires. It also runs on ABBA 0.24.1, where a row ABBA
refuses is reported without ABBA's own reason (ABBA keeps a list of failures from 0.25
on). Older ABBA versions (0.11 and earlier) are not supported. Runtime dependencies are
supplied by ABBA/Fiji and are not bundled into the plugin JAR. LangSlice works on
coronal sessions of ABBA's Allen Mouse V3 or V3p1 atlas.

## Build and install

With JDK 21+ and Maven:

```bash
cd connectors/fiji
mvn package dependency:build-classpath -Dmdep.outputFile=target/test-classpath.txt
```

After Maven has written the resolved dependency classpath, development builds can run
offline:

```bash
python connectors/fiji/build.py --java-home /path/to/jdk21 --test
```

The offline build starts from empty `target/classes` and `target/test-classes`
folders, so a renamed or deleted command cannot linger in the JAR or the SciJava
plugin index. It uses the exact Maven-resolved classpath, never every JAR found in
the Maven cache.

Copy `target/langslice-fiji-0.3.0.jar` into your Fiji `jars` folder and restart Fiji.
Do not copy the Maven dependency jars into Fiji. The connector and LangSlice must be
updated together: this version speaks worker protocol 3 and refuses any other with
"Update LangSlice and the connector together".

## Menu entries

- **Register › LangSlice…** in ABBA (and **Plugins › LangSlice › Open LangSlice…**):
  opens LangSlice on this ABBA project (below). `LangSliceService` registers the entry
  with ABBA's registration menu registry before the ABBA window is built, with a
  callable alias, since ABBA runs commands by class identifier; ABBA places it after a
  separator below its own entries. No upstream ABBA change is needed.
- **Plugins › LangSlice › Choose Python environment…**: the environment dialog.
- **Plugins › LangSlice › Stop following**: detaches the follower from this ABBA
  session without closing it; ABBA keeps what it has.

## What Register › LangSlice… does

The entry works on the slices selected in ABBA, or on every slice when none is
selected.

1. **Environment.** The saved LangSlice environment is used; when none is saved, the
   environment dialog opens first. The connector runs `setup.status` in it; when that
   fails, it says so and opens the dialog.
2. **Project folder.** The ABBA project's job lives in
   `~/.langslice/abba/<project key>/`. ABBA 0.25 keeps no accessor for its project
   file, so the key is computed from the session: the first 12 hexadecimal digits of
   sha256(atlas name, a newline, then the names of all the session's slices, sorted and
   joined by newlines). The same project therefore finds its folder again whatever is
   selected, after a restart too. Two ABBA projects that import the same files share a
   key and a folder.
3. **The one question.** When the folder's job has other sections than the selection,
   or when the key has no folder but another project folder (same atlas) shares slice
   names with the session (sections were added or removed since), one message box
   asks: **Continue the existing job (N sections)** or **Start a new job with the M
   selected slices**. Continuing another folder records the session's key in its
   `abba_project.json` (`keys`), so the question is not asked again. Starting a new job
   moves the old folder to `<folder>.old-<yyyyMMdd-HHmmss>` and removes the old job's
   steps from the selected sections (one ABBA Undo step brings them back). Closing the
   box changes nothing.
4. **A new job** (first use, or a new start): the connector removes this project's
   earlier LangSlice steps from the selection (none on first use), exports one
   calibrated 25 µm snapshot per section (`section_0001.tif …`, atlas order, one TIFF
   page per channel) into the project folder, writes `abba_sections.json` (snapshot to
   ABBA slice name) and asks the worker to **prepare** the job (`abba.prepare`, below),
   then writes `abba_project.json`. ABBA itself does not change. A later use continues
   the job instead.
5. **Follow.** The connector starts the follower on the job (below). A follower already
   following this job is kept, unless some of its rows were refused; then it attaches
   again, which retries them.
6. **The LangSlice window.** The connector starts `langslice gui <job folder> --tab
   prompt` (`--tab viewer` once the job has a checkpoint) as a process of its own; it
   is not ended with ABBA or Fiji. Its output goes to `gui.log` in the project folder,
   and its first lines (the window's address among them) to Fiji's log. A window
   process that ends within 3 seconds is reported in a message box with its first
   output lines.

Files in `~/.langslice/abba/<project key>/`:

| File | What it is |
| --- | --- |
| `section_0001.tif …` | the calibrated snapshots, the job's images |
| `langslice/` | the job folder |
| `abba_project.json` | `atlas`, `slice_names` (all the session's slices at creation), `sections` (snapshot to slice name), `created_at`, and `keys` (sessions that continued this folder) |
| `abba_sections.json` | snapshot filename to ABBA slice name, the job's sections |
| `gui.log` | the last LangSlice window's output |
| `langslice/logs/abba_applied.jsonl` | one line per applied checkpoint (below) |

## The follower

The follower is one worker process per followed ABBA session (`langslice serve
--stdio`, method `abba.follow`), started on a background thread with no timeout. The
worker reads the job's `state.json` as it changes and streams checkpoints; the
connector applies each as **one ABBA undo step**, through ABBA's native actions.
Several LangSlice writes between two of the worker's reads arrive as one checkpoint,
so one ABBA Undo step may hold several LangSlice steps. A LangSlice `undo` is a change
like any other and arrives as a forward step that puts the section back.

A checkpoint may carry, per section, a position, a new placement (flip, quarter turn,
affine) and a warp, and for the stack new cutting angles (`host_angles`). Each section
carries at most two steps of this job on top of its own registrations:

- **`LangSlice affine [<project key>]`**: ABBA's `AffineRegistration`;
- **`LangSlice warp [<project key>]`** on top of it: ABBA's BigWarp thin-plate spline
  (`BigWarpSource2DRegistration`; at most 1089 landmarks, its measured difference from
  LangSlice's deformation written in Fiji's log and kept in the step's parameters).

A replacement deletes the newest steps (the warp first, then the affine when the
placement changed) and appends new ones, only while this job's steps are still the
section's newest. Steps of another project key count as the section's own, as a
user's would. Cutting angles are applied through a small undoable action of the
connector's own, since ABBA's own angle command cannot be undone.

**Attach.** Each time the follower starts (first use, the menu entry again, ABBA
reopened), the connector removes this job's steps from the top of each section and
takes what remains as the section's baseline; the removal happens inside the first
checkpoint's undo step, and the worker's first checkpoint is the job's full state. So
ABBA is put where the job is in one undo step, and nothing about earlier Java objects
needs to survive a restart. A job that has not run yet (the worker sends `waiting`)
gets only the removal.

**Refused sections.** A section whose registrations were changed in ABBA (a step added,
removed or undone with ABBA's Undo) is refused and named in Fiji's log; its rows are
kept, merged with later checkpoints and retried, and never stop the others. A section
with this job's step under another step is refused at attach and nothing is removed
from it. When ABBA itself refuses an action, the reason ABBA recorded for that slice is
added. A warp the worker could not convert (`"warp": {"error": …}`) is such a row:
ABBA keeps the warp it had. Choosing the menu entry again attaches again: ABBA is
brought back to the job (a refused section once its foreign step on top is removed).
Each applied checkpoint is written to Fiji's log when something was refused, and to
`<job>/logs/abba_applied.jsonl` as one line `{at, applied, failed: {id: reason},
angles, angles_failed?}`; nothing reads that file back.

**Ending.** Closing ABBA's session (its `MultiSlicePositionerListener.closing`) or
**Stop following** closes the worker's stdin; the worker finishes, and is terminated
after 3 seconds if it does not. Nothing else ends the follower. The worker's stderr
lines go to Fiji's log.

Saved projects contain only ABBA's own registration types, so they reopen in ABBA
without LangSlice, and the warp step opens in BigWarp like any BigWarp registration.
Save the project with ABBA's normal Save command.

## The environment dialog

Plugins › LangSlice › Choose Python environment… (and automatically when no
environment is saved): the environment folder (an editable list and Browse), **Check
installation** (`setup.status`: the LangSlice version and its Python), **Save**
(enabled after a check that found protocol 3) and **Installation instructions** (the
LangSlice repository, whose README leads to the installation guide).

The list comes from conda's registry and the base environments of common conda and
mamba installations (`CONDA_EXE`, `MAMBA_ROOT_PREFIX`, and `miniforge3`, `miniconda3`,
`anaconda3`, `mambaforge` or `micromamba` in the home folder); the saved environment
is listed first. When available, `conda run --prefix` supplies activation variables
without an open terminal; otherwise the environment's Python is invoked directly.
Worker processes and their descendants are terminated when a request ends.

## Protocol

Protocol version 3. A request is one JSON line sent to `python -m langslice serve
--stdio`; stdin stays open until the result, and the worker streams events and then a
result or error. Images and registration outputs remain local files; no port is
opened. Three methods:

- `setup.status`: `{version, protocol_version: 3, python_executable,
  environment_prefix}`.
- `abba.prepare`: `{image_folder, pixel_size_um, positions_mm: {snapshot: AP mm},
  z_offset_mm, angles_deg: {pitch_deg, yaw_deg}, existing_warp: [snapshot…],
  registered: [snapshot…], channel_names, abba: {project_key, atlas, sections:
  {snapshot: slice name}, thickness_um, interval_um}}`. The worker writes the prepared
  job (`job.json`, no checkpoint) and answers `{job_folder}`. From ABBA:
  `positions_mm` is ABBA's `toAtlasZ` of each slice (BrainGlobe AP millimetres),
  `z_offset_mm` is `ReslicedAtlas.getZOffset()`, `angles_deg` are the session's
  stack-wide cutting angles (pitch = −degrees(rotateX), yaw = −degrees(rotateY)),
  `registered` lists the sections that carry registrations, `existing_warp` the ones
  carrying a spline or BigWarp step that is not LangSlice's, `channel_names` one name
  per page, and `thickness_um` / `interval_um` the medians of ABBA's slice thicknesses
  and position gaps (0 when ABBA gives none; no interval when the project folder
  already had a job, whose positions may be the agent's).
- `abba.follow`: `{job_folder}`; streams `checkpoint` (`host_updates`, `host_angles`),
  `log` (`message`) and `waiting` events until stdin closes, then answers `{}`.

## Tests

`build.py --test` runs two offline checks. `ConnectorSmokeTest` starts real
fake-worker subprocesses speaking protocol 3 and checks: events and results through
the pipe, paths containing spaces, timeouts, that a worker's stderr never reaches an
error and is drained for every request but the follower's; the follower (two streamed
checkpoints applied in order, its log lines and stderr in Fiji's log, a refused row
logged and written to `abba_applied.jsonl`, stopping that closes stdin, a worker that
ignores stdin's end terminated, a prepared job's `waiting`); the window process
(`gui JOB --tab prompt`, and one that ends at once reported with its output); the
project key (stable, order-free), finding a folder again, candidates, continuing one,
the start-over move; multi-page snapshots; and the menus (one Register entry and its
callable alias; Plugins › LangSlice holds its three commands once each).
`HostGeometrySmokeTest` checks native affine registration geometry and serialization,
the warp step's direction, its coordinate rows, its serialization and that BigWarp can
reopen it, the cutting-angle signs, and how a kept row merges with a newer one.

`HostSessionSmokeTest` is opt-in and needs ABBA's cached Allen V3p1 atlas (normally
`~/cached_atlas`). It runs a real headless ABBA 0.25.0:

```bash
java -Djava.awt.headless=true --add-opens=java.base/java.lang=ALL-UNNAMED \
  -cp "target/test-classes:target/classes:$(cat target/test-classpath.txt)" \
  org.langslice.fiji.HostSessionSmokeTest ~/cached_atlas
```

It imports two synthetic sections and checks: AP positions from ABBA's own `toAtlasZ`
against ABBA's atlas coordinate images read at the slice; a tilted session's export
(`angles_deg`, `existing_warp` for a user's own BigWarp step, `registered`, the `abba`
block, `abba_sections.json`); live checkpoints (position, affine step, warp step,
cutting angles) read back from ABBA; that a new placement removes the warp before
replacing the affine; that one ABBA Undo reverts exactly one checkpoint (angles
included) and Redo restores it; that a slice changed outside is refused, kept and
applied on retry; that a registration ABBA refuses fails with ABBA's own reason; native
save; and attach: this job's steps removed and the job applied again in one Undo step,
another key's step and the user's own kept, a section with this job's step under a
user's step refused with nothing removed. It then reopens the saved project in a
second JVM without the connector on the classpath (`ReloadCheck`): both steps, the
warp's landmarks, BigWarp editing, the position and the angles survive.

`E2EFollowCheck` runs everything together: a real headless ABBA, the real worker and
the agent CLI. The connector prepares a job from two synthetic sections and follows
it; `e2e/driver.py` stands in for the agent app, calling `langslice-job FOLDER VERB`
as processes (`init`, `position_sections`, `interactive_transform`, `fireants_syn`,
`undo`, `submit`); ABBA must show the job's positions, angles and steps with every row
applied. The session is then closed (the follower ends), the saved project reopened in
a new session and attached again: one checkpoint, identical geometry, and one ABBA
Undo reverts the whole sync. `tests/test_abba_e2e.py` runs `build.py --test` and this
check; it is marked `slow` and runs only with:

| Variable | Meaning |
| --- | --- |
| `LANGSLICE_ABBA_E2E=1` | opt in |
| `LANGSLICE_FIJI_JDK` | a JDK 21 home |
| `LANGSLICE_ABBA_ATLAS_CACHE` | ABBA's atlas cache (default `~/cached_atlas`) |

It also needs `target/test-classpath.txt` (Build, above).
