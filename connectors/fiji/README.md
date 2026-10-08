# LangSlice Fiji connector

This independently distributed Java plugin connects an existing ABBA installation
with a separately installed LangSlice Python environment. Python and model accounts
are not needed to load the plugin or open Setup.

The connector targets ABBA **0.24.x** (built and tested against 0.24.1) and **Java 21**,
the Java level ABBA 0.24 itself requires. Older ABBA versions (0.11 and earlier) are
not supported: upstream renamed the registration and source-processor packages.
Runtime dependencies are supplied by ABBA/Fiji and are not bundled into the plugin JAR.

## Build

With JDK 21+ and Maven:

```bash
cd connectors/fiji
mvn package dependency:build-classpath -Dmdep.outputFile=target/test-classpath.txt
```

After Maven has written the resolved dependency classpath, subsequent development
builds can run offline:

```bash
python connectors/fiji/build.py --java-home /path/to/jdk21 --test
```

The offline build starts from empty `target/classes` and `target/test-classes`
folders, so a renamed or deleted command cannot linger in the JAR or the SciJava
plugin index. It uses the exact Maven-resolved classpath, never every JAR found in
the Maven cache.

The offline smoke checks start real fake-worker subprocesses to exercise event and
result delivery, timeout handling, secret-safe authentication errors, and paths
containing spaces. They also check: the ABBA menu registration (one plain
`LangSlice Registration…` entry) and its callable alias; the job spec, preprocessing,
exported channel order and channel names built from the dialog's choices, and that
every choice persists; the image-model list read from `setup.status`; the cost line
for estimates and plain refusals; multi-page snapshot TIFFs; the `preprocess.preview`
request; that every saved checkpoint (not the initial one) reaches the live hook
while the run goes on, including after a stop; that event listeners receive
checkpoints and agent events in order and that a failing listener harms nothing;
native affine registration geometry and serialization; the warp step's
direction, both coordinate layouts, its serialization and that BigWarp can reopen
it; the cutting-angle signs; and how a kept row merges with a newer one.

`HostSessionSmokeTest` is opt-in and needs ABBA's cached Allen V3p1 atlas (normally
`~/cached_atlas`; see its class comment). It runs a real headless ABBA 0.24.1:

```bash
java -Djava.awt.headless=true --add-opens=java.base/java.lang=ALL-UNNAMED \
  -cp "target/test-classes:target/classes:$(cat target/test-classpath.txt)" \
  org.langslice.fiji.HostSessionSmokeTest ~/cached_atlas
```

It imports two synthetic sections and checks: AP positions from ABBA's own
`toAtlasZ` against ABBA's atlas coordinate images read at the slice; a tilted session's
`angles_deg`; `existing_warp` for a user's own BigWarp step; live checkpoints
(position, affine step, warp step, cutting angles) read back from ABBA; that a new
placement removes the warp before replacing the affine; that one ABBA Undo reverts
exactly one checkpoint (angles included) and Redo restores it; that a slice changed
outside the run is refused, kept and applied on retry; and native save. It then
reopens the saved project in a second JVM without the connector on the classpath
(`ReloadCheck`): both steps, the warp's landmarks, BigWarp editing, the position and
the angles survive.

`DialogPreview OUTPUT_FOLDER` (needs a display, no ABBA) renders the Registration
dialog's three tabs and both run windows from fake slices and a fake worker, as
`dialog_tasks.png`, `dialog_slices.png`, `dialog_preprocessing.png`,
`dialog_run_log.png` and `dialog_run_compact.png`. `SetupPreview` does the same for
Setup.

Copy `target/langslice-fiji-0.2.0.jar` into your Fiji `jars` folder and restart
Fiji. Install the connector by copying this JAR. Do not copy the Maven dependency jars into Fiji.

## Setup and lifecycle

Open Setup from Fiji's **Plugins → LangSlice → LangSlice setup…**, or with the
**Setup…** button of the Registration dialog. Its **Installation instructions**
button opens the LangSlice repository, whose README leads to the installation guide. Choose the conda environment,
check the installation, and connect a model account. The browser completes ChatGPT
sign-in. API keys are sent through the worker's stdin, never command-line arguments,
and are saved by LangSlice. A saved credential is explicitly distinguished from an
account validated online.

Environment discovery checks conda's registry and the base environments of
common conda and mamba installations (`CONDA_EXE`, `MAMBA_ROOT_PREFIX`, and
`miniforge3`, `miniconda3`, `anaconda3`, `mambaforge` or `micromamba` in the home
folder); Browse accepts an environment folder. A remembered environment is rechecked
when launching the agent. When ABBA was started with `langslice abba`, the
environment that started it (the `langslice.environment` system property) runs that
session's workers and is listed first in Setup; the user's saved choice is kept. When available, `conda run --prefix` supplies activation variables
without requiring an open terminal. Otherwise the environment's Python is invoked
directly. Worker processes and their descendants are terminated on cancellation.

`LangSliceService` registers one named command with ABBA's existing registration UI
registry before the ABBA window is built: ABBA shows it as one plain entry,
**Register → LangSlice Registration…**, and passes the live session as `mp`. ABBA
places externally registered entries at the end of the Register menu, after a
separator below its own entries (after the Spline entries); a plugin cannot choose the
position. No `PyCommandBuilder`, `abba_python`, or upstream ABBA modifications are
required.

## The Registration dialog

The dialog works on the slices selected in ABBA when it opens, or on every slice
when none is selected. It is not modal; the run starts from **Run** (**Copy prompt**
in Claude mode).

- **Top:** provider (ChatGPT or Claude), agent model (from the worker's `setup.status`
  list, editable), reasoning
  level, image resolution (Low/Medium/High/Auto), **Show agent log**, **Open agent
  viewer**, the image model, and **Save traces to** a folder (default
  `~/LangSlice/traces`), which keeps the run's full agent trace; the final message names
  the file. The image model list comes from `setup.status` (`image_models`, in the
  worker's order, **None (fit to the stain only)** included); an entry whose account is not connected says "(not set up)". It
  is enabled whenever Nonlinear is ticked, in both modes. **Open agent viewer** is
  enabled only when ABBA was started from Python (`langslice abba`), which registers a
  `LangSliceEvents` listener; otherwise it is disabled with the tooltip "Available when
  ABBA is started with `langslice abba`".
- **Tasks tab:** *Positioning* (section thickness and interval prefilled from ABBA's
  slices, notes for the agent).
  *Linear* (**Enable hemisphere flipping** with an optional hemisphere cue, the affine
  tool, max parallel slice transforms 1–4, **Enable slice angle estimation**, notes for
  the agent). *Nonlinear* (notes for the agent; ANTs SyN is the one deformable fit). Any combination of the three may run, Nonlinear alone included. When
  Nonlinear is on, Linear is off and some listed slices have no ABBA registration, Run
  asks "Slices … have no linear registration. Let the agent align them first?": Yes
  switches Linear on; No leaves them out of Nonlinear (out of the run entirely when
  Positioning is off too; otherwise listed in `nonlinear_skip`).
- **Slices tab:** one row per slice with its name, the number of ABBA registrations
  and an optional note on damaged tissue (a note the agent reads, not a damage mark); **Let the agent mark damaged regions**;
  **Allow the agent to overwrite existing transforms** (off by default:
  every listed slice that has registrations is sent as `locked`, so the agent keeps
  its in-plane alignment while its position may still move; a slice that carries the
  user's own spline or BigWarp step also keeps that warp).
- **Preprocessing tab:** Auto or Custom; in Custom a weight from 0 to 1 per channel,
  CLAHE on/off and CLAHE strength; **Let the agent drive preprocessing**; the snapshot
  pixel size (µm); the tip "Try to maximize contrast between different regions."; and a
  before/after preview of a chosen slice. Before is the snapshot as exported from ABBA
  (one channel in gray, several as a coloured overlay); after is the worker's
  `preprocess.preview` output, the grayscale image the agent sees.
- **Bottom:** the estimated cost from `linear.estimate`; when the worker gives no
  number, its plain reason ("Estimated cost: no estimate (…)"). **Setup…**, **Cancel**
  and **Run**.

Dialog settings are saved; per-slice damage notes are not. Settings are stored (Java preferences, node
`org/langslice/fiji/registration`) when a run starts, and restored next time; section
thickness and interval come from ABBA whenever ABBA's slices give them.

| Control | Request |
| --- | --- |
| Agent model; reasoning | `spec.model` (`openai-oauth/…`), `spec.reasoning` (omitted for "default") |
| Image model | `spec.nonlinear.provider` (`none` for None) and `spec.nonlinear.image_model` (null for None or the provider's default) |
| Image resolution | `spec.image_resolution` |
| Positioning | `spec.tasks` += `position`; `position.thickness_um`, `position.interval_um`, `position.notes` |
| Linear | `spec.tasks` += `transform`; `transform.flip`, `transform.hemisphere_cue`, `transform.automatic` (affine tool), `transform.interactive` true, `transform.angles`, `transform.max_parallel`, `transform.notes` |
| Nonlinear | `spec.tasks` += `nonlinear`; `nonlinear.notes` |
| Let the agent mark damaged regions; note on damaged tissue | `spec.agent_damage`; `damaged` (filename to note) |
| Allow the agent to overwrite existing transforms (off) | off: every listed slice with registrations is in `locked` |
| Let the agent drive preprocessing | `spec.agent_preprocessing`; every channel is exported |
| Auto/Custom, weights, CLAHE, strength | `preprocessing`; Custom exports only weighted channels unless the agent drives preprocessing |
| Snapshot pixel size (µm) | `pixel_size_um` |
| Save traces to | `trace_dir` |

The connector also sends, from ABBA: `positions_mm` (ABBA's `toAtlasZ` of each slice,
BrainGlobe AP millimetres), `z_offset_mm` (`ReslicedAtlas.getZOffset()`), `angles_deg`
(the session's stack-wide cutting angles, pitch = −degrees(rotateX), yaw =
−degrees(rotateY)), `existing_warp` (slices carrying a spline or
BigWarp step that is not LangSlice's) and `channel_names` (one per exported page).

Snapshots are multi-page TIFFs with one page per exported channel: every channel in
Auto or when the agent drives preprocessing, only channels with a weight above 0 in
Custom, in channel order, matching the request's `channel_weights`.

## The run: live in ABBA

The agent's work lands in ABBA as it goes, in ChatGPT and Claude mode alike: every
checkpoint after the initial one is applied when it arrives, as **one ABBA undo step**,
through ABBA's native actions. A checkpoint may carry, per slice, a position, a new
placement (flip, quarter turn, affine) and a warp, and for the stack new cutting
angles (`host_angles`). Each slice carries at most two LangSlice steps on top of the
registrations it had when the run started: a **LangSlice affine** step (ABBA's
`AffineRegistration`) and on top of it a **LangSlice warp** step (ABBA's BigWarp
thin-plate spline, `BigWarpSource2DRegistration`; at most 1089 landmarks, its measured
difference from LangSlice's deformation written in the log and kept in the step's
parameters). A replacement deletes the newest
steps (the warp first, then the affine when the placement changed) and appends new
ones; it is done only while LangSlice's steps are still the slice's newest
registrations. Cutting angles are applied through a small undoable action of the
connector's own, since ABBA's own angle command cannot be undone.

A slice whose registrations were changed in ABBA during the run (a registration added,
removed or undone) is refused and named in the log. Rows ABBA could not take are kept,
merged with the next checkpoint and retried; a failure never ends the run. At the end
the kept rows get one more try, and the run window offers **Retry failed updates**.
**Stop run** ends the agent; everything it saved before the stop is already in ABBA.

Saved projects contain only ABBA's own registration types, so they reopen in ABBA
without LangSlice, and the warp step opens in BigWarp like any BigWarp registration.
Save the project with ABBA's normal Save command.

With **Show agent log** on, the run window shows the agent's activity; with it off, a
compact window shows the status, the Stop button and the final message.

## Events for the agent viewer

`org.langslice.fiji.LangSliceEvents` is a small public registry for an in-JVM listener,
normally the Python launcher's JPype proxy when ABBA was started with `langslice abba`:
`addListener(Consumer<String>)`, `removeListener`, `hasListeners()` and `slices()` (the
run's snapshot filename to ABBA slice map). In both modes the connector passes
on every worker message verbatim (`checkpoint`, `agent_event`, `log`, each a JSON object
with its `kind`) and adds `run_started` (mode `openai-oauth` or `claude`, viewer, image folder, sections), `job`
(Claude mode), `applied` (what reached ABBA) and `run_finished`. Delivery runs on one
background thread, in order; a failing listener is logged and never affects the run.

## Protocol

The connector uses protocol version 1. A request is one JSON line sent to
`python -m langslice serve --stdio`; the worker streams events and then a result or
error. Images and registration outputs remain local files. No local HTTP server or
listening port is required for ChatGPT mode; Claude mode listens on one loopback port
with a random token for the live channel.
