# LangSlice Fiji connector

This independently distributed Java plugin connects an existing ABBA installation
with a separately installed LangSlice Python environment. Python and model accounts
are not needed to load the plugin or open Setup.

The connector targets ABBA **0.11.0** and Java **11 or newer**. Newer ABBA versions
must be checked before advertising compatibility: upstream has changed source
processor package names. Runtime dependencies are supplied by ABBA/Fiji and are
not bundled into the plugin JAR.

## Build

With JDK 11+ and Maven:

```bash
cd connectors/fiji
mvn package dependency:build-classpath -Dmdep.outputFile=target/test-classpath.txt
```

After Maven has written the resolved dependency classpath, subsequent development
builds can run offline:

```bash
python connectors/fiji/build.py --java-home /path/to/jdk --test
```

The offline build starts from empty `target/classes` and `target/test-classes`
folders, so a renamed or deleted command cannot linger in the JAR or the SciJava
plugin index. It uses the exact Maven-resolved classpath, never every JAR found in
the Maven cache.

The offline smoke checks start real fake-worker subprocesses to exercise event and
result delivery, timeout handling, secret-safe authentication errors, and paths
containing spaces. They also check: the ABBA menu registration (one `LangSlice`
entry) and its callable alias; the job spec, preprocessing and exported channel
order built from the dialog's choices, and that every choice persists; multi-page
snapshot TIFFs; the `preprocess.preview` request; `linear.estimate` detection;
that `linear.run` checkpoints are only remembered while `final_updates` is the one
application, with a stopped run keeping its last `updates_since_start`; and native
affine/spline registration geometry and serialization. They do not replace live
Fiji testing of registration, cancellation, undo and saved state.

`HostSessionSmokeTest` is opt-in and needs ABBA's cached Allen V3p1 atlas (see its
class comment). It imports a synthetic section into a real headless ABBA session and
checks calibration, `locked`/`damaged` request fields, multi-page export, the preview
snapshot, one final application reverted and restored by a single Undo/Redo, refusal
after an outside registration change, and native save/reload.

`DialogPreview OUTPUT_FOLDER` (needs a display, no ABBA) renders the Registration
dialog's three tabs and both run windows from fake slices and a fake worker, as
`dialog_tasks.png`, `dialog_slices.png`, `dialog_preprocessing.png`,
`dialog_run_log.png` and `dialog_run_compact.png`. `SetupPreview` does the same for
Setup.

Copy `target/langslice-fiji-0.1.0.jar` into your Fiji `jars` folder and restart
Fiji. This manual JAR installation is the development route; a public Fiji update
site has not been published. Do not copy the Maven dependency jars into Fiji.

## Setup and lifecycle

Open Setup from Fiji's **Plugins → LangSlice → LangSlice setup…**, or with the
**Setup…** button of the Registration dialog. Choose the conda environment,
check the installation, and connect a model account. The browser completes ChatGPT
sign-in. API keys are sent through the worker's stdin, never command-line arguments,
and are saved by LangSlice. A saved credential is explicitly distinguished from an
account validated online.

Environment discovery checks conda's registry and common installation folders;
Browse accepts an environment folder. A remembered environment is rechecked when
launching the agent. When available, `conda run --prefix` supplies activation variables
without requiring an open terminal. Otherwise the environment's Python is invoked
directly. Worker processes and their descendants are terminated on cancellation.

`LangSliceService` registers one named command with ABBA's existing registration UI
registry before the ABBA window is built: ABBA shows it as **Register → LangSlice →
LangSlice Registration…**, a submenu in the same Register menu as ABBA's DeepSlice
submenu, and passes the live session as `mp`. ABBA places externally registered
entries at the end of the Register menu, after a separator; a plugin cannot choose
the position. No `PyCommandBuilder`, `abba_python`, or upstream ABBA modifications
are required.

## The Registration dialog

The dialog works on the slices selected in ABBA when it opens, or on every slice
when none is selected. It is not modal; the run starts from **Run**.

- **Top:** provider (ChatGPT only), agent model (from the worker's `setup.status`
  list, editable; the connector's own list when an older worker gives none), image
  model, reasoning level, image resolution (Low/Medium/High/Auto), **Show agent log**,
  **Open agent viewer** (disabled, "Coming soon"), and **Save traces to** a folder
  (default `~/LangSlice/traces`), which keeps the run's full agent trace; the final
  message names the file.
- **Tasks tab:** *Positioning* (hemisphere flipping with an optional hemisphere cue,
  section thickness and interval prefilled from ABBA's slices, notes for the agent;
  DeepSlice and Bayesian position fit shown disabled). *Linear* (affine tool, max
  parallel slice transforms 1–4, notes for the agent; slice angle estimation shown
  disabled). *Nonlinear* is shown disabled: "Not yet available in ABBA".
- **Slices tab:** one row per slice with its name, the number of ABBA registrations
  and a **Damaged** checkbox with an optional note; **Let the agent flag damaged
  slices**; **Allow the agent to overwrite existing transforms** (off by default:
  every listed slice that has registrations is sent as `locked`, so the agent keeps
  its in-plane alignment while its position may still move).
- **Preprocessing tab:** Auto or Custom; in Custom a weight from 0 to 1 per channel,
  CLAHE on/off and CLAHE strength; the snapshot pixel size (µm); the tip "Try to
  maximize contrast between different regions."; and a before/after preview of a
  chosen slice. Before is the snapshot as exported from ABBA (one channel in gray,
  several as a coloured overlay); after is the worker's `preprocess.preview` output,
  the grayscale image the agent sees.
- **Bottom:** the estimated cost from `linear.estimate`, or "estimate unavailable"
  when the worker does not offer that method; **Setup…**, **Cancel** and **Run**.

Every choice except the per-slice damage checks is saved (Java preferences, node
`org/langslice/fiji/registration`) when a run starts, and restored next time; section
thickness and interval come from ABBA whenever ABBA's slices give them.

Snapshots are multi-page TIFFs with one page per exported channel: every channel in
Auto, only channels with a weight above 0 in Custom, in channel order, matching the
request's `channel_weights`.

During a run the user's ABBA session is not changed. Checkpoints are only
remembered. When the run ends, the result's `final_updates` are applied once, as one
undoable ABBA step. If the user stops the run (or it fails) after at least one
checkpoint, the run window offers **Apply partial result to ABBA**, which applies the
last checkpoint's `updates_since_start` the same way. The safety checks stay: a
slice whose registration count changed outside the run is refused. With **Show agent
log** on, the run window shows the agent's activity; with it off, a compact window
shows the status, the Stop button and the final message.

The connector uses protocol version 1. A request is one JSON line sent to
`python -m langslice serve --stdio`; the worker streams events and then a result or
error. Images and registration outputs remain local files. No local HTTP server or
listening port is required for the connector itself.
