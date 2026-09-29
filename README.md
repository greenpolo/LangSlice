<p align="center">
  <img alt="LangSlice" src="assets/logo/banner_black.svg" width="780">
</p>

<p align="center">
  <a href="https://github.com/greenpolo/LangSlice/blob/main/LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://github.com/greenpolo/LangSlice/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/greenpolo/LangSlice/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://langslice.readthedocs.io"><img alt="Documentation Status" src="https://readthedocs.org/projects/langslice/badge/?version=latest"></a>
</p>

<p align="center">
  <em>Register histological brain sections to BrainGlobe atlases.<br>
  A VLM agent orders, positions and aligns the sections; an image model corrects the atlas borders.</em>
</p>

> [!WARNING]
> LangSlice is in-development. Some atlases may produce mixed results; and
> horizontal and sagittal orientations haven't yet been thoroughly tested.
> LangSlice does not offer manual registration methods — it is meant to be
> used in combination with other registration software (ABBA, QUINT,
> PyNutil, BrainGlobe ecosystem, etc.). Export functions to these tools
> have not been fully developed yet.

<p align="center">
  <img alt="Three histology slices registered to the Allen mouse atlas" src="assets/promo_registration_combined.png" width="780">
</p>

## How it works

A VLM agent looks at the whole stack of sections through tool calls: it puts
them in cutting order, places each one at its atlas position, and aligns each
in-plane against the atlas at true physical scale. An image model is then shown
the atlas borders drawn on the tissue at that placement and asked to correct
them. Turning corrected borders into a nonlinear deformation is still being
designed, so by default no deformation is fitted and the linear placement is
what gets exported; an Elastix residual fit is available as an option. Results
export to VisuAlign-compatible JSON for QUINT / ABBA. The registration stages are
illustrated in [the nonlinear design](docs/nonlinear_design.md), and the planned
user-facing options in [the interface design](docs/interface_design.md).

## Quick start

For the new **plugin in an existing Fiji/ABBA installation**, follow
[Install LangSlice in ABBA](docs/abba_installation.md). Setup and account login
are available inside ABBA. This is currently a source-build preview; the Fiji
update site and conda package are not yet published.

For command-line use, download and extract the source and open a terminal in
the folder containing `environment.yml`:

```bash
conda env create -f environment.yml
conda activate langslice
langslice login  # browser sign-in; optional when using account setup inside ABBA

# Optional: pre-download an atlas (~500 MB) into ~/.brainglobe/
python -c "from brainglobe_atlasapi import BrainGlobeAtlas; BrainGlobeAtlas('allen_mouse_25um')"
```

For API-key providers, use the Fiji setup dialog or copy `.env.example` to `.env`
and add your provider key. The environment file installs LangSlice itself and its
dependencies; an editable install is only needed for development.

The CLI is grouped by method — `linear` for order, position and in-plane
alignment, `nonlinear` for generative-image registration:

```bash
# Linear: order, position and transform for a folder of sections
langslice linear run sections/

# Image-model border correction using the folder's saved linear alignment
langslice linear run sections/ --tasks nonlinear

# Standalone registration using a supplied linear placement
langslice nonlinear register slice.png --position 3.9 --initial-alignment placement.json
```

The image tool uses the fixed border-correction prompt plus optional per-slice
agent notes. It retains the first result without agent rejection and returns raw
output plus extracted borders on the original. It currently produces annotation
images, with deformation fitting kept separate. See
[the image-tool contract](docs/nonlinear_image_tool.md).

`nonlinear register` takes the position as an
argument, so it can follow `langslice linear run` or a placement made in
another tool — in QUINT/ABBA-style workflows it stands in for the manual
spline/BigWarp deformation step.

Nonlinear registration is exactly two border-based routes, chosen
automatically by whether a placement is supplied. With a supplied placement,
one image-generation call moves that placement's drawn atlas borders onto the
visible tissue; this is the production path, because nonlinear correction
needs a linear placement first. Without one, a local silhouette fit stands in
for the rough placement and the model instead draws boundaries from nothing
against an outlined grayscale atlas template, in one call (optionally two, for
an audit pass); this route remains for experiments. Neither route shows the
model a colored atlas map, and `--deformation` defaults to `none`. See
[the nonlinear design](docs/nonlinear_design.md).

Full CLI: `langslice --help`. Pipeline detail: [`docs/index.md`](./docs/index.md).

## ABBA integration

The independent [Fiji connector](fiji-plugin/README.md) adds
**Register > LangSlice > LangSlice Registration…** to an existing ABBA
installation, with account setup under **Plugins > LangSlice > LangSlice
setup…** and inside the dialog. The dialog runs the agent on the selected
slices with **Positioning** and **Linear** tasks, per-slice damage marks,
channel preprocessing with a preview, and an estimated cost; the result is
applied to ABBA as one undoable step when the run ends. It launches the
separately installed Python worker as needed.
See [installation and supported sessions](docs/abba_installation.md).

The following describes the older **Python-started ABBA launcher**, which remains
available for development and its existing companion viewers:

LangSlice runs inside [ABBA](https://abba-documentation.readthedocs.io) as a
registration plugin: position slices however you like (DeepSlice, QuickNII
import, manual), then apply LangSlice's nonlinear registration from ABBA's
`Register` menu like any built-in method — undoable, saved in the ABBA state.

The top-bar **LangSlice** menu configures and starts the linear agent on
selected slices (all slices if none are selected). Choose the model, reasoning
effort, slice interval and thickness, then enable **Ordering**, **Position**,
and **Transforms** independently. Transforms offers interactive adjustment by
the agent and automatic affine fitting as separate checkboxes. Interval and
thickness initially follow ABBA's current stack and can be overridden; interval
is the median current spacing, and thickness is ABBA's displayed thickness,
so check these against your cutting protocol.

The independent **Open agent viewer in ABBA** and **Open agent log** menu
options control which companion windows open during a run. The log appears as a
narrow sidebar with streamed assistant text, provider reasoning summaries,
expandable tool cards, and smaller image previews below. Browse earlier images,
enlarge a preview, or pause scrolling. It uses a local Chrome/Chromium app window,
with a Swing fallback. Reasoning summaries come directly from the provider;
LangSlice does not rewrite them, and encrypted reasoning is never displayed.
The viewer keeps bounded recent history. Enable a JSONL trace separately for
full persistent diagnostics.

The native agent viewer stays open across single-section and multi-section
work. Its upper overview uses ABBA's actual positioning atlas display, at the
session's existing display interval and channel settings. Sections retain their
native size and placement, with ABBA's green selection handles and dashed guides.
The lower focus panels show individual registered atlas overlays, up to four
sections per page. The agent's target changes update this separate viewer while
preserving the main ABBA camera and selection. Position and transform writes still reach
ABBA through its normal undoable actions. The overview and focus panels show
committed registrations; speculative candidate positions appear only in the
exact tool-image previews in the log, without moving sections or adding candidate
markers to the native viewer. Closing either companion leaves registration
running; **Show agent viewer** and **Show agent log** reopen them.

Menu runs currently support flat coronal Allen mouse sessions. Turn off ordering
when working on slices with existing registrations. The agent uses calibrated
snapshots of the loaded images and adds its corrections to the existing stack;
the window remains responsive while it works. Human edits during a run are not
read back by the agent. Results and snapshots are retained in the run directory
reported in the console; save the finished session with ABBA's **File > Save State**.

`langslice abba --linear FOLDER` runs the `linear` agent (order, position,
one in-plane transform per section) inside the same ABBA session instead: the
agent works exactly as it does headless, and every write it makes — a
reorder, a position, a flip, an affine tweak — appears live on the stack in
ABBA's BigDataViewer as it happens, undoable there like any other ABBA
action. `Register > LangSlice` stays available in the same session.

```bash
conda activate langslice
conda install -c conda-forge openjdk=11 maven
pip install -e ".[abba]"   # optional Python-started ABBA route
langslice abba             # launches the ABBA GUI with LangSlice installed
langslice abba --linear ./sections --save-state ./sections/run.abba
```

## Related Repositories

- **LangSlice-Training** — training infrastructure for LangSlice local models
  (SFT/RL pipelines, corpora, manifest tooling, docker training env).
- **SliceBench** — self-contained position-estimation benchmark.

Both depend on LangSlice; neither is required to run it.

## Links

- [**Documentation**](https://langslice.readthedocs.io) — full pipeline + harness internals

## Citation

If you use LangSlice in your work, please cite the project metadata in
[`CITATION.cff`](./CITATION.cff). GitHub renders a "Cite this repository"
button in the right sidebar from that file.

## License

BSD-3-Clause. See [`LICENSE`](./LICENSE).
