<p align="center">
  <img alt="LangSlice" src="assets/LangSlice_dark.png" width="780">
</p>

<p align="center">
  <a href="https://github.com/greenpolo/LangSlice/blob/main/LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://github.com/greenpolo/LangSlice/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/greenpolo/LangSlice/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://langslice.readthedocs.io"><img alt="Documentation Status" src="https://readthedocs.org/projects/langslice/badge/?version=latest"></a>
  <a href="https://huggingface.co/greenpolo/langslice-gemma-4-E4B"><img alt="Hugging Face Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-langslice--gemma--4--E4B-yellow"></a>
</p>

<p align="center">
  <em>Register histological brain sections to BrainGlobe atlases.<br>
  A VLM estimates position, an image-gen model lays down atlas colors, Elastix warps the rest.</em>
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

<p align="center">
  <img alt="Agent loop: inspect, explore atlas candidates, submit AP" src="assets/agent_loop.png" width="780">
</p>

A VLM agent (default [`langslice-gemma-4-E4B`](https://huggingface.co/greenpolo/langslice-gemma-4-E4B))
inspects the slice, explores candidate atlas planes through tool calls, and
submits an AP coordinate. Image generation then produces an atlas-colored
target from the histology, and itk-elastix recovers a dense B-spline
deformation. Results export to VisuAlign-compatible JSON for QUINT / ABBA.

<p align="center">
  <img alt="LangSlice registration pipeline: histology slice to atlas-colored target, dense warp, and overlay" src="assets/registration_pipeline_square.png" width="780">
</p>

## Quick start

```bash
conda env create -f environment.yml
conda activate langslice
pip install -e .
cp .env.example .env  # add AI Studio / Vertex / OpenAI keys

# Optional: pre-download an atlas (~500 MB) into ~/.brainglobe/
python -c "from brainglobe_atlasapi import BrainGlobeAtlas; BrainGlobeAtlas('allen_mouse_25um')"
```

The CLI is grouped by method — `linear` for position estimation and affine
anchoring, `nonlinear` for generative-image registration:

```bash
# Linear: order, position and transform for a folder of sections
langslice linear run sections/

# Nonlinear: registration at a known atlas position
langslice nonlinear register slice.png --position 3.9
```

The two are independent. `nonlinear register` takes the position as an
argument, so it can follow `langslice linear run` or a placement made in
another tool — in QUINT/ABBA-style workflows it stands in for the manual
spline/BigWarp deformation step.

Full CLI: `langslice --help`. Pipeline detail: [`docs/index.md`](./docs/index.md).

## ABBA integration

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
pip install -e ".[abba]"   # adds abba-python (needs the env's OpenJDK + Maven)
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
- [**langslice-gemma-4-E4B**](https://huggingface.co/greenpolo/langslice-gemma-4-E4B) — the v1.0 fine-tune

## Citation

If you use LangSlice in your work, please cite the project metadata in
[`CITATION.cff`](./CITATION.cff). GitHub renders a "Cite this repository"
button in the right sidebar from that file.

## License

BSD-3-Clause. See [`LICENSE`](./LICENSE).
