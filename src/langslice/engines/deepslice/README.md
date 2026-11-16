# DeepSlice as a LangSlice tool

A LangSlice engine (`docs/developers/plugins.md`) that offers DeepSlice 1.2.8
(Carey et al. 2023, Nat Commun) to the registration agent as one tool, `deepslice_place`.
Design: Fable's proposal with Nash's decisions of 2026-10-19 (SliceBench
`notes/2026-10-19_deepslice_brainways_plugin_design/fable_proposal.md`, "Nash's decisions"
at its top). DeepSlice runs in LangSlice's process (Nash, 2026-11-15: "I think we're safe
to just import them, rather than keeping them separate.").

DeepSlice's inference is vendored here from DeepSlice 1.2.8
(https://github.com/PolarBean/DeepSlice, commit b0d6965ee1861403bc5686eccab4f226c8b15c8b;
Copyright (c) 2023 Harry Carey; MIT License, `LICENSE`): exactly what the tool uses, the
mouse network and its weights, the image reading, the prediction and the ensemble, the angle
vote and the spacing along an order. The `deepslice` package is not a dependency (it pins
`urllib3==1.26.6`).

- `plugin.py`: the declaration, the factory and the call (which sections, which parts).
- `engine.py`: one run in DeepSlice's own order, the QUINT files, `EngineRun`.
- `network.py` (vendored): the weights (fetched and checked), the Keras model, the images,
  the prediction and the ensemble.
- `alignment.py` (vendored): the angle vote and the spacing, DeepSlice's pandas steps
  ported to numpy operation for operation.
- `proposal.py`: the run's answer as per-section rows, read through LangSlice's
  registration reader; the stack's angle pair.
- `writes.py`: what a call writes, as one undo step.
- `reply.py`: the reply's words, `settings.json` and `rows.json`.

## Install and use

```bash
pip install "langslice[engines]"     # tensorflow 2.21, tf-keras 2.21
```

It starts off: switch it on with `--enabled deepslice_place` (`enabled=["deepslice_place"]` in Python). Its options are the entry of the job's `plugin_config`
under `langslice.engines.deepslice.plugin:tools`.
TensorFlow 2.21 has no wheels for Intel Macs. TensorFlow caps `h5py` below 3.15; the uv lock
overrides that cap (`[tool.uv]` in `pyproject.toml`) so that it moves no version of the
other installs, and the network reads its weights alike on h5py 3.16 (below); pip honours
the cap where the extra is installed. Options (`--plugin-config
'langslice.engines.deepslice.plugin:tools=JSON'`): `ensemble` (true: the paper's
two-network model, DeepSlice's own default for mouse), `long_edge` (3072: the long edge in
pixels of the image written for DeepSlice; a file of that size or smaller goes in with its
delivered pixels, DeepSlice shrinks to 299 itself), `cache_folder` (where the weights are
kept; empty is `~/.cache/langslice/deepslice`). Any other option is refused when the plugin
loads. The tool's pictures come back only when the call sets `view` true.

**In the process.** Nothing of TensorFlow is imported until the tool runs. A call is refused
`ENGINE_MISSING` when a package of the extra is missing (the message names it and
`pip install "langslice[engines]"`; nothing written, no call folder) or when the weights
cannot be fetched. TensorFlow runs on the CPU: every GPU is hidden from it before it sets up
its devices, so it never takes the GPU memory the fits on torch use. `TF_USE_LEGACY_KERAS=1`
and `TF_CPP_MIN_LOG_LEVEL=2` are set unless the environment sets them. One DeepSlice run at
a time per process, each in a fresh Keras session: without the reset, a second model in the
same process answered differently from the first by up to 32 atlas voxels on a test stack,
with no error (Keras renames a rebuilt model's layers, and the weights are loaded by layer
name). What Keras prints is caught in the call's `deepslice_stdout.log`; TensorFlow's own
log lines go to the process's stderr. A run has no time limit.

**Weights.** Fetched once, on the first call, from the EBRAINS bucket DeepSlice's metadata
names (`data-proxy.ebrains.eu/api/v1/buckets/deepslice/weights/`):
`xception_weights_tf_dim_ordering_tf_kernels.h5`, `Allen_Mixed_Best.h5`,
`Synthetic_data_final.hdf5` (about 260 MB together), each checked by its SHA-256 (the files
DeepSlice 1.2.8 itself downloads) and kept in `~/.cache/langslice/deepslice` (or
`cache_folder`); a download goes to a `.part` file renamed into place once checked, so a
broken download leaves nothing. Checked on Windows.

## The tool

`deepslice_place(sections, order, bad_sections, write, view)`, kind write, group
Positioning, task `angles` (it exists wherever `position_sections` does), writes
`position`, `angles` and `affine`. Offered only on `allen_mouse_*um` and
`ccfv3augmented_mouse_25um`, coronal jobs only; on any other atlas or plane it is absent and
no other tool's text names it.

| Argument | Meaning | Default |
|---|---|---|
| `sections` | filenames to run; the angle vote and the spacing use exactly these | every section |
| `order` | the cutting order, anterior to posterior, covering every section run; empty: each section keeps its own predicted position, no spacing | `[]` |
| `bad_sections` | filenames left out of DeepSlice's angle vote and spacing fit; `[]`: none | every section with marked damage |
| `write` | list of `"positions"`, `"angles"`, `"transform"`; `[]`: write nothing, show the proposal | every part the job lets the agent write |
| `view` | the positioning picture | false |

- **Input**: each section as the agent oriented it (quarter turn, then flip), the delivered
  pixels, no preprocessing (`section_raster(raw=True)`). The answer is read back with
  `read_registration(registered_on="oriented")`, so the six in-plane numbers are for the
  oriented render.
- **Interval sign**. With `order` anterior first, DeepSlice's depth falls along the index,
  so the interval it is given is NEGATIVE: the plugin passes `-interval` (the job's positive
  micrometres) and the agent never sees or chooses a sign. A job without an interval passes
  none and DeepSlice estimates it (reported in the reply as `estimated_by_deepslice`).
- **Bad sections**: a section with a marked damage region is a DeepSlice bad section unless
  the call names its own list. DeepSlice still predicts it, leaves it out of the angle vote
  and spacing fit, gives it the stack's angles and, with `order`, places it from its
  neighbours. DeepSlice's vote fails (`FIT_FAILED`, an `IndexError` of its own) when two
  sections or fewer are left to vote.
- **`transform`** (each section's in-plane transform from DeepSlice's anchoring) exists in
  the tool's text and its `write` only while the transform task is on and `elastix_affine`
  is on (not disabled, affine not frozen on every section). It replaces the section's
  transform and clears its deformation; a section with no pixel size has no numbers (the
  row says so); a frozen affine is that row's `FROZEN`. Not special-cased for damaged
  sections (Nash: "Doesn't work on damaged sections, but that's fine").
- **Writes**: one `with ctx.writing()` block, one undo step. `set_positions` (rows without a
  frozen position, plus the stack's angles, refused whole when any section's angles are
  frozen: then the positions still land and the reply says `FROZEN`), then, read after
  those writes at the plane each section now holds, `set_transforms`. A changed position
  or angle clears the deformations it moves, as `position_sections` does.
- **Reply** (no metric or confidence): `written`, `stack` (pitch and yaw now and proposed,
  what became of the angles, the spacing used and whether DeepSlice estimated it, the bad
  sections and why), `columns`/`rows` (per section in proposed order: section,
  position_mm, change_mm, own_pitch_deg, own_yaw_deg, bad, the in-plane change in words
  where a transform is proposed, written), then `refused`, `clamped` and `view` when they
  have something to say. The in-plane change reads the proposed transform against the one
  the section holds (the identity when none): turn, scale, shift as a share of the width
  and height. With `view` off the reply ends with one sentence pointing to the positioning
  view of `look`; with it on, the positioning picture of the written sections or, with
  nothing written, drawn at the proposal and captioned "not written".
- **Every call** (propose-only included) saves `plugins/deepslice_place/calls/<n>/`:
  `images/` (the images written), `engine/` (`final.json` and `own.json`, QUINT files of
  the final and the own anchorings with the section ids as filenames;
  `deepslice_stdout.log`, and `deepslice_error.log` with the traceback when DeepSlice
  fails), `out.json` (the run's request and answer: each section's number, bad flag, size
  and both anchorings, the two passes of the angle vote, the spacing, versions and
  timings), `settings.json` (arguments, bad sections and why, signed interval passed,
  versions, timings) and `rows.json`.

## The run (`engine.py`)

In DeepSlice's own order: each image's anchoring (`DSModel.predict`: Keras's generator at
299 x 299, grey, divided by its standard deviation, one image a batch; the primary network,
and with the ensemble the mean of both), the table sorted by cutting index when an order is
given and by predicted depth otherwise, a copy (`own`), the angle vote over the good
sections applied to every section, twice (`propagate_angles`), with an order the spacing
(`enforce_index_spacing`: the good sections' depths reordered along the index, then even
steps of the interval given or DeepSlice's weighted estimate, fitted to the predicted
depths). A table without an order is numbered in its final order, as DeepSlice's
`save_predictions` numbers it. The spacing reported is the interval passed, or the estimate
in micrometres, and DeepSlice's reading of the reordered depths' direction. DeepSlice's own
vote fails (`FIT_FAILED`, an `IndexError`) when two sections or fewer are left to vote, and
so does this one. The rows (`proposal.py`) round position to 4 decimals (mm) and angles to
3, as the importer does; the stack pair is the median over the placed rows.

## Checked (2026-11-15, Windows, CPU; no model quota)

On LSD M02 (38 sections, pixel size 4.4 um/px given so that transforms exist):

- The vendored engine against the saved DeepSlice 1.2.8 runs of 2026-10-19 (Linux, the
  `deepslice` package) on the same images: plain on Allen and on the augmented atlas, a
  marked-damage bad section (pair -3.077 / 0.368), the true order with the interval (-200
  um passed) and with DeepSlice's estimate (-209.31 um, `caudal-rostro`): three bit-identical,
  two within 3.84e-4 atlas voxel (0.01 um) and 7.5e-7 degrees. Those saved runs differ from
  each other by the same 3.97e-4 voxel on identical images (TensorFlow on the CPU, run to
  run), so every vendored answer is bit-identical to one of DeepSlice's own. The same on
  h5py 3.14 and 3.16.
- The tool against the subprocess tool it replaced (SliceBench `comparators/plugins/deepslice/`,
  the `deepslice` package in an environment of its own), one fresh job each, five calls
  (propose-only; propose-only with `view`; `bad_sections` A_09; `order` with the job interval
  200; the default `write`): every reply identical, every `rows.json` table identical (its
  raw anchorings bit-identical in four calls, within 4e-4 voxel in one), the 7 pictures
  pixel-identical, and after the default write all 38 positions, transforms and the stack
  angles (-3.055 / 0.315) identical.
- The declaration rendered for six jobs (Linear on and off, `elastix_affine`,
  `interactive_transform` and `mark_damage` disabled, ...): description, `TOOLS.md` text,
  argument texts and schema, and the function declaration ADK sends (Gemini API and Vertex),
  identical to the subprocess plugin's.
- Weights: 260 MB fetched from EBRAINS into `~/.cache/langslice/deepslice`, the SHA-256s
  matching; then about 16 s for 38 sections.

Measured with the subprocess plugin on 2026-10-19 (Linux), the in-process run giving the
same numbers: plain median position error 174 um against the truth; `order` with no job
interval (DeepSlice estimates 209.3 um) 231 um; `order` with the job interval 200 87 um,
worst 482 um; the filename order of the interleaved series 2248 um. A mirrored section's own
yaw changes sign (DeepSlice has one handedness), and a mixed stack skews the voted yaw: turn
and flip sections before the call. On `ccfv3augmented_mouse_25um` every position is the
Allen job's + 0.350 mm, the angles equal.
