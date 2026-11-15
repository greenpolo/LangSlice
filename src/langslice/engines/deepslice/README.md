# DeepSlice as a LangSlice tool

A LangSlice tool plugin (`docs/developers/plugins.md`) that offers DeepSlice 1.2.8
(Carey et al. 2023, Nat Commun) to the registration agent as one tool, `deepslice_place`.
Design: Fable's proposal with Nash's decisions of 2026-10-19 (SliceBench
`notes/2026-10-19_deepslice_brainways_plugin_design/fable_proposal.md`, "Nash's decisions"
at its top). DeepSlice runs in LangSlice's process (Nash, 2026-11-15: "I think we're safe
to just import them, rather than keeping them separate.").

- `plugin.py`: the declaration, the factory and the call (which sections, which parts).
- `engine.py`: DeepSlice itself, imported when the tool runs: the staging, the calls in
  DeepSlice's own order, the QUINT files, `EngineRun`.
- `proposal.py`: the run's answer as per-section rows, read through LangSlice's
  registration reader; the stack's angle pair.
- `writes.py`: what a call writes, as one undo step.
- `reply.py`: the reply's words, `settings.json` and `rows.json`.

## Install and use

```bash
pip install "langslice[engines]"     # deepslice 1.2.8, tensorflow 2.21, tf-keras 2.21
langslice linear run ... --plugins langslice.engines.deepslice.plugin:tools
```

or `langslice.create_job(..., plugins=["langslice.engines.deepslice.plugin:tools"])`.
TensorFlow 2.21 has no wheels for Intel Macs. DeepSlice 1.2.8 pins `urllib3==1.26.6`, so
the extra puts that version in LangSlice's environment. Options (`--plugin-config
'langslice.engines.deepslice.plugin:tools=JSON'`): `ensemble` (true: the paper's
two-network model, DeepSlice's own default for mouse), `long_edge` (3072: the long edge in
pixels of the image written for DeepSlice; a file of that size or smaller goes in with its
delivered pixels, DeepSlice shrinks to 299 itself). The tool's pictures come back only when
the call sets `view` true.

**In the process.** Nothing of DeepSlice or TensorFlow is imported until the tool runs. A
call is refused `ENGINE_MISSING` (nothing written, no call folder) when a package of the
extra is missing (the message names it and `pip install "langslice[engines]"`), when a
weight file is missing and DeepSlice's weights folder cannot be written, or when a weight
file there is not an HDF5 file (an interrupted download: delete it). TensorFlow runs on the
CPU: every GPU is hidden from it before it sets up its devices, so it never takes the GPU
memory the fits on torch use. `TF_USE_LEGACY_KERAS=1` (DeepSlice imports
`tensorflow.keras`, which TensorFlow 2.21 gives as `tf_keras` only with it) and
`TF_CPP_MIN_LOG_LEVEL=2` are set unless the environment sets them. One DeepSlice run at a
time per process, each in a fresh Keras session: without the reset, a second `DSModel` in
the same process answered differently from the first by up to 32 atlas voxels on a test
stack, with no error (Keras renames a rebuilt model's layers, and DeepSlice loads its
weights by layer name). What DeepSlice prints is caught in the call's
`deepslice_stdout.log`; TensorFlow's own log lines go to the process's stderr. A run has no
time limit.

**Weights.** DeepSlice downloads its mouse weights on first use into its package's
`DeepSlice/metadata/weights/` (`xception_weights_tf_dim_ordering_tf_kernels.h5`,
`Allen_Mixed_Best.h5`, `Synthetic_data_final.hdf5`; about 260 MB together, from EBRAINS),
as it does outside LangSlice; the first call takes that long longer. Checked on Windows
(the venv's site-packages).

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
  `images/` (the images written), `engine/` (`staged/`, hard links or copies under
  DeepSlice's names; DeepSlice's own `deepslice.json` / `.csv`; `final.json` and `own.json`,
  QUINT files of the final and the own anchorings with the section ids as filenames;
  `deepslice_stdout.log`, and `deepslice_error.log` with the traceback when DeepSlice
  fails), `out.json` (the run's request and answer: each section's staged name, number,
  bad flag, size and both anchorings, the two propagation passes, the spacing, versions and
  timings), `settings.json` (arguments, bad sections and why, signed interval passed,
  versions, timings) and `rows.json`.

## The run (`engine.py`)

Staging: with `order`, `staged/section_sNNN.png` (NNN = 1-based rank in `order`; DeepSlice
reads one `_s<digits>` per name); without, the section id (+ `.png` if it has no image
suffix) and `section_numbers=False`. Bad sections are matched by an anchored, escaped
pattern on the staged name (DeepSlice matches by regular-expression search). Calls, in
DeepSlice's order: `DSModel("mouse")`, `predict`, a copy (`own`), `set_bad_sections` (only
if any), `propagate_angles`, with `order` `enforce_index_spacing(section_thickness)`,
`save_predictions`. The spacing reported is DeepSlice's own reading of the batch on a copy
(`enforce_section_ordering`, `calculate_brain_center_depths`,
`determine_direction_of_indexing`, `calculate_average_section_thickness` when estimated).
The rows (`proposal.py`) round position to 4 decimals (mm) and angles to 3, as the importer
does; the stack pair is the median over the placed rows.

## Checked (2026-11-15, Windows, CPU; no model quota)

Against the subprocess plugin this replaced (SliceBench `comparators/plugins/deepslice/`,
same DeepSlice and TensorFlow), on LSD M02 (38 sections, pixel size 4.4 um/px given so that
transforms exist):

- Engine against the saved 2026-10-19 runs (Linux) on the same images: plain and Allen /
  augmented, marked-damage bad section (pair -3.077 / 0.368), true order with the interval
  (-200 um passed) and with DeepSlice's estimate (-209.31 um, `caudal-rostro`): three runs
  bit-identical, two within 3.84e-4 atlas voxel (0.01 um) of the anchorings and 7.5e-7
  degrees of the angle pair, the run-to-run spread TensorFlow already showed on the same
  inputs there.
- Tool against the subprocess tool, one fresh job each, five calls (propose-only;
  propose-only with `view`; `bad_sections` A_09; `order` with the job interval 200; the
  default `write`): every reply identical, every `rows.json` table identical (its raw
  anchorings within 4e-4 voxel), the 7 pictures pixel-identical, and after the default
  write all 38 positions, transforms and the stack angles (-3.055 / 0.315) identical.
- The declaration rendered for six jobs (Linear on and off, `elastix_affine`,
  `interactive_transform` and `mark_damage` disabled, ...) identical to the subprocess
  plugin's.
- First call with no weights: 260 MB downloaded into the venv's DeepSlice package; then
  about 20 s for 38 sections.

Measured with the subprocess plugin on 2026-10-19 (Linux) and unchanged since, the
in-process run giving the same numbers: plain median position error 174 um against the
truth; `order` with no job interval (DeepSlice estimates 209.3 um) 231 um; `order` with the
job interval 200 87 um, worst 482 um; the filename order of the interleaved series 2248
um. A mirrored section's own yaw changes sign (DeepSlice has one handedness), and a mixed
stack skews the voted yaw: turn and flip sections before the call. On
`ccfv3augmented_mouse_25um` every position is the Allen job's + 0.350 mm, the angles equal.
