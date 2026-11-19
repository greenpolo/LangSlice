# STalign as LangSlice tools

A LangSlice engine (`docs/developers/plugins.md` in LangSlice) that offers STalign 2.0
(Clifton et al. 2023, Nat Commun; JEFworks-Lab main 8a84bf1f, GPL-3.0) to the
registration agent as three tools. Design: Fable's proposal, adopted by Nash on
2026-10-18.

- `plugin.py`: the adapter (binds the tools its rows of the tool table declare).
- `engine.py`: the STalign calls, in LangSlice's process; STalign is imported when a tool runs,
  never when the plugin loads.

## STalign itself

STalign is vendored: `src/langslice/_vendor/stalign/` holds the Python files of its `STalign/`
package at JEFworks-Lab/STalign commit 8a84bf1facc5173f7155400867a09c302125904e, unchanged,
with its GPL-3.0 `LICENSE` and a header in `__init__.py` naming the commit. Its optional C++
engine (`STalign/_cpp`, the compiled `_stalign_cpp`) is not included, so STalign runs its
pure-torch engine on the CPU as on the GPU (`backend_info()` reports `torch`). It needs nothing
beyond LangSlice's own dependencies (numpy, torch, matplotlib); `pandas`, `requests` and `pynrrd`
are imported only by STalign's atlas download and analysis helpers, which LangSlice does not call.

Needs the LangSlice plugin seams of branch `stalign-plugin-seams` (background work for
plugins, `BUSY`, `FitFrame.applied_map`, `plane.native_to_um`,
`ToolContext.angles_from_plane`, the `angles` task).

## Use

```bash
```

Its tools are on in every job; `--disabled` switches one off. Its options are the entry of the job's `plugin_config`
under `langslice.engines.stalign.plugin:tools`.
Options (`--plugin-config 'langslice.engines.stalign.plugin:tools=JSON'`): `device` (`auto`:
the GPU at float32 when there is one, else the CPU at float64), `threads` (4, STalign's
`num_threads`). A fit on the GPU takes a slot of the machine's GPU queue through the plugin
API's `gpu_turn` (`LANGSLICE_GPU_SLOTS` fits at once, else the GPU's memory over 2 GB), so a
call's sections, each its own piece of background work, fit at once. `stalign_plane_search` runs no fit
iterations: its multi-start pre-alignment is CPU work, so it always runs on the CPU at float64,
takes no GPU turn, and searches a call's sections at once (`map_sections`, `threads` each). The fit draws no figures
(`draw=False`) and reads its energy per iteration from the `Esave` entry of STalign's result,
which the vendored `_result` adds.

## The tools

- **`stalign_lddmm`** (write, Nonlinear, background; arguments `sections, section_image, atlas_image, mask, kernel_um, fit, displacement_um, sorting, iterations, view`): STalign's 2-D `LDDMM` per section, one
  piece of background work each; a write to a section whose fit runs is refused
  (`BUSY`). `kernel_um` is STalign's `a` (500), the width of the kernel that smooths
  the velocity, not a Gaussian sigma on the displacement as `smoothness_um` is elsewhere.
  `fit="warp"` (default) holds the transform and bends only; on a section
  that holds a deformation (a landmark warp, an earlier fit) the atlas is first carried through
  it, so the landmark move is kept (the 2026-10-18 test: kept on all six displaced-tissue
  sections, while STalign's own landmark term failed on one), and the bend is written with
  `start="current"`: the section keeps that deformation's steps before its own and the record
  names it as its parent. `fit="both"` is refused on a
  section with a deformation (`PRECONDITION`). Each landing is its own undo step; the
  deformation keeps the call's `settings` as its step's and `engine_settings` (the target
  and atlas grids, 10 and 50 um, and the settings STalign chose) as the record's
  `engine.native_parameters`, which the notice also carries. The notice
  gives what was fitted, the displacement summary and flags, and what STalign estimated (the share of tissue its sorting left out), and names the files of STalign's mismatch record (`energy.csv`, `mismatch.png`); the
  record itself is not sent (Nash: "just a readable file in the job, not something that gets
  dumped into context automatically"). Pictures: the overlay and the final sorting.
- **`stalign_plane_search`** (read, Positioning, task `angles`): STalign 2.0's 3-D plane search
  only (`LDDMM_3D_to_slice(niter=0, prefit=True)`: 27 trial planes, the best refined), from the
  section's current placement, against the BrainGlobe volume at 50 um (`../_atlas_average.py`,
  kept in the core's atlas cache). Gives each section's pitch, yaw and position in the stack's
  convention (`ToolContext.angles_from_plane`), `at_search_edge`, the sections grouped by the
  sign of their yaw (a mirrored section reverses it; how small a yaw still tells its sign
  depends on the stack, so no threshold is set), the
  middle pitch and yaw magnitude. Writes nothing: the agent writes what it keeps with
  `position_sections`. The full 3-D affine and warp are not run (the warp cannot be stored, and
  dropping it made the plane worse on the one section measured).

Inputs, as `plugin.py`'s module text says: the preprocessed channel scaled slide 0 / tissue 1
at 10 um, the atlas image at 50 um with the published two-channel recipe, marked damage (with
the native margin) painted as atlas background, the published prefit recipe values for the
plane search (sigmas 2, muA 3, muB 0). Sorting defaults are STalign's own (`classic`, sigmaM
1, sigmaB 2, sigmaA 5); on the sections tried so far they set nothing aside.

## Checked (2026-10-18)

- Rat sagittal s139 with a noisy landmark warp: `stalign_lddmm` bend landed in 33 s (GPU),
  started from the landmark warp; a `mark_damage` on the section while it ran was refused
  `BUSY`.
- Carey 04-0180 with the cutting angles set to 0: `stalign_plane_search` on 4 sections, middle
  pitch 11.0 deg and yaw magnitude 5.2 deg against the raters' 10.8 and 5.5 (LangSlice's own
  convention, end to end); one posterior brainstem section read +1.9 deg yaw (the wrong sign).
- Mirror test (20 sections mirrored left-right): the yaw sign reversed on 39 of 40 readings;
  the other read 0.6 deg.

## Limits (the engine's own)

Takes no section mask (it sorts the section's pixels itself; a named mask is refused `BAD_ARGUMENT`). One image pair (channels are fitted jointly by STalign but the tool gives one). Its final sorting is saved with the deformation. Brightness: slide 0, tissue 1.
