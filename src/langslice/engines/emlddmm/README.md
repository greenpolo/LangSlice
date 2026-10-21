# emlddmm as LangSlice tools

A LangSlice tool plugin (`docs/plugins.md` in LangSlice) that offers emlddmm (Tward et al. 2025,
Nat Commun; twardlab/emlddmm @ 9c4dc80 plus our fork's e426ece and 6ff62b0, Apache-2.0) to the
registration agent as two tools. Design: Fable's proposal of 2026-10-19
(`_local/notes/2026-10-19_comparator_plugin_review/emlddmm_proposal.md`), with one change Nash
made: the slot spacing is an argument of the agent's, not a plugin setting.

- `plugin.py`: the adapter, run in LangSlice's process (imports only `langslice.plugin_api`).
- `engine.py`: the emlddmm calls, run under `~/.claude-science/conda/envs/rigid_emlddmm/bin/python -I`
  with `emlddmm.py` from `comparators/src/emlddmm_fix`. Nothing of emlddmm is copied or edited:
  the energy record is taken by wrapping the fork's `interp` (it finds the calling `emlddmm` frame's
  local `Esave` list). That environment has no `brainglobe_atlasapi`, so the engine reads the atlas
  folder under `~/.brainglobe` (`reference.tiff`, `additional_references/`, `annotation.tiff`)
  directly.

Needs the LangSlice plugin seams of branch `stalign-plugin-seams` (background work,
`SECTION_BUSY`, `FitFrame.applied_map`, `plane.native_to_um`, `angles_from_plane`, the `angles`
task). No further seam was needed.

## Use

```bash
langslice linear run ... --tool-plugin langslice.engines.emlddmm.plugin:tools
```

or `langslice.create_job(..., plugins=["plugin:tools"])`. Options (`--tool-plugin-config`):
`python`, `emlddmm_src`, `brainglobe_dir`, `device` (`auto`), `dtype` (`float64`), `section_um`
(58.88, the example's pixel), `section_weight` (`gap`, or a thickness in um), `memory_limit_gb`
(24), `threads`, `timeout_s`. Engine calls take the GPU lock file STalign's engine takes
(`~/.cache/langslice-stalign/gpu.lock`).

## The tools

- **`emlddmm_stack`** (write, Nonlinear, background; arguments `sections`, `slot_um`,
  `section_image`, `atlas_image`, `mask`, `smoothness_um`, `rigid`, `warp`, `warp_freedom`, `iterations`, `sorting`, `view`):
  the joint fit of the placed sections on the stack's current plane (`lock_slice_axis`,
  `out_of_plane=False`, slice matching), one piece of background work over the whole stack (every
  section in it is `SECTION_BUSY` until it lands), one undo step. Per section: a transform and
  its cut of the warp; a section that holds a deformation, or whose affine is frozen, keeps its
  transform and gets the rigid change folded into its deformation; a section with a frozen
  deformation is fitted for the stack but not written. The start reply names the work, the stack
  size, sections left out, sections that will not get a transform, the largest snap to a slot and
  the device. The notice names the sections not (fully) written with the reason, the sections with
  warp flags and the energy file (or says it was not captured), and the overlays of up to four
  written sections (flagged first, then most moved) with how far each moved. Everything per
  section is in `sections.csv`.
- **`emlddmm_plane`** (read, Positioning, task `angles`; arguments `sections`, `slot_um`,
  `atlas_image`, `iterations`, `view`): the same stack with the plane free and no warp
  (`lock_slice_axis` false, warp never started, `up_vector` as published). Proposes pitch, yaw and
  every section's position (one scale and shift for the stack), with each section's residual turn
  and shift beyond the stack's median. Writes nothing. Pictures: the overlays of up to four
  sections drawn at the proposed plane.

`slot_um` has no default; the description tells the agent to get the order and rough positions
right, then set it to the section thickness or about half the interval between neighbours. The
call is refused `STACK_TOO_LARGE` when slots x rows x columns (the grid is the union of the slides'
footprints, so slide size counts) x 0.4 GB per million voxels (float64) is over `memory_limit_gb`,
naming the slot spacing that would fit; two sections in one slot is `BAD_ARGS`. Sections whose
planes differ by under 1 degree (an imported placement's rounding) count as one stack; the largest
tilt is kept in `settings.json`. The borrowed Nissl `ccfv3augmented_nissl` is built in the engine
as LangSlice builds it (the augmented template on the atlas grid, 0.35 mm AP offset, zero outside
the brain).

Inputs, as `plugin.py`'s module text says: the preprocessed channel as it is in [0, 1] (the Fig
3 benchmark's rule; background 1 on a white slide, 0 on a black one) at 58.88 um, resampled through the section's current registration; the atlas image at 50 um over its
mean; marked damage (with the native margin) and everything outside the image get weight 0; each
section weighs as the gap it stands for (`section_weight`, as the Fig 3 runner).

## Job files (never put into the agent's context)

`plugins/<tool>/calls/<n>_<tool>/`: `settings.json` (asked, chosen, plugin options, what emlddmm
chose, whether its energy was captured, device, peak GPU memory, seconds), `sections.csv` (stack), `A.json`, `energy.csv`
and `energy.png`, `images/` (each section as the engine saw it, its final sorting and the atlas
under the fit), `weights.npz`, `in.npz`, `out.npz`, `log.txt`.

## Checked (2026-10-19)

Carey 04-0180 (copied jobs), GPU; see `_local/notes/2026-10-19_comparator_plugin_review/emlddmm_try/`.
Five sections, default iterations, 34 s: on s0154 (right cortex split off the midbrain) an exact
landmark warp survives the fit (its landmarks 4 um from their atlas points before, median 33 um
after); without it the fit takes the same points from 229 to 113 um. The whole 20-section stack
(125 s): 46 um with the landmark warp, 229 to 122 um without.

## Limits (the engine's own)

Takes a section mask as soft weights (`W0 *= weights`). Estimates its own three-class sorting, saved with each deformation it writes. One smooth 3-D warp for the stack, no per-section bends; needs dense, evenly spaced sections. Brightness: the channel as it is, 0-1.

A missing interpreter or source folder is refused at call time with `ENGINE_UNAVAILABLE`.
