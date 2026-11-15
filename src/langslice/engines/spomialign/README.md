# SPOmiAlign as a LangSlice tool

A LangSlice tool plugin (`docs/developers/plugins.md` in LangSlice) that offers SPOmiAlign
(Wang et al. 2026, Brief Bioinform) to the registration agent as one tool,
`spomialign(sections, fit, rigid, section_image, atlas_image, mask, smoothness_um, matches, view)`.

- `plugin.py`: the adapter (imports `langslice.plugin_api` and the two modules below).
- `matching.py`: RoMa's matches as SPOmiAlign takes them (the grid, the edge weighting, the
  suppression, the ranking).
- `engine.py`: the fits on the matched pairs, and one call's run in the machine's GPU queue.

All three run in LangSlice's process. SPOmiAlign's method (wangyiyuyang/SPOmiAlign, commit
c7f8b7a) is written here from what its `align_and_process_images` does; none of its code is
copied. Its matcher, RoMa, is its own package `romatch` (MIT), at the release SPOmiAlign
vendors (0.1.2, its model code unchanged), installed with LangSlice's `engines` extra:

```bash
pip install "langslice[engines]"
```

torch, SimpleITK and OpenCV are LangSlice's own dependencies. RoMa and torch are imported
only when a call runs; a call without RoMa installed is refused `ENGINE_MISSING`, naming the
extra.

## Use

```bash
langslice linear run ... --plugins langslice.engines.spomialign.plugin:tools
```

or `langslice.create_job(..., plugins=["langslice.engines.spomialign.plugin:tools"])`.
Option (`--plugin-config 'langslice.engines.spomialign.plugin:tools=JSON'`): `device` (a
torch device name; default the first GPU when torch sees one, else the CPU). A call, the
matcher's loading included, holds the machine's GPU queue (the core's `gpu_turn`: the lock
file `LANGSLICE_GPU_LOCK`, else `~/.cache/langslice-stalign/gpu.lock`, which FireANTs, STalign
and emlddmm take too), and lets the matcher go before it gives the queue back. RoMa's weights
load from torch's hub cache (`TORCH_HOME`, else `~/.cache/torch`; on Windows
`%USERPROFILE%\.cache\torch`), `hub/checkpoints`.

## What a call does

`fit` (`"transform"`, `"warp"`, `"both"`) and `rigid` choose the engine's mode:
`"transform"` is SPOmiAlign's `affine`, or with `rigid` its `rigid`; `"warp"` its
`bspline`; `"both"` its `affine+bspline` default (`affine_warp` below). `rigid` with
another `fit` is refused `BAD_ARGS`.

1. **Refusals first**, per section, before the engine runs: `FROZEN` (the affine for
   rigid, affine and affine_warp; the deformation for warp and affine_warp),
   `INVALID_LINEAR_PLACEMENT` (no transform: SPOmiAlign starts from the placement), and
   `HAS_DEFORMATION` for rigid, affine or affine_warp on a section that holds a
   deformation (a new transform would clear it; a landmark warp is a deformation).
2. **Canvas.** Rigid, affine and affine_warp: the atlas plane at the section's position and
   cutting angles, upsampled to about 1536 px on its long edge, with the section's
   preprocessed channel resampled onto it through its current transform (this stands in for
   the published user's hand-picked plate and hand rotation). Warp on a section that holds a
   deformation: the fit grid, with the atlas resampled onto it through that deformation
   (`FitFrame.applied_map`), so the warp lands on top of a landmark warp, as `stalign_lddmm`
   does.
3. **Atlas image.** The atlas image the agent names (default `nissl`: LangSlice's borrowed
   Nissl `ccfv3augmented_nissl` where the job offers it, on an Allen CCFv3 atlas, else
   `template`, which on `ccfv3augmented_mouse` is already the averaged Nissl; Nash
   2026-10-19) is rendered as SPOmiAlign's published reference looks, a brightfield Nissl plate: white background, dense tissue dark. This is
   the same rendering as the method-alone arm (`comparators/rigid/spomialign_runner.py`).
4. **Matching.** SPOmiAlign's own steps: both images resized to 1152 x 864, RoMa outdoor
   matches (coarse resolution 560, upsampled to the grid), edge weighting (constants as the
   authors fixed them), 3 x 3 grid NMS. RoMa's optional fused local-correlation CUDA kernel
   (`fused-local-corr`, Linux only) is off everywhere: RoMa computes the same local
   correlation in plain torch, a little slower and with more GPU memory. Damage is
   handled as the native fits handle it: the marked regions are painted as background in
   the atlas image, and matches whose atlas end falls in them or within the native margin
   (0.1 mm, scaled to the atlas's brain size) are dropped; then the `matches` most confident
   of the rest are kept (the paper's K, "typically 5000"; the default). The refill after the
   drop is the plugin's rule.
5. **Fit**, by transform:
   - `rigid`: `fit_rigid_transform`.
   - `affine`: least squares.
   - `warp`: SPOmiAlign's `bspline` mode, the displacement field alone on the current
     registration. Despite the name it is no B-spline: each pair's displacement, filled in
     between the pairs (TELEA inpainting) and blurred with a Gaussian of `smoothing_px`
     (default 15, `fit_bspline_transform`'s) on the 1152 x 864 grid.
   - `affine_warp`: its `affine+bspline` default.

   None rejects outlier pairs: the dropped homography mode was the only one with RANSAC.
   The linear part is written through `transform_from_matrix`, the full map through
   `deformation_from_map`, in one undo step: a warp on a section that holds a deformation
   with `start="current"`, so the section keeps that deformation's steps before its own and
   the record names it as its parent; the call's `settings`, and the engine's own numbers
   (`engine_settings`: the matcher, the mode, the 1152 x 864 working grid and its
   micrometres per pixel, `smoothing_sigma_px`, the pairs, the canvas), with the record, as
   its step's settings and its `engine.native_parameters`; each row carries
   `engine_settings` too.

Job files, per call under `plugins/spomialign/calls/<n>_<mode>/`, per section in
`<section>/`: `atlas.png` (the atlas image as SPOmiAlign saw it,
the damage zone outlined), `section.png`, `settings.json`, `matches.csv` (every pair used,
and those passed over in the damage zone, in pixels of the two PNGs, with confidence) and
`matches.png` (the 300 most confident pairs as dots of one colour on both images). The
reply names the folder; nothing from it is sent.

RoMa's weights (`roma_outdoor.pth`, 445 MB; `dinov2_vitl14_pretrain.pth`, 1.2 GB) download
into torch's hub cache on first use; when they are missing and cannot be downloaded the
reply names their URLs.

Licence and credit: the method is SPOmiAlign's (Wang et al. 2026, Brief Bioinform), whose
repository has no licence file, so no SPOmiAlign code is copied here; these modules are
written from what its code does. RoMa (Edstedt et al. 2024, CVPR) is MIT, DINOv2's weights
Apache-2.0.

## Checked

2026-10-17, Carey 04-0180:

- **Engine control.** An image paired with itself returns the identity within
  0.3 px. Known shifts of 25 and 30 px are recovered at the canvas centre to
  within a few px, with 1-3 % spurious scale: SPOmiAlign's unweighted affine.
- **Damage exclusion.** With `CTX:right` marked on s0220, the right cortex and
  hippocampus reach SPOmiAlign as background and 4 of the top 5000 matches still
  fell in the zone and were replaced. That cortex is intact on s0220, so the fit
  shifts: marking intact tissue costs the fit.
- **Atlas image differs from the alone arm.** On Carey, the method-alone arm reads
  the atlas's local `single_animal_nissl.tiff` (atlasapi v1 folder); the plugin's default
  there is the atlas's template, the averaged Nissl, a smoother picture. The two arms
  therefore match against different Nissl images.

2026-10-19:

- **Landmarks kept.** Rat sagittal s139 with a noisy landmark warp (the displaced
  cerebellum): `affine_warp` refused `HAS_DEFORMATION`; `warp` started from the landmark
  warp and the cerebellum stayed in the atlas cerebellum, while the forebrain moved into its
  outlines (10 s on the GPU, model load included).
- **Carey s0314, rigid then warp (an older run with smoothing_px 30).** Ventricles and dorsal cortex closer;
  the 300 most confident pairs sit in dorsal cortex and striatum, almost none ventrally, so
  the ventral warp is filled in from far away and stretched the ventrolateral edge
  (flags FOLDS 0.5 %, visceral cortex expanded 2.4x).

2026-11-15, the move into LangSlice's process: on this machine the in-process engine gives
bit-identical results to the subprocess engine it replaced (the same `in.npz` of five saved
calls, rigid and warp, and four calls of the tool on the three golden sections: replies,
section maps and `matches.csv` identical), and the same results run after run.

## Limits (the engine's own)

One RGB image pair per call. No weighting in the matcher: a section mask can only drop matches whose section end has weight under 0.5 (the reply says "approximated by dropping matches"), and it estimates no mask of its own (`estimated` is null). Brightness: 8-bit, Nissl rendered dark on white. `smoothness_um` (a Gaussian sigma on the displacement field, as `fireants_syn`'s is) becomes `smoothing_px` per section from the canvas geometry.

A call without the `engines` extra is refused with `ENGINE_MISSING`; so is one whose RoMa
weights are missing from torch's hub cache and cannot be downloaded (the reply names their
URLs). Any other failure to load the matcher or to run on the device is `FIT_FAILED`.
