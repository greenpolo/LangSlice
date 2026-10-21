# SPOmiAlign as a LangSlice tool

A LangSlice tool plugin (`docs/plugins.md` in LangSlice) that offers SPOmiAlign
(Wang et al. 2026, Brief Bioinform) to the registration agent as one tool,
`spomialign(sections, transform, section_image, atlas_image, mask, smoothness_um, matches, view)`. Revised
2026-10-19 after Fable's review
(`_local/notes/2026-10-19_comparator_plugin_review/spomialign_review.md`).

- `plugin.py`: the adapter, run in LangSlice's process (imports only
  `langslice.plugin_api`).
- `engine.py`: SPOmiAlign's matching and fitting, run in the `rigid_spomialign`
  conda environment, from `comparators/src/spomialign` (commit c7f8b7a).

## Use

```bash
langslice linear run ... --tool-plugin langslice.engines.spomialign.plugin:tools
```

or `langslice.create_job(..., plugins=["plugin:tools"])` with this folder on
`PYTHONPATH`. Options (`--tool-plugin-config 'langslice.engines.spomialign.plugin:tools=JSON'`): `python` (the
`rigid_spomialign` interpreter; default
`~/.claude-science/conda/envs/rigid_spomialign/bin/python`), `src`, `device`,
`timeout_s`. RoMa's weights load from `~/.cache/torch/hub/checkpoints`.

## What a call does

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
   matches, edge weighting (constants as the authors fixed them), 3 x 3 grid NMS. Damage is
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
   `deformation_from_map`, in one undo step.

Job files, per call under `plugins/spomialign/calls/<n>_<transform>/`: `in.npz`, `out.npz`,
`log.txt`; per section in `<section>/`: `atlas.png` (the atlas image as SPOmiAlign saw it,
the damage zone outlined), `section.png`, `settings.json`, `matches.csv` (every pair used,
and those passed over in the damage zone, in pixels of the two PNGs, with confidence) and
`matches.png` (the 300 most confident pairs as dots of one colour on both images). The
reply names the folder; nothing from it is sent.

RoMa's weights (`roma_outdoor.pth`, 445 MB; `dinov2_vitl14_pretrain.pth`, 1.2 GB) download
into torch's hub cache on first use; when they are missing and cannot be downloaded the
reply names their URLs.

Licence: SPOmiAlign's repository has no licence file (checked 2026-10-19); Nash is asking
the authors. Its bundled RoMa is MIT, DINOv2 Apache-2.0. This adapter and engine copy no
SPOmiAlign code. In LangSlice, SPOmiAlign's code is to ship inside LangSlice's own tree (Nash
2026-10-19: "Within our code"), once the licence allows.

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

2026-10-19 (`_local/notes/2026-10-19_comparator_plugin_review/spomialign_try/`):

- **Landmarks kept.** Rat sagittal s139 with a noisy landmark warp (the displaced
  cerebellum): `affine_warp` refused `HAS_DEFORMATION`; `warp` started from the landmark
  warp and the cerebellum stayed in the atlas cerebellum, while the forebrain moved into its
  outlines (10 s on the GPU, model load included).
- **Carey s0314, rigid then warp (an older run with smoothing_px 30).** Ventricles and dorsal cortex closer;
  the 300 most confident pairs sit in dorsal cortex and striatum, almost none ventrally, so
  the ventral warp is filled in from far away and stretched the ventrolateral edge
  (flags FOLDS 0.5 %, visceral cortex expanded 2.4x).

## Limits (the engine's own)

One RGB image pair per call. No weighting in the matcher: a section mask can only drop matches whose section end has weight under 0.5 (the reply says "approximated by dropping matches"), and it estimates no mask of its own (`estimated` is null). Brightness: 8-bit, Nissl rendered dark on white. `smoothness_um` becomes `smoothing_px` per section from the canvas geometry.

A missing interpreter or source folder is refused at call time with `ENGINE_UNAVAILABLE`.
