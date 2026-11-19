# Brainways as a LangSlice tool

A LangSlice engine (`docs/developers/plugins.md`) that offers the agent one tool,
`brainways_ap`: the automatic search of Brainways 0.1.16.3 (Kantor et al. 2025,
Neuropsychopharmacology; https://github.com/bkntr/brainways, GPL-3.0) over the atlas's coronal
planes, which gives each section's position along the slicing axis and nothing else (no cutting
angles, no in-plane fit, no hemisphere). Design: Fable's proposal with Nash's decisions of
2026-10-19 (SliceBench `notes/2026-10-19_deepslice_brainways_plugin_design/fable_proposal.md`:
Brainways for rat, `whs_sd_rat_39um` only, coronal only, position only,
`position_mm = ap * 0.039`).

The network runs in LangSlice's own process. Only the part of Brainways that finds the position
is here, vendored from commit `1c43b0b708564974ce057167b6f83ed967dd3b04` (version 0.1.16.3)
under the GPL-3.0 (`LICENSE`, the license text Brainways ships); the rest of Brainways (napari,
aicsimageio, TensorFlow, its project files) is not needed, so its pins (zarr below 2.16, numpy
below 2, TensorFlow below 2.20) do not reach LangSlice.

- `plugin.py`: the adapter (imports `langslice.plugin_api`, `images` and `network`).
- `images.py` (vendored): the thumbnail (`brainways.utils.image`: `get_resize_size`,
  `resize_image`, `slice_to_uint8`, copied) and the network's input (the albumentations chain of
  `brainways.model.model_utils.load_model`, written with numpy and OpenCV).
- `network.py` (vendored): the weights, the backbone (timm's `tf_efficientnetv2_s`, features
  only, its last stage, written as functional torch on the checkpoint's tensors), the siamese
  head and the binary search (`brainways.model.siamese.siamese_model.SiameseModel.predict`).

Needs torch (a core dependency of LangSlice), numpy and OpenCV; neither timm, albumentations nor
huggingface_hub. A call where torch does not import is refused with `ENGINE_UNAVAILABLE`, naming
`pip install "langslice[engines]"`.

## Use

```bash
```

It starts off: switch it on with `--enabled brainways_ap` (`enabled=["brainways_ap"]` in Python). Its options are the entry of the job's `plugin_config`
under `langslice.engines.brainways.plugin:tools`.
Options
(`--plugin-config 'langslice.engines.brainways.plugin:tools=JSON'`): `device` (`auto`, the GPU
when torch sees one; `cuda`; `cpu`), `long_edge` (3072: the long edge in pixels of the one
channel handed to Brainways, which makes its own 1024-pixel thumbnail), `cache_folder` (where
the weights are kept; default `~/.cache/langslice/brainways`). An unknown option fails the
plugin's load.

## The tool

`brainways_ap(sections, channel, write, view)`, kind write, group Positioning, task `position`,
writes `position`. Offered only on `whs_sd_rat_39um`, coronal jobs; on any other atlas or plane
it is absent and no other tool's text names it. Its description, arguments and replies are those
of the SliceBench plugin it was moved from (`comparators/plugins/brainways/`).

| Argument | Meaning | Default |
|---|---|---|
| `sections` | filenames to run; each section is placed independently | every section |
| `channel` | the raw channel Brainways reads | the section's first raw channel |
| `write` | list holding `"positions"`; `[]` writes nothing and shows the proposal | `["positions"]` |
| `view` | the positioning picture | false |

- **Input**: ONE raw channel of each section as the agent oriented it (quarter turn, then flip),
  the delivered pixels (`section_raster(raw=True, images=[name])` at `long_edge`).
- **Writes**: `set_positions`, one undo step; a frozen position is that row's `FROZEN` and the
  rest land; a changed position clears the deformation it moves. The angles are never touched.
- **Reply**: `written`, `columns` / `rows` (per section in proposed order: section,
  `position_mm`, `change_mm`, written), `channel`, and when they apply `refused`, `clamped`
  (`id`, `asked_mm`, `used_mm`) and, without a picture, one sentence pointing to the positioning
  view of `look`. No metric or confidence.
- **Every call** saves `plugins/brainways_ap/calls/<n>/`: `images/` (the channel handed to the
  network, as PNGs), `settings.json` (arguments, channel used, device, GPU, model id, weights
  revision and SHA-256, torch version, timings) and `rows.json` (the table and each section's
  plane index `ap`, `position_mm`, `seconds`).

A refused call: `BAD_ARGUMENT` (a section named twice, an unknown `write` part or channel, `device`
`cuda` without a GPU), `ENGINE_UNAVAILABLE` (torch does not import; the weights cannot be fetched),
`FIT_FAILED` (the atlas is not `whs_sd_rat_39um` at 1024 x 512 x 512, 39 um).

## What the vendored path does

1. Thumbnail: fit the channel into 1024 x 1024 keeping its aspect (halve while at least twice
   the target, then bilinear), clip at the levels that saturate 0.1 % of the pixels at each end
   (a 1024-bin histogram whose end bins are dropped when they hold more than their neighbours),
   stretch to 0-255.
2. Network input, for the thumbnail and for each atlas plane (the atlas's `template` divided by
   its largest value, float32): resize to 224 x 224 by area, clip at the 0.1 and 99.8
   percentiles and stretch to 0-1, round to 8 bits, repeat in three channels, subtract the
   ImageNet mean and divide by its deviation.
3. Embedding: EfficientNetV2-S (timm `tf_efficientnetv2_s`: "same" padding, batch-norm epsilon
   1e-3, SiLU, squeeze-and-excitation), its last stage (256 channels) averaged over the plane.
4. Head: the section's and the plane's embeddings concatenated, a 512-256-1 perceptron, `tanh`:
   negative when the section lies before the plane.
5. Search: binary search over planes 122 to 800 (Brainways' rat limits); the answer is the
   plane the search ends on, `ap`, and `position_mm = ap * 0.039`.

Brainways pins numpy below 2, and its input chain's results depend on numpy 1's type promotion
(its percentiles take the quantile index in float32 and interpolate in float64; a section is
stretched in float64, an atlas plane in float32). Real Brainways run under numpy 2 moves 1 of the
43 Kondo F5 sections by 2 planes. `images.py` spells each step's types out, so it gives numpy 1's
result under numpy 2.

Weights: Brainways' own checkpoint `5redxpw6/state_dict.pt` (the model `model_ids.json` names for
`whs_sd_rat_39um`), fetched over HTTPS from the Hugging Face repository `brainways/siamese` at
revision `a0f3b1957b59b2ca2c02f7d094725c6dcb77783d` (`HF_ENDPOINT` honoured), checked against its
SHA-256 `a3835674...89f0fd`, kept under `cache_folder/<revision>/5redxpw6/`. The plain download
needs no Xet client, so `HF_HUB_DISABLE_XET` plays no part.

## Parity with Brainways 0.1.16.3 (2026-11-15)

Brainways 0.1.16.3 itself (commit `1c43b0b`, installed without its dependencies beside torch
2.14.1, timm 1.0.31, albumentations 2.0.5, OpenCV 4.10 and numpy 1.26.4, in a throwaway
environment) against this package (numpy 2.3.5, torch 2.14.1+cu130), on the Kondo rat stacks
(`kondo_ofc_rat_coronal`, channel 0, as SliceBench's Brainways comparator runs them):

- GPU (RTX 5090): all 1189 sections of the 25 stacks of the saved comparator run
  (`results/paper_figures/Fig2_v30/panel_d_right/comparators__brainways__kondo_ofc_rat_coronal_ch0`,
  Linux) give the same plane; the 43 F5 sections of the engine check of 2026-10-19 too (max
  difference 0 planes).
- CPU: 204 sections (F1, F5, F10) give the same plane as Brainways on the CPU (max difference
  0). The CPU and the GPU differ by 1 or 2 planes on 3 of them, in Brainways as here.
- Bitwise on the CPU, 43 F10 PHAL sections and 97 atlas planes: the thumbnails, the network
  inputs, the embeddings and 4171 head outputs are equal.
