# Synthetic Histology — Library & Tool Index

Cleaned index of every library/tool surfaced by the deep-research pass in
[`synthdataideas.md`](synthdataideas.md). The original report had broken
link formatting (`urlNAMEhttps://...` concatenated tokens, stray `citeturn`
markers); this file preserves the substance with proper Markdown.

Verdicts use the report's vocabulary:

- **slot in** — drops into the existing NumPy/SciPy pipeline as an additive
  module without replacing core procedural logic.
- **partially replace** — would take over one stage (e.g. final texture pass).
- **replace** — broader rewrite, only worth it if you want what it gives you.

---

## Recommended stack (top-line)

The report's executive recommendation, in build order:

1. **Keep the atlas-aligned procedural core.** Your AP labels, section
   metadata, and damage maps live there.
2. **Add cell-density and cell-type priors** at 25 µm:
   [atlas-densities](#atlas-densities) +
   [abc_atlas_access](#abc_atlas_access).
3. **Use morphology synthesis only where silhouette matters** (Purkinje,
   cortical pyramidal belts, hippocampal principal layers, ependymal edge):
   [MorphIO](#morphio) + [NeuroTS](#neurots).
4. **Stop hand-coding optics.** Use [microsim](#microsim) as the default
   imaging backend; reach for [PSF-Generator](#psf-generator) or
   [pySTED](#psted) for explicit PSF / bleaching.
5. **Standardize modality appearance** with pathology libraries instead of
   ad-hoc color code: [torchstain](#torchstain) (lightweight in-pipeline),
   [HistomicsTK](#histomicstk) (stain deconvolution), and either
   [TIAToolbox](#tiatoolbox) or [Slideflow](#slideflow) when you want a
   broader toolkit.
6. **If you add learned detail, keep it local and bounded.** Train
   [pix2pix](#pytorch-cyclegan-and-pix2pix) or [img2img-turbo](#img2img-turbo)
   on small tiles, or use [CAX](#cax) as a mask-bounded NCA residual.

---

## Procedural microscopy simulators

### DeepTrack2
- **Repo:** https://github.com/DeepTrackAI/DeepTrack2
- **License:** MIT
- **Status:** active; latest release 2025-03-26
- **Verdict:** slot in
- Modular Python library for generating, manipulating, and analyzing
  microscopy image pipelines. Slot it in as a configurable forward-model
  and augmentation layer behind your atlas-aligned tissue synthesis.

### microsim
- **Repo:** https://github.com/tlambert03/microsim
- **License:** BSD-3
- **Status:** active; PyPI releases through 2026-03-10
- **Verdict:** slot in (recommended optics backend)
- Light-microscopy simulator that explicitly separates ground truth, optical
  image formation, and digital image formation: PSF convolution, detector
  noise, downsampling. The cleanest way to stop hand-rolling camera noise,
  dynamic range, and output-resolution effects.

### SyMBac
- **Repo:** https://github.com/georgeoshardo/SyMBac
- **License:** GPL-2.0
- **Status:** active; PyPI releases 2026-03
- **Verdict:** reference, not adopt (license + scope)
- Procedural object generation + microscope forward model + perfect labels
  pattern. Read it for architectural ideas; not a fit as a dependency.

### pySTED
- **Repo:** https://github.com/FLClab/pySTED
- **License:** MIT
- **Status:** active
- **Verdict:** slot in (fluorescence branch only)
- Fluorescence/STED simulator with explicit fluorophore, point-scanning,
  and photobleaching modeling. Useful as a specialized photophysics module
  for the fluorescence branch; doesn't help Nissl/DAPI/brightfield/ISH.

### python-microscopy
- **Repo:** https://github.com/python-microscopy/python-microscopy
- **License:** GPL-3
- **Status:** active; releases 2026-03
- **Verdict:** replace (broader than wanted)
- Large optical microscopy environment with acquisition simulation,
  detectors, stages, lasers, and a simulator. Treat as a reference
  implementation when you need hardware-level realism, not as a clean
  NumPy module.

---

## Cell-type and morphology priors

### abc_atlas_access
- **Repo:** https://github.com/AllenInstitute/abc_atlas_access
- **License:** Allen non-commercial BSD-variant ("AS IS")
- **Status:** active; releases continuing through 2026
- **Verdict:** slot in (data source)
- Official access layer for the Allen Brain Cell Atlas. 4M-cell whole-brain
  MERFISH dataset + example notebooks. Use for AP-dependent cell-type
  mixture priors and region-conditioned transcriptomic texture maps.

### atlas-densities
- **Repo:** https://github.com/openbraininstitute/atlas-densities
- **License:** Apache-2.0
- **Status:** active
- **Verdict:** slot in (recommended density priors)
- Toolchain for producing voxelwise mouse-brain cell-type density volumes
  from Allen data plus literature values. Commands explicitly target
  `annotation_25.nrrd`, so it drops into a 25 µm pipeline directly.

### MorphIO
- **Repo:** https://github.com/openbraininstitute/MorphIO
- **License:** Apache-2.0
- **Status:** active; latest release 2026-03-25
- **Verdict:** slot in (morphology I/O)
- Reader/writer for SWC, ASC, and H5 neuron morphologies. Use as the I/O
  layer for turning exemplar reconstructions into slice-intersected
  silhouettes, soma diameters, arbor statistics, and class-specific
  templates.

### NeuroTS
- **Repo:** https://github.com/openbraininstitute/NeuroTS
- **License:** Apache-2.0
- **Status:** active
- **Verdict:** partially replace (where silhouette matters)
- Neuronal-tree synthesis from statistical distributions extracted from
  reconstructed cells. Replace blob-only local texture with explicit
  cell-class morphology for Purkinje layers, pyramidal cortical strata,
  hippocampal principal-cell belts, and ventricular-edge ependymal cells.

### AllenSDK
- **Repo:** https://github.com/AllenInstitute/AllenSDK
- **License:** Allen
- **Status:** "selective maintenance mode"; latest tagged release 2023-11-30
- **Verdict:** slot in (data extraction only)
- Canonical SDK for reading and processing Allen Brain Atlas data. Use as a
  metadata bridge — don't build new simulator code on top of it.

---

## Optics, PSF, and artifact toolkits

### PSF-Generator
- **Repo:** https://github.com/Biomedical-Imaging-Group/psf_generator
- **License:** MIT
- **Status:** released 2025
- **Verdict:** slot in
- PyTorch library implementing scalar and vectorial microscope PSF models,
  including Fourier- and Richards–Wolf-style propagators. Use as a fast PSF
  engine for fluorescence/IHC branches and aberration-aware blur.

(microsim and pySTED also listed under "Procedural simulators" — both are
relevant here for noise/detector and bleaching respectively.)

---

## Histology normalization and augmentation

### TIAToolbox
- **Repo:** https://github.com/TissueImageAnalytics/tiatoolbox
- **License:** BSD-3
- **Status:** active; releases through late 2024, repo updates into late 2025
- **Verdict:** slot in (post-render normalization)
- Computational pathology toolbox with Macenko, Reinhard, Ruifrok, and
  Vahadane stain normalization. Robust post-render normalization and
  pathology preprocessing layer.

### HistomicsTK
- **Repo:** https://github.com/DigitalSlideArchive/HistomicsTK
- **License:** Apache-2.0
- **Status:** active; dev releases through 2026-04
- **Verdict:** slot in (best for H&E/IHC stain handling)
- Pathology toolkit with explicit APIs for color deconvolution,
  deconvolution-based normalization, and color augmentation. Best verified
  open-source package for H&E/IHC stain separation and color-grounded
  augmentation.

### torchstain
- **Repo:** https://github.com/EIDOSLAB/torchstain
- **License:** MIT
- **Status:** active; updates noted late 2025
- **Verdict:** slot in (cleanest NumPy fit)
- NumPy/PyTorch/TensorFlow backends implementing Macenko, Reinhard,
  modified Reinhard, multi-target Macenko, and Macenko-based augmentation.
  NumPy backend is first-class — best fit for a NumPy/SciPy pipeline.

### Slideflow
- **Repo:** https://github.com/slideflow/slideflow
- **License:** Apache-2.0
- **Status:** active
- **Verdict:** slot in (norm submodules) or grow into a larger stack
- Digital pathology platform; `slideflow.norm` provides efficient NumPy
  Macenko / Reinhard / Vahadane / HSV augmentation, plus optional
  CycleGAN-based normalization. Use only the normalization submodules, or
  let it grow into a broader pathology preprocessing stack.

### PathML
- **Repo:** https://github.com/Dana-Farber-AIOS/pathml
- **License:** GPLv2 (commercial licensing available)
- **Status:** active
- **Verdict:** slot in technically; not first choice (license)
- Stain normalization and deconvolution. Skip in favor of the permissive
  alternatives above unless its specific features are needed.

---

## Optional neural detail modules

All of these preserve label control if used as **bounded post-processors**
trained on small patches conditioned on procedural masks/coarse renders.

### pytorch-CycleGAN-and-pix2pix
- **Repo:** https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- **License:** BSD-style
- **Status:** active in 2025
- **Verdict:** partially replace (final texture stage)
- Paired pix2pix from procedural masks/coarse renders to real histology
  patches. Preserves AP labels and region masks upstream while adding
  learned realism downstream.

### img2img-turbo
- **Repo:** https://github.com/GaParmar/img2img-turbo
- **License:** MIT (note: SD-Turbo base model has its own checkpoint license)
- **Status:** active
- **Verdict:** partially replace (fast last-mile refiner)
- One-step image-to-image with very fast 512×512 inference; supports paired
  and unpaired translation.

### stylegan2-pytorch
- **Repo:** https://github.com/lucidrains/stylegan2-pytorch
- **License:** MIT
- **Status:** active
- **Verdict:** replace microtexture synthesizer only
- Train class-specific texture priors for white matter, cortex, cerebellum,
  hippocampus, or ventricle backgrounds; composite under existing masks.

### CAX
- **Repo:** https://github.com/maxencefaldor/cax
- **License:** MIT
- **Status:** active
- **Verdict:** slot in (mask-bounded local refinement)
- JAX library for neural cellular automata (growing, conditional,
  unsupervised, attention-based, diffusing) with GPU/TPU acceleration.
  Cleanest current option for an NCA-based stochastic detail layer that
  evolves locally within region masks rather than replacing global
  geometry.

---

## Reference points (not build targets)

The report calls these out as historically important but **not** the center
of gravity for a modern Python stack. Read for ideas about decomposing
specimen / optics / acquisition; build on the modern libraries above.

- **SimuCell** (2012) — flexible microscopy-image framework.
- **CytoPacq** (2019) — web interface over older simulators.

---

## Per-modality emphasis

The report's modality-specific guidance, condensed:

| Modality | Strongest fit |
|----------|---------------|
| IHC / brightfield | [HistomicsTK](#histomicstk) + [Slideflow](#slideflow) + [TIAToolbox](#tiatoolbox) |
| Nissl | Same as IHC/brightfield |
| Fluorescence / DAPI | [microsim](#microsim) / [pySTED](#psted) optics + light color/intensity perturbation |
| ISH | Tone, illumination, tissue-background perturbation (less stain-transfer machinery) |

---

## Open gaps the report did not close

1. **No actively maintained, permissively licensed package models classical
   paraffin/cryosection gross defects** — folds, knife chatter, tissue
   bites, ventricle blowout, mounting bubbles — as first-class operators.
   Modern tooling is much better at microscope physics and stain/style
   normalization than at section damage. The hand-rolled damage layer in
   the existing pipeline remains strategically valuable.
2. **No single package bundles whole-brain mouse region-specific density
   priors, morphology templates, and soma size distributions** in one
   actively maintained, permissively licensed API. The best path is fusion:
   Allen spatial/transcriptomic data + Open Brain Institute density and
   morphology libraries + your own procedural rendering logic.
