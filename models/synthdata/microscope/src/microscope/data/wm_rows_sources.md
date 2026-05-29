# White-matter row model — sources & inspiration

Compiled 2026-05-29. Documents the biological grounding and statistical-method
inspiration behind `microscope/wm_rows.py` — the interfascicular
oligodendrocyte-row placement model for the DAPI microscopy simulator.
Synthesized from a multi-agent literature deep-read (2026-05-29; recorded in the
project memory `project_wm_nuclei_spatial_model`) and QC'd against a real lab
Keyence BZ-X800 DAPI white-matter tract (`M04_002_001`).

This file documents the *placement* model only. Cell **densities** are sourced in
`SOURCES.md`; nucleus **sizes** in `size_priors.md`; cell **shapes** are real
COSEM/OpenOrganelle segmentations (see `cosem_shapes.py`).

---

## The model in one line

White matter in DAPI = a **3-D Poisson process of finite, fiber-aligned
oligodendrocyte CHAINS** (segmented "bead" rows of nuclei) plus interstitial
astrocytes and microglia. `wm_rows.py` seeds chains uniformly in the slab volume,
orients each by a fiber azimuth + 3-D dispersion, and traces segmented
Gamma-renewal bead rows that are truncated where they exit the section faces.

## Why (the un-traceability problem)

Gray matter ≈ a hard-core Poisson point cloud — easy. White matter is anisotropic
AND, in a real 50 µm section, individual axons are **hard to trace**: some leave
the slab and are sliced off, some leave the focal plane (defocus), some collide.
The model reproduces that from the 3-D structure rather than faking it in 2-D.

## Model feature → source

| Feature in `wm_rows.py` | Grounding / inspiration |
|---|---|
| Oligodendrocytes line up in **rows along axons**; rows are **segments of ~8 contiguous cells** capped by a **solitary interfascicular astrocyte**; **~60 µm segment length**; **~15 µm between-row** core-to-core spacing | Suzuki & Raisman 1992, rat fimbria — **[1]** |
| Mouse corpus-callosum confirmation of the interfascicular-row arrangement | Tanaka et al. 2021, mouse CC, EM — **[2]** |
| **within-row ~8 µm < between-row** (the anisotropy invariant); **shifted-Gamma renewal** bead spacing; nuclei as offspring strung along parent line-segments; uniform-in-volume seeding | Poisson line / segment cluster point processes — **[3]**; closest published nucleus analog — **[4]** |
| **Out-of-plane dip / 3-D orientation dispersion** — the "hard to trace" realism (slab-face truncation, defocus, crossings) | fiber orientation-dispersion from diffusion-MRI numerical phantoms: Watson distribution (NODDI) **[5]**; ConFiG **[6]**; MEDUSA (explicitly places oligodendrocytes/astrocytes among dispersed axons) **[7]** |
| gentle in-plane **waviness** (mean-reverting heading walk) | organic departure from straight lines; consistent with measured fiber waviness/undulation in **[6]** |

## Citations

**Biological grounding (confirmed):**

- **[1] Suzuki M, Raisman G (1992).** *The glial framework of central white matter
  tracts: segmented rows of contiguous interfascicular oligodendrocytes and
  solitary astrocytes ... in the adult rat fimbria.* Glia. **PMID 1478731.**
  → Directly parameterizes `seg_mean` (~8 oligos/segment), `mu_within` (60 µm / 8
  ≈ 7.5 µm), the inter-segment astrocyte spacer, and the ~15 µm between-row spacing.
- **[2] Tanaka T, et al. (2021).** *Large-scale electron-microscopic volume imaging
  of interfascicular oligodendrocytes in the mouse corpus callosum.* Glia
  69:2488–2502. **doi:10.1002/glia.24055.** → mouse-specific confirmation
  (also cited in `size_priors.md`).

**Statistical / methodological inspiration:**

- **[3] Poisson line cluster point processes** — Møller, Safavimanesh & Rasmussen,
  *The cylindrical K-function and Poisson line cluster point processes*
  (Biometrika, 2016); and segment / germ-grain processes (Chiu, Stoyan, Kendall &
  Mecke, *Stochastic Geometry and Its Applications*, 3rd ed., 2013). → the formal
  model class: points clustered around randomly placed lines/segments.
- **[4] Hierarchical columnar point process for pyramidal-cell nucleoli** —
  arXiv:1908.05065. → closest published analog (aligned columns of nuclei via a
  cluster-plus-interaction model); we use only the forward/generative half.
- **[5] Zhang H, et al. (2012).** *NODDI: practical in vivo neurite orientation
  dispersion and density imaging.* NeuroImage 61:1000–1016. → the **Watson
  distribution** for fiber orientation dispersion (our per-chain dip σ ≈ dispersion).
- **[6] Callaghan R, et al. (2020).** *ConFiG: Contextual Fibre Growth to generate
  realistic axonal packing for diffusion MRI simulation.* NeuroImage.
  **PMCID PMC7903162.** → realistic WM fibre geometry, dispersion, undulation.
- **[7] Ginsburger K, et al. (2019).** *MEDUSA: a GPU-based tool to create realistic
  phantoms of the brain microstructure using tiny spheres.* NeuroImage.
  **PMID 30849528.** → numerical phantom that explicitly places glial cells among
  dispersed axons.

## Method & integrity note

Sources were assembled via a multi-agent deep-read workflow (8 parallel literature
readers + adversarial verification, 2026-05-29). **[1]** (PMID) and **[2]** (DOI)
are confirmed; exact volume / page / DOI for **[3]–[7]** should be re-verified
before any publication use. The model is a forward generator inspired by these
works, not a re-implementation of any one of them. Validated visually against the
real lab tract `M04_002_001` (single-plane render ≈ real: faint crossing wisps in
a hazy mid-blue floor).
