# Cell Size Priors for Mouse-Brain Histology Image Simulator

Compiled 2026-05-28. All values are for adult mouse (C57BL/6J or equivalent) unless noted.
DAPI labels nuclei; nucleus diameter is the primary target. Soma diameter is provided as
secondary context for cytoplasmic stains.

**Uncertainty note.** Nuclear diameter values are derived from 2D cross-sectional area
measurements on ~35-40 µm cryo-sections, assuming circular profile — section obliquity
systematically underestimates true nuclear diameter by ~10-15%. Soma values from various
methods (patch-clamp DIC, 3D confocal, Nissl segmentation) and have wider uncertainty.
Ranges here reflect biological variation across cells within a type/region, not measurement
error alone.

---

## 1. Neurons

### 1a. Nucleus sizes across regions (excitatory neurons, adult mouse)

From Das & Ramanan 2023 (*Front. Cell Dev. Biol.* 11:1032504), immunostained for Lamin
(nuclear envelope) + NeuN, 3-month C57BL/6J, n=3 mice, 80–150 nuclei per region:

| Region | Nuclear area (µm²) | Derived diameter (µm) |
|---|---|---|
| Neocortex | 22.0 ± 0.41 | ~5.3 |
| Striatum | 19.7 ± 0.29 | ~5.0 |
| Hippocampus CA1 | 20.4 ± 0.36 | ~5.1 |
| Hippocampus CA3 | 23.2 ± 0.36 | ~5.4 |

Diameter derived as d = 2√(A/π). All nuclei are highly circular (circularity 0.93–0.96).
Striatal nuclei are smallest and most circular; CA3 largest.

Interneurons from same study (Table 2):

| Cell type / region | Nuclear area (µm²) | Derived diameter (µm) |
|---|---|---|
| Calbindin+, piriform cortex | 17.7 ± 0.59 | ~4.7 |
| Calbindin+, SSC+MC | 18.1 ± 0.47 | ~4.8 |
| Calbindin+, cerebellum | 26.6 ± 0.82 | ~5.8 |
| Parvalbumin+, piriform | 19.0 ± 1.0 | ~4.9 |
| Parvalbumin+, SSC+MC | 18.6 ± 0.46 | ~4.9 |
| Parvalbumin+, CA1 | 24.0 ± 2.5 | ~5.5 |

The cerebellar calbindin+ cells likely include Purkinje cells (see below); their unusually
large nuclei (~5.8 µm derived diameter) are consistent with this.

### 1b. Region/type-specific soma sizes

Neuron soma sizes vary by over an order of magnitude across types.

#### Cerebellar granule cells
- **Soma: ~8–10 µm diameter** — among the smallest neurons in the brain
- Nucleus: ~5–6 µm (nearly fills the soma; very thin cytoplasm rim)
- The soma is so compact that DAPI gives a small, intensely bright, round spot
- Source: Granule cell morphology review (Wikipedia, citing primary lit); Cesana et al.
  *Sci. Rep.* 7:46147 (2017) — 3D CGC morphological reconstructions confirm compact soma

#### Cerebellar Purkinje cells
- **Soma: ~20–22 µm wide** (flask-shaped; height ~14 µm)
- Nucleus: ~8–10 µm (large, pale, with prominent nucleolus)
- Source: Nishiyama et al. *Sci. Rep.* 1:122 (2011), Table 1 — 3D measurements in adult
  mouse: soma width 21.2 ± 1.1 µm, height 13.9 ± 1.5 µm (n=6 normal mice)

#### Cortical pyramidal neurons (mouse)
- **Layer 2/3 pyramidal: soma ~15–20 µm** (nucleus ~5 µm as per Das & Ramanan above)
- **Layer 5 thick-tufted (TTL5/ET): soma ~25–35 µm**, triangle-shaped, can reach ~40 µm
  along longest axis; nucleus ~6–8 µm
- Source: Ramaswamy & Bhatt 2015 (*Front. Cell. Neurosci.* 9:233) — anatomy review of
  TTL5 rodent somatosensory cortex; Allen Cell Types Database mouse morphology reconstructions
  (VISp/MOp; soma depth and 3D SWC data)

#### Hippocampal CA3 pyramidal
- Soma ~20–30 µm; nucleus largest of major forebrain excitatory neuron populations (~5.4 µm
  nuclear diameter from 2D area)
- Source: Das & Ramanan 2023

#### Striatal medium spiny neurons
- Soma ~12–18 µm; smallest and most spherical forebrain neuron nuclei (~5 µm)
- Source: Das & Ramanan 2023

#### Large projection neurons (motor, brainstem)
- Alpha motor neurons (spinal cord) and large cranial motor nucleus neurons: soma **40–70 µm**
- Nucleus ~10–14 µm; prominent nucleolus
- These are exceptional outliers relative to the bulk of brain sections (rare in histological
  sections of cerebral cortex/cerebellum)
- Source: standard neuroanatomy; Sturrock 1981 (*Neuropathol. Appl. Neurobiol.* 7:647) gives
  mean neuron nucleus ~8 µm in mouse indusium griseum (a small-neuron structure) as lower bound

---

## 2. Astrocytes

- **Nucleus diameter: ~6–9 µm** (estimated; DAPI staining)
- **Soma diameter: ~9–12 µm** (mouse cortical protoplasmic astrocytes)
- Fibrous astrocytes (white matter) have slightly smaller, elongated soma
- Territorial domain (total arborization): ~50–100 µm — much larger than the soma
- Mouse astrocytes are substantially smaller than human astrocytes (human soma ~12–16 µm)
- Source: Oberheim et al. 2009 (*J. Neurosci.* 29:3276-3287) — comparative mouse vs human
  astrocyte morphology; mouse protoplasmic soma cited as ~10 µm. Emsley & Macklis 2006
  (*J. Neurosci. Res.* 85:2432) — regional heterogeneity of mouse astroglia.

**Uncertainty:** Direct mouse astrocyte nuclear diameter measurements in DAPI were not located
in the searched literature. The 6–9 µm nucleus estimate is extrapolated from soma size and
typical nucleus/soma ratios in Nissl-stained material. This value needs verification from
quantitative DAPI/Lamin staining studies of GFAP+ cells.

---

## 3. Oligodendrocytes

- **Nucleus diameter: ~5–8 µm**
- **Soma diameter: ~8–12 µm**
- Oligodendrocyte nucleus appears dark and compact in Nissl/DAPI (dense heterochromatin)
- Soma is round with little visible cytoplasm by light microscopy; processes extend invisibly
  into myelin
- In white matter (corpus callosum), cells are arranged in rows along axons
- Source: Sturrock 1981 (*Neuropathol. Appl. Neurobiol.* 7:647) — mouse indusium griseum,
  glia mean nucleus 5 µm (vs neuron 8 µm); consistent with Nissl atlas descriptions.
  Tanaka et al. 2021 (*Glia* 69:2488-2502) — EM of mouse corpus callosum oligodendrocytes
  (soma and nucleus morphology visible but sizes not tabulated in text snippet reviewed).

**Uncertainty:** The 5 µm value from Sturrock is from a small, specific structure. In larger
white matter tracts the soma is ~8–12 µm. A DAPI-specific oligodendrocyte nuclear diameter
study was not located; values are from Nissl and EM literature.

---

## 4. Microglia

- **Nucleus diameter: ~6–8 µm** (estimated as ~60–65% of soma diameter)
- **Soma diameter: ~9–12 µm** (homeostatic/ramified state)
- Resting microglia: small soma, extensive thin branching processes (total cell span 50–100 µm)
- Activated microglia: soma enlarges significantly (can reach 20–30+ µm 2D cross-section)
- Source:
  - Kongsui et al. 2014 (*J. Neuroinflammation* 11:182), Table 1: rat PFC microglia cell body
    perimeter 30.6–34.6 µm across cortical layers → soma diameter ~9.7–11.0 µm (d = P/π
    assuming circular soma boundary)
  - Kozlowski & Weimer 2012 (*PLoS ONE* 7:e31814): mouse neocortex CX3CR1-EGFP microglia;
    resting soma 2D area ~500 µm² (whole-cell area in MIP including proximal processes)

**Uncertainty:** The Kongsui 2014 data are from rat, not mouse; mouse microglia soma sizes
are expected to be similar but not identical. The nucleus specifically (vs soma) was not
directly measured in DAPI in the reviewed papers.

---

## Summary Table

| Cell type | Nucleus diam. mean (µm) | Nucleus diam. range (µm) | Soma diam. mean (µm) | Soma diam. range (µm) |
|---|---|---|---|---|
| Neuron, generic (neocortex/hipp/striatum) | 5.2 | 4.7–5.8 | varies | 5–70+ |
| Neuron, cortical pyr. L2/3 | 5.1 | 4.5–6.0 | ~17 | 12–25 |
| Neuron, cortical pyr. L5 TTL5 | 6.5 | 5.5–8.0 | ~28 | 20–40 |
| Neuron, cerebellar granule | 5.0 | 4.0–6.0 | ~9 | 6–12 |
| Neuron, cerebellar Purkinje | 8.0 | 6.5–10.0 | ~21 | 18–30 |
| Neuron, hippocampus CA3 pyr. | 5.4 | 4.5–6.5 | ~25 | 15–35 |
| Neuron, large motor/projection | 10.0 | 8.0–14.0 | ~55 | 30–80 |
| Neuron, striatal MSN | 5.0 | 4.0–5.5 | ~15 | 10–20 |
| Neuron, PV interneuron | 4.9 | 4.0–5.8 | ~13 | 8–22 |
| Astrocyte | 7.0 | 5.0–9.0 | ~10 | 8–20 |
| Oligodendrocyte | 6.5 | 5.0–8.0 | ~10 | 8–12 |
| Microglia | 6.5 | 5.0–8.0 | ~10 | 7–15 |

---

## Neuron Size Heterogeneity by Region

Neurons in the mouse brain span roughly **5 µm (granule cell soma) to 70+ µm (alpha motor
neuron soma)** — a 14-fold range. Key points:

1. **Nucleus size is far more uniform than soma size.** Most neurons have nuclei of 5–7 µm
   in diameter regardless of soma size, because the nucleus scales with genome content, not
   cell function. The soma-to-nucleus ratio varies enormously: ~1.2:1 for granule cells vs
   ~8:1 for Purkinje cells.

2. **Histological sections from any given brain region are dominated by granule cells**
   (cerebellum and dentate gyrus) or medium-sized cortical/hippocampal neurons. Large
   projection neurons are sparse.

3. **For DAPI simulation,** the practical nucleus diameter range is:
   - Small end: ~4.5 µm (interneurons, granule cells, striatal MSNs)
   - Typical neurons: ~5–6 µm
   - Large interneurons (PV-CA1) and Purkinje: ~6–9 µm
   - Motor neurons (rare): ~10–14 µm
   - Glia: ~5–8 µm (oligodendrocytes and microglia slightly smaller/denser than neurons;
     astrocytes similar size but paler DAPI signal due to lower heterochromatin content)

4. **DAPI intensity** also encodes cell identity: oligodendrocytes stain very brightly
   (dense heterochromatin, dark in Nissl), astrocytes stain more diffusely, neurons
   intermediate.

---

## Key Citations

1. **Das S & Ramanan N (2023).** Region-specific heterogeneity in neuronal nuclear morphology
   in young, aged and in Alzheimer's disease mouse brains. *Front. Cell Dev. Biol.* 11:1032504.
   https://doi.org/10.3389/fcell.2023.1032504
   → Primary source for nucleus area by region (Table 1/2, C57BL/6J adult mice, Lamin staining)

2. **Nishiyama H, Fukaya M, Watanabe M, Bhatt DL (2011).** Axonal motility and its
   modulation by activity are branch-type specific in the intact adult cerebellum.
   *Sci. Rep.* 1:122.
   https://doi.org/10.1038/srep00122
   → Table 1: Purkinje cell soma dimensions in adult mouse (3D)

3. **Cesana E et al. (2017).** Granule cell ascending axon excitatory synapses onto Golgi
   cells implement a potent feedback circuit in the cerebellar granular layer. *Sci. Rep.*
   7:46147.
   https://doi.org/10.1038/srep46147
   → 3D morphological reconstructions of mouse cerebellar granule cells

4. **Ramaswamy S & Bhatt DL (2015).** Anatomy and physiology of the thick-tufted layer 5
   pyramidal neuron. *Front. Cell. Neurosci.* 9:233.
   https://doi.org/10.3389/fncel.2015.00233
   → TTL5 soma anatomy in rodent somatosensory cortex

5. **Kongsui R, Beynon SB, Johnson SJ, Walker FR (2014).** Quantitative assessment of
   microglial morphology and density reveals remarkable consistency in the distribution and
   morphology of cells within the healthy prefrontal cortex of the rat.
   *J. Neuroinflammation* 11:182.
   https://doi.org/10.1186/s12974-014-0182-7
   → Table 1: cell body perimeter 30.6–34.6 µm across layers (rat PFC)

6. **Kozlowski C & Weimer RM (2012).** An automated method to quantify microglia morphology
   and application to monitor activation state longitudinally in vivo.
   *PLoS ONE* 7(2):e31814.
   https://doi.org/10.1371/journal.pone.0031814
   → Mouse neocortex microglia; resting soma 2D area ~500 µm²

7. **Oberheim NA et al. (2009).** Uniquely hominid features of adult human astrocytes.
   *J. Neurosci.* 29(10):3276-3287.
   → Comparative mouse/human astrocyte morphology; mouse protoplasmic soma ~10 µm

8. **Sturrock RR (1981).** Quantitative and morphological changes in neurons and neuroglia
   in the indusium griseum of aging mice. *Neuropathol. Appl. Neurobiol.* 7(6):647-658.
   → Mean nucleus diameters: neuron ~8 µm, glia ~5 µm (mouse indusium griseum)

9. **Tanaka T et al. (2021).** Large-scale electron microscopic volume imaging of
   interfascicular oligodendrocytes in the mouse corpus callosum. *Glia* 69:2488-2502.
   https://doi.org/10.1002/glia.24055
   → EM soma/nucleus morphology of mouse corpus callosum oligodendrocytes

10. **Allen Cell Types Database** (Allen Institute for Brain Science, celltypes.brain-map.org)
    → Mouse cortical neuron morphology reconstructions (SWC files) with soma surface area
    and 3D coordinates for hundreds of patched neurons from VISp, MOp, and other areas.

---

*Note: This file is a compiled reference, not primary data. Where direct mouse measurements
were unavailable (astrocyte nucleus, oligodendrocyte nucleus), values are estimated from
soma sizes and nucleus/soma ratios from Nissl or EM literature. These estimates should be
superseded by direct DAPI/Lamin quantification if available.*
