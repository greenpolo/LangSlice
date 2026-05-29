# Mouse-brain cell-density sources (for microsim DAPI/histology simulator)

Compiled 2026-05-28. **REAL published data only** — no fabricated or estimated density values.
Where a source does not provide a value, the cell is left blank (NaN) in the table.

Target: per-region ABSOLUTE cell densities (cells/mm³) keyed to Allen Mouse Brain CCFv3 region
IDs, split into neuron / astrocyte / oligodendrocyte / microglia (+ total). DAPI stains all
nuclei, so both total and per-type matter.

Output files in this directory:
- `density_by_region.csv` / `density_by_region.json` — compiled per-region table (955 rows).
- raw source files (see below).
- `allen_structure_graph_1.json` — Allen CCF structure hierarchy used as the name↔ID↔acronym crosswalk.

---

## Region-ID system

All rows are keyed to **Allen Mouse Brain Atlas / CCFv3 `structure id`** (the integer `id` in the
Allen `structure_graph` ontology, graph_id=1), plus `acronym` and full `name`.
Crosswalk source: `http://api.brain-map.org/api/v2/structure_graph_download/1.json`
(downloaded as `allen_structure_graph_1.json`, 1327 structures).
This is the same ontology used by brainglobe `allen_mouse` atlases, so mapping to brainglobe IDs
later is the identity map on `id`.

Mapping method: source region **names** were matched to the Allen ontology after a
comma/punctuation-insensitive, lowercased normalization (Erö stripped commas from AMBA names, e.g.
"Frontal pole cerebral cortex" → Allen "Frontal pole, cerebral cortex"). Murakami already ships
Allen structure IDs directly and was joined on ID.

---

## 1. PRIMARY (per-type backbone) — Erö et al. 2018

- **Citation:** Erö C, Gewaltig M-O, Keller D, Markram H (2018) *A Cell Atlas for the Mouse Brain.*
  Front. Neuroinform. 12:84. doi:10.3389/fninf.2018.00084
- **License:** CC-BY (Frontiers open access).
- **Files used (downloaded here):**
  - `ero2018_DataSheet2.CSV` — **per-region DENSITIES (cells/mm³)**. Columns: Regions, Cells,
    Neurons, Glia, Excitatory, Inhibitory, Modulatory, **Astrocytes, Oligodendrocytes, Microglia**.
    955 rows.
  - `ero2018_DataSheet1.CSV` — same regions, **absolute COUNTS** (used to derive Erö's implied
    region volume = count/density, kept as `ero_region_volume_mm3`, `ero_total_count`).
- **Exact download URL:** the supplementary "Data Sheet" files are served by Frontiers but only via
  JS on the article page; they were retrieved from the Europe PMC mirror of the open-access record
  **PMC6280067**:
  `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6280067/supplementaryFiles`
  (a ZIP whose members include `Data_Sheet_1.CSV`, `Data_Sheet_2.CSV`, `Data_Sheet_3.docx`).
  Article landing page / supplementary anchor:
  `https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00084/full#supplementary-material`
- **Units:** cells per mm³ (Data_Sheet_2); absolute cell number (Data_Sheet_1).
- **Methodology — MODEL-DERIVED.** Cell positions placed algorithmically from whole-brain Allen
  Nissl + gene-expression (ISH) stains; glia split into astro/oligo/microglia using marker ISH
  (e.g. Aldh1l1/GFAP, Olig2, Iba1-type markers) and literature glia fractions. Not direct counts.
- **Coverage:** 737 AMBA regions expanded to **955 rows** (includes cortical layer subdivisions and
  the full hierarchy). 953/955 names mapped to Allen IDs; the 2 unmatched are source typos
  ("Nucleus of reunions"→reuniens, "mammilothalmic tract"→mammillothalamic), which were hardcoded
  so all 955 map. This is the ONLY one of the four sources that gives a per-region
  astrocyte/oligodendrocyte/microglia split, so it is the backbone for those columns.
- **Note:** a published Corrigendum exists for this article; values here are from the supplementary
  data files as distributed with the OA record.

**Columns sourced from Erö 2018:** `total_density_cells_per_mm3`, `neuron_density`,
`astrocyte_density`, `oligodendrocyte_density`, `microglia_density`, `glia_density`,
`ero_region_volume_mm3`, `ero_total_count`.

## 2. CORROBORATING (neuron + volume) — Rodarie et al. 2022

- **Citation:** Rodarie D, Verasztó C, Roussel Y, Reimann M, Keller D, Ramaswamy S, Markram H,
  Gewaltig M-O (2022) *A method to estimate the cellular composition of the mouse brain from
  heterogeneous datasets.* PLoS Comput Biol 18(12):e1010739. doi:10.1371/journal.pcbi.1010739
- **License:** CC-BY (PLoS open access). Code: https://github.com/BlueBrain/atlas-densities (Apache-2.0).
- **File used (downloaded here):** `rodarie2022_s011.xlsx` (PLoS supplementary **S11 File**),
  sheet **"Densities BBCAv2"** — 861 regions.
  Columns: Brain region, PV/SST/VIP/Rest-inh/GAD67/Non-inh densities [mm⁻³] (+ std),
  **Neuron [mm⁻³]**, **Volumes [mm³]**.
  Also downloaded `rodarie2022_s010.xlsx` (S10 File — the GABAergic literature-measurement input
  table; not merged, kept for provenance).
- **Exact download URLs (verified):**
  - S11: `https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1010739.s011&type=supplementary`
  - S10: `https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1010739.s010&type=supplementary`
- **Units:** cells/mm³ (densities), mm³ (volumes).
- **Methodology — MODEL-DERIVED** (updates the Blue Brain Cell Atlas = "BBCAv2"). Rodarie re-derives
  neuron density and splits inhibitory neurons (PV/SST/VIP/rest) via constrained optimization over
  ISH + literature. **Its public supplement does NOT republish the astro/oligo/microglia split** — only
  neurons + inhibitory subtypes + region volume. So Rodarie is used as (a) the explicit region-volume
  source and (b) an independent cross-check on Erö's neuron density.
- **Coverage:** 861 regions, all mapped to Allen IDs (860 overlap Erö).

**Columns sourced from Rodarie 2022:** `rodarie_neuron_density`, `rodarie_region_volume_mm3`
(also `region_volume_mm3`, see merge note), `rodarie_gad67_density`, `rodarie_pv_density`,
`rodarie_sst_density`, `rodarie_vip_density`.

## 3. INDEPENDENT MAGNITUDE ANCHOR (measured total cells) — Murakami et al. 2018

- **Citation:** Murakami TC, Mano T, Saikawa S, et al. (2018) *A three-dimensional
  single-cell-resolution whole-brain atlas using CUBIC-X expansion microscopy and tissue clearing.*
  Nat Neurosci 21:625–637. doi:10.1038/s41593-018-0109-1
- **License:** © Springer Nature (article paywalled). Supplementary files are publicly downloadable
  from the Springer static content host. (Reused here under fair use as a numeric anchor; redistribute
  with care — not an open-access CC license.)
- **File used (downloaded here):** `murakami2018_MOESM5_suppTable3.xlsx` = **Supplementary Table 3**,
  "Total cell numbers for 8-week-old C57BL/6N male mice in each brain area." Columns: graph order,
  **ABA ID**, name, acronym, R,G,B, and total cell COUNT for **three independent brains** (8wk01,
  8wk02, 8wk03). 677 regions.
- **Exact download URL (verified):**
  `https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-018-0109-1/MediaObjects/41593_2018_109_MOESM5_ESM.xlsx`
  (Supp Tables 1 & 2 are MOESM3/MOESM4 at the same path; not needed.)
- **Units:** absolute total-cell COUNT per region (nuclear PI stain → all nuclei, i.e. comparable to
  a total/DAPI count). **NOT a density.** Keyed directly by Allen ABA structure ID (100% ID-match).
- **Methodology — DIRECTLY MEASURED** (CUBIC-X tissue clearing + light-sheet + automated nuclear
  detection). This is the only measured (non-model) source, so it anchors the Erö/Rodarie magnitudes.
- **Coverage:** 677 regions (664 overlap Erö by ID).

**Columns sourced from Murakami 2018:** `murakami_total_count_mean` (mean of 3 brains),
`murakami_total_count_sd` (sample SD across 3 brains), `murakami_total_density` (see conversion note),
`murakami_is_leaf_region`.

---

## Merge / derivation notes (unit conversions performed)

1. **`region_volume_mm3`** = Rodarie's explicit `Volumes [mm³]` where available, else Erö's implied
   volume (Erö total count ÷ Erö total density). Used for the Murakami density conversion.
2. **`murakami_total_density` = mean Murakami count ÷ `region_volume_mm3`.** This is the only computed
   unit conversion. **IMPORTANT CAVEAT:** Murakami's Supp Table 3 lists high-level *parent/aggregate*
   regions (e.g. "grey", "Cerebral cortex" CTX, "Thalamus" TH, "Midbrain" MB, "fiber tracts") holding
   only the cells *directly assigned* to that node (unparcellated remainder), NOT the recursive sum of
   their children. Dividing those parent counts by the full region volume gives an artifactually low
   density. Therefore `murakami_total_density` is populated **only for leaf regions** (no children in
   the Allen graph; `murakami_is_leaf_region == True`); for non-leaf regions it is left blank but the
   raw `murakami_total_count_mean` is retained for transparency. 639 leaf-region densities result.
3. No other values were transformed — Erö and Rodarie densities are carried through verbatim
   (rounded to 6 decimals).
4. **Fiber tracts / white matter:** 93 regions have `neuron_density == 0` in Erö (e.g. internal
   capsule, corpus callosum). This is anatomically correct (white matter has glia but ~no neuronal
   somata), not missing data — those rows still carry nonzero glia densities.

## Coverage summary of compiled table

- **955 rows** (= Erö's full region set), every row keyed to an Allen CCFv3 structure id.
- `neuron_density`, `astrocyte_density`, `oligodendrocyte_density`, `microglia_density`,
  `total_density_cells_per_mm3`: **955/955 present** (from Erö).
- `rodarie_neuron_density`: 861/955. `region_volume_mm3`: 955/955.
- `murakami_total_count_mean`: 665/955; `murakami_total_density` (leaf only): 639/955.
- **663 regions have all three sources** (Erö per-type + Rodarie neuron + Murakami measured).
- **Whole-CCFv3 coverage:** this covers the AMBA/CCFv3 *gray-matter region hierarchy plus major fiber
  tracts and cortical layer subdivisions* (the 737-region Erö parcellation expanded to 955). It does
  NOT include every one of the ~1327 nodes in the full Allen ontology (very fine leaf structures and
  some ventricular/spinal nodes absent from Erö are not present). For voxel-complete coverage, use the
  NRRD volumes below.

## Magnitude agreement (sanity)

- **Rodarie neuron / Erö neuron** (n=861 overlapping regions): median ratio **0.86**, mean 0.84
  (p10 0.49, p90 1.14). Rodarie's BBCAv2 revises whole-brain neuron count downward vs Erö 2018;
  same order of magnitude, good agreement.
- **Murakami total density / Erö total density** (n=639 leaf regions): median ratio **0.83**, mean
  0.88 (p10 0.54, p90 1.31). The independent CUBIC-X measurement runs ~17% below the Erö model total
  on average but is the same magnitude across the brain → Erö/Rodarie magnitudes are corroborated.
- Whole-brain (Allen id 8 "grey"): Erö total 243,077 cells/mm³, neuron 167,290, astro 12,059,
  oligo 38,919, microglia 24,809 cells/mm³ — consistent with published mouse whole-brain values
  (~71.7M neurons, ~108M total cells over the brain).

---

## 4. VOXEL-VOLUME (NRRD) availability verdict — AVAILABLE, directly downloadable

**Verdict: YES.** Precomputed CCFv3-native per-cell-type density volumes in NRRD exist and are
directly, anonymously downloadable. A voxel volume is richer than this per-region table (it has
within-region spatial structure), so for the simulator these NRRDs are the preferred resource; the
per-region CSV is a convenient region-summary fallback.

- **Host:** Blue Brain Open Data on the AWS Registry of Open Data — public S3 bucket
  `s3://openbluebrain/` (region us-west-2). No AWS account required:
  `aws s3 ls --no-sign-request s3://openbluebrain/...` or plain HTTPS.
  Registry: https://registry.opendata.aws/bluebrain_opendata/
- **Path:** `Model_Data/Brain_atlas/Mouse/resolution_25_um/version_1.1.0/Cell_densities/`
- **Per-cell-type density files (units = cells/mm³, per the folder README):**
  | file | size | HTTPS URL |
  |---|---|---|
  | total cell density (Nissl-corrected) | 239 MB | `https://openbluebrain.s3.us-west-2.amazonaws.com/Model_Data/Brain_atlas/Mouse/resolution_25_um/version_1.1.0/Cell_densities/overall_cell_density_correctednissl.nrrd` |
  | neuron_density.nrrd | 203 MB | `.../Cell_densities/cells/neuron_density.nrrd` |
  | glia_density.nrrd | 235 MB | `.../Cell_densities/cells/glia_density.nrrd` |
  | astrocyte_density.nrrd | 228 MB | `.../Cell_densities/cells/astrocyte_density.nrrd` |
  | oligodendrocyte_density.nrrd | 227 MB | `.../Cell_densities/cells/oligodendrocyte_density.nrrd` |
  | microglia_density.nrrd | 216 MB | `.../Cell_densities/cells/microglia_density.nrrd` |
  | (inhibitory subtypes) gad67+/pv+/sst+/vip+ _density.nrrd | ~190–200 MB each | `.../Cell_densities/inhibitory_neurons/*.nrrd` |
  (full prefix for the `cells/` rows:
  `https://openbluebrain.s3.us-west-2.amazonaws.com/Model_Data/Brain_atlas/Mouse/resolution_25_um/version_1.1.0/Cell_densities/cells/`)
- **Companion ontology (keys the voxels):**
  `.../version_1.1.0/Parcellation_ontology/mba_hierarchy.json` and the matching annotation volume
  `.../version_1.1.0/Annotation_volume/annotation_ccfv3_l23split_barrelsplit_validated.nrrd`.
- **Format (verified from the NRRD header):** NRRD0005, `type: double`, `encoding: gzip`, 3 dims,
  `space directions (25,25,25)` → **25 µm isotropic**, `sizes: 566 320 456`, `space origin
  (-350,0,0)`, little-endian, generated by pynrrd 2024-06-18. The 566 AP extent (vs the stock Allen
  CCFv3 528) indicates this is the **BBP-extended CCFv3a** template space (Piluso et al. 2024/2025;
  Zenodo 13640418 / 15176439), aligned to CCFv3.
- **Resolution:** 25 µm isotropic. (A 10 µm version of the *annotation*/Nissl exists in the Zenodo
  extended-atlas deposits; the density NRRDs on S3 are 25 µm.)
- **License:** **CC-BY-4.0** (stated on the AWS registry entry; produced by the BBP / Open Brain
  Institute atlas pipeline, https://github.com/BlueBrain/bbp-atlas-pipeline).
- **NOT downloaded here** (per the task — verdict only): each file is ~200–240 MB. These were not
  pulled into this directory; download on demand from the URLs above.

### Related (newer) resource, for awareness
A 2024/2026 successor atlas (Verasztó/Roussel et al., "A multimodal spatial atlas of transcriptomic,
morphological, and electrophysiological cell type densities," PLoS Comput Biol 10.1371/journal.pcbi.1014106;
data at github.com/BlueBrain/Molsys-transcriptomic-atlas, Zenodo 15176439, same S3 bucket) provides
finer transcriptomic-type densities at 25 µm. Not used here (this task targets the 4 broad classes).

---

## What is NOT in this dataset / gaps

- Erö is the sole per-region source for the astro/oligo/microglia split — there is no second
  independent per-region glia table in these sources to cross-check the glia split (Rodarie didn't
  republish it; Murakami only measured total nuclei). The glia split should be treated as model-derived,
  single-source. (The S3 NRRDs are the same Blue Brain lineage, not independent.)
- `*_sd` columns: Erö's Data Sheets do not ship per-region SDs for the broad cell classes, so there
  are no `neuron_sd`/`astrocyte_sd` etc. The only SDs available are Murakami's across-3-brains SD
  (`murakami_total_count_sd`) and Rodarie's per-inhibitory-subtype stds (in the raw `rodarie2022_s011.xlsx`,
  not merged into the broad-class columns).
- Murakami density is a derived count÷volume and only valid for leaf regions (see merge note 2).
- The table does not cover every fine Allen leaf structure (see coverage summary). Use the NRRDs for
  voxel-complete coverage.
