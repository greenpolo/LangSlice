# SliceBench coronal — data licenses

Verified 2026-05-17 against the authoritative source for each dataset
(Allen Terms of Use page, EBRAINS Knowledge Graph license_info field,
Figshare record, Zenodo record). This file covers every coronal brain
across the `tiny`, `small`, and `large` bench tiers in `data/slicebench.json`.

## Summary table

| Dataset (`dataset` key) | Brains in tiny/small/large | Source | License | Commercial use? | License files |
|---|---|---|---|---|---|
| `allen_connectivity` | 1 (large) | Allen Mouse Brain Connectivity Atlas | Allen Institute Terms of Use | **No — noncommercial only** | `data/datasets/coronal/allen_connectivity/{LICENSE,NOTICE}` |
| `allen_dev_coronal` | 1 (large) | Allen Developing Mouse Brain Atlas (ADMBA) | Allen Institute Terms of Use | **No — noncommercial only** | `data/datasets/coronal/allen_dev_coronal/{LICENSE,NOTICE}` |
| `deepslice_gt` | 2 (m287 in all three tiers; ISH_Calb1 in large) | Figshare 22802411 | CC BY 4.0 (nested Allen NC inside ISH_Calb1) | Yes for non-Allen subjects; NO for ISH_Calb1 | `data/datasets/coronal/deepslice_gt/{LICENSE,NOTICE}` |
| `m01_m03` | 3 (M01 in all three tiers; M02, M03 in large) | Walsh lab, internal/unpublished | **UNPUBLISHED — not redistributable** | **Must be swapped out before public release** — see `project_slicebench_publish_m01_swap` memory | (no LICENSE — will be removed) |
| `m04_m09` | 1 (M04 in large) | Walsh lab, internal/unpublished | **UNPUBLISHED — not redistributable** | Same — must be swapped out | (no LICENSE — will be removed) |
| `rat_tract_eval` | 3 (F1/PHAL in all three tiers; F1/BDA + F10/BDA in large) | EBRAINS Kondo et al. (DOI 10.25493/2MX9-3XF) | CC BY 4.0 | Yes (with attribution) | `data/datasets/coronal/rat_tract_eval/{LICENSE,NOTICE}` |
| `zenodo_pnnpv` | 3 (CC2B in all three tiers; AL1A, CC1A in large) | Zenodo 7419283 (Boutte / Lasztoczi) | CC BY 4.0 | Yes (with attribution) | `data/datasets/coronal/zenodo_pnnpv/{LICENSE,NOTICE}` |

License conventions used:
- **CC BY 4.0** — full shared text in `slicebench/licenses/CC-BY-4.0.txt`, each per-dataset LICENSE points at it
- **Allen Institute ToU** — relevant permission-grant excerpt embedded in each Allen dataset's LICENSE; canonical at https://alleninstitute.org/terms-of-use/
- **SliceBench code** — MIT, at `slicebench/LICENSE`
- **Master attribution index** — `slicebench/NOTICE`

## Verification per dataset

### `allen_connectivity` — Allen Mouse Brain Connectivity Atlas
- **License**: Allen Institute Terms of Use, noncommercial only with citation.
- **Source**: https://alleninstitute.org/terms-of-use/ and
  https://alleninstitute.org/citation-policy/
- **Quote** (from Allen Citation Policy): "To use our resources in your
  noncommercial research or for noncommercial purposes, you must cite them in
  accordance with this policy."
- **Citation format** (per Allen guidance): "© [year] Allen Institute for
  Brain Science. Allen Mouse Brain Connectivity Atlas. Available from
  connectivity.brain-map.org."
- **Commercial implications**: SliceBench cannot be redistributed for
  commercial purposes while this dataset is included. If a commercial-use
  version is needed later, remove Allen-sourced brains.

### `allen_dev_coronal` — Allen Developing Mouse Brain Atlas (ADMBA)
- **License**: Same Allen Institute Terms of Use as above.
- **Source**: https://alleninstitute.org/terms-of-use/ +
  https://developingmouse.brain-map.org/
- **Citation format**: "© [year] Allen Institute for Brain Science. Allen
  Developing Mouse Brain Atlas. Available from developingmouse.brain-map.org."

### `deepslice_gt` — DeepSlice ground-truth manuscript-companion dataset
- **License**: **CC BY 4.0**.
- **Source**: Figshare record 22802411
  (https://figshare.com/articles/dataset/22802411).
- **Quote** (CC BY 4.0): "You are free to: Share — copy and redistribute the
  material in any medium or format. Adapt — remix, transform, and build upon
  the material for any purpose, even commercially. The licensor cannot
  revoke these freedoms as long as you follow the license terms" (full text:
  https://creativecommons.org/licenses/by/4.0/legalcode).
- **Citation format**: Carey, H., Pegios, M., Martin, L. *et al.* DeepSlice:
  rapid fully automatic registration of mouse brain imaging to a volumetric
  atlas. *Nat Commun* **14**, 5884 (2023). Plus the figshare DOI for the GT
  bundle specifically.

### `m01_m03` and `m04_m09` — Walsh lab internal slices (M01–M09)
- **License**: NONE — unpublished primary data, not licensed for redistribution.
- **Action required before public release**: SWAP OUT these brains for
  equivalent public-pool brains. See `project_slicebench_publish_m01_swap`
  memory for the action plan (replace with public-pool equivalents matching
  the mixed-staining mouse-coronal profile, then sanity-check the comparison
  ranking via mask-out re-score before publish).
- **Until swapped**: do not commit these images to a public repo; do not
  publish bench results that name M01–M09 brains without disclosing they are
  internal.

### `rat_tract_eval` — EBRAINS Kondo orbitofrontal anterograde tracing
- **License**: **CC BY 4.0**, free access.
- **Source verification**: EBRAINS Knowledge Graph metadata field
  `license_info.value = "Creative Commons Attribution 4.0 International"`
  for KG instance cc17c126-dec2-486c-a23d-deedf9102269 (DOI
  10.25493/2MX9-3XF).
- **Covers** all three subjects we use: F1/PHAL, F1/BDA, F10/BDA. Image
  paths embed the DOI (`10_25493_2MX9-3XF`) for traceability.
- **Citation format**: Kondo, H. *et al.* Anterogradely labeled axonal
  projections from the orbitofrontal cortex in rat. EBRAINS (2022).
  https://doi.org/10.25493/2MX9-3XF. Plus the Nature Scientific Data
  descriptor: doi:10.1038/s41597-023-02527-y.

### `zenodo_pnnpv` — Boutte / Lasztoczi perineuronal-net + parvalbumin
- **License**: **CC BY 4.0**.
- **Source**: Zenodo record 7419283
  (https://zenodo.org/records/7419283).
- **Covers** all three subjects we use: CC2B, AL1A, CC1A.
- **Citation format**: per the Zenodo record's "Cite as" block.

## Net commercial-use status of SliceBench

While the bench includes Allen-sourced brains, the whole bench inherits
the most-restrictive license (noncommercial). To publish a commercial-OK
benchmark, the only public-pool replacements that would work are
DeepSlice GT, EBRAINS Kondo, and Zenodo PNN/PV — all CC BY 4.0.

Recommended publish-time stance for v1.0:
1. Keep Allen brains (high-quality registration, broad species coverage).
2. License the SliceBench *repo* (code, manifests, scoring) under CC BY 4.0
   or MIT, but make clear in the README that data redistribution inherits
   each source's license.
3. Swap M01–M09 for public-pool brains per
   `project_slicebench_publish_m01_swap`.
4. Disclose noncommercial-only status driven by Allen content.
