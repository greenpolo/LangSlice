---
name: choosing-engines
description: What each of LangSlice's fitting engines is (elastix_affine, fireants_syn, stalign_lddmm, spomialign, emlddmm_stack), what it takes and what it is built to handle, how to compare two fits, and how to fit sections that are partial or in pieces. Read it before the warp stage, or when a fit looks wrong.
---

# Choosing engines

Every fit starts from the section's current registration, so the placement
under it must be close first (`registration-playbook`). No engine has been
shown to be best on every kind of section. Know what each one does, pick the
one whose design fits the section, look at the result, and try another where
it falls short.

## The engines

| Tool | What it fits | How it matches the section to the atlas | Missing tissue |
|---|---|---|---|
| `elastix_affine` | the in-plane transform only | brightness, pixel by pixel ([elastix](https://elastix.dev)) | no term of its own: leave the lost regions out with `mark_damage` |
| `fireants_syn` | a warp | brightness, pixel by pixel; SyN, the deformable method of ANTs, on the graphics card (FireANTs, Jena et al. 2026, Nature Communications) | no term of its own: `mark_damage` |
| `stalign_lddmm` | a warp, or with `fit="both"` the transform and a warp | brightness, pixel by pixel; LDDMM (STalign, Clifton et al. 2023, Nature Communications) | an EM sorting of every pixel into tissue, background and artifact (`sorting`, `"em"` for sections with missing tissue) |
| `spomialign` | the transform, rigid or affine, a warp, or both in one call | matched points: a dense matcher trained on natural images (RoMa, on DINOv2 features) pairs distinctive points, and the fit follows the pairs (SPOmiAlign, Wang et al. 2026, Briefings in Bioinformatics) | tissue with no counterpart gives no confident pairs and does not pull the fit |
| `emlddmm_stack` | one smooth 3-D warp of the whole stack, and each section's transform | brightness; all sections at once (emlddmm, Tward et al. 2025, Nature Communications) | an EM sorting into tissue, background and artifact, so holes and debris are left out |

All of them leave out `mark_damage` regions; all but `stalign_lddmm` also
take a `mask` made with `make_mask` (`damage-and-masks`).

What the design means in practice:

- **Brightness engines** (elastix, FireANTs, STalign, emlddmm) work best when
  the section image looks like the atlas image: process the section so its
  contrast runs like the atlas's (`registration-playbook`).
- **SPOmiAlign** compares structure, not brightness, so it was built to align
  sections whose stain looks nothing like the atlas. It takes under a minute,
  and with `fit="both"` it fits the transform and the warp in one call from the
  placement. It draws the atlas as a brightfield Nissl, dark on white, and its
  matcher still sees polarity: on one DAPI brain, an inverted section image
  (dark tissue on light) brought it closer to the human registration than the
  image as delivered, while on ISH, DAB and Nissl, already dark on light,
  inverting made it worse.
- **Sections with lost or mangled tissue:** an engine with a missing-data
  term (STalign with `"em"`, emlddmm, SPOmiAlign's matching) can carry on where
  marking every lost region is impractical. elastix and FireANTs rely on the
  marks.

## Comparing two fits

On a hard section, fit a second engine and put both registrations side by
side with `look(registrations={"<section>": <n>, ...})`; keep the better with
`restore_registration`.

Judge the inner borders of distinct regions (the hippocampus, the anterior
commissure, the corpus callosum, the ventricle walls, cerebellar folia):
people judge a registration by those. The outer outline is the easy part, and
pulling it tight can leave the inner borders further from their tissue. But a
fit that pushes the outline off the tissue into the background is wrong too.

## SPOmiAlign on partial sections

`spomialign` can fit an affine and a warp at once (fit `"both"`). On a
partial section, or one in displaced pieces, that affine can shrink and shear
the atlas off the tissue. Example: `examples/spomialign_partial.jpg`.

- When that happens, a rigid transform first (`fit="transform"`,
  `rigid=true`), then `"warp"`, tends to hold.
- A rigid fit keeps the size of the placement it starts from, so the size
  needs to be right first; `elastix_affine` or `interactive_transform` can
  set it.

## emlddmm on a dense stack

`emlddmm_stack` fits one smooth 3-D warp to the whole stack at once, so
neighbouring sections inform each other. It suits dense, evenly spaced series
and gives every section a good start. It cannot bend a single section on its
own, and sections usually need local bending beyond it, so a per-section
warp after it (`fireants_syn`, `stalign_lddmm` or `spomialign` fit `"warp"`)
is the norm. Look at each section after it. Example: `examples/emlddmm_then_local.jpg`.
