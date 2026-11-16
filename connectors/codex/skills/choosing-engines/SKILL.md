---
name: choosing-engines
description: How to choose and compare LangSlice's warp engines (fireants_syn, spomialign, stalign_lddmm, emlddmm_stack) and how to fit sections that are partial or in pieces. Read it before the warp stage, or when a warp looks wrong.
---

# Choosing engines

Every warp starts from the section's current registration, so the affine
under it must be right first (`registration-playbook`).

## No engine is best everywhere

Fitted from the same affine, the engines give different warps, and none came
closest to human registrations on every kind of data: SPOmiAlign more often
on DAB and ISH series, FireANTs and STalign on DAPI fluorescence, STalign on
Nissl. The differences between them are usually small. Start with one engine,
look, and compare with a second only where the first looks wrong.

Example: `examples/same_section.jpg`, one section through three engines from
one affine.

## Comparing two warps

On a hard section, fit a second engine and put both registrations side by
side with `look(registrations={"<section>": <n>, ...})`; keep the better with
`restore_registration`. Example: `examples/side_by_side.jpg`.

Judge the inner borders (ventricle walls, hippocampus, cortical layers,
folia), not how tightly the outer outline hugs the tissue. A warp that pulls
the outline tight onto the tissue edge can leave the inner borders further
from their tissue than a looser warp does.

## SPOmiAlign on partial sections

`spomialign` can fit an affine and a warp at once (fit `"both"`). On a
partial section, or one in displaced pieces, that affine can shrink and shear
the atlas off the tissue. Example: `examples/spomialign_partial.jpg`.

- Fit the transform rigid first (`fit="transform"`, `rigid=true`), check the
  overlay, then fit `"warp"`.
- A rigid fit keeps the size of the placement it starts from. Check the size
  against an intact edge first; if it is off, set the size with
  `elastix_affine` or `interactive_transform`, then fit rigid or warp.

## emlddmm on a dense stack

`emlddmm_stack` fits one smooth 3-D warp to the whole stack at once, so
neighbouring sections inform each other. It suits dense, evenly spaced series.
It cannot bend a single section on its own: after it, look at every section,
and fit a per-section warp (`fireants_syn` or `spomialign` fit `"warp"`) only
where the overlay still shows a local mismatch, often at the ends of the
stack. Keep that warp only if the overlay improves; on most sections emlddmm
alone is as good. Example: `examples/emlddmm_then_local.jpg`.
