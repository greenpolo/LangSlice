---
name: damage-and-masks
description: How and when to mark damaged LangSlice sections with mark_damage, what a mark does inside a fit, and what is not damage (moved pieces, tears, holes). Read it when a section has lost tissue, or has tears, holes or pieces that moved.
---

# Damage and masks

## What a mark does

`mark_damage` names atlas regions the section has lost. Every fit then paints
those regions to background in the atlas image it compares, and leaves them,
plus a 0.1 mm margin, out of the comparison. The tissue that survives drives
the fit. Example: `examples/paint_and_mask.jpg` shows the atlas as the fit
sees it before and after a mark, and the two fits that follow.

## When to mark

- **Mark a large loss before the first fit.** When a section has lost a large
  part (most of a cortex, the frontal cortex over the olfactory bulbs, one
  hemisphere), mark it before `elastix_affine`. Unmarked, the affine shrinks
  the atlas onto what is left and pulls the surviving tissue into the gap.
  Marking afterwards does not undo a bad affine: if you marked late, put the
  placement from before that affine back (`restore_registration`, or set it
  by hand), then fit again.
- **Small losses need no mark.** A frayed edge or a small nick does not move a
  fit.
- **Mark only what is missing.** Marking a region that is mostly intact takes
  its tissue out of the fit and makes the affine worse.
- **A coarse mark is fine.** When most of a large region is gone, mark the
  whole region on that side (for example `Isocortex:right`) rather than
  looking up the subregions that are lost. The part of it that survives still
  lands in place. Example: `examples/coarse_mark.jpg`.
- **A far start is not fixed by a mark.** If the section starts far off its
  place (a large turn, or a shift of half a millimetre or more), set it close
  by hand with `interactive_transform` first; a fit refines a placement that
  is already close.

## Exclude what is lost; fit everything else

Mark the lost regions and fit the whole section. Do not try to fit only a few
surviving regions with `restrict_to` instead: those regions fit, and the rest
of the section stays where the rough start put it. Example:
`examples/exclude_vs_include.jpg`.

## Masks

A mark names atlas regions; a mask (`make_mask`) names section pixels.
`elastix_affine` and `fireants_syn` leave its pixels under one half out of
the comparison, `emlddmm_stack` weights each pixel by it, and `spomialign`
drops the matches that end where it is under one half. `stalign_lddmm` takes
no mask. STalign (`sorting` `"em"`) and emlddmm also sort
every pixel into tissue, background and artifact on their own, so holes and
debris drop out without a mask. In tests a mask over a tear or hole did not
help a FireANTs warp (below).

## A section too mangled to mark

When so much is torn or folded that no set of marks describes what is left,
an automatic affine may never land. Set the transform by hand with
`interactive_transform`, or fit with an engine that sorts out missing tissue
itself (`choosing-engines`).

## What is not damage

Example of all three cases: `examples/not_damage.jpg`.

- **Missing tissue:** mark it.
- **A piece that moved but is intact** (a flap of cortex folded out, a
  cerebellum lying beside the brainstem): do not mark it. Marking it throws away
  tissue that can still be placed. Move it back with landmarks (skill
  `landmarks`), or leave it to the warp when it lies close to its place.
- **A tear or a hole inside intact tissue:** leave it unmarked. A
  whole-section warp crosses a tear without bending into it. Around a hole it
  can pull the rim in slightly; neither a mark nor a `make_mask` mask fixed
  that in tests, and both made the fit worse.

## Checking

After every fit on a damaged section, look at the overlay and judge the
surviving inner borders (ventricles, hippocampus, fibre tracts), not the
outline over the missing part. The outline of a marked region is drawn where
the atlas has it, over empty slide; that is expected.
