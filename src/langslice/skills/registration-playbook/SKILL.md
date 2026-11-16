---
name: registration-playbook
description: How an expert works a LangSlice registration from start to submit. The order of the stages, what to check after each, which section and atlas images to fit, and which of the other LangSlice skills to read when a section needs it. Read it before the first tool call of a registration.
---

# Registration playbook

This is how to work one brain. The job statement in your first message says
what to do and which tools this job has; this skill says how an expert goes
about it. The other skills go deeper on one topic each (list at the end).
Open the example pictures named here with your file-reading tool: each shows
what the advice looks like on a real or staged section.

## The order of the stages

1. **Place.** Give every section its position along the slicing axis, the
   stack's cutting angles, and the right handedness (a section mounted
   mirrored is flipped before anything is fitted). See `placement`.
2. **Mark what is missing.** On a section that has lost a large part (most of
   a cortex, the frontal cortex over the bulbs, one hemisphere), mark the lost
   regions with `mark_damage` before the first fit. See `damage-and-masks`.
3. **In-plane affine.** Fit or set each section's in-plane transform. An
   affine always comes before a warp, damaged sections included.
4. **Warp.** Fit a deformation only on a section whose affine is right. Keep it
   only where its overlay is better than the affine's. See `choosing-engines`
   for which engine, and `landmarks` for pieces that moved and small
   structures a warp left off.
5. **Final review**, then `submit` (below).

A fit refines a placement that is already close. When a fit lands badly, the
cause is usually the stage before it: a wrong position, wrong angles, a
mirrored section, or a large loss left unmarked. Go back and fix that stage
rather than retrying the fit with other settings.

## Look after every change

- After every fit or manual change, look at the section's overlay, and zoom
  where the borders matter. Keep the change only if the overlay is better.
- Judge the inner borders: ventricle walls, the hippocampus and dentate gyrus,
  cortical layers, fibre tracts, cerebellar folia. How tightly the outer
  outline hugs the tissue edge is a poor guide: a fit can pull the outline onto
  the edge while the inner borders move away from their tissue.
- To compare two registrations of a section, draw them side by side with
  `look(registrations={...})` and keep the better with `restore_registration`.

## Section and atlas images

- **Atlas image:** fit against `nissl` (the fits' default) for every stain.
  Against human registrations the `template` image made affine fits worse for
  DAPI, DAB and Nissl sections, and failed more often from a rough start.
- **Inverting:** a brightfield stain (dark tissue on a light slide) runs
  opposite to the atlas pictures. Invert it with `make_image` to look at it
  if that helps you read it, but fit on the image as delivered. elastix and
  FireANTs give the same fit either way, and SPOmiAlign's warp ended further
  from human registrations on inverted ISH, DAB and Nissl.
  Example: `examples/invert.jpg`.
- Example of choosing the atlas image by what the section shows:
  `examples/atlas_image_choice.jpg`.

## Final review

Before `submit`, draw every section's final overlay, look at them as a
stack, then zoom on the regions the study cares about. Fix anything that
looks off, then submit. A section you have not looked at since its last
change is not finished.

## The other skills

| Skill | Read it when |
|---|---|
| `placement` | placing sections: positions, cutting angles, mirrored sections |
| `damage-and-masks` | a section has lost tissue, or has tears, holes or moved pieces |
| `landmarks` | a piece of tissue has moved off its place, or a small structure stays off its outline after a warp |
| `choosing-engines` | choosing or comparing warp engines, partial sections, whole-stack fits |
