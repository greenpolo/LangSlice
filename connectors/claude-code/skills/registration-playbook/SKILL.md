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

## The usual order of the stages

1. **Place.** Give every section its position along the slicing axis, the
   stack's cutting angles, and the right handedness (a section mounted
   mirrored is flipped before anything is fitted). See `placement`.
2. **Mark what is missing.** On a section that has lost a large part (most of
   a cortex, the frontal cortex over the bulbs, one hemisphere), mark the lost
   regions with `mark_damage` before the first fit. See `damage-and-masks`.
3. **In-plane affine.** Fit or set each section's in-plane transform. A warp
   only bends locally, so it needs the section roughly in place first. Some
   engines fit the transform and the warp in one call (`choosing-engines`).
4. **Warp.** Fit a deformation once the section's alignment is about right,
   and keep it where its overlay is better. See `choosing-engines`
   for which engine, and `landmarks` for pieces that moved and small
   structures a warp left off.
5. **Final review**, then `submit` (below).

A fit refines a placement that is already close. When a fit lands badly, the
cause is often the stage before it: a wrong position, wrong angles, a
mirrored section, or a large loss left unmarked. Fixing that stage usually
does more than retrying the fit with other settings.

## Look after every change

- After every fit or manual change, look at the section's overlay, and zoom
  where the borders matter. Keep the change only if the overlay is better.
- Judge the inner borders of distinct regions: the hippocampus and dentate
  gyrus, the anterior commissure, the corpus callosum and other fibre tracts,
  the ventricle walls, cerebellar folia. People judge a registration at a
  glance by those. The outer outline is the easy part: a fit can pull it onto
  the tissue edge while the inner borders move away from their tissue.
- To compare two registrations of a section, draw them side by side with
  `look(registrations={...})` and keep the better with `restore_registration`.

## Section and atlas images

Make the two images the fit compares look alike.

- **Atlas image:** pick the atlas image that looks most like the section:
  `nissl` (the fits' default) where the atlas has one. Against human
  registrations the `template` image made elastix fits worse for DAPI, DAB
  and Nissl sections, and failed more often from a rough start. Example:
  `examples/atlas_image_choice.jpg`.
- **Section image:** process the section with `make_image` so it looks like
  the atlas image the engine compares (one channel, contrast, polarity).
  elastix and FireANTs gave the same fit on a stain and on its inverse.
  SPOmiAlign draws the atlas dark on white, so fluorescence tends to match
  better inverted, and a brightfield stain (already dark on light) as
  delivered (`choosing-engines`).

## Sections an automatic fit cannot place

Some sections are so torn, folded or mangled that no set of damage marks
describes what is left, and an automatic affine may never land however its
settings change. Setting the transform by hand with `interactive_transform`,
or an engine with a missing-data term (`choosing-engines`), can get further.

## Final review

Before `submit`, draw every section's final overlay, look at them as a
stack, then zoom on the regions the study cares about. Fix anything that
looks off, then submit. A section's last change is only known to be good
once its overlay has been looked at.

## The other skills

| Skill | Read it when |
|---|---|
| `placement` | placing sections: positions, cutting angles, mirrored sections |
| `damage-and-masks` | a section has lost tissue, or has tears, holes or moved pieces |
| `landmarks` | a piece of tissue has moved off its place, or a small structure stays off its outline after a warp |
| `choosing-engines` | choosing or comparing warp engines, partial sections, whole-stack fits |
