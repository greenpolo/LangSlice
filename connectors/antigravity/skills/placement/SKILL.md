---
name: placement
description: How to place LangSlice sections along the slicing axis, estimate the stack's cutting angles, and find sections mounted mirrored. Read it when placing sections or when a section's borders fit one hemisphere and miss the other.
---

# Placement

Placement is each section's position along the slicing axis, the stack's
cutting angles (pitch and yaw), and each section's handedness. Every later fit
builds on it: a fit refines a placement that is already close and does not
correct a wrong plane.

## Positions

- **Keep the order the filenames' section numbers give** unless the
  anatomy clearly says otherwise. Sections are usually numbered in cutting
  order; when a name holds several numbers (an image id, then `s0132`),
  the section number is the one that steps along the stack, and it can run
  backwards.
- **Fit the positions to the file numbers.** Once each section has a first
  position (from `deepslice_place` or by eye), list each position beside the
  section number in its filename. Real stacks fall on one line: its slope is
  the spacing, a doubled step marks a missing section, and a section far off
  the line is the one to look at again. Read positions off the line, and always
  look again at the two ends of the stack. Example:
  `examples/position_vs_index.jpg`.
  - Use a robust line (one that ignores the sections far off it), not a
    least-squares fit, which the outliers drag.
  - Use the line only if most sections sit within half an interval of it. If
    they do not, the filenames follow slides or carry another number (an image
    id, a date): find the real section number, or drop the line.
- **Check a doubtful position against fast-changing landmarks.** When a
  section may be a few hundred micrometres off, draw the borders of structures
  that change quickly along the axis at the candidate plane and at a few planes
  either side, and move the section only to a plane where a landmark visibly
  sits on its own tissue and sits off it at the written plane. Example:
  `examples/compare_sheet.jpg`. Landmarks by level:
  - forebrain: anterior commissure, lateral ventricles, corpus callosum,
    hippocampus
  - diencephalon: fasciculus retroflexus, habenula, mammillothalamic tract
  - midbrain: periaqueductal grey, interpeduncular nucleus, substantia nigra
  - hindbrain: inferior colliculus, fourth ventricle, cerebellar peduncles

  If no landmark shows in the stain, keep the position: a guess from a faint
  section moved good positions away in tests.

## DeepSlice

On whole coronal mouse sections, `deepslice_place` (when the job has it) can
give a quick estimate of the whole placement: positions, the stack's cutting
angles and a rough in-plane transform. Many people start a registration with
it; whether it helps here is yours to judge, since it reads each section on
its own, without the atlas, and can be noisy.

What it can be used for:

- **Spotting order problems.** Run without `order`, each section keeps the
  position DeepSlice predicts, so a section that falls out of the filename
  order stands out. With `order`, it keeps that order and spaces the sections
  evenly instead.
- **A first cutting angle.** It votes one pitch and yaw for the whole stack.
  It cannot see a mirrored section, but the reply lists each section's own
  angles, so one with the opposite yaw sign shows.
- **A first transform** on intact sections; it is poor on damaged ones.

Damaged or mangled sections, and those beyond the atlas's ends (the tip of
the bulbs, the caudal medulla), throw its estimates off; `bad_sections`
leaves them out of its vote and spacing (sections with marked damage are left
out by default). `write` chooses which parts land: an empty list only shows
the proposal, and writing just `angles` or `transform` keeps the positions.

## Cutting angles

A left-right difference that an in-plane fit cannot remove (the borders fit
one hemisphere and miss the other, at every in-plane setting) is a cutting
angle. Estimate it with `stalign_plane_search` rather than by eye:

1. Search 6 to 8 sections from the middle of the stack. Leave out the
   olfactory bulbs and the caudal medulla: their angles read wrong on their own.
2. Write the middle pitch and yaw with `position_sections`.
3. Search once more from the written angles. If the second pass agrees with
   the first to about a degree, keep it.

Then judge it:

- If the angles found are within about 3 degrees of flat and the sections fit
  as well at flat, keep flat.
- If the yaws found are small (under about 1.5 degrees) and of mixed sign,
  write yaw 0 rather than following the sign from pass to pass.
- With `emlddmm_plane`, repeat until two passes agree: one pass lands short.

Example: `examples/angle_before_after.jpg`.

## Mirrored sections

A section mounted mirrored fits at the opposite yaw sign: the plane search on
its own gives it the stack's yaw with the sign reversed. Example:
`examples/mirrored_section.jpg`.

- This tells you something only when most sections share one sign at a yaw of
  2 degrees or more. Then a section of the other sign is likely mirrored:
  flip it with `interactive_transform`, fit it again, and search it again. It
  should now read the stack's sign.
- When the yaws scatter around zero, the sign says nothing. Decide from
  asymmetric anatomy instead: a notch or a marker cut, an asymmetric tear, a
  structure present on one side only. Flipping "the smaller group" made
  near-symmetric stacks worse than flipping nothing in tests.

## Drawing the comparisons

- **Positions against file numbers:** list each section's written position
  beside the number in its filename, in filename order, and look for the line
  they share, the steps that are doubled, and the sections off it.
- **Landmarks at candidate planes:** `grep_atlas_view` draws the atlas at
  several positions with the landmark regions' borders highlighted. To see
  them on the section itself, write the candidate position with
  `position_sections`, look and zoom on the landmarks, and put the position
  back.
