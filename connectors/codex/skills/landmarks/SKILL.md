---
name: landmarks
description: How to use LangSlice's set_landmarks and landmark_triangulation for moving a detached or displaced piece of tissue back to its place with anchors, picking endpoints that are accepted, and fixing one small structure a warp left off its outline. Read it when a piece of tissue has moved off its place, or a structure stays off its outline after a warp.
---

# Landmarks

A landmark pair goes from a point on the section to the place where the same
point is drawn on the atlas. `landmark_triangulation` turns a section's pairs
into a mesh of triangles and moves each triangle so its pairs meet. The mesh
also holds four fixed corners just outside the image. An **anchor** is a pair
whose two endpoints are the same point: it holds that tissue where it is.

## A detached piece: landmarks first, then the warp

A dense warp (`fireants_syn` and the other engines) refines borders that are
already close. It does not carry a piece of tissue across a gap: left to
FireANTs alone, a piece lying off its place stays where the affine put it.
Example: `examples/fireants_alone.jpg`.

The recipe (example: `examples/detached_good.jpg`):

1. Put pairs on the piece: points on its outline and on internal landmarks
   you can see, each to the place where the atlas draws it.
2. Put anchors on the intact tissue around the piece, most of all along the
   gap it has to cross, and a few further out.
3. Apply with `landmark_triangulation` and look: the piece should sit in its
   outline and the rest should not have moved.
4. Then fit `fireants_syn` to refine the borders. It keeps the landmark move.

Without anchors, the move spreads over the intact tissue: the pairs pull the
mesh between them and the image corners, and the whole section is dragged
along. Example: `examples/detached_no_anchors.jpg`.

## Keep anchors off the path

Put anchors around the piece, never between the piece and its atlas place.
An anchor the piece has to pass folds the mesh: LangSlice outlines folded
triangles in magenta in the `landmark_triangulation` picture. If you see
magenta, move or delete the anchors in the way and apply again. Example:
`examples/anchors_on_path_fold.jpg`.

When the piece's atlas place lies beyond the edge of the image, the piece also
folds against the fixed corners, whatever the anchors. Leave such a piece
and record it with `note`.

## Picking endpoints

- Read endpoints off a close `zoom`, not the whole-section picture. The zoom's
  grid labels are in the whole picture's pixels, which is what
  `set_landmarks` takes.
- Put an outline pair's atlas endpoint on or just inside the drawn border. A
  point outside the atlas is refused, and the refusal offers the nearest point
  inside it. Example: `examples/picking_points.jpg`.

## One small structure off its outline after a warp

Rarely, after a good whole-section warp, a small structure that is easy to see
on its own still sits off its outline: the anterior commissure, the
fasciculus retroflexus, the mammillothalamic tract, the habenula. When a close
zoom shows it clearly off:

1. Pair its edges (all four sides) and its centre to the matching places on
   its drawn outline. Pairs on the two ends alone pull it the wrong shape.
2. Put a ring of anchors a short distance around it, where the warp is
   already right.
3. Apply `landmark_triangulation`. It is composed onto the warp; only the
   tissue inside the ring moves.

Example: `examples/region_fix.jpg`. Do not fit a new affine afterwards: a new
transform clears the warp and the landmark move with it.

Most sections never need this. Do it only where a zoom shows the structure off
its outline, not as a routine step.
