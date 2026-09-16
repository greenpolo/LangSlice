# Nonlinear registration: correct placed borders

The default design starts from a rough atlas placement and asks the image model
to move its yellow region boundaries onto the visible tissue. A supplied linear
agent or host placement takes precedence; it is never replaced by a silhouette fit.

## Two entry routes

| Starting information | Image-generation calls | Sequence |
| --- | --- | --- |
| Linear-agent or host placement | 1 | Placed borders + clean histology → corrected borders → residual fit |
| Position only, standalone | 2 | Color atlas + grayscale atlas + histology → initial color registration → placed borders + clean histology → corrected borders → residual fit |

The standalone first call retains the existing color-map prompt and registration.
The second call is a separate edit request with two attachments, not a conversation
claiming to remember the first reply. Automatic tissue-outline placement is only
the explicit model-free `provider="none"` diagnostic, not the standalone default.

## What the correction model sees

1. The histology photograph with roughly placed, thin yellow atlas boundaries.
2. The identical photograph and frame without those lines.

The prompt asks the model to slide or bend the lines to match visible anatomy,
keep already-correct boundaries, and retain the photograph and boundary identities.
It does not ask for another color map. The atlas position, cutting angles,
orientation and supplied placement are established before this request.

The output photograph is not treated as anatomical ground truth: the raw reply
is saved, its yellow boundaries are extracted, and those boundaries are displayed
over the original input photograph. Elastix fits a residual deformation from
the rough borders to the corrected ones. Original atlas region IDs are warped
with nearest-neighbor sampling, not reconstructed from yellow lines.

## Supplying a placement

The Python candidate entry point accepts `initial_atlas_to_slice`, a finite,
invertible 3×3 affine mapping native atlas annotation pixel centers into pixels
of the supplied photograph. The atlas grid is sampled at the requested position
and cutting angles, then transformed by `image_axes` and the explicit
`atlas_mirror_lr` option. The matrix refers to that oriented grid.

`langslice.registration_handoff.prepare_linear_registration` prepares this
contract from an existing linear section state; `run_linear_registration`
performs the handoff. These are callable host interfaces, not an automatically
enabled tool in the linear agent's toolbox. They preserve the linear placement,
including shear, physical calibration and section orientation, without changing
the linear state. Their image frame is the oriented rendered section, not the
original acquisition TIFF.

ABBA already provides aligned atlas-coordinate channels. Its adapter draws those
placed labels directly on the section and uses the same one-call correction core.

Do not infer left–right reflection from a nearly symmetric tissue silhouette.
Mirroring must be explicit and consistent for labels, grayscale anatomy and
reference maps. A correction request cannot repair an incorrectly specified
atlas coordinate system reliably.

## Geometry and outputs

The final transform composes the initial placement with the residual fit.
For the two-call route, this includes the first call's full nonlinear coordinate
map, not just an affine approximation. Resize rounding, padding and pixel-center
offsets are retained in both axes.

Saved artifacts distinguish:

- the rough borders and the exact correction prompt;
- the untouched model reply;
- extracted model borders over the original photograph;
- the fitted atlas labels and their borders over that same photograph;
- residual deformation and composed slice-to-native-atlas correspondences.

Compatibility names such as `generated_segmentation.png` now contain the raw
border-correction reply on the default route. Read `output_kind` and `workflow`
metadata instead of assuming a color-map image. Canonical-frame VisuAlign markers
and direct native-atlas correspondences include explicit coordinate-frame metadata.
Inverse histology-to-atlas renders are not currently computed by the border route.

The file-based API may downsample the acquisition image before registration.
It converts supplied placements into that prepared frame; returned markers retain
the prepared-image coordinate convention. Session metadata `api_image_frame`
records both image sizes and the exact acquisition-to-prepared resize matrix.
Direct native-atlas coordinates are unaffected by that image resize.

## Review and limits

Inspect the raw reply first, then the extracted boundaries on original tissue,
then the fitted atlas overlay. A good model correction can be degraded by its
subsequent fit. Metrics complement these separate visual checks; neither fit
success nor agreement with a reference proves anatomical correctness.

Only one draw per stage is currently supported on the border route. The explicit
Python `registration_mode="colormap"` route retains older color-map experiments.
Offline correction replay via `generated_image` requires supplied placement;
without it, replay is rejected rather than silently making the initial model call.

The yellow extractor can confuse naturally saturated yellow tissue with drawn
lines. Boundary fitting is not region-identity-aware and does not certify topology.
For two-stage composition, sampling beyond the first-stage coordinate map uses
nearest-edge extension, recorded in metadata. These remain review considerations,
not claims that every visually excellent reply transfers perfectly to atlas labels.
