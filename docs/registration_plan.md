# Registration pipeline

The supported design is [border refinement](nonlinear_design.md): correct the
borders of an atlas placement against the original histology, on exactly two
routes selected by whether a placement is supplied.

## Active files

- `nonlinear/image_gen_registration.py`: model-facing canvas geometry, the
  outlined-atlas template, and `generate_registration_candidate` (public
  dispatcher).
- `nonlinear/border_registration.py`: route selection, rough placement, the
  one/two atlas-route model calls, and transform composition.
- `nonlinear/border_refinement.py`: the shared correction request, line
  extraction and residual fit both routes use.
- `nonlinear/prompts.py`: the three prompt texts (route "supplied", and route
  "atlas" pass 1 / pass 2).
- `registration_handoff.py`: top-level bridge from linear state without
  coupling the sibling methods.
- `integrations/abba.py`: host-placed borders through the shared correction
  core (route "supplied" only).
- `nonlinear/runtime.py`: orchestration and debug artifacts.
- `nonlinear/types.py`: result and annotation data classes.
- `nonlinear/providers.py`: image-generation transport adapters.

## Pipeline

Use a supplied placement when available (route "supplied"); otherwise fit a
local silhouette placement and let the model draw boundaries from nothing
against an outlined grayscale atlas template, with an optional second
corrective call (route "atlas"). Either way: send the (rough or drawn)
borders and the clean histology through the shared correction core, extract
corrected lines, fit the residual deformation, and compose it with the
complete initial placement.

Markers are sampled from the composed mapping, not B-spline coefficients.
Review the raw correction, extracted borders on original tissue, and fitted
atlas separately. There is no hosted-router retry loop and no colormap
stage.
