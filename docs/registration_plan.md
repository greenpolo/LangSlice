# Registration pipeline

The supported design is [border refinement](nonlinear_design.md): correct the
borders of an atlas placement against the original histology, on exactly two
routes selected by whether a placement is supplied. Route "supplied" is the
production path: nonlinear correction needs a linear placement first, from the
linear agent or the host. Route "atlas" remains for experiments.

## Active files

- `core/nonlinear/image_gen_registration.py`: model-facing canvas geometry, the
  outlined-atlas template, and `generate_registration_candidate` (public
  dispatcher).
- `core/nonlinear/border_registration.py`: route selection, rough placement, the
  one/two atlas-route model calls, and transform composition.
- `core/nonlinear/border_refinement.py`: the shared correction request and line
  extraction both routes use.
- `core/nonlinear/border_fit.py`: the fit of the corrected lines, through the
  deformable package (`deformable/`).
- `core/nonlinear/prompts.py`: the three prompt texts (route "supplied", and route
  "atlas" pass 1 / pass 2).
- `core/handoff.py` (re-exported by `core/nonlinear/registration_handoff.py`): bridge from
  linear state without coupling the sibling methods.
- `core/nonlinear/registration_tool.py`: the linear agent's `trace_borders` tool, which
  runs route "supplied" on the handoff and keeps the first reply (no fit).
- ABBA: the Fiji connector's runs use the linear agent's own tools (an
  applied deformation lands in ABBA as a warp step on top of the affine,
  `core/abba_warp.py`); the abba-python plugin was removed 2026-10-04.
- `core/nonlinear/runtime.py`: orchestration and debug artifacts.
- `core/nonlinear/types.py`: result and annotation data classes.
- `providers/images.py`: image-generation transport adapters.

## Pipeline

Use a supplied placement when available (route "supplied"); otherwise fit a
local silhouette placement and let the model draw boundaries from nothing
against an outlined grayscale atlas template, with an optional second
corrective call (route "atlas"). Either way: send the (rough or drawn)
borders and the clean histology through the shared correction core, extract
corrected lines, optionally fit the residual deformation (the deformable
package, `deformation="deformable"`), and compose it with the complete
initial placement. The CLI default is no fit (`deformation="none"`, an
identity residual); see the [interface design](interface_design.md).

Markers are sampled from the composed mapping, not B-spline coefficients.
Review the raw correction, extracted borders on original tissue, and fitted
atlas separately. There is no hosted-router retry loop and no colormap
stage.
