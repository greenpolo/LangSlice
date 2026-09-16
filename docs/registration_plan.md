# Registration pipeline

The supported design is [border refinement](nonlinear_design.md): correct the
borders of an existing rough atlas placement against the original histology.

## Active files

- `nonlinear/image_gen_registration.py`: public dispatcher and retained color-map initialization.
- `nonlinear/border_registration.py`: one-call supplied-placement or two-call standalone orchestration and transform composition.
- `nonlinear/border_refinement.py`: shared correction prompt, extraction and residual fit.
- `registration_handoff.py`: top-level bridge from linear state without coupling the sibling methods.
- `integrations/abba.py`: host-placed borders through the shared correction core.
- `nonlinear/runtime.py`: orchestration and debug artifacts.
- `nonlinear/types.py`: result and annotation data classes.
- `nonlinear/providers.py`: image-generation transport adapters.

## Pipeline

Use supplied placement when available. Otherwise generate and register a color map
first. Send placed borders on histology and the clean histology to the correction
model, extract corrected lines, fit the residual deformation, and compose it with
the complete initial registration.

Markers are sampled from the composed mapping, not B-spline coefficients.
Review the raw correction, extracted borders on original tissue, and fitted atlas
separately. There is no hosted-router retry loop.
