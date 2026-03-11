# Registration Architecture Plan

## Goal

Define a registration runtime that is separate from AP estimation, uses one agent pass to propose anatomical correspondences, and uses deterministic code to turn that single landmark set into both QC-friendly affine outputs and a production nonlinear warp.

## Core Decision

AP estimation and registration are separate LangSlice features and separate runtimes. `langslice/vlm/estimator.py` stays scoped to AP estimation only. Registration uses one agent pass that emits paired anatomical correspondences, then one deterministic solver path that vets those correspondences, computes `AffineResult` from them for QC and backward-compatible export, and computes `NonlinearResult` from the same vetted landmarks via regularized TPS.

TPS already includes an affine component, so there is no separate affine-agent stage and no separate nonlinear-agent stage.

## Design Principles

1. Keep AP and registration decoupled. Registration consumes an AP choice, whether model-derived or manual.
2. Ask the agent for evidence, not transforms. The agent returns paired landmarks with labels and confidence, not affine or nonlinear parameters.
3. Use one landmark set end to end. Vet once, solve once, derive all downstream artifacts from the same vetted correspondences.
4. Treat affine as a deterministic byproduct. `AffineResult` remains useful for preview, QC, and backward-compatible export, but it is not a separate runtime.
5. Treat nonlinear export to ABBA as unproven. Compatibility is a hard feasibility gate, not an assumed deliverable.
6. Keep the registration runtime self-contained and free of deprecated external affine backends.

## Pipeline Overview

```text
1. AP runtime                  -> position_mm
2. Registration agent runtime  -> paired anatomical correspondences
3. Deterministic solver        -> vet and prune correspondences
                               -> compute AffineResult from same landmarks
                               -> compute regularized TPS NonlinearResult from same landmarks
4. Review and export           -> QC artifacts, debug artifacts, backward-compatible affine export
```

## Registration Runtime

Inputs:

- histology slice image
- atlas slice at the selected AP position
- atlas metadata needed to name structures consistently

Agent responsibility:

- produce one set of paired anatomical correspondences between slice and atlas
- attach structure labels so correspondences are inspectable by humans
- attach confidence or rationale fields that help deterministic pruning
- avoid inventing a staged affine result or a staged nonlinear result

Expected output shape:

- slice point `(x, y)`
- atlas point `(x, y)`
- anatomical label
- confidence tag

Target behavior:

- favor landmarks on structures that are visible, distinct, and spatially distributed
- skip damaged or ambiguous anatomy rather than forcing full coverage

## Deterministic Solver

The deterministic solver owns all geometry.

1. Vet and prune correspondences.
   - reject duplicate, clustered, or weakly distributed landmarks
   - drop low-confidence or high-residual outliers when enough stronger points remain
   - fail early when coverage is too sparse for a stable solve
2. Compute `AffineResult` from the vetted landmarks.
   - use the same landmarks that will feed TPS
   - keep this result for QC, debug, and backward-compatible export paths
   - expose residuals and parameter bounds for review
3. Compute `NonlinearResult` from the same vetted landmarks.
   - fit one regularized TPS
   - keep the solver regularized to reduce foldovers and overfitting
   - treat the TPS as the actual nonlinear registration result

`NonlinearResult` is new. It should store the vetted landmarks, TPS parameters, and QC metadata needed for replay, review, and export experiments.

## QC Gates

- Landmark count gate, fail when too few correspondences survive pruning.
- Landmark spread gate, fail when points do not cover enough of the tissue extent.
- Label consistency gate, fail or warn when correspondences imply obvious anatomical mismatches.
- Affine residual gate, prune or fail when landmark error is too high.
- Affine parameter sanity gate, warn or fail on implausible rotation, scale, shear, or translation.
- TPS validity gate, fail on foldovers or other invalid warp behavior.
- TPS smoothness gate, warn when the solution looks underconstrained or overfit.

## Debug/Review Artifacts

Save enough data to inspect every registration run and replay solver behavior:

- raw agent correspondences
- vetted and pruned correspondences
- overlay images with landmark labels
- affine residual tables and parameter summaries
- TPS QC metrics and any fold or smoothness warnings
- serialized inputs needed to reproduce a run later

When `LANGSLICE_VLM_DEBUG_DIR` is set, registration should emit the same class of review artifacts as AP estimation, extended for correspondence and warp review.

## Implementation Phases

### Phase 1, Registration Prototype

- [ ] Define registration-specific types, including new `NonlinearResult`
- [ ] Create a registration runtime separate from `langslice/vlm/estimator.py`
- [ ] Implement one agent pass that returns paired anatomical correspondences
- [ ] Implement deterministic vetting and pruning of correspondences
- [ ] Compute `AffineResult` from vetted landmarks
- [ ] Compute regularized TPS `NonlinearResult` from the same vetted landmarks

### Phase 2, QC and Review Tooling

- [ ] Add QC gates for spread, residuals, affine sanity, and TPS validity
- [ ] Save debug and review artifacts for raw and vetted correspondences
- [ ] Expose enough artifacts in the GUI or logs to compare affine and TPS outcomes from one run

### Phase 3, Runtime Validation

- [ ] Validate the new runtime on representative damaged and intact slices
- [ ] Compare solver stability across reruns and edge cases
- [ ] Confirm that affine outputs remain usable for the existing backward-compatible export path
- [ ] Compare behavior against archived pre-removal affine outputs where useful

### Phase 4, ABBA Nonlinear Export Feasibility Gate

- [ ] Research the exact ABBA or BigWarp nonlinear serialization contract
- [ ] Test whether `NonlinearResult` can be converted without loss or hidden assumptions
- [ ] Treat failed compatibility as a stop condition for nonlinear ABBA export claims
- [ ] Only proceed with nonlinear export work if feasibility is proven

### Phase 5, Cleanup

- [ ] Remove obsolete staged-affine or staged-nonlinear design assumptions from code and docs
- [ ] Remove any temporary prototype-only registration shims once the runtime stabilizes

## Key Architectural Boundaries

- `langslice/vlm/estimator.py` remains AP-only.
- Registration lives in its own runtime and should not be folded back into the AP estimator.
- The agent produces correspondences only.
- Deterministic code owns vetting, affine fitting, TPS fitting, QC, and debug artifacts.
- `AffineResult` stays as an existing result type for QC and backward-compatible export.
- `NonlinearResult` is the new nonlinear registration result type.
- Export logic must not assume nonlinear ABBA compatibility until Phase 4 proves it.

## Explicit Non-Assumptions

- Nonlinear ABBA export is not solved.
- A separate affine-agent stage is not part of the design.
- A separate nonlinear-agent stage is not part of the design.
- TPS does not need an upstream affine agent output to work.
- Registration does not redefine AP estimation ownership or scope.

## Open Questions

- What minimum landmark count and spatial spread produce stable TPS behavior on damaged tissue?
- What confidence or rationale fields from the agent are most useful for deterministic pruning?
- What regularization strategy gives the best tradeoff between local flexibility and warp stability?
- What exact nonlinear format, if any, will ABBA accept from LangSlice without manual intervention?
