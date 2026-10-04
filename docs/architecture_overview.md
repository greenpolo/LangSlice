# Architecture Overview

LangSlice is one installable Python package, `langslice`. Local-model training
lives in the sibling `LangSlice-Training` repository and the position benchmark
in `SliceBench`; both depend on this package.

## Package Layout

The package is laid out by layer (folder move, 2026-10-04). Each layer
imports only the layers below it; import-linter enforces it
(`[tool.importlinter]` in `pyproject.toml`, run by
`tests/test_import_layers.py`), and every `ignore_imports` entry there is a
known violation to be removed by moving code.

| Layer | Package | May import |
|---|---|---|
| hosts | `src/langslice/hosts/` | everything below |
| doors : agent | `src/langslice/doors/`, `src/langslice/agent/` | each other, ops, job, core, providers |
| ops | `src/langslice/ops/` | job, core |
| job | `src/langslice/job/` | core |
| core | `src/langslice/core/` | core only |
| providers (beside) | `src/langslice/providers/` | core types; never job, ops, doors, agent, hosts |

Core, job and ops never import `providers`, `google.*`, `litellm`, `openai`
or `mcp`. The two registration methods are task groups over this one
layout, not packages: the linear agent environment and the image-model
border route (`core/nonlinear/`).

- `src/langslice/core/` -- the core library: the stack state
  (`state.py`), the job spec (`spec.py`), the workspace (`workspace.py`:
  the atlas, the section files, render caches), the renders, captions,
  canvas, sheets and pictures the tools send with their layers
  (`coordinate_map`), the in-plane fits (`transform.py`), the deformable
  machinery (`deformation.py`), the opening, display and appearance options,
  and `handoff.py` (a linear section state as the calibrated placement the
  nonlinear border correction and the deformable fit take, and its
  fingerprint). Shared foundations:
  - `core/atlas/` -- BrainGlobe atlas loading, slice extraction, and colored
    region maps.
  - `core/space.py` -- coordinate and orientation conventions.
  - `core/affine.py` -- the shared in-plane affine core: the silhouette
    (image-moments) fit of a section onto an atlas section, the
    rotation/scale/translate matrix builder, the normalized six-number
    parameter convention, and `pixel_center_map`. `silhouette_affine` is the
    one silhouette wrapper the linear transform tools and `quick_affine`
    share.
  - `core/oblique.py` -- arbitrary-plane (cutting-angle) sampling of an
    atlas volume and the (pitch, yaw) fitter.
  - `core/deformable/` -- the library deformable fit (ANTs SyN or Elastix
    B-spline) of a placed atlas plane: the linear agent's `fit_deformable`
    and the fit of the image model's lines.
  - `core/image_prep.py` -- image normalization, metadata detection,
    downsampling, and the host multichannel blend (`host_preprocess`).
  - `core/nonlinear/` -- generative-image registration: exactly two
    border-based routes (placed-border correction when a placement is
    supplied; otherwise a silhouette-placed, model-drawn-boundary route
    against an outlined grayscale atlas template), the fit of the model's
    lines through `core/deformable/` (`border_fit.py`), affine/nonlinear
    result types, the silhouette-based `quick_affine` preview,
    `registration_tool.py` (the linear agent's optional image-model
    border-correction tool; the image model is an argument) and
    `registration_handoff.py` (route "supplied" for one section with the
    image model passed in).
- `src/langslice/job/` -- the job: one `Job` (`job.py`) owning the state,
  undo, the checkpoint (`checkpoint.py`) and the submit gates; the job
  folder's layout, history, index, migrations, saved views, the public files
  (`formats.py`) and the QUINT/QuickNII/VisuAlign export (`quint.py`).
- `src/langslice/ops/` -- the verbs: every write and every viewing read as a
  function on the job; `registry.py` lists them for every door.
- `src/langslice/doors/` -- the doors over the verbs: one declaration per
  verb (`declarations.py`) that the agent tools, the MCP tools, the agent
  CLI and the library are built from; `tools/` (the native agent tools:
  toolbox, argument shapes, `view` options, ADK message packaging), `mcp/`
  (the MCP server), `cli/` (every `langslice` command, one module per
  group, including the agent CLI `langslice job FOLDER VERB`, `ops`,
  `schema`; `docs/agent_cli.md`), `library.py` (`langslice.open_job`),
  `card.py` (the job folder's `AGENTS.md` / `CLAUDE.md`).
- `src/langslice/agent/` -- the ADK driver of the linear agent environment:
  the run engine, the ADK session, plugins and model resolution, the job
  statement, the trace, the cost estimate, live events.
- `src/langslice/providers/` -- model access: Gemini API keys, OpenAI API
  keys, and ChatGPT subscription sign-in (`openai-oauth`), named in
  `registry.py`, where `resolve_image_model` turns a provider name into the
  image-edit call a door hands to the operations; `images.py`, the image
  transport.
- `src/langslice/hosts/` -- host connectors in LangSlice's own environment:
  `integrations/` (the abba-python registration plugin and the live linear
  mirror) and `api/` (the Pydantic engine contract, runtime wrappers, and the
  stdio service used by non-Python clients).
- `src/langslice/cli.py` -- the `langslice` command's entry point
  (`langslice.cli:main`).
- `src/langslice/linear/`, `atlas/`, `nonlinear/`, `integrations/`, `adk/`
  and a few top-level modules (`space.py`, `oblique.py`, `affine.py`,
  `image_prep.py`, `registration_handoff.py`) are compatibility shims for
  the sibling repos; LangSlice imports none of them.

## Where The Methods Fit

`linear` and `nonlinear` can be used separately. `nonlinear` takes a slice
position as input and does not care where that position came from, so it can
follow `langslice linear run` or a placement made in another tool. In
QUINT/ABBA-style workflows, linear placement happens first in the host tool and
LangSlice-nonlinear stands in for the manual spline/BigWarp deformation step.

## Engine Contract

The Python package is the source of truth for LangSlice runtime behavior. The
engine contract is defined with Pydantic models in `src/langslice/hosts/api/models.py`.

`langslice serve --stdio` runs the newline-delimited JSON engine service. It
accepts `version`, `register.run`, `quick_affine.run`, and `export.run`, plus
`setup.status`, `setup.login`, `setup.api_key`, `linear.run`, `linear.estimate`,
`preprocess.preview` and `nonlinear.abba` for the independent Fiji connector. It emits progress/log/data event envelopes
and returns either result or error envelopes. The connector starts a worker in
the selected Python environment; it does not require `abba_python`. See
[the connector design](abba_plugin_design.md) and
[installation instructions](abba_installation.md).

## Linear: Order, Position, Transform

`langslice linear run` is ONE agent environment over a folder of sections: one
`StackState`, one toolbox, one job statement, and one ADK session that ends at
`submit` or the turn budget. A host fills a `JobSpec` -- which of `reorder`,
`position`, `transform` and the optional `nonlinear` are on, plus the knobs
each one exposes -- and the toolbox is built from it: a task that is off
contributes no tools, and its answer comes from `spec.inputs` instead. Hosts
present these tasks to users as Positioning (`reorder` + `position`), Linear
(`transform`) and Nonlinear; see [the interface design](interface_design.md).

Every write tool checkpoints the whole state, so a run that dies resumes with
the state it had (the agent is re-seeded, not replayed), and every write is
undoable (`undo`/`redo`, one tool call = one step; the history is saved beside
the checkpoint and survives a resume). One job object (`job/job.py`) owns
the state, undo, checkpoint and submit gates; the agent's tools and the MCP
server both sit on it, and it picks up a state file a script changed on disk. The results file
uses the checkpoint's schema, so the CLI, the checkpoint and any host adapter
read one shape. Corrections (order, flips, rotations), positions, cutting
angles and per-section transforms are proposals -- the host applies them, and
the user's image files are never modified.

Interactive alignment runs in the main session, sharing the context that placed
and ordered the stack. `adjust_transforms` writes and shows one to four sections
in one undo step, including before/after or side-by-side views. A dependent
refinement waits for its first picture. The linear agent handles affine
alignment. With the optional `nonlinear` task it also gets
`trace_borders`, which sends one section's placed atlas borders to the
image model for correction and keeps the first reply; it fits no deformation
and changes no transform. See [the image-tool contract](nonlinear_image_tool.md).

Picture-returning writes show what they did: `orient_slices` the re-oriented
sections, `set_positions` placements not already seen at the same position,
orientation and cutting angles in a full-canvas atlas-bearing view, and the
transform tools their overlays. A section-only or zoomed comparison does not
suppress the full placement. A compare and position write planned in the same
model round both return their images because neither sibling result was
visible when planned. There is no
preview-then-apply anywhere; every write is undoable instead. Transform
results send the useful physical parameters once; normalized matrices,
derived representations and adjustment history stay local.

## Nonlinear: Border Refinement

1. Prefer the atlas placement supplied by the linear agent or host tool.
   Without placement, fit a local silhouette placement instead.
2. Send rough yellow atlas borders over histology plus the identical clean
   histology to the image model (route "supplied"); or, with no placement,
   send the clean histology plus an outlined grayscale atlas template and
   ask the model to draw the boundaries from nothing (route "atlas", with an
   optional second corrective call).
3. Extract corrected yellow lines and overlay them on the original photograph.
4. Optionally fit the residual deformation to the corrected lines with the
   deformable package (`deformation="deformable"`: lines against the atlas
   family borders, Elastix B-spline) and warp the placed atlas labels. The CLI
   default is `deformation="none"`: the residual is identity.
5. Compose the initial affine placement with that residual for native-atlas
   correspondences and VisuAlign markers.
6. Return separate raw, corrected-border and fitted-atlas review artifacts.

Each route uses one image-generation call, except route "atlas" with an
optional second audit call. Route "supplied" is the production path: nonlinear
correction needs a linear placement first. Route "atlas" remains for
experiments. ABBA shares the border-correction core and
retains its host placement. See
[the design](nonlinear_design.md).

## Debugging

Set `LANGSLICE_VLM_DEBUG_DIR` for run artifacts. Set
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` for redacted ADK request captures. Set
`LANGSLICE_TRACE_DIR` (or `langslice linear run --trace-dir`, or the worker's
`trace_dir`) for a full-content trace of each linear agent session.
