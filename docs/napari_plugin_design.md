# napari connectors and the LangSlice worker

Proposed design (2026-09-16), modelled on the accepted Fiji connector in
`abba_plugin_design.md`: LangSlice stays in its own user-managed environment,
and each napari host gets a thin connector plugin that talks to the LangSlice
worker over the existing JSON-lines protocol. No napari host ever imports
`langslice` in-process.

## Why out-of-process, and why one folder per host

napari plugins conflict with each other at the dependency level, not just the
UI level. Measured on 2026-09-16:

| Host | Latest | Pins that bite |
| --- | --- | --- |
| brainglobe-registration | 0.0.6 (2026-02-12) | Python >= 3.11, `napari>=0.4.18,!=0.6.0`, `itk-elastix>=0.24` |
| brainways | 0.1.16.3 (2025-08-28) | `napari[all]>=0.5`, `numpy<2`, `tensorflow<2.20`, `jpype1==1.5.0`, `albumentations==2.0.5`, `aicsimageio==4.14.0` |
| napari itself | 0.9.1 (2026-09) | PyQt6 default; PyQt5 dropped Q4 2026 |

LangSlice's own stack (ADK, Gemini and OpenAI SDKs, itk-elastix, brainglobe
tooling) cannot be pip-resolved into a brainways environment, and the
installation matrix would be unmaintainable even where it could. The Fiji
connector already solved this: the host holds only a connector, the worker
holds LangSlice, and a JSON-lines pipe joins them. The napari connectors reuse
that worker, protocol and setup flow unchanged.

One connector per host, each an independent distribution, so a user installs
exactly the connector for the napari environment they already have. A shared
connector core carries what every host needs and depends on nothing but
`qtpy` and `napari`.

## Layout

```
connectors/napari/
  README.md                       one-paragraph map, install matrix
  langslice-napari-core/          shared, host-agnostic (qtpy + napari only)
    langslice_napari_core/
      worker.py                   Python port of connectors/fiji WorkerClient
      environments.py             Python port of EnvironmentDiscovery
      setup_widget.py             Setup dock: env picker, check, login, API key
      activity.py                 progress/transcript dock with cancel
  brainglobe-registration/        napari-langslice-brainglobe-registration
  brainways/                      napari-langslice-brainways
  <next host>/
```

Each host folder is a complete plugin: `pyproject.toml`, `napari.yaml`
(npe2 manifest, `napari.manifest` entry point), a `widgets.py` that
contributes the LangSlice dock, an `adapter.py` that translates between the
host's data model and worker requests, and tests that run against a fake
worker and a fake host. Hosts are never a dependency of the core.

`brainreg` is not a target: it is a whole-brain 3D pipeline with no
per-section entry point (brainreg-napari merged into it at 1.0.0). Its
`brainreg.json` sidecar and `registered_*.tiff` naming are the ecosystem's de
facto interchange convention, and brainglobe-registration is converging on
them, so the brainglobe-registration connector writes results in that shape.

## What each connector does

Both connectors follow the Fiji connector's Registration dialog, which
implements [the interface design](interface_design.md): one dock with the
Positioning, Linear and Nonlinear tasks, per-slice damage marks, channel
preprocessing with a preview, and an estimated cost. Setup and accounts live
in the shared core dock. (This proposal predates that dialog; its original two
actions, a linear agent and nonlinear boundary refinement, map onto those tasks.)

Worker methods used, all already in protocol version 1:

| Method | Used for |
| --- | --- |
| `setup.status` / `setup.login` / `setup.api_key` | shared setup dock |
| `linear.run` | positions and in-plane affines from calibrated section snapshots |
| `linear.estimate` / `preprocess.preview` | the dock's cost line and preprocessing preview |
| `register.run` | one section at a supplied position, angles and `initial_atlas_to_slice` |
| `export.run` | QUINT JSON when a host has no native result store |

`nonlinear.abba` stays ABBA-only: it needs per-pixel atlas coordinate channels
that only ABBA's resliced atlas provides. napari hosts carry a position plus
angles plus an in-plane affine, which is exactly the `register.run` contract.

### brainglobe-registration

Host model: one sample layer, one atlas layer, atlas `atlas_pitch` /
`atlas_yaw` / `atlas_roll`, an AP slice chosen on the viewer slider, and an
Elastix parameter set. Results saved brainreg-style.

- Linear: snapshot the sample layer with its pixel size, run `linear.run`,
  push the returned position onto the slider and the in-plane affine onto the
  sample layer's `affine`. Cutting angles map to the pitch/yaw spin boxes once
  the linear engine's angle updates are verified for hosts.
- Nonlinear: `register.run` with the host's current position, angles and
  layer affine; write `registered_atlas.tiff`, `registered_hemispheres.tiff`
  and a `langslice.json` parameter record next to them, and add the warped
  annotation as a labels layer.
- Open: brainglobe-registration has no confirmed headless API; the adapter
  drives its widget state. A source audit of its widget module tree is the
  first implementation task.

### brainways

Host model: a project of subjects, each a list of `SliceInfo` whose
`BrainwaysParams` hold `AtlasRegistrationParams(ap, rot_frontal,
rot_horizontal, rot_sagittal, hemisphere, confidence)`,
`AffineTransform2DParams(angle, tx, ty, sx, sy, cx, cy)` and
`TPSTransformParams(points_src, points_dst)`. Brainways uses its own ResNet
position model, not DeepSlice.

- Linear: export the subject's low-resolution slice images, run `linear.run`,
  write `ap` and the affine fields back into each `SliceInfo.params`.
- Nonlinear: `register.run` per slice; the returned atlas-to-slice pairs
  become `TPSTransformParams`, which is brainways' native nonlinear store.
  This is the cleanest fit of the two hosts.
- Open: brainways' dataclasses are frozen and its project writer is internal;
  the adapter must go through its public project API. Sign and axis
  conventions for `rot_*` need the same probe-style measurement the ABBA
  mirror used; never hardcode.

## Setup and credentials

Identical to the Fiji connector: the connector loads without Python, the
setup dock discovers conda environments, checks `setup.status` for
`protocol_version == 1`, offers browser login or API key entry, and remembers
only the environment path in napari settings. Credentials stay in LangSlice's
user settings. Either side may be installed first.

## Worker changes this needs

- `linear.run` currently refuses non-coronal planes, nonzero host angles and
  atlases outside Allen Mouse with Fiji-specific error text. Those checks
  should say what is unsupported, not which host is asking, and the coronal
  and flat-plane limits should be lifted as the linear engine's angle path
  is verified.
- A `register.run` result already carries the warped atlas path and
  correspondence count; the connectors additionally need the paired
  atlas-to-slice points that `nonlinear.abba` returns, so `register.run` gains
  an optional `landmarks: true` flag returning `source_points` /
  `target_points` in acquisition pixels.
- No new transport, no new setup methods.

## Distribution

Each host connector is its own PyPI distribution and napari-hub entry:
`napari-langslice-brainglobe-registration`, `napari-langslice-brainways`.
The core is `langslice-napari-core`. The worker is the same `langslice`
environment the Fiji route installs (`environment.yml`). The `.github`
package workflow gains one build per connector.

## Acceptance

Same list as the Fiji connector, per host: both installation orders, absent
or moved environment, credential-free startup, login cancel, protocol
mismatch, paths with spaces, selected and all sections, calibrated geometry,
host undo where the host has it, and save/reload of the host project with
LangSlice results in place. Offline tests use a fake worker and never call a
model provider.

## The host you may be forgetting

Candidates found, none yet confirmed as the intended one:

- **DMC-BrainMap** (Karolinska, Cell Reports Methods 2026): napari, 2D
  sections, its own SHARPy-track registration with a batch atlas-plane
  assigner. Closest to "feed a per-section atlas position".
- **atlastrack** (PyPI, first release ~2026-09-03): per-section atlas level,
  manual or DeepSlice, headless core with an optional napari front end.
  Only seen in one search result; unverified.
- **brainglobe-segmentation** and **cellfinder**: consume brainreg output,
  do not accept per-section positions. Not connector targets.
- There is no `napari-deepslice`; DeepSlice is a standalone package aimed at
  QuickNII.
