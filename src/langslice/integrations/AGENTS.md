# LangSlice `integrations/ — ABBA / QUINT`

Package guide for `src/langslice/integrations/`. The repo-level `CLAUDE.md` holds the
project-wide rules; this file holds what is specific to this package. `AGENTS.md`
here is a verbatim copy — edit one, mirror to the other.

- `integrations/` — one module per external registration ecosystem.
  `quint.py`: QUINT/QuickNII/VisuAlign-compatible JSON export (anchoring
  vectors; file-based, formerly top-level `export.py`).
  `abba.py`: LangSlice as an abba-python registration plugin
  (`enable_langslice_registration`, `register_selected_slices`). Implements
  ABBA's `SimpleRegistrationPlugin` socket from Python via JPype: the fixed
  image requested from ABBA is its atlas *coordinate* channels (per-pixel
  AP/DV/ML mm), so the adapter samples the BrainGlobe volumes at exactly
  those coordinates — no offset constants, no axis assumptions (the measured
  ABBA↔brainglobe AP offset is ~0.99, not 1.0; never hardcode it). The
  nonlinear result returns as a serializable invertible thin-plate-spline
  (`InvertibleWrapped2DTransformAs3D` — the plain wrapper is not invertible)
  sampled from the Elastix deformation field, and lands on the slice's
  registration stack like a native step (undo, state save/reload included).
  Reopening a saved state requires the plugin registered first, else ABBA
  drops the step. `install_gui` adds a `Register > LangSlice` menu entry via
  ABBA's registration-plugin UI registry (PyCommandBuilder dialog; needs the
  `pyimagej-scijava-command` artifact, test-scope in ABBA, added as a scyjava
  endpoint before JVM start — `run_gui_session` handles it), and
  `langslice abba` is the CLI launcher: full ABBA GUI with LangSlice in the
  Register menu, one command. Ships in the recommended
  `langslice` conda env — environment.yml carries openjdk 11 + maven, and
  `pip install -e ".[abba]"` adds abba-python (a separate env from the user's
  `abba`/`deepslice` envs). Live-session spikes: `_local/abba_spike/`
