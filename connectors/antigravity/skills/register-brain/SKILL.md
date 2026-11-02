---
name: register-brain
description: Set up and run a LangSlice registration of one brain (a folder of histology section images to a BrainGlobe atlas): create the job with the user's choices, hand it to a registration subagent, then check and export the results. Use when the user asks to register, align or place brain sections in an atlas.
---

# Register a brain with LangSlice

You (the main session) prepare the job, launch a registration subagent, check
what it produced, and handle import and export with other software. You never
do the registration yourself: no positions, transforms or fits written by you.
Registration is always the subagent's, and it registers exactly one brain.

## 1. Ask the user (only what is not already said)

If the user's message holds a `langslice-job FOLDER init` command from the
LangSlice window's prompt generator, those settings are the user's choices:
run that command as given (change only its quoting to suit your shell, and add
any setting it names as left to you), and ask only about what its request says
it leaves out, and which subagent to use.

- The folder of section images.
- Tasks: Positioning (order and position along the atlas), Linear (in-plane
  alignment), Nonlinear (a deformation per section). Default is positioning
  plus linear.
- Atlas (default `allen_mouse_25um`) and cutting plane (coronal, sagittal,
  horizontal).
- Nonlinear only: an image model (`openai-oauth`, `openai-api`, `gemini-api`)
  or none (`--nonlinear-provider none`, fits to the stain alone).
- Section pixel size in um if the files do not carry it (`--pixel-size-um`),
  and section thickness and interval (`--position-thickness-um`, `--position-interval-um`).
- An existing registration to start from (QuickNII or VisuAlign JSON/XML,
  DeepSlice CSV/JSON/XML, or a LangSlice `registration.json`): `--registration
  FILE`. With it and no `--tasks`, the placement is kept and only the
  nonlinear step runs.
- Notes about the brain (a tear, a damaged section).
- Which subagent the user allows (step 3).

`init` takes the job flags `langslice-job schema init` lists (each with its
help, default and choices), among them `--notes TEXT` (the user's notes,
saved in the job). Check that `langslice-job`
runs (`langslice-job ops`); if it does not, follow the `setup` skill and
stop.

## 2. Create the job

```bash
langslice-job FOLDER init --tasks position,transform --atlas allen_mouse_25um \
  --pixel-size-um 4.0 --nonlinear-provider none
```

It prints one JSON object (`ok`, `result`, `artifacts` with the job's `card`
and `state`, `warnings`). Report any warning to the user (for example what a
`--registration` file did not place). A job already in the folder is
continued; changed inputs are refused with `INPUTS_CHANGED` and other job
settings with `SETTINGS_CHANGED`, which both name `--no-resume`: ask the user
before starting over.

## 3. Launch exactly one registration subagent

Pick the agent the user allows:

| Subagent | Can do | Use when |
| --- | --- | --- |
| `register` | the LangSlice MCP tools (`call_mcp_tool` on the plugin's `langslice_mcp` server) and view_file, which opens the pictures Antigravity saves from their replies: no shell, no file writes | default |
| `register-scripting` | run_command, view_file, write_to_file, replace_file_content and the web: the `langslice-job` command line, scripts with the LangSlice Python library, and packages installed in the job's own environment | the user wants scripting |

Use `invoke_subagent` with `TypeName` set to that agent's name and
`Workspace` set to `inherit`. Make its prompt with `langslice-job FOLDER launch --role register` (or `--role
register-scripting` for `register-scripting`) and pass `result.prompt` as the
prompt, unchanged: nothing added, shortened or summarised. The user's notes
belong in `init --notes` (the prompt holds them). Run `launch` again for each
launch: it reflects the job as it stands. Do not give the subagent any other
task. Run one subagent per job at a time.

The model is the user's choice: the agents inherit this session's. Without a
chat, either agent runs headless, started in the job folder:
`agy -p "<result.prompt of launch --role register>" --agent register --model MODEL --effort LEVEL --output-format stream-json`
(or `--agent register-scripting`).

When the session ends, the plugin copies its transcript into
`FOLDER/trace/`.

## 4. Check the result yourself

When the subagent returns, do not trust its summary alone:

- `langslice-job FOLDER status`: the job's state, tasks and open items;
  confirm the job was submitted.
- Read `FOLDER/registration.json` (derived from `state.json`, the truth) and
  the exports under `FOLDER/exports/` (`quicknii.json`, `visualign.json`).
  Submit writes no per-section maps: `langslice-job FOLDER export_maps`
  writes them when they are wanted. `docs/reference/file_formats.md` in the
  LangSlice repository describes every file.
- Look at the pictures the job saved (`views/`, paths in `artifacts`) with
  view_file, and tell the user in plain words what looks right and what does not.
  If a picture and a number disagree, say so.
- Changes the user asks for go through another subagent run on the same job
  (every write is one undo step), not through your own edits.

## 5. Import and export with other software

Starting from another program's registration is `init --registration FILE`
(step 1).

When the user's message names software they will use the results in,
prepare the results for it once the job is submitted and you have checked
it (step 4), then tell the user which files to open and where they are. An
export writes the job as it stands: after a correction, export again. The
job folder is `langslice/` inside the image folder unless `init --job-dir`
put it elsewhere.

- **QUINT tools (PyNutil, QuickNII, VisuAlign) and VERSO.** Run
  `langslice-job FOLDER export_maps`. In the job folder's `exports/` it
  writes `visualign.json` (the placement and the warps: give this one to
  PyNutil, VisuAlign and VERSO), `quicknii.json` (the placement only, for
  QuickNII) and a PNG copy of each section beside them
  (`<stem>_sNNN.png`). Keep these files together: the JSON files name the
  PNGs. QuickNII and VisuAlign run on Windows and macOS only; PyNutil and
  VERSO run anywhere.
- **DMC-BrainMap.** The export is in LangSlice's Python library. Save this
  script as `scripts/export_dmc.py` in the job folder:

  ```python
  import sys

  import langslice

  print(langslice.open_job(sys.argv[1]).export_dmc_brainmap())
  ```

  and run it with LangSlice's own Python:
  `langslice-job FOLDER python scripts/export_dmc.py FOLDER` (the job's
  `env/`, made on first use, imports LangSlice). It writes the animal
  folder `exports/dmc_brainmap/<animal>/` in the job folder, named after
  the image folder, and prints what it wrote. DMC-BrainMap holds the linear
  placement only: when `warps_left_out` names sections, tell the user that
  their warps are left out of this export and that PyNutil on
  `exports/visualign.json` keeps them.
- **ABBA.** A job started from ABBA (**Register › LangSlice…** in ABBA) is
  already in ABBA: see the `abba` skill. For any other job there is no
  correct way to put LangSlice's results into ABBA. Never import
  `quicknii.json` into ABBA: ABBA misreads its image sizes and cutting
  angles, and the sections land millimetres off and out of order. To use
  ABBA, start the job from ABBA.

The LangSlice documentation (https://langslice.readthedocs.io) has a page
for each: QUINT, VERSO, DMC-BrainMap and ABBA. Point the user to the one
they need.
