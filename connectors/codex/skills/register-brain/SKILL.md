---
name: register-brain
description: Set up and run a LangSlice registration of one brain (a folder of histology section images to a BrainGlobe atlas): create the job with the user's choices, hand it to a registration subagent, then check and export the results. Use when the user asks to register, align or place brain sections in an atlas.
---

# Register a brain with LangSlice

You (the main session) set up the job, launch a registration subagent, and
check what it produced. You never do the registration yourself: no positions,
transforms or fits written by you. The subagent registers exactly one brain.

## 1. Ask the user (only what is not already said)

- The folder of section images.
- Tasks: Positioning (order and position along the atlas), Linear (in-plane
  alignment), Nonlinear (a deformation per section). Default is positioning
  plus linear.
- Atlas (default `allen_mouse_25um`) and cutting plane (coronal, sagittal,
  horizontal).
- Nonlinear only: an image model (`openai-oauth`, `openai-api`, `gemini-api`)
  or none (`--image-provider none`, fits to the stain alone).
- Section pixel size in um if the files do not carry it (`--pixel-size-um`),
  and section thickness and interval (`--thickness`, `--interval`).
- An existing registration to start from (QuickNII or VisuAlign JSON/XML,
  DeepSlice CSV/JSON/XML, or a LangSlice `registration.json`): `--registration
  FILE`. With it and no `--tasks`, the placement is kept and only the
  nonlinear step runs.
- Notes about the brain (a tear, a damaged section).
- Which subagent the user allows (step 3).

`langslice --help`, `langslice job FOLDER init --help` list every flag. Check
that `langslice` runs (`langslice version`); if it does not, tell the user how
the plugin README says to install it, and stop.

## 2. Create the job

For the CLI and scripting subagents:

```bash
langslice job FOLDER init --tasks position,transform --atlas allen_mouse_25um \
  --pixel-size-um 4.0 --image-provider none
```

It prints one JSON object (`ok`, `result`, `artifacts` with the job's `card`
and `state`, `warnings`). Report any warning to the user (for example what a
`--registration` file did not place). A job already in the folder is
continued; changed inputs are refused with `INPUTS_CHANGED`, which names
`--fresh`: ask the user before starting over.

For the MCP subagent the job is saved with the same flags, and the command
prints the id to give the subagent (on stderr: `Saved job ID in DIR`):

```bash
langslice claude prepare FOLDER --tasks position,transform --notes "Section 12 has a tear."
```

## 3. Launch exactly one registration subagent

Pick the variant the user allows, by how much access it gets:

| Subagent | Can do | Use when |
| --- | --- | --- |
| `register_mcp` | only LangSlice's MCP tools | the user wants no shell or file access (default) |
| `register_cli` | Bash + Read | the user accepts a shell for `langslice job` |
| `register_scripting` | Bash + Read + Write + Edit | the user also wants scripts with the LangSlice Python library |

Ask Codex to spawn that custom agent by name (`register_mcp`, `register_cli`
or `register_scripting`; the agent files are in `connectors/codex/agents/`).
The request names only the job:
"Register the brain in job id ID" (MCP) or "Register the brain in job folder
FOLDER" (CLI, scripting). Add the user's notes only if they were not already
saved in the job. Do not add registration advice of your own: the subagent
works from LangSlice's own job statement or job card. Do not give it any
other task. Run one subagent per job at a time.

Shell restriction: Codex rules govern commands that run outside the sandbox
only. If the user wants the CLI variants limited, point them to
`connectors/codex/README.md` (rules file and sandbox settings) before launching.

## 4. Check the result yourself

When the subagent returns, do not trust its summary alone:

- `langslice job FOLDER status`: the job's state, tasks and any open items;
  confirm the job was submitted.
- Open `FOLDER/registration.json` (derived from `state.json`, the truth) and
  the maps and exports under `FOLDER/exports/` (`quicknii.json`,
  `visualign.json`) and each section's maps. `docs/file_formats.md` in the
  LangSlice repository describes every file.
- Look at the pictures the job saved (`views/`, paths in `artifacts`) by opening the images, and tell the user in plain words what looks right and what does not.
  If a picture and a number disagree, say so.
- Changes the user asks for go through another subagent run on the same job
  (every write is one undo step), not through your own edits.

To export without registering: `langslice job FOLDER export_maps` writes each
placed section's maps, `exports/quicknii.json`, `exports/visualign.json` and
`registration.json` from the job as it stands. To bring the registration into
ABBA or QuPath, use the `abba` skill.
