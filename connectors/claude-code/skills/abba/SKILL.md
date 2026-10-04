---
name: abba
description: Work with ABBA (Aligning Big Brains and Atlases, in Fiji) from a coding agent: import section images, save and load state, export registrations, regions and QuickNII datasets. Use when the user mentions ABBA, Fiji, QuPath projects or QuickNII alongside a LangSlice job.
---

# ABBA from a coding agent

This is for the main session. Registration itself belongs to the LangSlice
subagents (`register-brain` skill), not to this skill.

## Two ways to reach a running ABBA

1. The LangSlice dialog inside ABBA ("LangSlice Registration"): the user picks
   tasks, atlas and model there and LangSlice applies the result through ABBA's
   own actions. Prefer this when the user is sitting at ABBA.
2. Fiji's own MCP server (fiji/fiji-llm, a preview): in Fiji, run
   `Help > Assistants > Manage Fiji MCP Server...`, start the server (default
   port 9090, URL `http://localhost:9090/mcp`) and register it, for example
   `claude mcp add --transport http fiji http://localhost:9090/mcp`. The dialog's
   "Copy" selector gives the exact registration text. Fiji must be running;
   the tools fail while it is closed. The Fiji tools are separate from
   LangSlice's: never give them to a registration subagent.

Without either, ABBA is driven by the user; give them the exact menu steps.

## ABBA 0.24.x commands (menu path under Plugins > BIOP > Atlas)

Start ABBA first (`ABBA - ABBA Start`); every command below acts on that
ABBA session.

| Command | Use |
| --- | --- |
| `Multi Image To Atlas > Import > ABBA - Import With Bio-Formats` | image files; asks position of the first slice (mm) and spacing between slices (mm) |
| `... > Import > ABBA - Import QuPath Project` | a `.qpproj`, same position and spacing |
| `... > Import > ABBA - Import QuickNII Project` | a QuickNII `.json` (not exact, per ABBA's own warning) |
| `... > File > ABBA - Save State` | writes a `.abba` file; it must not exist yet |
| `... > File > ABBA - Load State` | reads a `.abba` file |
| `... > Inspect > ABBA - Get State` | the state as JSON (compact; `full` for every action) |
| `... > Export > ABBA - Export Registrations To QuPath Project` | registrations into the open QuPath project |
| `... > Export > ABBA - Export registered slices as QuickNII dataset` | output folder, pixel size, channels, prefix; resamples the images |
| `... > Export > ABBA - Export Regions To File` | one ROI set zip per slice, named by an ontology property |

Names and parameters were read from the ABBA source at tag
`ImageToAtlasRegister-0.24.1`; other ABBA versions may differ.

## Using LangSlice results with ABBA

- Register in ABBA's frame inside the LangSlice dialog, or register a folder
  with the `register-brain` skill and import the result: `langslice job
  FOLDER init --registration FILE` takes a QuickNII or VisuAlign file or a
  LangSlice `registration.json`; `langslice job FOLDER export_maps` writes
  `exports/quicknii.json` and `exports/visualign.json`.
- ABBA's coordinates differ from BrainGlobe's by a measured offset of about
  0.985 mm; LangSlice's ABBA integration fits it at start-up. Never hardcode
  it, and never convert positions between the two by hand.
- Save State before any bulk change.
