---
name: abba
description: Work with ABBA (Aligning Big Brains and Atlases, in Fiji) from a coding agent: import section images, save and load state, export registrations, regions and QuickNII datasets, and bring a LangSlice job's files into or out of ABBA. Use when the user mentions ABBA, Fiji, QuPath projects or QuickNII.
---

# ABBA from a coding agent

For the main session. ABBA is handled with ABBA's own commands; LangSlice does
not mirror or drive ABBA here. Registration of a LangSlice job is the
subagent's work (`register-brain` skill).

## Reaching a running ABBA

Fiji's own MCP server (fiji/fiji-llm, a preview): in Fiji, run
`Help > Assistants > Manage Fiji MCP Server...`, start the server (default port
9090, URL `http://localhost:9090/mcp`) and register it with the coding agent;
the dialog's "Copy" selector gives the registration text (for Claude Code, for
example, `claude mcp add --transport http fiji http://localhost:9090/mcp`).
Fiji must be running; the tools fail while it is closed. Never give these Fiji
tools to a registration subagent. Without the server, ABBA is driven by the
user: give them the exact menu steps.

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

## With a LangSlice job

- Into LangSlice: export from ABBA (QuickNII dataset, or the registration
  file) and start the job from it: `langslice-job FOLDER init --registration
  FILE`.
- Out of LangSlice: `langslice-job FOLDER export_maps` writes
  `exports/quicknii.json` and `exports/visualign.json`. ABBA takes the
  linear placement through `ABBA - Import QuickNII Project` with
  `quicknii.json`; ABBA has no VisuAlign import, so `visualign.json` (the
  deformations) is for VisuAlign itself.
- ABBA's coordinates differ from BrainGlobe's by a measured offset of about
  0.985 mm. Never hardcode it or convert positions by hand.
- Save State before any bulk change.
