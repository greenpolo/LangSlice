---
name: abba
description: Work with ABBA (Aligning Big Brains and Atlases, in Fiji) from a coding agent: import section images, save and load state, export registrations, regions and QuickNII datasets, and bring a LangSlice job's files into or out of ABBA. Use when the user mentions ABBA, Fiji, QuPath projects or QuickNII.
---

# ABBA from a coding agent

For the main session. A job started from ABBA (the user chose **Register ›
LangSlice…** in ABBA) is an ordinary LangSlice job in
`~/.langslice/abba/<project>/` whose images are ABBA's snapshots: the user
pastes the LangSlice window's message, and `langslice-job FOLDER init`
continues the prepared job (ABBA's positions, cutting angles, pixel size and
protected registrations are already in it; do not pass them again). While
ABBA is open, every write to that job appears in ABBA as one ABBA Undo step;
nothing else is needed to "send results to ABBA". Registration itself is the
subagent's work (`register-brain` skill).

## Reaching a running ABBA

Fiji's own MCP server (fiji/fiji-llm, a preview): in Fiji, run
`Help > Assistants > Manage Fiji MCP Server...`, start the server (default port
9090, URL `http://localhost:9090/mcp`) and register it with the coding agent;
the dialog's "Copy" selector gives the registration text (for Antigravity, for
example, `agy mcp add --type http fiji http://localhost:9090/mcp`).
Fiji must be running; the tools fail while it is closed. Never give these Fiji
tools to a registration subagent. Without the server, ABBA is driven by the
user: give them the exact menu steps.

## ABBA 0.25.x commands (menu path under Plugins > BIOP > Atlas)

Start ABBA first (`ABBA - ABBA Start`); every command below acts on that
ABBA session.

| Command | Use |
| --- | --- |
| `Multi Image To Atlas > Import > ABBA - Import With Bio-Formats` | image files; asks position of the first slice (mm) and spacing between slices (mm) |
| `... > Import > ABBA - Import QuPath Project` | a `.qpproj`, same position and spacing |
| `... > Import > ABBA - Import QuickNII Project` | a QuickNII `.json` with positions and in-plane affines; approximate, per ABBA's own warning: one slicing angle (the median) for all slices |
| `... > File > ABBA - Save State` | writes a `.abba` file; never overwrites an existing one |
| `... > File > ABBA - Load State` | reads a `.abba` file into the current session; the images must still be where they were |
| `... > Inspect > ABBA - Get State` | the state as JSON (compact; `full` for the state file's content), once the queued work is done or after `timeout_s` |
| `... > Inspect > ABBA - Wait` | waits for queued actions (at most `timeout_s`), then reports how many are pending and what failed; registration commands return before their actions finish, so call it after them |
| `... > Export > ABBA - Export Registrations To QuPath Project` | regions and registration of each slice imported from a QuPath project, into that project's folder |
| `... > Export > ABBA - Export registered slices as QuickNII dataset` | output folder, pixel size, channels, image name; resamples the images with all registrations applied |
| `... > Export > ABBA - Export Regions To File` | one ROI set zip per slice, ROIs named by an ontology property |

Names and parameters were read from the ABBA source at tag
`ImageToAtlasRegister-0.25.0`; other ABBA versions may differ.

## With a LangSlice job that did not start from ABBA

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
