---
name: abba
description: Work with ABBA (Aligning Big Brains and Atlases, in Fiji) from a coding agent: drive a running ABBA through its MCP server (abba-mcp), import section images, save and load state, export registrations and regions, continue a LangSlice job started from ABBA, and bring an ABBA registration into LangSlice. Use when the user mentions ABBA, Fiji or QuPath projects.
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

## Reaching a running ABBA: abba-mcp

abba-mcp is ABBA's MCP server, released with ABBA 0.25.0 (installation:
https://abba-documentation.readthedocs.io/en/stable/installation/installation.html#install-assistant).
It needs `uv`. Register `uvx abba-mcp` as a local (stdio) MCP server:

```bash
claude mcp add abba -- uvx abba-mcp
# Fiji without a window:
claude mcp add abba -e ABBA_MCP_HEADLESS=true -- uvx abba-mcp
```

In another client: server `abba`, command `uvx`, argument `abba-mcp`.

- The first Fiji tool call downloads Java, Fiji and ABBA into `~/.abba-mcp`,
  which takes minutes: tell the user, and call the tool again. That Fiji is
  abba-mcp's own; it does not use or change the user's Fiji.
- If a Fiji already answers on port 9090 running ABBA 0.25.0 (the user's own
  Fiji, its MCP server started from **Help › Assistants › Manage Fiji MCP
  Server…**), abba-mcp drives that one instead. A Fiji there running another
  ABBA version is refused, with a message to close it.
- Only the user's own Fiji has the LangSlice connector installed (ABBA's
  **Register › LangSlice…**); the Fiji abba-mcp downloads does not.
- Before the first command, read abba-mcp's guide with `abba_docs_read`
  (`guides/abba.md`: the workflow and its traps; `abba_docs_search` searches
  its docs). Commands run by title with `fiji_module_run`;
  `fiji_module_describe` gives a command's inputs, defaults and choices.
  ABBA's documentation for this version:
  https://abba-documentation.readthedocs.io/en/v0.25.0/

Never give these tools to a registration subagent. Without the server, ABBA
is driven by the user: give them the exact menu steps.

## ABBA 0.25.x commands (menu path under Plugins > BIOP > Atlas)

Start a session first (`Open Atlas`, then `ABBA - ABBA Start`; without a
window, `ABBA - Align Big Brains and Atlases (no GUI)`); every command below
acts on that ABBA session.

| Command | Use |
| --- | --- |
| `Multi Image To Atlas > Import > ABBA - Import With Bio-Formats` | image files; asks position of the first slice (mm) and spacing between slices (mm) |
| `... > Import > ABBA - Import QuPath Project` | a `.qpproj`, same position and spacing |
| `... > Import > ABBA - Import QuickNII Project` | a QuickNII `.json` (DeepSlice writes one too) with positions and in-plane affines; approximate, per ABBA's own warning: one slicing angle (the median) for all slices. Never LangSlice's `quicknii.json` (below) |
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
- Out of LangSlice: there is no correct way to put such a job's results into
  ABBA. Never import its `quicknii.json` with `ABBA - Import QuickNII
  Project`: ABBA reads the JSON's width and height (the original image's) as
  the size of the PNG copy, so large images shrink toward a corner, and it
  takes one median cutting angle along each image's own axes, so the
  sections land millimetres off and out of order. To use ABBA, start the job
  from ABBA. For other programs, see step 5 of the `register-brain` skill.
- ABBA's coordinates differ from BrainGlobe's by a measured offset of about
  0.985 mm. Never hardcode it or convert positions by hand.
- Save State before any bulk change.
