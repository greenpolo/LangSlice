---
name: abba
description: Work with ABBA (Aligning Big Brains and Atlases, in Fiji) from a coding agent: drive a running ABBA through its MCP server (abba-mcp), import section images, save and load state, export registrations and regions, open a LangSlice registration in ABBA, and bring an ABBA registration into LangSlice. Use when the user mentions ABBA, Fiji or QuPath projects.
---

# ABBA from a coding agent

For the main session. LangSlice and ABBA meet through files: a LangSlice
registration is exported as an ABBA project that ABBA loads, and an ABBA
registration is exported from ABBA and given to a LangSlice job (the last
section here). Registration itself is the subagent's work (`register-brain`
skill).

## Reaching a running ABBA: abba-mcp

abba-mcp is ABBA's MCP server, released with ABBA 0.25.0 (installation:
https://abba-documentation.readthedocs.io/en/stable/installation/installation.html#install-assistant).
It needs `uv`. Register `uvx abba-mcp` as a local (stdio) MCP server:

```bash
agy mcp add abba uvx abba-mcp
# Fiji without a window:
agy mcp add --env ABBA_MCP_HEADLESS=true abba uvx abba-mcp
```

In another client: server `abba`, command `uvx`, argument `abba-mcp`.

- The first Fiji tool call downloads Java, Fiji and ABBA into `~/.abba-mcp`,
  which takes minutes: tell the user, and call the tool again. That Fiji is
  abba-mcp's own; it does not use or change the user's Fiji.
- If a Fiji already answers on port 9090 running ABBA 0.25.0 (the user's own
  Fiji, its MCP server started from **Help › Assistants › Manage Fiji MCP
  Server…**), abba-mcp drives that one instead. A Fiji there running another
  ABBA version is refused, with a message to close it.
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

## With a LangSlice job

- Into LangSlice: export from ABBA (QuickNII dataset, or the registration
  file) and start the job from it: `langslice-job FOLDER init --registration
  FILE`.
- Out of LangSlice: write the job as an ABBA project and load it in ABBA.
  Save this as `scripts/export_abba.py` in the job folder:

  ```python
  import sys

  import langslice

  if __name__ == "__main__":
      print(langslice.open_job(sys.argv[1]).export_abba())
  ```

  and run it with `langslice-job FOLDER python scripts/export_abba.py
  FOLDER`. It writes `exports/<image folder>.abba`, which ABBA opens with
  `ABBA - Load State` on the Allen Brain Adult Mouse Brain CCF 2017 atlas,
  coronal; `exports/abba_export.json` names any section left out and why.
  The user can also choose **Export ‣ ABBA project (.abba)** in the
  LangSlice window. Never import the job's `quicknii.json` with `ABBA -
  Import QuickNII Project`: ABBA misreads its image sizes and cutting
  angles, and the sections land millimetres off and out of order.
- ABBA's coordinates differ from BrainGlobe's by a measured offset of about
  0.985 mm. Never hardcode it or convert positions by hand.
- Save State before any bulk change.
