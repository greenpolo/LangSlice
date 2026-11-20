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

If the user's message holds a `langslice job FOLDER init` command from the
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

`init` takes the job flags `langslice job FOLDER schema init` lists (each with its
help, default and choices), among them `--notes TEXT` (the user's notes,
saved in the job). Check that `langslice job`
runs (`langslice job FOLDER schema init`); if it does not, or LangSlice's MCP tools are
missing, follow the `langslice:setup` skill and stop.

## 2. Create the job

```bash
langslice job FOLDER init --tasks position,transform --atlas allen_mouse_25um \
  --pixel-size-um 4.0 --nonlinear-provider none
```

It prints one JSON object (`ok`, `result`, `artifacts` with the job's `card`
and `state`, `warnings`). Report any warning to the user (for example what a
`--registration` file did not place). A job already in the folder is
continued; changed inputs, and other job settings, are refused with `STALE_INPUT`,
whose message names `--no-resume`: ask the user
before starting over.

## 3. Launch exactly one registration subagent

Pick the agent the user allows:

| Subagent | Can do | Use when |
| --- | --- | --- |
| `langslice:register` | only LangSlice's MCP tools, whose replies carry the pictures, and its registration skills (with `Read` for their example pictures); no shell, writing or web | default |
| `langslice:register-scripting` | the same MCP tools, plus Bash + Read + Write + Edit for scripts of its own with the LangSlice Python library in the job folder (a starting transform it computes, an image it cleans, a method no tool has) | the user wants scripting |

Use the Agent tool with that `subagent_type`, and the `model` and `effort`
the user chose, if any (otherwise it inherits this session's). Make its
prompt with `langslice job FOLDER launch --role register` (or `--role
register-scripting` for `langslice:register-scripting`) and pass `result.prompt`
as the subagent's entire prompt, unchanged: nothing added, shortened or
summarised (the one exception: your brief of the user's review comments,
after it, step 4). The user's notes belong in `init --notes` (the prompt holds them).
Run `launch` again for each launch: it reflects the job as it stands. Do not
give the subagent any other task. Run one subagent per job at a time. The
plugin's MCP server holds one job at a time for this session, so run one
registration subagent at a time, of either kind; brains registered side by
side each need a Claude Code session of their own.

You have LangSlice's MCP tools too; do not call them. They are the subagent's.
`langslice job` is yours, not the subagent's: it sets the job up and checks
it (`init`, `launch`, `status`, `export_maps`) and runs no registration tool.

When the subagent finishes, the plugin copies its transcript into
`FOLDER/trace/`.

Shell restriction: a plugin subagent cannot limit Bash to one command. If the
user wants `langslice:register-scripting` to run only the job's own Python,
tell them to add the permission rules from the plugin README to their
settings before launching.

Advisor: if you have an advisor tool, the subagent gets it too, and nothing in
the plugin can remove it. Before launching, tell the user to turn it off with
`CLAUDE_CODE_DISABLE_ADVISOR_TOOL=1` (plugin README) and restart Claude Code.

## 4. Check the result yourself

When the subagent returns, do not trust its summary alone:

- `langslice job FOLDER status`: the job's state, tasks and open items;
  confirm the job was submitted.
- Read `FOLDER/registration.json` (derived from `state.json`, the truth) and
  the exports under `FOLDER/exports/` (`quicknii.json`, `visualign.json`).
  Submit writes no per-section maps: `langslice job FOLDER export_maps`
  writes them when they are wanted. `docs/reference/file_formats.md` in the
  LangSlice repository describes every file.
- Look at the pictures the job saved (`views/`, paths in `artifacts`) with
  Read, and tell the user in plain words what looks right and what does not.
  If a picture and a number disagree, say so.
- Changes the user asks for go through another subagent run on the same job
  (every write is one undo step), not through your own edits.
- Comments the user left in the LangSlice window (its Review: a box drawn on
  a section and a few words on what is wrong there) are kept in the job
  folder's `review.json`. When the user asks you to look at them or fix them,
  run `langslice job FOLDER review_comments`: it lists each comment (its
  section, words, box and whether it is still `open` or already `revised`)
  and saves a picture of each commented section with its boxes outlined and
  numbered (`picture_files`). Look at those pictures with Read and judge
  what each comment asks for. Then launch a registration subagent on the job
  as in step 3, with the `launch` prompt first and, after it, your own brief
  of what to fix, in your own words. A job the user does not name is the
  one they last opened in the window: the newest `folder` in
  `~/.langslice/recent_jobs.jsonl`. When the subagent returns, run
  `review_comments` again: a comment whose section changed is `revised`.

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
  `langslice job FOLDER export_maps`. In the job folder's `exports/` it
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
  `langslice job FOLDER python scripts/export_dmc.py FOLDER` (the job's
  `env/`, made on first use, imports LangSlice). It writes the animal
  folder `exports/dmc_brainmap/<animal>/` in the job folder, named after
  the image folder, and prints what it wrote. DMC-BrainMap holds the linear
  placement only: when `warps_left_out` names sections, tell the user that
  their warps are left out of this export and that PyNutil on
  `exports/visualign.json` keeps them.
- **ABBA.** Write the job as an ABBA project. Save this script as
  `scripts/export_abba.py` in the job folder:

  ```python
  import sys

  import langslice

  if __name__ == "__main__":
      print(langslice.open_job(sys.argv[1]).export_abba())
  ```

  and run it with `langslice job FOLDER python scripts/export_abba.py
  FOLDER`. It writes `exports/<image folder>.abba`, which ABBA opens with
  **File ‣ Load State** on the Allen Brain Adult Mouse Brain CCF 2017 atlas,
  coronal; `exports/abba_export.json` names any section left out and why.
  The user can also choose **Export ‣ ABBA project (.abba)** in the
  LangSlice window. Never import `quicknii.json` into ABBA: ABBA misreads
  its image sizes and cutting angles, and the sections land millimetres off
  and out of order. To start a job from an ABBA registration, export it from
  ABBA (QuickNII dataset, or the registration file) and pass it to `init
  --registration FILE` (step 1).

The LangSlice documentation (https://langslice.readthedocs.io) has a page
for each: QUINT, VERSO, DMC-BrainMap and ABBA. Point the user to the one
they need.
