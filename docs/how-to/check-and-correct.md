# Check and correct a registration

Check every registration before you use it. The registration agent works
alone, and it can be wrong in ways it does not notice. This page shows where
to look, what to look for, and how to have a mistake put right.

## Read your agent's report

When the registration agent finishes, your agent checks the job itself. It
reads the job's status and the result files, and looks at the saved
pictures. It then tells you, in plain words, what looks right and what does
not.

Treat that report as a starting point. Look at the pictures yourself.

## Look at the pictures

Every picture the registration agent saw is saved and numbered. Two kinds
matter most:

- **Overlays.** The section with the atlas borders drawn on it. The borders
  are black lines with a thin white edge, or white lines with a black edge
  on dark tissue.
- **The positioning picture.** A millimetre ruler along the cutting axis,
  with atlas sections above it and your sections below it. A line joins each
  section to its position.

### Open the LangSlice window

The LangSlice window shows a job's pictures, newest first, beside a list of
what the agent did. Ask your agent:

```text
Open the LangSlice window on /data/brains/M04.
```

The window opens like a browser window. It shows:

- the newest picture, large, with its caption; click **Follow** to keep
  showing the newest;
- one row per tool call, with the sections and the pictures it saved;
- the **Sections** table: each section's position, cutting angles, mirroring
  and alignment, and the number of the registration it holds;
- the **Undo** list: the steps that can be undone, each named by the call
  that made it;
- your notes and the agent's.

The window only shows the job. Nothing in it changes the registration.
Closing it does not stop a run. See [The LangSlice window](../developers/gui.md)
for all of its parts.

<details>
<summary>Under the hood: opening the window and finding the files</summary>

The command is `langslice gui FOLDER`, where `FOLDER` is the image folder
or its `langslice` job folder. Without a folder it lists the jobs opened
recently on this computer. It keeps running until the window has been
closed for five seconds.

The pictures are files in the job folder: `views/` holds pictures of
several sections, and `sections/<name>/views/` the pictures of one section.
Each picture is a folder named by its number, the tool that drew it and its
mode, holding `view.jpg`.

</details>

### What to check

Judge each section by its internal anatomy, not by its outline alone. An
outline can fit while the inside is wrong.

1. **Position.** On the positioning picture, does each section sit at the
   atlas level that shows the same structures? Check structures that change
   quickly along the axis: the anterior commissure, the hippocampus, the
   ventricles.
2. **Order and spacing.** Do the lines from the ruler run in the order you
   cut the sections? Is the spacing even, apart from gaps where sections
   are missing?
3. **Left and right.** Is any section mirrored? Check your hemisphere mark,
   if you have one.
4. **In-plane alignment.** On each overlay, do the borders follow the
   midline, the ventricle walls, the corpus callosum and the cortical
   layers?
5. **Warps.** On a warped section, do the borders follow the tissue without
   folding or tearing it? A warp can pull borders onto the wrong structure.
6. **Damage.** Are the regions the agent marked as missing really missing?
   Marked regions are left out of every fit.

A picture is the evidence. When the agent's summary and a picture disagree,
trust the picture.

## Have a mistake corrected

Tell your agent what is wrong, by filename, in terms of the anatomy. For
example:

```text
M04_s07.tif is placed too far anterior: its hippocampus is visible, but the
atlas section at that position has none. Also, M04_s12.tif is mirrored.
```

Your agent starts a new registration run on the same job. That run begins
from the registration as it stands, with your request in its notes. It keeps
everything else. Every change it makes is a new step that can be undone.

<details>
<summary>Under the hood: how a correction run starts</summary>

Your agent adds the request to the job's notes with `langslice-job FOLDER
init --notes TEXT`, keeping the job's other settings. It then makes a new
launch prompt with `langslice-job FOLDER launch` and starts a registration
agent with it. The job reopens from its saved state.

</details>

## Undo a change

Every change to the registration is one step. You can go back in two ways.

- **Undo the last steps.** Ask for the last steps to be undone, for example
  "undo the last two steps". Steps are undone newest first, across the whole
  job. The last 50 steps are kept, also after the job is closed. The window's **Undo** list
  shows which call made each step.
- **Put one section back.** Each registration a section has held is
  numbered and kept. Ask for a section to go back to an earlier number, for
  example "put M04_s07.tif back to registration 2". This changes that section
  only. The **Sections** table in the window shows each section's current
  number.

These requests go through a registration run too, as corrections do.

## After a correction

Corrections update the job's result files when the run submits. The
per-section maps are written only on request, so ask your agent to write
them again after a correction. See
[Export results for cell counting](export-results.md).
