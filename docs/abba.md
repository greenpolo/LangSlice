# ABBA

The Fiji connector starts a LangSlice job from the sections you selected in
ABBA and opens the LangSlice window on it. Every change the agent saves then
appears in ABBA, as a native ABBA registration you can undo. The registration
itself runs in your AI app (Codex, Claude Code or Antigravity), from the
message the LangSlice window writes for you.

Supported sessions: ABBA 0.25 (0.24.1 also works) on Java 21, the Allen Mouse
V3 or V3p1 atlas, coronal sections. Tilted cutting angles are supported.

## Install

1. Install LangSlice's Python environment as in
   [Install LangSlice](getting-started/install.md) (Miniforge, then
   `conda env create -f environment.yml`). Keep the environment installed:
   ABBA uses it.
2. Build the connector with its
   [build instructions](https://github.com/greenpolo/LangSlice/blob/main/connectors/fiji/README.md#build-and-install).
   Copy `langslice-fiji-0.3.0.jar` into Fiji's `jars` folder, remove any older
   `langslice-fiji` jar from it, and restart Fiji. The connector and
   LangSlice must be updated together; a mismatch says **Update LangSlice and
   the connector together**.
3. The first time you use the connector, it asks for the LangSlice
   environment folder (the folder of the environment, not its Python program
   or the LangSlice source folder). Choose it, click **Check installation**,
   then **Save**. **Plugins ‣ LangSlice ‣ Choose Python environment…** opens
   the same window later.

Sign in to an AI provider, if you need one, in the LangSlice window's
**Settings** tab, not in Fiji.

## Use

1. In ABBA, import your sections and select the ones to register. With none
   selected, every section is used.
2. Choose **Register ‣ LangSlice…** (also **Plugins ‣ LangSlice ‣ Open
   LangSlice…**). The first time, the connector exports one snapshot per
   section and prepares the job with ABBA's positions, cutting angles and
   pixel size. ABBA itself does not change. The snapshots have the detail
   LangSlice works at when you start a job from the image files: each shows
   ABBA's image frame at 3072 pixels on its long side (about 4 µm per pixel
   for the Allen mouse atlas), or at your images' own pixel size when theirs
   is coarser.
3. The LangSlice window opens on the **Prompt generator**. Its **From ABBA**
   block shows the atlas, plane, pixel size and the snapshot file for each
   ABBA slice; these are fixed by ABBA. Thickness and interval are filled in
   from ABBA; check them. **Keep the N sections' existing ABBA registrations**
   is ticked: sections you already registered in ABBA keep their in-plane
   alignment and warps (their positions may still move). Untick it to let the
   agent redo them.
4. Add what you know (damaged sections, notes), click **Copy** and paste the
   message into a new chat in your AI app. The agent sets up the job and runs
   the registration.
5. Watch ABBA. Each change the agent saves appears as one ABBA undo step.
   Several changes saved within half a second of each other arrive as one
   step; the LangSlice window's **Agent viewer** lists every change on its own.
   An undo the agent makes arrives as a new ABBA step that puts the section
   back.
6. Save the project with ABBA's own save command (**ABBA - Save State**). The saved project contains only ABBA's
   own registration types, so it reopens without LangSlice, and the warp opens
   in BigWarp like any BigWarp registration.

Each section carries at most two LangSlice steps on top of its own
registrations: `LangSlice affine [<key>]` and `LangSlice warp [<key>]`, where
`<key>` names the project.

### Coming back to a project

Choosing **Register ‣ LangSlice…** again on the same project continues its
job: no new export, and the window opens on the **Agent viewer** once the job
has run. ABBA is brought back in line with the job first, in one undo step,
also after ABBA was closed and the project reopened. The job is the truth:
a LangSlice step you undid in ABBA comes back.

When your selection differs from the job's sections, or you added or removed
sections since, one box asks: **Continue the existing job (N sections)** or
**Start a new job with the M selected slices**. Starting a new job moves the
old job's folder aside (it is kept, renamed with the date) and removes the old
job's steps from the selected sections, in one ABBA undo step. Closing the box
changes nothing.

Two ABBA projects that import the same image files share one job folder; use
**Start a new job** in the second.

### Editing a section while the agent works

Avoid editing the registrations of sections the agent is working on. Adding,
removing or undoing a step of a section in ABBA makes that section refused.
The agent's later changes to it are kept but not applied, and Fiji's **Log**
window names the section. To
bring it back in line, remove any step you added on top of it and choose
**Register ‣ LangSlice…** again.

Closing ABBA, or **Plugins ‣ LangSlice ‣ Stop following**, stops the updates
and nothing else. The agent's run and the job go on, and the next
**Register ‣ LangSlice…** brings ABBA up to date.

## Where the files are

Each ABBA project's job is in `~/.langslice/abba/<key>/`, in the
`.langslice` folder of your home folder. It holds:

- the snapshots, `section_0001.tif` and so on;
- `langslice/`, the [job folder](reference/file_formats.md), with the exports;
- `abba_sections.json`, which snapshot is which ABBA slice;
- `gui.log`, the LangSlice window's last output.

The job folder's `logs/abba_applied.jsonl` records each update: what ABBA
applied, and what it refused with ABBA's own reason.

## Limits

- Coronal sessions of ABBA's Allen Mouse V3 or V3p1 atlas only.
- One cutting angle for the whole stack, as in ABBA.
- The agent works on the snapshots, at most 3072 pixels across, not on your
  full-resolution images; a zoom cannot show more detail than a snapshot has.

## Troubleshooting

**The LangSlice window does not open.** A message box shows its first output
lines. Choose **Plugins ‣ LangSlice ‣ Choose Python environment…** and click
**Check installation**; it shows the LangSlice version found. In a terminal
(the window where you type commands), `conda run -n langslice langslice
version` should print it too.

**A section did not update in ABBA.** Fiji's **Log** window names it and gives the
reason. See [Editing a section while the agent works](#editing-a-section-while-the-agent-works).

<details>
<summary>Under the hood</summary>

The connector starts `langslice serve --stdio` in the chosen environment: it
prepares the job (`abba.prepare`) and then follows it (`abba.follow`),
reading the job's files as the agent saves them. It also starts
`langslice gui <job folder> --tab prompt`. Nothing opens a network port. The
worker protocol and the attach step are in
[architecture](developers/architecture.md#fiji-connector-and-worker) and the
connector's own
[README](https://github.com/greenpolo/LangSlice/blob/main/connectors/fiji/README.md).

</details>
