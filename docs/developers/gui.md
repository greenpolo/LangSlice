# The LangSlice window

The LangSlice window writes the message that starts a registration in your
agent app, and shows the registration while it runs: the pictures the
agent looks at, what it is doing, the sections' current placements and the
steps it can undo. It reads the job folder, so it works the same whether the
run is LangSlice's own agent, a coding agent (Claude Code, Codex,
Antigravity) or a script. Closing the window does not stop the run.

## Opening it

```bash
langslice gui                       # the jobs opened recently on this computer
langslice gui ~/data/brain1         # an image folder (its langslice/ job folder) ...
langslice gui ~/data/brain1/langslice  # ... or the job folder itself
langslice linear run ~/data/brain1 --gui   # start a run and open the window on it
langslice gui ~/data/brain1 --tab prompt   # open on the prompt generator, filled in from the job
```

LangSlice serves the window from this computer only. With the `window` extra
installed (`pip install "langslice[window]"`, which adds pywebview) it opens as
a desktop window of its own, titled LangSlice, with LangSlice's icon and its
own entry in the taskbar or dock; no browser is involved (on Windows it is
drawn by WebView2, which Windows 11 includes). Without the extra it opens as
a browser window without an address bar, using Chrome, Chromium, Edge or
Brave; without one of those, as a tab in the default browser; if that fails
too it prints the address to open. `--no-window` only prints the address, and
`--port N` serves on a fixed port. The command keeps running until Ctrl-C, or
until the window has been closed for five seconds.

## What it shows

The window has three tabs: **Agent viewer**, **Prompt generator** and
**Settings** (top right). It opens on the agent viewer, or on the prompt
generator with `--tab prompt`. Opened without a folder and with no connection
chosen, it opens on Settings. In the job picker and its tooltips a path under
your home folder is written from `~`.

## The agent viewer

The agent viewer has two views, chosen by **Simple | Extended** at the
viewer's own top right corner, under the top bar (or `v`); the window remembers the choice.

**Simple**, the view it starts in, shows three things: the agent speaking,
the sections it is working on, and where they sit from front to back. The
agent's latest words are in a speech bubble of a fixed height beside the
LangSlice mark and a small figure for the agent's model, with the model's
name and its state (Working, Thinking, Done) under them; a summary of the
agent's reasoning is its thinking, a message what it says. The words come
in one after another; the figure walks while a tool runs, looks up while it
thinks and talks while its words come in. Under the bubble is the call
running now, in plain words. Up to three sections are shown large with
their name and position; more are dealt out as a grid of small pictures,
and the finished run shows every section. These pictures are drawn quiet
(`calm.py`): the main stain alone (a channel named for DAPI, Hoechst or
Nissl, else an RGB file's blue), the tissue cut out of the scanner's tile
floor and the atlas borders a soft light layer. A new focus is shown once
its pictures are ready (at most 4 s), and the stage that was fades over
the new one. Under the pictures, a front-to-back line holds a dot per
section: dim in stack order until placed, then at its position; a dot
glides when the agent moves its section (one moved alone leaves a fading
mark, a reshuffle of many moves front to back as a wave). Hovering a dot
names its section; clicking shows it.

**Extended** answers three questions, top to bottom: what is the agent
doing, what does the registration look like, and how did it get here. It
adds the progress meters, the journal of tool calls, the agent's pictures,
the table and the steps. The parts below are Extended's unless they say
otherwise.

- **Job.** The job picker at the top right lists the folders given on the
  command line and the jobs opened recently on this computer
  (`~/.langslice/recent_jobs.jsonl`; `LANGSLICE_RECENT_JOBS` names another
  file, or `off`), newest first. Each is named by its images folder and the
  folder above it (`M01/images`), with its path beside it. The window shows
  the most recently changed job until you pick one; the address then names
  it (`#job=<n>`).
- **What the agent is doing now.** The band across the top. Its first line
  is the status (Running, Waiting, Idle after ten minutes without a change,
  Submitted or Failed) and the run in one sentence: the model, the plane and
  the atlas. Under it, large, are the agent's latest words: the first
  paragraph of its newest message or reasoning summary (the whole of it on
  hover and in the journal). The last line names the tool running now, or
  the last one, in plain words, with how many sections it works on. On the
  right, three meters count the sections that are placed in the atlas,
  aligned to it and warped to fit; a task the job has switched off has no
  meter. A coding agent keeps its own words in its app, so for such a run
  the band shows what the agent is doing in their place.
- **Pictures.** The sections the agent's latest call worked on (a change,
  `look` or `zoom`; before the first, every section), four to a page (the
  arrows, or `[` and `]`). Each panel shows its section under its current
  registration with the atlas borders, drawn by the window from the job
  folder, and drawn again whenever that section's registration changes; a
  section not placed yet is shown alone. While a panel waits for another
  section's picture, the one it showed stays, dimmed. Its title gives the panel's
  number, the section, its position and cutting angles, and how it was
  aligned, whether it is warped and whether it is damaged. The wheel zooms,
  dragging pans and **Enlarge** shows it large. **Follow the agent** (on to
  start with; `f`) keeps the panels on the agent's latest call. Clicking a
  section in the strip, the table or the journal shows it and stops
  following.
- **Review.** The button above the pictures (or `r`) shows one section at a
  time, every section in order; the arrow keys flip through them and `Esc`
  ends the review. Drag a box around a part that looks wrong and type a few
  words under it; `Enter` keeps the comment in the job (`review.json`),
  `Esc` drops it. Every panel shows its section's comments, a box with the
  words above it; once the section has changed since you wrote one, it is
  faded and says *revised*. The `×` beside a comment removes it. The button
  counts the comments still open. Then tell your agent to look at the
  comments: it reads them with `langslice-job FOLDER review_comments`, which
  saves a picture of each commented section with its boxes, and briefs a
  registration agent on what to fix.
- **Where the sections sit in the brain.** The strip under the pictures: the
  atlas along its cutting axis, as thumbnails of the atlas template at
  regular steps over a millimetre ruler, and below them every placed section
  as a thumbnail in up to three rows, with a tick on the ruler at its
  position. The sections the agent is working on are marked in the window's
  one accent colour, each shown one with its panel's number, and a bar along
  the axis spans them; sections not placed yet are grouped at the right end.
  Point at a section for its name and position, and click it to look at it.
  The wheel zooms, dragging pans and a double-click shows the whole axis.
  The divider above the strip sets its height.
- **Table.** Order, filename, position, cutting angles, flipped, turned,
  transform, deformation method, damage and registration number; click a
  row to show that section. A job saved by an older LangSlice is read as
  every other way in opens it; one whose state this LangSlice cannot read (a
  newer one's) is shown from its raw fields.
- **Steps.** The steps that can be undone and redone, each named by the call
  that made it; your notes from the job's settings and the run's; and the
  spend (tokens and the dollar estimate, when the run records them).
- **The run so far.** The journal, on the right: a timeline of the agent's
  own words, its messages and reasoning summaries, each at its minute of the
  run. The job statement asks every agent to write, before each step, one
  plain sentence saying what it is about to do and why. Between them, one
  quiet line per tool call, named in plain words, with its section or how
  many it worked on and how many pictures it saved; open a line for the
  inputs; the tool's own name is beside its plain words. Calls of one tool in a row share one line
  (`×n`) that opens to show each. The journal follows its end until you
  scroll away from it.
- **The agent looked at.** Under the journal, the pictures each call saved,
  four at a time; the arrows go through the calls, **Newest** returns to the
  latest, and a click enlarges a picture with its caption, the words the
  agent read with it (then the arrow keys go through every picture of the
  job).

The agent's words come from LangSlice's own agent trace when the job folder
holds one. For a run by Claude Code they come from Claude Code's transcript
of the session, read while it is written: LangSlice looks under
`~/.claude/projects` for the session (or subagent) whose `start_job` call
names the job's folder, and shows its messages and reasoning summaries from
that call on. It reads those files and writes nothing;
`LANGSLICE_CLAUDE_TRANSCRIPTS` names another folder, or `off`. Other apps'
runs show their tool calls only.

The figure beside the agent's words is its model maker's: Clawd, Claude
Code's mascot, for a Claude model, OpenAI's mark for a GPT model, Google
Gemini's for a Gemini model, and a plain face otherwise. The marks belong to
their owners and only say whose model is speaking.

The panels and the strip show each section as its file holds it, its
channels in their own colours, not the contrast-enhanced image the fits
read. The pictures the agent looked at are shown as the agent saw them.

The window is neutral around the pictures, with one accent colour for what
is happening now; **Light | Dark** in the top bar switches its look (dark to
start with, remembered). The agent's own words are set in a serif face, the
window's own in a sans-serif face.

The viewer changes nothing of the registration: it writes only the review
comments, to their own file, and the Export menu's files.

## Export

**Export** in the top bar writes the shown job's registration as a file for
another program, under the job's `exports/` folder:

- **ABBA project (.abba)**: `exports/<image folder>.abba`, which ABBA 0.25
  opens with **File ‣ Load State** on its Allen CCFv3 atlas, started coronal.
  Each section opens from its own image file at its position and the stack's
  cutting angles, with its in-plane placement as an affine registration and
  its warp, when it has one, as a BigWarp landmark registration you can edit
  with **Edit last registration**. A warp takes up to a minute per section to
  convert; several convert at once. `exports/abba_export.json` lists the
  sections left out and why, and how far each warp's landmarks are from
  LangSlice's deformation.
- **QuickNII (.json)** and **VisuAlign (.json)**: `exports/quicknii.json`
  and `exports/visualign.json` with the section images they name, the files
  `submit` writes (both are written together).

While a file is written the menu says so; then it gives the file's path,
**Open folder** and **Copy path**, or why nothing was written. An ABBA
project needs the Allen mouse atlas, coronal sections and one pair of
cutting angles for the whole stack.

## The prompt generator

The prompt generator turns what you know about a brain into one message to
paste into your agent app (Claude Code, Codex or Antigravity, with LangSlice
installed). Only the folder of section images is needed; anything you leave
blank, the agent decides or asks you about.

- **Images.** The folder of section images. *How to copy a folder's path*
  says how on Windows, macOS and Linux; the quotes Windows' "Copy as path"
  adds are taken off.
- **Atlas** and **cutting plane.** Every atlas BrainGlobe describes, grouped
  into mouse, rat and other, starting on `allen_mouse_25um`, with
  BrainGlobe's own description beneath it; the plane is one of three
  buttons, and **Let the agent decide** leaves it to the agent.
- **Sections.** Thickness and interval in micrometres (whole numbers), and the
  pixel size, needed only when the image files do not record it.
- **What to do.** Place along the atlas, align in the plane, warp to fit
  (the job's tasks; the first two are on to start with).
- **What you already know.** That the file names sort in cutting order; that
  the sections were cut straight (the cutting plane stays untilted); that
  every section is the right way up and unmirrored (no flips or quarter
  turns); and damaged sections, one per line as `file name: what is wrong`.
- **Notes for the agent.** Saved with the job; the registration agent reads
  them.
- **Where you'll use the results.** QUINT (PyNutil, QuickNII, VisuAlign),
  VERSO and DMC-BrainMap, none ticked to start with. Each one ticked is
  named in the message's request, for your agent: once the registration is
  checked, it prepares the files that program opens and tells you where
  they are. The command does not change, so the registration agent is never
  told. ABBA is not listed: a job reaches ABBA only by starting from ABBA.
  A job prepared in ABBA has no such card.
- **Advanced** (closed to start with): every tool the registration agent can
  use, grouped by task, each with a short description and its own name, all
  on. Turning a tool or one of its options off takes it away from the agent.
  The tools of a task you did not choose are greyed; so are the options that
  something you said in *What you already know* leaves nothing to do. Some
  tools cannot be turned off. Below the warp's tools, **Image model**: the
  provider and model that draw a section's borders when the agent traces
  them (`--nonlinear-provider`, `--nonlinear-image-model`). It offers the
  providers a job accepts (your ChatGPT account, an OpenAI or Gemini API key,
  or none, which leaves the warp to fit each section to its stain alone) and
  the models LangSlice lists for each, says whether this computer has the
  provider's key or sign-in, and is greyed unless the run warps with border
  tracing on. Not set, the job uses LangSlice's default.

With `--tab prompt` and a folder, the form starts from that folder's job
(its `job.json`): its folder, atlas, plane, sizes, tasks, tools turned off,
what is kept on every section, damage notes, facts and notes.

The form is a column of numbered cards. The message beside it updates as
you type and is shown whole, and **Copy** puts it on the clipboard. It is a short request and the job's `langslice-job FOLDER init`
command, holding only what you set:

````text
Please register my brain sections with LangSlice, following its register-brain skill: create the job with the init command below, then launch the registration and check the result. The command leaves out the pixel size (when the image files do not record it): choose it yourself, or ask me when the images do not tell you. When the registration is done, prepare the results for the QUINT tools (PyNutil, QuickNII, VisuAlign) and DMC-BrainMap. The command is quoted for bash; if your shell is another, quote it for yours.

```bash
langslice-job /data/brain1 init \
  --tasks position,transform,nonlinear \
  --atlas allen_mouse_25um \
  --plane coronal \
  --position-thickness-um 40 \
  --position-interval-um 120 \
  --disabled zoom,trace_borders \
  --frozen '{"*":["orientation"]}' \
  --damaged '{"s012.tif":"left cortex torn"}' \
  --facts 'The file names sort in the order the sections were cut.' \
  --notes 'Injection in the right striatum.'
```
````

Each line of the command is one of `init`'s settings (`langslice-job schema
init` lists them all): `--tasks` what to do, `--disabled` the tools and
options turned off, `--frozen` what stays as it is on every section,
`--damaged` the damage notes, `--facts` the cutting order and `--notes` your
notes; `--nonlinear-provider` and `--nonlinear-image-model` appear only when
you choose an image model. A choice of tools the job would refuse (a task
with none of the tools that do it, a tool without the pictures it needs,
`look` with no kind of picture) shows in red instead of a message, with what
to turn back on. The command is written for bash; on Windows the agent
adjusts its quoting.

## From ABBA

A job prepared in ABBA (its `job.json` names ABBA as its host; see
[ABBA](../abba.md)) opens on the prompt generator with a **From ABBA** block
at the top:

- the snapshot folder, the atlas (and ABBA's name for it), the cutting plane
  and the snapshot pixel size, as fixed text. ABBA set them and the job holds
  them, so their fields below are greyed and the message leaves them out;
- **Keep the N sections' existing ABBA registrations (their positions may
  still move)**, ticked, when ABBA had registered some sections. Ticked, the
  message adds nothing and the job keeps them as ABBA prepared it; unticked,
  the command carries `--frozen '{}'`, which lets the agent redo their
  alignment and warps too. A box ticked under *What you already know* writes
  `--frozen` with the kept property on every section, and, while the
  registrations are kept, those sections' own entries beside it;
- each snapshot file with its ABBA slice name and whether ABBA had registered
  or warped it, so a damaged section can be named by its file.

Thickness and interval start from ABBA's slice spacing, when it had one;
check them.

## Settings

Only you set these, in the window; no agent tool reads or changes them.
Three cards, in this order:

- **Connections.** How LangSlice reaches a model: your ChatGPT account (the
  **Sign in with ChatGPT** button opens the sign-in page in your browser), or
  an OpenAI, Claude (Anthropic) or Gemini API key pasted into its card. Each
  card says whether it is signed in, has a key saved, or has a key set in the
  environment. Click a card to choose it. Keys are saved on this
  computer in `~/.langslice/provider_credentials.json`, the sign-in in
  `~/.langslice/openai_auth.json` (`LANGSLICE_OPENAI_AUTH` names another
  file); a saved key is never shown again, and saving a new one replaces it.
  A key set in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GEMINI_API_KEY`) wins over a saved one.
- **Agent model.** The model and reasoning effort for the chosen connection,
  starting at LangSlice's default where it has one (the ChatGPT account's
  `gpt-6.1-sol`, also offered for the OpenAI API key). The Claude and Gemini
  keys have no LangSlice default: choose a model. `langslice linear run`
  started without `--model` uses the chosen connection's saved model, and its
  saved effort unless `--reasoning` is given; with nothing saved it uses its
  own default model.
- **Registration runs in.** *This app* (your agent app runs registration) or
  *LangSlice, using the connection above* (LangSlice's own agent runs it). It
  is greyed until the chosen connection is signed in or has a key, and then
  starts at LangSlice. The choice is saved in the settings file, but nothing
  in LangSlice reads it yet: it changes neither your agent app nor
  `langslice linear run`.

The choices are saved in `~/.langslice/settings.json`
([file formats](../reference/file_formats.md#settings-langslicesettingsjson)); `LANGSLICE_SETTINGS` names another file, or
`off` for none.

## How it works

The window is a page served by `langslice gui` on `127.0.0.1` under a random
address, reachable only from this computer. The agent viewer watches the job folder's files
(`logs/events.jsonl`, `views.jsonl`, `trace/`, `state.json`,
`history/undo.json`, `job.json`, `exports/spend.json`; see
[file formats](../reference/file_formats.md)) four times a second while a page looks at the
job, and serves only the folders it lists. The focus panels and the strip's
thumbnails are drawn by the window itself, with LangSlice's own picture code,
from the job's settings, images, state and saved deformations, without
opening the job: nothing is written to the folder. The pictures are kept in
memory and drawn again only when a section's registration changes. The prompt generator's choices
(the tools, the tasks, the atlases) come from the installed LangSlice, so a
tool a newer version renames or removes is never offered. The viewer and the
prompt generator only read; the Settings tab's changes, the review comments
(`review.json` in the job folder) and the Export menu are the only requests
that write (Export only under the job's `exports/` folder and
`registration.json`, never the job's state), and the window accepts them
only from its own page.
