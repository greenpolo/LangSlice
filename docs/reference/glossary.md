# Glossary

The computing terms and LangSlice terms used in these pages, each in one or
two sentences.

```{glossary}
:sorted:

ABBA
  Aligning Big Brains & Atlases, a Fiji plugin for registering brain sections
  to an atlas by hand and with automatic tools. LangSlice can run inside it
  and hand its result back to ABBA.

Affine
  A transform of a section's image within its own plane that rotates,
  scales, shears and shifts it, keeping straight lines straight. LangSlice
  stores it as six numbers per section; together with the
  {term}`orientation` it is the in-plane part of the
  {term}`linear registration`.

Agent
  An AI model that works in a loop: it reads what it is given, calls
  {term}`tools <tool>`, looks at what they return and decides the next step
  until the task is done. In LangSlice an agent places the sections, judging
  the anatomy from the pictures the tools return.

Agent CLI
  `langslice-job`, LangSlice's command for coding agents: each call runs one
  {term}`tool` on a {term}`job folder` and prints its answer as {term}`JSON`.
  A coding agent with a {term}`terminal` uses it in place of {term}`MCP`.

API key
  A secret code from an AI company that lets a program use its models, billed
  per use and separate from a {term}`subscription`. LangSlice reads a key
  from `OPENAI_API_KEY` or `GEMINI_API_KEY`, set on your computer or in a
  `.env` file, or from the keys ABBA's setup saved.

Atlas
  A 3-D reference brain with its regions labelled, here a
  {term}`BrainGlobe` atlas such as the Allen mouse brain atlas
  (`allen_mouse_25um`). Registering a section finds the place in the atlas
  that it shows.

Atlas plane
  The 2-D slice through the 3-D atlas at a section's {term}`position` and
  {term}`cutting angles`: the image of the atlas that the section is matched
  against.

BrainGlobe
  An open-source project that packages brain atlases of many species in one
  format. LangSlice downloads its atlases from BrainGlobe on first use.

Checkpoint
  The job's saved state, `state.json` in the job folder, written after every
  change. A run that stops resumes from it.

Coding agent
  An {term}`agent` that works on files and runs commands on your computer,
  such as Codex, Claude Code or Antigravity. You talk to it in its app; it
  sets up LangSlice jobs and starts the registration {term}`subagent`.

Command
  A line of text typed in a {term}`terminal` that runs a program, such as
  `langslice gui`.

Conda environment
  A folder of software kept apart from the rest of the computer, made by the
  conda tool. LangSlice and its registration engines are installed in one
  named `langslice`.

Coordinate map
  An image the size of a section whose pixels hold atlas coordinates (in
  micrometres) or atlas region ids: `coords.tif` and `labels.tif` in the
  section's folder. Cell-counting software can read it to assign each cell to a
  region.

Cutting angles
  The tilt of the plane the sections were cut in, relative to the atlas's
  flat plane, given as pitch and yaw in degrees. The whole stack shares one
  plane, unless a registration imported from elsewhere gives each section its
  own.

Damage mark
  An atlas region recorded as missing or folded over on a section, with the
  `mark_damage` tool. Every fit leaves marked regions out.

DeepSlice
  A tool that estimates each section's atlas position and cutting angles with
  a trained neural network. LangSlice can import its results.

Deformation
  A smooth warp, on top of the {term}`linear registration`, that bends the
  atlas onto the section region by region, to follow tissue that is
  stretched, compressed or shifted locally. It is the result of the
  {term}`nonlinear registration` task.

Elastix
  A widely used open-source image-registration program. LangSlice's
  `elastix_affine` tool uses it to refine a section's {term}`affine`
  automatically.

Exports
  The files LangSlice writes for other software: `registration.json`, and
  QuickNII and VisuAlign files in the job folder's `exports/`. The
  {term}`coordinate maps <coordinate map>` are written on request.

Fiji
  A distribution of the ImageJ image-analysis program with plugins included.
  {term}`ABBA` runs in it.

FireANTs SyN
  The method of LangSlice's `fireants_syn` tool: ANTs SyN (symmetric
  normalization), the most widely used diffeomorphic (smooth and invertible)
  registration in neuroimaging, run by the FireANTs software on a GPU when one is present.
  It fits a {term}`deformation` by matching the stain of the section with an
  atlas image.

Frozen property
  A property of a section (position, cutting angles, orientation, affine or
  deformation) that the job forbids the agent to change. The job keeps the
  value you supplied, and every tool refuses a change.

GPU
  A graphics processor. The deformable fits run about a second per section
  on a GPU, and more slowly on the main processor (CPU) when there is none.

Harness
  The program around a model that gives it tools, runs the tool calls and
  keeps the conversation. Codex, Claude Code and Antigravity are harnesses,
  and so is LangSlice's built-in agent (`langslice linear run`).

Image model
  An AI model that edits images. LangSlice's optional `trace_borders` tool
  asks it to draw the atlas borders onto a section's anatomy, then fits a
  {term}`deformation` to the drawn lines.

Job
  One registration of one brain: its settings, its sections and their
  registration, kept in a {term}`job folder`.

Job folder
  The folder `langslice` that LangSlice makes beside your section images. It
  holds the job's settings, the registration, every picture the agent was
  shown, the undo history and the {term}`exports`.

Job statement
  The text that tells the registration agent its job: the atlas, the tasks,
  what is supplied and frozen, the tools by name and your notes. LangSlice
  writes it from the job's settings.

JSON
  A plain-text format for structured data, with names and values in braces
  and brackets: `{"s01.tif": 5.2}`. LangSlice's settings and results files use
  it.

Landmark
  A pair of points, one on the atlas and one on the section, that mark the
  same anatomical spot. A set of landmark pairs is applied as a warp
  (`landmark_triangulation`), as VisuAlign does.

LangSlice window
  The window `langslice gui` opens, which shows a job while it runs: the
  pictures the agent looks at, the tool it is running, and the sections'
  placements.

Launch prompt
  The message that starts the registration {term}`subagent`, written by
  `langslice-job FOLDER launch` from the job. The main agent passes it on
  unchanged.

Linear registration
  The placement of a section that keeps straight lines straight: its
  {term}`position`, {term}`cutting angles`, {term}`orientation` and
  {term}`affine`. The Positioning and Linear tasks produce it.

Main agent
  The {term}`agent` you talk to in your app. It asks what you know, creates
  the job, starts the registration {term}`subagent` and checks and exports the
  result; it does not register the sections itself.

Mask
  A per-pixel weight from 0 to 1 over a section, made by the `make_mask` tool,
  that tells a fit which tissue to compare with the atlas. Pixels under 0.5,
  such as holes, tears and stains, are left out of the fit.

MCP
  Model Context Protocol, an open standard by which an AI app uses tools that
  another program provides. LangSlice's MCP server (`langslice mcp`) runs on
  your computer, and your app's agent calls LangSlice's tools through it.

Miniforge
  A free installer for the conda tool, from the conda-forge community. It is
  the first thing to install before LangSlice.

Model
  One AI system of an AI company, such as a GPT, Claude or Gemini model. The
  agent's model reasons and calls tools; the {term}`image model` draws.

Nonlinear registration
  The task that gives each section a {term}`deformation` on top of its
  {term}`linear registration`, with FireANTs SyN, landmarks or image-model
  traces.

Orientation
  How a section's image sits before its {term}`affine`: whether it is
  mirrored left-right (flip) and turned by a quarter turn (0, 90, 180 or 270
  degrees).

Overlay
  A picture of a section under its current registration with the atlas
  region borders drawn on it. It is the picture to judge a registration by.

Picture number
  The number LangSlice gives every picture it shows the agent. A later tool
  call names a picture by it, and the picture is saved under it in the job
  folder.

Position
  Where a section sits along the atlas's {term}`slicing axis`, in millimetres
  from the edge of the atlas volume (for a coronal series, from the anterior
  edge).

Preprocessed image
  The grayscale image of each section that LangSlice makes when the job is
  created, with the channels blended and the local contrast enhanced. The
  pictures and the fits use it unless the agent chooses another
  {term}`section image`.

Provider
  How LangSlice reaches a model: `openai-oauth` (signing in with ChatGPT),
  `openai-api` or `gemini-api` (an {term}`API key`), or `none` for no image
  model.

QuickNII
  A program for placing sections in an atlas by hand, from the QUINT workflow
  of the University of Oslo. Its files hold each section's linear placement;
  LangSlice reads and writes them.

Reasoning effort
  How long a model thinks before it answers, on models that offer a choice
  (low to max). It applies to LangSlice's built-in agent; a coding-agent app
  sets its own.

Registration
  Finding, for every point of a section, the atlas point it shows. In
  LangSlice it is the {term}`linear registration` plus an optional
  {term}`deformation`.

Registration number
  The number each registration a section has held is given. `status` shows
  the one held now, and `restore_registration` puts an earlier one back.

Section
  One image file of one brain section, named by its filename.

Section image
  A grayscale picture of a section that the pictures and fits read by name:
  the raw channels of the image file, the {term}`preprocessed image`, or one
  the agent makes with `make_image`.

Slicing axis
  The axis along which the sections follow one another: anterior-posterior
  for coronal sections, medial-lateral for sagittal, dorsal-ventral for
  horizontal.

Stack
  All the sections of one brain, in one folder.

Subagent
  An {term}`agent` that another agent starts for one task, with a
  conversation of its own. The {term}`main agent` in your app starts a
  registration subagent, which registers one brain.

Submit
  The tool that ends a registration run. It checks that every section has
  what the job requires, then writes the {term}`exports`.

Subscription
  A plan with an AI company (ChatGPT, Claude or Gemini) that you sign in to,
  free or paid. A registration run counts against its usage limits.

Task
  One part of a registration the agent can be given: Positioning (`position`),
  Linear (`transform`) or Nonlinear (`nonlinear`). A task that is off takes
  its answer from what you supply.

Terminal
  The window where you type {term}`commands <command>`: Terminal on a Mac,
  Miniforge Prompt on Windows.

Token
  The unit in which AI companies count the text and images a model reads and
  writes. Usage limits and API prices are set in tokens.

Tool
  A function an {term}`agent` can call, such as `look` (draw a picture) or
  `elastix_affine` (fit a section). The [tools page](tools.md) lists every
  LangSlice tool.

Tool plugin
  Python code that adds tools of its own to a job's toolbox, such as a
  published registration method wrapped as a tool.

Transcript
  The record of an agent's conversation, saved in the job folder's `trace/`
  folder.

Undo
  Taking back the last change to the registration. Every change is one undo
  step, and the history is kept in the job folder.

Usage limit
  How much model use a {term}`subscription` allows in a period. A
  registration run uses your own.

VisuAlign
  A QUINT program that refines a QuickNII placement with landmark pairs into
  a nonlinear registration. LangSlice imports the linear placement from its
  files, and writes its {term}`deformations <deformation>` in its format.
```
