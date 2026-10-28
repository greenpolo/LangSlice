# Tell the agent what you already know and what it must not change

You know things about your brain that the agent can only guess. Examples
are the pixel size, the section interval, a torn section, or positions you
have already checked. Tell your agent in plain words before the registration
starts. This page shows what to say and what each statement becomes.

## How your words reach the registration

Two agents are involved:

- **Your agent** is the one you chat with in Codex, Antigravity or Claude
  Code. It sets up the job and checks the result. It never registers
  sections itself.
- **The registration agent** is a second agent that your agent starts. It
  places and fits the sections, then hands the job back.

Your agent writes what you say into the job's settings when it creates the
job. The registration agent receives those settings, word for word, in the
first message it reads. Settings are saved in the job folder, so a later
correction run reads them too.

Say everything in your first message, or answer your agent's questions. For
example:

```text
Register the brain in /data/brains/M04. Coronal sections, Allen mouse atlas.
The images are 0.65 µm per pixel. Sections are 40 µm thick and every sixth
section was mounted, so they are 240 µm apart. The files are in cutting
order. M04_s07.tif has a torn left cortex. Never mirror a section.
```

## What you can say

The statements fall into four kinds. They differ in how firmly they bind the
agent.

| Kind | What it does |
|---|---|
| Notes and facts | The agent reads them and takes them into account. Nothing enforces them. |
| Known values | Values the agent receives instead of finding them itself. It may still revise positions, order and angles. |
| Fixed values ("frozen") | Values the agent may not change. Every tool refuses a change. |
| Tools switched off | The agent is not given that tool or option at all. |

### Notes and facts

| You say | It becomes |
|---|---|
| Anything about the brain or the run: "the olfactory bulbs were removed", "be careful with the ventral sections" | The job's notes, shown to the agent as **User notes** |
| A fact about the stack: "sections were cut on a vibratome" | A line in the agent's **Run facts** |
| "M04_s07.tif has a torn left cortex" | A damage note on that section |
| "The right hemisphere has a notch in the cortex" | The hemisphere cue, which the agent uses to tell left from right |
| A note for one task, such as positioning or the in-plane alignment | That task's notes, shown with the task |

A damage note does not remove anything from the fits by itself. The agent
decides which atlas regions are missing and marks them. Marked regions are
left out of every fit.

Name sections by filename in every note. "Section 7" is ambiguous: the agent
knows the sections only by filename.

Notes can be changed later without starting the job over.

### Known values

| You say | It becomes | How the agent uses it |
|---|---|---|
| "The images are 0.65 µm per pixel" | The pixel size, for every file or per file | Every section is drawn at that size. It replaces the size stored in the files. |
| "Sections are 40 µm thick" | The section thickness | Advice only |
| "They are 240 µm apart" | The section interval | Advice only: the agent places each section by its anatomy, and spacing may vary where sections are missing |
| "They must be exactly 240 µm apart" | The interval, made strict | `submit` is refused until neighbouring sections are within 10% of the interval |
| "The files are in cutting order" or a list of filenames in order | The stack's starting order | The first pictures show this order. Positions still decide the final order. |
| "These are the positions" (a table of filename and millimetres) | Each section's starting position, in millimetres along the cutting axis from the edge of the atlas volume (the front edge, for coronal sections) | The agent starts from them |
| "The cutting angle is 3° pitch, 0° yaw" | The stack's cutting angles | The agent starts from them |
| "I already registered these in QuickNII" | An imported registration | See [Refine an existing registration](refine-existing.md) |

When you don't know a value, say nothing. The agent infers positions and
spacing from the anatomy and records its estimates in its notes.

### Fixed values

A section has five properties the agent could change. You can fix any of
them, for every section or for named ones.

| Property | What it is | Fixed at, when you give no value |
|---|---|---|
| `position` | Where the section sits along the cutting axis | You must give the positions |
| `angles` | The cutting angles (pitch and yaw) | A flat cut, 0° and 0° |
| `orientation` | Mirroring and quarter turns, together | Not mirrored, not turned |
| `affine` | The in-plane alignment: rotation, scale, shear and shift | The section as the file shows it |
| `deformation` | The nonlinear warp on top of the alignment | No warp |

For example:

- "Keep my positions as they are" fixes `position` on every section, with
  your positions as the values.
- "The cut is not tilted" fixes `angles` on every section.
- "Don't warp M04_s12.tif" fixes `deformation` on that one section.

A tool call that would change a fixed property is refused, and the agent is
told why. The registration agent's first message lists what is fixed.

### Tools switched off

You can take a tool or one of its options away from the agent. The most
useful case is mirroring:

- "Never mirror a section" switches off the mirroring option. The agent then
  has no way to flip a section left to right in its in-plane tool.
- For a stronger rule, fix `orientation` instead. That also fixes quarter
  turns.

Switching a tool off keeps it out of the agent's instructions. Fixing a
property is the guarantee. The full list of tools and options is in
[Tools and options](../reference/tools.md).

## Check what your agent wrote

Your agent's setup reply names the settings it used. To see them exactly,
ask your agent to show the job's launch prompt: the message the registration
agent starts from. LangSlice saves it in the job folder, under
`logs/launches/`. Your notes are under **User notes**, and fixed values under
**Run facts**.

The [LangSlice window](../developers/gui.md) shows your notes too, in its
**Notes** panel.

## Change a setting after the job exists

- **Notes** can be changed at any time. The next registration run reads the
  new notes.
- **Any other setting** is part of the job. Changing it means starting the
  job over, which discards the work done so far. Your agent asks before it
  does that.

<details>
<summary>Under the hood: the job settings your agent writes</summary>

Your agent creates the job with `langslice-job FOLDER init` and the flags
below, then makes the launch prompt with `langslice-job FOLDER launch`. The
settings are saved in the job folder's `job.json`.

| What you said | Flag |
|---|---|
| Notes | `--notes TEXT` |
| A fact | `--fact TEXT` (repeatable) |
| A note for one task | `--position-notes`, `--transform-notes`, `--nonlinear-notes` |
| Damage notes | `--damaged '{"M04_s07.tif": "torn left cortex"}'` |
| Hemisphere cue | `--hemisphere-cue TEXT` |
| Pixel size | `--pixel-size-um 0.65`, or `'{"a.tif": 0.65, "b.tif": 0.5}'` |
| Thickness, interval | `--thickness 40 --interval 240`, plus `--strict-interval` |
| Order | `--order '["s03.tif", "s01.tif", "s02.tif"]'` |
| Positions | `--positions '{"s01.tif": 5.2, "s02.tif": 5.44}'` |
| Cutting angles | `--pitch 3 --yaw 0`, or `--section-angles` per section |
| Fixed values | `--frozen '{"*": ["position", "angles"], "s12.tif": ["deformation"]}'` |
| Tools off | `--disable interactive_transform:flip` |

`"*"` means every section. The JSON values can also be file paths. Running
`init` again with other settings is refused with `SETTINGS_CHANGED` or
`INPUTS_CHANGED`; `--fresh` starts the job over. Every flag is in
[Job settings](../reference/settings.md).

</details>
