# What the AI decides, where it fails, and how to check it

LangSlice's agent makes the same judgements a person makes when registering
sections by hand. It can be wrong in the same ways, and in some of its own.
You can see every choice it made and undo any of them. Before you use a
registration in a paper, look at it.

## What the agent decides on its own

Unless you supply or lock a value, the agent chooses:

- the position of each section along the cutting axis, and so the order of
  the stack;
- the cutting angles of the stack;
- whether each section is mirrored or turned by a quarter turn;
- each section's rotation, scale, shear and shift;
- which sections get a deformation, by which method, and with which settings;
- which regions a fit pays attention to;
- which regions are damaged or missing, which every fit then leaves out;
- which version of each section image the fits read, for example one with
  more contrast;
- which atlas image the fits compare against (by default a Nissl image of
  the atlas);
- when the work is finished.

It decides by looking at pictures, the same overlays you would look at. The
registration methods report what they did, but no score of how well the
result matches. The agent judges that by eye.

The agent is given a short fixed role, the same for every job, and a job
statement written for your stack. The statement says what to do, lists the
tools, gives facts about the sections and the atlas, and includes your notes.
It also says that LangSlice sets no limit on time or effort and that quality
comes first.

## Every change is recorded

Nothing the agent does is hidden or final.

- **Undo.** Every change is one undo step. Undo and redo work across
  sessions, so you can step back after the agent has finished.
- **Earlier registrations.** Each registration a section has held (its
  orientation, transform and deformation) is numbered and kept. You can put
  any of them back on that section alone, without undoing other sections.
- **Pictures.** Every picture the agent looked at is saved in the job folder
  and numbered. Its caption states the position, angles and image it shows.
- **Tool calls.** Every tool call, from any route into LangSlice, is logged
  in the job folder.
- **Transcripts.** LangSlice's own agent saves its full conversation in the
  job folder. The Claude Code and Antigravity setups copy the agent's
  conversation there when its session ends.
- **Version.** The registration file names the LangSlice version that wrote
  it.

The [LangSlice window](../developers/gui.md) shows the pictures, the tool
calls and the undo steps while a run goes on and afterwards. The files are
described in [Job folder and file formats](../reference/file_formats.md).

## What you can lock

You can give LangSlice what you already know and stop the agent from changing
it. Each section has five properties you can lock (*freeze*):

| Property | What it is |
|---|---|
| Position | Where the section sits along the cutting axis |
| Angles | The cutting angles |
| Orientation | Mirroring and quarter turns, together |
| Affine | The in-plane rotation, scale, shear and shift |
| Deformation | The deformation on top of the linear placement |

A locked property is refused for every change, from every tool and every
route, scripts included. A stage you switch off locks its properties on every
section. [Tell the agent what you already know](../how-to/tell-the-agent.md) gives
the steps, and [Job settings](../reference/settings.md) lists the settings.

You can also switch off single tools or options. That keeps the agent's
instructions shorter. It is not a lock: a script can still write past a
switched-off tool. Use a frozen property when the agent must not change
something.

## Where it fails

These are the failures to look for. Each follows from how the methods work.

- **A section at the wrong position.** The agent places each section by
  comparing its anatomy with atlas planes. Where neighbouring planes look
  alike, it can pick the wrong one. The fits do not move a section along the
  cutting axis, so a wrong position stays wrong until someone moves it.
- **A mirrored section.** On a nearly symmetric section, left and right
  cannot be told apart from the anatomy. If one hemisphere is marked, by an
  injection site or a notch, tell LangSlice what marks it, or lock the
  orientation.
- **The wrong size.** Without a pixel size from the file or from you,
  LangSlice matches the image's width to the atlas plane's width. A section
  that holds only part of the brain, such as one hemisphere, then starts at
  the wrong scale. An automatic fit can also stretch it over atlas it does
  not hold. LangSlice warns the agent when a fit changes the scale or shear
  by a large amount.
- **A fit from too far away.** Elastix and FireANTs refine a placement that
  is already close. Started far off, they can settle on a wrong match.
- **Damaged tissue.** Regions marked as damaged are left out of every fit,
  so intact tissue inside a marked region is not fitted. Torn or folded
  tissue that is not marked can pull a fit off.
- **A sparse stain.** A fit compares one section image with one atlas image.
  A tracer or reporter channel shows little of the anatomy the atlas draws,
  and gives a fit little to work with.
- **Folds in a landmark warp.** Pairs whose atlas points cross over each
  other fold the tissue over itself. LangSlice keeps the warp, as VisuAlign does, and
  reports the folded area and the pairs that cause it.
- **Image-model traces.** The image model can redraw the tissue as well as
  the borders, and yellow tissue can confuse the reading of its lines. A
  fit to its lines can make a good trace worse.
- **A finished run that is still wrong.** When the agent finishes, LangSlice
  checks only that the job is complete: every section has a position and a
  transform.
  It does not check that they are right.
- **A run that stops.** A run stops when the AI provider refuses more work.
  Reaching your plan's usage limit is the usual cause (see
  [Plans and usage limits](plans-and-usage.md)). The work so far is saved,
  and the job continues from there when you start it again.

## How to check a result

Check the result the way you would check a registration made by a student:
look at every section.

1. **Open the overlay of every section.** An overlay is the section with the
   atlas borders drawn on it. The agent's last overlays are in the job
   folder, and you can ask your AI to draw any section again.
2. **Judge the inside, not the outline.** An outline can match while the
   inside is wrong. Look at landmarks you trust: the ventricles, the corpus
   callosum, the hippocampus, cortical layers, fibre tracts.
3. **Check the positions across the stack.** The positioning picture shows
   every section beside atlas planes, in order. Look for a section whose
   anatomy belongs between two others, and for uneven spacing.
4. **Check the orientation.** Make sure a marked hemisphere ended on the
   side you expect.
5. **Correct what is wrong.** Tell the agent what to fix, undo steps, or put
   back an earlier registration of a section. The QuickNII and VisuAlign
   files also open in those programs for manual correction. The
   guide [Check and correct a registration](../how-to/check-and-correct.md)
   gives the steps.

```{figure} img/check_ventricles.jpg
:alt: Close-up of the lateral ventricles with atlas borders. Before the deformation, each section landmark (circle) sits apart from its atlas landmark (diamond) and the atlas ventricle outlines lie beside the ventricles. After it, circles and diamonds coincide and the outlines follow the ventricle walls.

Judge alignment on internal anatomy. Before the deformation (left), the
atlas ventricle outlines lie beside the ventricles in the tissue; after it
(right), they follow the ventricle walls. Circles mark section points of the
agent's landmark pairs and diamonds their atlas points. Section F1_PHAL_s097
from Kondo et al. (EBRAINS, CC BY 4.0).
```

LangSlice does not score its own results. Judge them by looking, against the
histology.
