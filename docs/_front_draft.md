# LangSlice

**Give LangSlice a folder of brain-section images. An AI agent places each
section in a brain atlas and draws the atlas region borders on it. You can
then assign cells and signal to brain regions.**

% Picture: a finished LangSlice registration in which the agent chose every
% section's position and in-plane alignment (tasks position and transform,
% nothing supplied or frozen; no deformation). Made from a copy of the job
% folder of SliceBench run deepslice_gt_joint_sol61_med_20261005_ISH_Calb1,
% section 0253_71662493 (Allen ISH image 71662493, Calb1, from the DeepSlice
% ground-truth series), atlas allen_mouse_25um: `look(mode="overlay",
% sections=["0253_71662493.png"], resolution=2000)` through
% langslice.open_job on the copy, then cropped to the section (caption strip
% and scale bar off) and resized to 1200 px wide in gray. The copy's job.json
% and state.json had settings the current code no longer reads removed. The
% original run folder was not modified.

```{figure} img/front_registered_section.png
:alt: A coronal mouse section through the dorsal hippocampus with atlas region outlines drawn over it. The outlines follow the dentate gyrus and CA1 cell layers, the corpus callosum, the midline and the outer edge of the cortex.
:width: 100%

A coronal mouse section placed by LangSlice in the Allen mouse brain atlas
(CCFv3), with the atlas region borders drawn in black. The agent chose the
section's position and aligned it in-plane. Check every result the same way:
the outlines should follow the anatomy you can see in the tissue. Section
image: Allen Mouse Brain Atlas, in situ hybridization for Calb1, image
71662493, © Allen Institute for Brain Science, mouse.brain-map.org.
```

:::{admonition} Before you start
:class: important

- **Best tested on coronal sections of the Allen mouse atlas.** LangSlice
  also takes sagittal and horizontal sections and the other atlases of
  [BrainGlobe](https://brainglobe.info), which are less tested.
- **You need a computer and one AI app.** LangSlice installs on Windows, Mac
  or Linux. You talk to it through one of three AI apps: Codex (ChatGPT),
  Antigravity (Gemini) or Claude Code (Claude). No graphics card is needed.
- **Your AI provider sees your sections.** The agent looks at pictures of
  your sections, so they are sent to OpenAI, Google or Anthropic, whichever
  runs your app. If you switch on the optional image model, its provider sees
  them too. Your original image files stay on your computer, unchanged.
- **It uses your AI plan.** A registration counts against your AI plan's
  usage limits.
- **You check the result.** The agent works on its own, and its placements
  can be wrong. You review every section before you use it.
:::

## Where do you want to go?

**New to LangSlice?**
: [Install it and connect your AI app](getting-started/index.md).

**Which AI app should I use?**
: [Choose by the AI you already use](getting-started/connect.md).

**How do I check that a registration is right?**
: [How-to guides](how-to/index.md).

**How does the agent decide where a section goes?**
: [Concepts](concepts/index.md).

**What does a setting or a file mean?**
: [Reference](reference/index.md).

**Something went wrong?**
: [Troubleshooting](troubleshooting.md).

**How do I cite LangSlice, or ask for help?**
: [Cite and get help](help.md).

**Are you an AI agent installing LangSlice?**
: [For agents](agents/index.md).

**Writing your own code with LangSlice?**
: [For developers](developers/index.md).

## Cite LangSlice

Baughman, N. (2026). *LangSlice: VLM-driven registration of histological brain
sections to BrainGlobe atlases* (version 0.1.0) [Computer software].
<https://github.com/greenpolo/LangSlice>

[Cite and get help](help.md) has the citation in other formats.
