# LangSlice

```{image} ../assets/logo/banner_light.svg
:alt: LangSlice
:class: only-light langslice-banner
```

```{image} ../assets/logo/banner_dark.svg
:alt: LangSlice
:class: only-dark langslice-banner
```

**Give LangSlice a folder of brain-section images. An AI agent places each
section in a brain atlas and draws the atlas region borders on it. You can
then assign cells and signal to brain regions.**

% Picture: one LangSlice registration typed in the logo's characters (built by
% figure-factory/projects/langslice-readme/scripts/build_ascii_card.py, which
% records the steps). Section M04_C_07 of the Walsh lab LSD_910 series (DAPI),
% atlas allen_mouse_25um: the agent's placement from a 2026-10-08 LangSlice run
% plus a FireANTs SyN fit (detail fine, other settings default) run through
% langslice.open_job on a copy of that job; labels from job.section_maps. Left
% of the atlas midline the DAPI image as character density; right of it the
% atlas regions, borders in pink, acronyms written in.

```{figure} img/front_ascii_section.png
:alt: A coronal mouse brain section drawn in text characters. The left hemisphere shows the tissue as character density in lavender; the right hemisphere is tinted by atlas region, with pink strokes along region borders and region abbreviations such as RSP, CA1, DG, CA3, TH and HY written into the regions.
:width: 100%

One coronal section of a mouse brain, typed out in text. Left: the section's
DAPI image. Right: the same section as LangSlice registered it to the Allen
mouse brain atlas (CCFv3), with the region borders in pink and the regions
named.
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
: [Check and correct a registration](how-to/check-and-correct.md).

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

```{toctree}
:hidden:

Getting started <getting-started/index>
How-to guides <how-to/index>
ABBA <abba>
Concepts <concepts/index>
Reference <reference/index>
Troubleshooting <troubleshooting>
Cite and get help <help>
For agents <agents/index>
For developers <developers/index>
```
