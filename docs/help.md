# Cite and get help

## Cite LangSlice

If you use LangSlice in your research, cite it as:

> Baughman, N. (2026). *LangSlice: VLM-driven registration of histological
> brain sections to BrainGlobe atlases* (version 0.1.0) [Computer software].
> <https://github.com/greenpolo/LangSlice>

For BibTeX:

```bibtex
@software{baughman_langslice_2026,
  author  = {Baughman, Nash},
  title   = {LangSlice: VLM-driven registration of histological brain sections to BrainGlobe atlases},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/greenpolo/LangSlice},
  license = {BSD-3-Clause}
}
```

The same record is in the file `CITATION.cff` at the top of the LangSlice
folder. On the [GitHub page](https://github.com/greenpolo/LangSlice), the
**Cite this repository** button gives it in APA and BibTeX.

Cite the version you used: `langslice version` prints it.

### Also cite

- **The atlas.** Each BrainGlobe atlas has its own citation, listed in
  [BrainGlobe's atlas documentation](https://brainglobe.info). For the
  default atlas, `allen_mouse_25um`, that is the Allen Mouse Brain Common
  Coordinate Framework (CCFv3).
- **The registration engines your registration used.** In-plane fits use
  [Elastix](https://github.com/SuperElastix/elastix). Deformations use
  [FireANTs](https://github.com/rohitrango/FireANTs).

### For your methods section

Say which AI app and model ran the registration, what the agent was allowed
to change, and how you checked the result. A starting point to adapt:

> Sections were registered to the [atlas name, resolution] atlas with
> LangSlice [version] (Baughman, 2026), run from [app] with [model]. An AI
> agent set each section's position along the [anterior–posterior] axis, its
> cutting angles and its in-plane alignment[, and fitted a deformation to
> each section]. [Known values supplied to the agent, if any.] Every section's
> registration was inspected against the histology. [How many sections were
> corrected, and how.]

## Get help

### Check the guides first

- [Troubleshooting](troubleshooting.md) lists common problems by the message
  or symptom you see.
- [Getting started](getting-started/index.md) walks through installing and
  connecting your AI app.

### Report a problem on GitHub

Open an issue at <https://github.com/greenpolo/LangSlice/issues>. You need a
free GitHub account. Include:

- the LangSlice version (`langslice version`);
- your operating system (Windows, macOS or Linux, and its version);
- which AI app you use;
- what you did, step by step, and what happened;
- the full message, copied as text.

GitHub issues are public. Leave out images or details you have not
published.

### Ask the community

The [image.sc forum](https://forum.image.sc) is where users of image analysis
tools such as ABBA, QuPath and BrainGlobe ask questions. It suits general
questions about registering sections and using the results.
