# Refine an existing registration from QuickNII, VisuAlign or ABBA

If you already placed your sections in another program, LangSlice can start
from that placement instead of from nothing. By default it keeps your
placement exactly and adds a nonlinear warp per section.

## What LangSlice can read

| Made with | File to give |
|---|---|
| QuickNII | The series file, JSON or XML |
| VisuAlign | The series file, JSON |
| DeepSlice | Its prediction file, CSV, JSON or XML |
| ABBA | The JSON file of ABBA's QuickNII export (below) |
| An earlier LangSlice job | Its `registration.json` |

LangSlice reads the linear placement only: each section's position, cutting
angles, mirroring and in-plane alignment. **VisuAlign's markers are not
read.** Your VisuAlign adjustments are left out, and LangSlice says so when
it reads the file. The agent fits its own warp.

## Steps

1. Put the section images in one folder, as in
   [Prepare your images](prepare-images.md). They can be the images you
   registered, or larger versions of them: the placement is carried over by
   fractions of each image's width and height.
2. Tell your agent where the registration file is, and what to do with it.
   For example:

   ```text
   Register the brain in /data/brains/M04, starting from my QuickNII file
   /data/brains/M04/quicknii_series.json. Keep my placement and add a
   nonlinear warp to each section.
   ```

3. Your agent creates the job from the file and tells you what it found.
   The report lists the sections placed, the entries in the file that
   matched no image, the images with no entry, and any warnings. Read this before the run
   starts. Every section needs a placement before the job can be submitted.
4. The registration agent then fits a warp to each section.
5. Check the result as in [Check and correct a registration](check-and-correct.md).

### Let the agent revise your placement

By default your placement is fixed: the agent may not move, turn or realign
any section. To let it improve the placement too, say so:

```text
Start from my QuickNII file, but you may correct positions and the in-plane
alignment before warping.
```

The agent then starts from your placement and may change it.

## How files are matched to images

Each entry in the registration file is matched to one image, by the first
rule that finds one:

1. the same filename;
2. the same name without its ending, ignoring capitals (so `s01.png` matches
   `S01.tif`);
3. the same QuickNII section number, `_s` followed by digits (`brain_s012`).

An entry that matches two images, or two entries that match one image, stop
the import, and your agent tells you which.

## Pixel size

LangSlice needs each image's pixel size, as for any job. When neither you
nor the files give one, it uses the size the imported placement implies.
Tell your agent the pixel size when you know it.

## When an import is refused

The whole import stops, and no job is created, when:

- the file was made for another atlas than the job's;
- the file places no section at all.

A single section is refused, and listed in the report, when its plane is
tilted more than 45° from the job's cutting plane. It then has no placement,
which must be resolved before the job can be submitted.

A registration file cannot be combined with positions, angles or
alignments given another way.

## From ABBA

LangSlice reads ABBA's registrations through ABBA's QuickNII export:

1. In ABBA, select the sections and run **Export ‣ ABBA - Export registered
   slices as QuickNII dataset**. Choose an output folder and a pixel size.
2. ABBA writes one image per section (`Section_s000.jpg`, ...) and a series
   file (`Section.json`) into that folder.
3. Give your agent that folder as the image folder and `Section.json` as the
   registration file. Tell it the pixel size you chose in ABBA.

:::{note}
ABBA's export resamples each section with all of its ABBA registrations
applied, including any warp, and saves 8-bit images at the pixel size you
chose (40 µm by default). LangSlice then works on those exported images, not
on your original scans, and its results describe the exported images.
:::

Inside ABBA, the LangSlice Registration dialog also starts from the
sections' existing ABBA registrations. See [ABBA](../abba.md).

<details>
<summary>Under the hood: the job settings</summary>

Your agent runs `langslice-job FOLDER init --registration FILE`. With
`--registration` and no `--tasks`, the only task is `nonlinear`, so the
imported positions, angles, mirroring and alignment are fixed on every
section. `--tasks position,transform,nonlinear` lets the agent revise them.
`--registration` cannot be combined with `--positions`, `--transforms`,
`--orientation`, `--pitch`, `--yaw`, `--section-angles` or `--angles`.
DeepSlice's CSV names no atlas; LangSlice reads it as DeepSlice's mouse
target. The import rules are in
[Job folder and file formats](../reference/file_formats.md).

</details>
