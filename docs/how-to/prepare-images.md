# Prepare your images

LangSlice registers one folder of section images, one brain per folder. This
page says which files it reads, how it treats channels, and what to tell it
about pixel size and section order.

## Put one brain in one folder

1. Make a folder for the brain, for example `M04`.
2. Put one image file per section in it. Each file is one section.
3. Leave other images out of the folder. LangSlice reads every image file it
   finds there as a section, including overviews and slide labels.

LangSlice reads files with these endings:

| Ending | Notes |
|---|---|
| `.tif`, `.tiff` | Plain, tiled and pyramidal (whole-slide) TIFF and OME-TIFF |
| `.png` | |
| `.jpg`, `.jpeg` | |

It does not open microscope formats such as `.czi`, `.vsi`, `.ndpi` or
`.svs`. Export them to TIFF first, for example with Fiji (**File ‣ Save As ‣
Tiff…**) or with the microscope's own software.

A slide scan that holds several sections must be cut into one file per
section first.

:::{warning}
A TIFF saved with LZW compression does not open with the standard
installation. The job is created, but the next step fails with the message `<COMPRESSION.LZW: 5> requires the 'imagecodecs' package`.
Save the images uncompressed, or ask your agent to install the
`imagecodecs` package into the LangSlice environment.
:::

LangSlice never changes your image files. It writes its work into a new
folder, `langslice`, inside the image folder. When it cannot write there, it
uses a folder in your home folder (`~/.langslice/jobs/`) instead.

<details>
<summary>Under the hood: image size</summary>

Large images are fine. LangSlice works on a smaller copy of each section.
For a pyramidal TIFF, that is the smallest level at least 1536 pixels long.
Other files are shrunk to at most 3072 pixels on their long side. Close-up views ("zooms")
read the original file at full resolution. The maps you
[export](export-results.md) can be written at full resolution too.

</details>

## Keep the channels that show anatomy

Every channel in the file reaches the agent:

- A colour file (RGB) gives three channels, `red`, `green` and `blue`.
  When the three are identical, it gives one, `gray`.
- A multi-page TIFF gives one channel per page, named `ch1`, `ch2` and so on.

When the job starts, LangSlice blends the channels into one picture, called
`preprocessed`. Each channel is weighted by how much tissue it covers, and
local contrast is enhanced. The agent looks at this picture first and fits
the atlas to it. It can also look at each channel alone and make its own
blends.

A channel with sparse signal, such as a tracer or a reporter, shows little
of the anatomy the atlas draws. Keep a counterstain (DAPI, Nissl, NeuN) or
another channel that shows the cytoarchitecture in the file.

## Know the pixel size

The agent draws each section at its true size beside the atlas, so it needs
the pixel size in micrometres. LangSlice reads it from the TIFF file when the
file records it (OME-TIFF and TIFFs saved by Fiji usually do). PNG and JPEG
files never carry it.

To check a TIFF, open it in Fiji and choose **Image ‣ Properties…**. The
**Pixel width** is the pixel size when the unit is µm or microns.

When the files don't carry it, tell your agent the pixel size, for example
"the images are 0.65 µm per pixel". Different sizes per file are fine.

Without a pixel size, LangSlice stretches each section's width to the width
of the atlas section. Tissue that is not a whole section, such as one
hemisphere, is then drawn at the wrong size. The job's files mark such a
section `uncalibrated`.

## Name the files in cutting order

LangSlice lists the files in natural order: numbers in names are compared as
numbers, so `s2.tif` comes before `s10.tif`. This is the order of the first
pictures the agent sees.

The agent places every section by its anatomy. It never takes a position
from the file order or from a nominal interval. Positions decide the final
order. The picture of the stack labels each section with its filename, so a
file order that disagrees with the anatomy shows there.

Sections are known by filename everywhere: in the pictures, in the agent's
replies and in the results. Name a section by its filename when you talk
about it, for example "M04_s07.tif is torn", never "section 7".

When you know the cutting order and the files are not named in it, tell your
agent. See [Tell the agent what you already know](tell-the-agent.md).

## What next

- [Tell the agent what you already know and what it must not change](tell-the-agent.md)
- The full list of job settings: [Job settings](../reference/settings.md)
