# How LangSlice registers a brain

Registering a section means finding where it sits in the atlas. That is the
atlas plane it was cut from, and where each point of the tissue lies on that
plane.
LangSlice does this for a whole stack of sections from one brain. An AI agent
does the work, with the same kinds of tools a person uses in QuickNII,
VisuAlign or ABBA. It looks at pictures of the sections and the atlas, places
each section, runs registration methods, and checks each result by looking
at it.

Your image files are never changed. Every result is stored as numbers in a
job folder beside your images (see [Your data](your-data.md)).

## The three stages

LangSlice works in the order a person would. Each stage builds on the one
before it.

| Stage | What is decided | Kind of transform |
|---|---|---|
| 1. Order and position | Where each section sits along the cutting axis, and the cutting angles of the stack | Choice of atlas plane |
| 2. In-plane alignment | How each section is turned, mirrored, scaled and shifted onto its atlas plane | Linear (affine) |
| 3. Deformation (optional) | How each part of the section is bent onto the atlas, for tissue that shrank, stretched or shifted unevenly | Nonlinear |

An *affine* transform moves the whole section at once: rotation, scaling,
shear and shift, with straight lines staying straight. A *deformation* lets
different parts of the section move by different amounts.

### 1. Order and position

The agent compares each section with atlas planes along the cutting axis and
writes its position in millimetres. Positions are measured from the anterior
edge of the atlas volume in a coronal atlas. The order of the stack follows
the positions, not the filenames. The stack shares one pair of cutting angles
(pitch and yaw) for sections cut at an angle to the atlas.

LangSlice does not guess positions from filename order or from the nominal
section interval. The agent places each section on its own anatomy. If you
know the section thickness or interval, you can supply them as a hint. The
spacing is enforced only if you ask for it.

```{figure} img/stages_positioning.jpg
:alt: Six sections lined up below a millimetre ruler, each joined by a line to its position, with atlas planes above the ruler for comparison. The sections run from olfactory bulb level at 9 mm to caudal cortex at 23 mm.

The positioning picture: atlas planes above the ruler, the sections below in
order of position, each joined to the position the agent wrote. Rat brain F1
from Kondo et al. (EBRAINS, CC BY 4.0).
```

### 2. In-plane alignment

The agent fits each section onto the atlas plane at its position. It can
mirror the section, turn it by a quarter turn, rotate, scale, shear and shift
it. It can set these values by hand or run Elastix (below) to fit them.

LangSlice draws each section at its true size when it knows the pixel size,
from the image file's metadata or from your settings. Without a pixel size,
the image's width is matched to the atlas plane's width, and the pictures say
`uncalibrated`.

### 3. Deformation

A linear placement cannot follow tissue that was distorted unevenly when it was
cut, mounted or dried. A deformation corrects this, section by section, on
top of the linear placement. The agent gives a section a deformation only
where it improves on the linear placement. A deformation is fitted to one placement. Moving the section to a new
position or new cutting angles, or changing its in-plane alignment, removes
it.

```{figure} img/stages_overlays.jpg
:alt: The same rat coronal section twice, with atlas borders drawn on it. On the left, after in-plane alignment, the atlas outlines of the lateral ventricles sit beside the ventricles in the tissue. On the right, after the landmark deformation, they sit on them.

One section after in-plane alignment (left) and after a deformation built from
four landmark pairs (right). Black lines are the atlas borders. The shaded
patches at the top are regions the agent marked as damaged. Atlas outlines
below the tissue are regions this section does not hold. Section F1_PHAL_s097
from Kondo et al. (EBRAINS, CC BY 4.0); in this run the positions came with
the dataset and the agent did the alignment and the deformation.
```

## Which stages a run does

You choose the stages when you set up a job. By default LangSlice does stages
1 and 2. A stage you switch off takes its answer from what you supply, such
as positions from an earlier registration. A deformation needs a linear
placement first, so stage 3 needs either stage 2 or a supplied placement.
You can also lock any of these values so the agent cannot change them (see
[What the AI decides](what-the-ai-decides.md#what-you-can-lock)).

## The methods the agent can use

The agent chooses which method to use on each section, and with which
settings. There is no fixed recipe. Each method runs on your computer except
image-model tracing, which sends pictures to an image model.

### Placing by hand

The agent sets positions, cutting angles and the in-plane transform directly,
as you would drag a section in QuickNII or ABBA. It reads the result off a
picture of the section with the atlas borders drawn on it, called an
*overlay*.

### Elastix affine

[Elastix](https://elastix.dev/) finds the affine transform that makes the
section image most similar to an atlas image. By default the atlas image is
a Nissl image of the atlas. Elastix refines a placement that is already
close; it does not fix a section that is far off. The agent can ask it to
fit to selected regions only, such as the hippocampus.

<details>
<summary>Under the hood</summary>

The similarity measure is Mattes mutual information. Mutual information
rewards images whose intensities predict each other, so a section does not
need the same contrast as the atlas image. The edge channel adds the outlines
of both images to the comparison. LangSlice warns the agent when a fit
changes the scale or shear by a large amount. A section that holds only part
of the brain can be stretched over atlas it does not hold. Full
options: [Tools and options](../reference/tools.md).

</details>

### FireANTs SyN

[FireANTs](https://github.com/rohitrango/FireANTs) fits a smooth deformation
that makes the section image more similar to the atlas image. It uses SyN
(symmetric normalization), the deformable method of the ANTs package. It runs
on the graphics card (GPU) when the computer has one, and on the processor
otherwise. Like Elastix, it refines a placement that is already close.

<details>
<summary>Under the hood</summary>

SyN is built to give a diffeomorphic deformation: smooth and invertible.
LangSlice runs it with mutual information plus an edge channel, from a
coarse level to a fine one. The agent sets the smoothness in micrometres.
It can restrict a fit to selected regions, which then moves only those
regions and a margin around them. The reply reports how far the tissue moved
and any folding, and flags implausible fits. It gives no score of how well
it matches. For a
rat or a developing mouse brain, every distance is scaled by the size of the
atlas brain relative to the adult mouse. Full options:
[Tools and options](../reference/tools.md).

</details>

### Landmarks and triangulation

The agent marks pairs of matching points: a point on the atlas and the point
in the tissue where that anatomy actually lies. LangSlice then warps the
section so that each pair meets, the way VisuAlign does. This method moves
tissue that lies far from its place, which FireANTs cannot reach. The agent
is told to move such tissue with landmarks first and then refine it with
FireANTs, which keeps the landmark move.

<details>
<summary>Under the hood</summary>

The warp is a Delaunay triangulation of the pairs' section points, plus four
fixed points a little outside the image corners, as VisuAlign builds it. Each
triangle is moved by its own affine transform. A pair therefore moves the
tissue out to its neighbouring pairs, so tissue that is already right needs
pairs too. A warp needs at least three pairs. Where triangles fold over, the
warp is kept, as VisuAlign keeps it, and the reply reports the folded area.

</details>

### Image-model border tracing

This method is optional. The agent sends an image model two pictures of a
section. One is the section alone; the other has the atlas borders drawn on
it in yellow. The model redraws the borders where it sees them in the
tissue. LangSlice reads the redrawn lines back and fits a FireANTs
deformation that moves the atlas onto them. The agent decides which sections,
if any, to trace. The method is offered only when an image model is
connected; [Your data](your-data.md#image-model-tracing) says which ones and
what they receive.

An image model can redraw the tissue itself as well as the borders, and a fit
can spoil a useful trace. The agent gets back two pictures: the traced lines
on the section and the fitted borders. The job folder keeps every prompt
sent, the model's raw reply, the extracted lines and the fit, so you can
check each one.

## What comes out

At the end the agent calls `submit`. LangSlice checks that every section has
what the job asked for: a position and an in-plane transform for every
section, damaged ones included. It does not check whether they are right;
that is what the pictures are for. It then writes QuickNII and VisuAlign
files, which open in those programs. Per-section maps of atlas coordinates
and region labels are written on request. The formats are in
[Job folder and file formats](../reference/file_formats.md).

## What it does not do

- It registers 2D sections. It does not register a 3D volume as a volume.
- It works with BrainGlobe atlases in the coronal, sagittal and horizontal
  planes. Coronal sections and the Allen mouse atlas are the most tested
  case.
- It does not tell left from right on a nearly symmetric section by itself.
  If something marks one hemisphere, such as an injection site or a notch,
  tell LangSlice what it is.
- It cannot register anatomy that is missing from the section. The agent
  marks missing or folded regions as damaged, and every fit leaves them out.
