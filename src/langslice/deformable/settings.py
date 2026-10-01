"""The few settings a deformable fit exposes, and what each named level means.

Every knob is a short named ladder rather than a raw engine number, so an agent
or a host dialog can pick "firmer" without knowing that ANTs and Elastix
regularize in different units. Physical units (millimetres) are used wherever
an engine allows it, so changing the detail level does not change how stiff a
fit is. The similarity metric and step size are fixed per engine and image
pairing (:func:`metric_for`), not exposed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast, get_args

Engine = Literal["ants", "elastix"]
#: The 2026-10-01 ceiling test dropped ``stiff`` (never the best-looking fit; it
#: left enlarged ventricles unfilled).
Stiffness = Literal["soft", "medium", "firm"]
#: ``standard`` is what the linear tool always uses. ``coarse`` looked the same
#: in the ceiling test at a fraction of the time and stays for quick previews
#: and tests; ``fine`` (10 um) was 4-5x slower with no visible gain and was
#: dropped.
Detail = Literal["coarse", "standard"]
#: ``ara``: the atlas's own BrainGlobe reference (for Allen, the ARA average
#: template). ``nissl``: ABBA's cached Allen Nissl volume (ABBA hosts only).
#: ``borders``: every region boundary from the annotation. ``borders_merged``:
#: the color-family set the image model is shown (``trace_borders``). The
#: grayscale images pair with the stain, the border images with model lines
#: (:func:`metric_for`).
AtlasImage = Literal["ara", "nissl", "borders", "borders_merged"]
#: ``stain``: the preprocessed section photograph. ``lines``: the image
#: model's extracted boundary lines on that photograph's grid.
SectionImage = Literal["stain", "lines"]
Metric = Literal["mutual_information", "mean_squares"]
#: Label-map channels added to the image pair (ANTs only). ``none``: the image
#: pair alone. ``auto``: what can be detected from the stain without a model —
#: the tissue footprint and empty interior holes near atlas ventricles, paired
#: with the placed atlas footprint and ventricular system. ``model``: the image
#: model's lines turned into named regions (:mod:`langslice.deformable.regions`),
#: paired with the placed merged atlas regions.
Labels = Literal["none", "auto", "model"]
#: Optional ANTs preprocessing of the section stain image before the fit.
Preprocess = Literal["n4", "denoise"]

BORDER_IMAGES: frozenset[str] = frozenset({"borders", "borders_merged"})
INTENSITY_IMAGES: frozenset[str] = frozenset({"ara", "nissl"})


@dataclass(frozen=True)
class AntsStiffness:
    """SyN regularization as Gaussian sigmas in millimetres.

    ANTs takes both as variances in voxels of the working grid; the adapter
    converts (variance = (sigma_mm / voxel_mm) ** 2), so a level means the
    same physical smoothness at every detail level.
    """

    update_sigma_mm: float
    total_sigma_mm: float


@dataclass(frozen=True)
class ElastixStiffness:
    """B-spline control-point spacing and bending-energy weight."""

    grid_spacing_mm: float
    bending_weight: float


#: Soft lets a region bend at roughly the scale of a large nucleus; firm keeps
#: the residual to smooth, larger-scale shape change.
ANTS_STIFFNESS: dict[str, AntsStiffness] = {
    "soft": AntsStiffness(update_sigma_mm=0.05, total_sigma_mm=0.0),
    "medium": AntsStiffness(update_sigma_mm=0.08, total_sigma_mm=0.02),
    "firm": AntsStiffness(update_sigma_mm=0.12, total_sigma_mm=0.04),
}
ELASTIX_STIFFNESS: dict[str, ElastixStiffness] = {
    "soft": ElastixStiffness(grid_spacing_mm=0.25, bending_weight=0.01),
    "medium": ElastixStiffness(grid_spacing_mm=0.4, bending_weight=0.05),
    "firm": ElastixStiffness(grid_spacing_mm=0.6, bending_weight=0.2),
}
#: Mean squares on soft line images is orders of magnitude smaller than mutual
#: information, so the same bending weight would freeze the fit. The penalty
#: is scaled by this factor whenever the metric is mean squares.
ELASTIX_MEAN_SQUARES_BENDING_SCALE = 0.02


@dataclass(frozen=True)
class DetailLevel:
    """Working resolution and optimization effort.

    The fit runs on the section resampled to *working_um* (never finer than
    the section's own pixels); the field is interpolated back to the section
    grid afterwards.
    """

    working_um: float
    ants_iterations: tuple[int, ...]
    elastix_resolutions: int
    elastix_iterations: int


DETAIL: dict[str, DetailLevel] = {
    "coarse": DetailLevel(working_um=40.0, ants_iterations=(80, 50, 25),
                          elastix_resolutions=2, elastix_iterations=300),
    "standard": DetailLevel(working_um=20.0, ants_iterations=(100, 70, 50, 25),
                            elastix_resolutions=3, elastix_iterations=500),
}
#: ANTs halves the grid at each level before the last (shrink factors
#: 2^(n-1)..1, smoothing n-1..0 voxels): the extra coarse levels are what give
#: SyN its capture range; more iterations at a level rarely change the result
#: because its convergence test stops it first.

#: Gaussian sigma, in micrometres, that turns a one-pixel line into a soft
#: ridge both images share. A wider ridge widens the capture range (how far a
#: misplaced border can be and still be pulled in) at the cost of precision.
#: Fixed since the 2026-10-01 ceiling test: 30 and 60 um looked the same,
#: 120 um slightly worse and added folds with Elastix.
LINE_SOFTENING_UM = 60.0

#: Default margin, in micrometres, around the structures a restricted
#: (sequential) fit is limited to: their own edges plus enough surroundings to
#: show where those edges should go.
DEFAULT_NEIGHBOURHOOD_UM = 300.0


@dataclass(frozen=True)
class FitSettings:
    """One candidate's choices. ``exclude`` names regions (BrainGlobe acronyms
    or ids) left out of the fit with all their descendants; an entry may name
    one side only (``"CTX:left"``, :mod:`langslice.atlas.sides`), as may a
    ``structures`` entry."""

    engine: Engine = "ants"
    stiffness: Stiffness = "medium"
    detail: Detail = "standard"
    atlas_image: AtlasImage = "ara"
    section_image: SectionImage = "stain"
    exclude: tuple[str | int, ...] = field(default_factory=tuple)
    labels: Labels = "none"
    #: Restrict this fit to these structures' neighbourhood (acronyms or ids,
    #: descendants included); empty fits the whole section. With a previous
    #: record this is one step of a sequential, region-by-region fit.
    structures: tuple[str | int, ...] = field(default_factory=tuple)
    #: Margin around *structures* the restricted fit may see and move, in um.
    neighbourhood_um: float = DEFAULT_NEIGHBOURHOOD_UM
    preprocess: tuple[Preprocess, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name, kind in (("engine", Engine), ("stiffness", Stiffness), ("detail", Detail),
                           ("atlas_image", AtlasImage), ("section_image", SectionImage),
                           ("labels", Labels)):
            value = getattr(self, name)
            if value not in get_args(kind):
                raise ValueError(f"{name} must be one of {get_args(kind)}, got {value!r}")
        if not (self.neighbourhood_um >= 0 and self.neighbourhood_um < 5000):
            raise ValueError("neighbourhood_um must be a non-negative width below 5 mm")
        for name in ("exclude", "structures", "preprocess"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        unknown = [step for step in self.preprocess if step not in get_args(Preprocess)]
        if unknown:
            raise ValueError(f"Unknown preprocessing steps: {unknown}")
        if self.preprocess and self.section_image != "stain":
            raise ValueError("Preprocessing applies to the stain image only")
        if self.labels != "none" and self.engine != "ants":
            raise ValueError(
                "Label-map channels are ANTs-only: Elastix has no multi-label metric "
                "with per-channel masks that matches ANTs' label registration"
            )
        if self.labels == "auto" and self.section_image != "stain":
            raise ValueError("Automatic labels are detected from the stain image")
        metric_for(self.section_image, self.atlas_image)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("exclude", "structures", "preprocess"):
            result[name] = list(getattr(self, name))
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FitSettings:
        data = dict(value)
        # Records saved before 2026-10-02 carry the then-adjustable softening.
        data.pop("line_softening_um", None)
        for name in ("exclude", "structures", "preprocess"):
            data[name] = tuple(data.get(name, ()))
        return cls(**cast(dict[str, Any], data))


def metric_for(section_image: str, atlas_image: str) -> Metric:
    """The fixed similarity metric for an image pairing.

    Lines against borders are the same kind of picture (soft ridges on zero),
    so intensities should simply agree: mean squares. A photograph against a
    grayscale atlas image has an unknown intensity relationship (brightfield
    tissue is dark where the template is bright): mutual information. The
    crossed pairings are refused: model lines against a grayscale image share
    no structure, and a stain against atlas borders is the pairing the
    2026-10-01 ceiling test found worst (it stayed at the linear placement and
    left enlarged ventricles unfilled) — borders are for traced lines.
    """
    if section_image == "lines":
        if atlas_image not in BORDER_IMAGES:
            raise ValueError("Model lines can only be fitted against a borders atlas image")
        return "mean_squares"
    if atlas_image in BORDER_IMAGES:
        raise ValueError("Atlas borders are fitted against traced lines only; fit a stain "
                         "against a grayscale atlas image (ara or nissl)")
    return "mutual_information"
