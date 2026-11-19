"""The few settings a deformable fit exposes, and what each named level means.

Every knob is a short named ladder rather than a raw engine number, so an agent
or a host dialog can pick "firmer" without knowing that FireANTs and Elastix
regularize in different units. Physical units (millimetres) are used wherever
an engine allows it, so changing the detail level does not change how stiff a
fit is. The similarity metric follows from the image pairing, the engine and,
for stain fits, ``stain_metric`` (:func:`metric_for`); the step size is fixed.
The smoothness of the warp is one number in micrometres (``smoothness_um``).

Every distance here is the adult mouse brain's. A fit multiplies them by the
atlas's brain scale (``core.atlas.brain_size.brain_scale``: 1.0 for the adult
CCFv3 brain, about 1.7 for the Waxholm rat, 0.7 for DeMBA P4), working pixels
included, so a rat section is fitted as a mouse section drawn at a smaller
scale: the same number of working pixels across the brain, the same smoothing,
margins and capture range relative to its anatomy.
"""

from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field
from typing import Any, Literal, cast, get_args

Engine = Literal["fireants", "elastix"]
Detail = Literal["coarse", "standard", "fine"]
AtlasImage = str
SectionImage = Literal["stain", "lines"]
Metric = Literal["local_correlation", "mutual_information", "mean_squares"]
StainMetric = Literal["local_correlation", "mutual_information"]
Labels = Literal["none", "model"]

BORDER_IMAGES: frozenset[str] = frozenset({"borders", "borders_merged"})


DEFAULT_SMOOTHNESS_UM = 80.0
TOTAL_SIGMA_FRACTION = 0.25
SMOOTHNESS_BOUNDS_UM = (5.0, 2000.0)


@dataclass(frozen=True)
class DetailLevel:
    """Working resolution and optimization effort.

    The fit runs on the section resampled to *working_um* times the brain
    scale (never finer than the section's own pixels); the field is
    interpolated back to the section grid afterwards.
    """

    working_um: float
    iterations: tuple[int, ...]


DETAIL: dict[str, DetailLevel] = {
    "coarse": DetailLevel(working_um=40.0, iterations=(80, 50, 25)),
    "standard": DetailLevel(working_um=20.0, iterations=(100, 70, 50, 25)),
    "fine": DetailLevel(working_um=10.0, iterations=(100, 70, 50, 25, 10)),
}

LINE_SOFTENING_UM = 60.0

CORRELATION_RADIUS_UM = 80.0
ELASTIX_STAIN_METRIC: Metric = "mutual_information"
EDGE_SIGMA_UM = 30.0
EDGE_CHANNEL_WEIGHT = 1.0

DEFAULT_NEIGHBOURHOOD_UM = 300.0


@dataclass(frozen=True)
class FitSettings:
    """One candidate's choices. ``exclude`` names regions (BrainGlobe acronyms
    or ids) left out of the fit with all their descendants; an entry may name
    one side only (``"CTX:left"``, :mod:`langslice.core.atlas.sides`), as may a
    ``structures`` entry."""

    engine: Engine = "fireants"
    smoothness_um: float = DEFAULT_SMOOTHNESS_UM
    detail: Detail = "standard"
    atlas_image: AtlasImage = "nissl"
    section_image: SectionImage = "stain"
    exclude: tuple[str | int, ...] = field(default_factory=tuple)
    labels: Labels = "none"
    structures: tuple[str | int, ...] = field(default_factory=tuple)
    neighbourhood_um: float = DEFAULT_NEIGHBOURHOOD_UM
    stain_metric: StainMetric = "mutual_information"
    stain_edges: bool = True
    stiffness: InitVar[Any] = None

    def __post_init__(self, stiffness: Any = None) -> None:
        if stiffness is not None:
            raise ValueError(
                "stiffness is gone: give smoothness_um, the warp's smoothness in micrometres"
            )
        if not (
            isinstance(self.smoothness_um, (int, float))
            and SMOOTHNESS_BOUNDS_UM[0] <= self.smoothness_um <= SMOOTHNESS_BOUNDS_UM[1]
        ):
            raise ValueError(
                "smoothness_um must be a number of micrometres between "
                f"{SMOOTHNESS_BOUNDS_UM[0]:g} and {SMOOTHNESS_BOUNDS_UM[1]:g}"
            )
        for name, kind in (
            ("engine", Engine),
            ("detail", Detail),
            ("section_image", SectionImage),
            ("labels", Labels),
            ("stain_metric", StainMetric),
        ):
            value = getattr(self, name)
            if value not in get_args(kind):
                raise ValueError(f"{name} must be one of {get_args(kind)}, got {value!r}")
        if not isinstance(self.atlas_image, str) or not self.atlas_image:
            raise ValueError(f"atlas_image must name an atlas image, got {self.atlas_image!r}")
        if not (self.neighbourhood_um >= 0 and self.neighbourhood_um < 5000):
            raise ValueError("neighbourhood_um must be a non-negative width below 5 mm")
        for name in ("exclude", "structures"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.labels != "none" and self.engine != "fireants":
            raise ValueError(
                "Label-map channels are SyN-only: Elastix has no multi-label metric "
                "with per-channel masks that matches the label-map channels"
            )
        metric_for(self.section_image, self.atlas_image)

    @property
    def metric(self) -> Metric:
        """The similarity metric this setting fits with (:func:`metric_for`)."""
        return metric_for(
            self.section_image, self.atlas_image, engine=self.engine, stain_metric=self.stain_metric
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("exclude", "structures"):
            result[name] = list(getattr(self, name))
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FitSettings:
        """Rebuild from :meth:`to_dict` output; ``ValueError`` naming any unknown key."""
        data = dict(value)
        legacy = data.pop("stiffness", None)
        if legacy is not None and "smoothness_um" not in data:
            data["smoothness_um"] = {"soft": 50.0, "medium": 80.0, "firm": 120.0}.get(
                legacy, DEFAULT_SMOOTHNESS_UM
            )
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown fit setting key(s) {unknown}")
        for name in ("exclude", "structures"):
            data[name] = tuple(data.get(name, ()))
        return cls(**cast(dict[str, Any], data))


def neighbourhood_um_for(scale: float) -> float:
    """The default neighbourhood of a restricted fit at brain *scale*."""
    return DEFAULT_NEIGHBOURHOOD_UM * scale


def metric_for(
    section_image: str,
    atlas_image: str,
    *,
    engine: str = "fireants",
    stain_metric: str = "mutual_information",
) -> Metric:
    """The similarity metric for an image pairing, engine and stain metric.

    Lines against borders are the same kind of picture (soft ridges on zero),
    so intensities should simply agree: mean squares. A photograph against a
    grayscale atlas image has an unknown intensity relationship (brightfield
    tissue is dark where the template is bright, and stains differ region by
    region): mutual information over the section by default, which asks only
    that intensities be predictable from each other, or local
    correlation, which asks that they agree up to a scale and offset within
    each small window. Elastix has no local correlation metric, so its stain
    fits use :data:`ELASTIX_STAIN_METRIC`. The crossed pairings are refused: model
    lines against a grayscale image share no structure, and a stain against
    atlas borders is the worst pairing (it stays at the linear placement and
    leaves enlarged ventricles unfilled): borders are for traced lines.
    """
    if section_image == "lines":
        if atlas_image not in BORDER_IMAGES:
            raise ValueError("Model lines can only be fitted against a borders atlas image")
        return "mean_squares"
    if atlas_image in BORDER_IMAGES:
        raise ValueError(
            "Atlas borders are fitted against traced lines only; fit a stain "
            "against a grayscale atlas image"
        )
    if stain_metric == "mutual_information":
        return "mutual_information"
    return "local_correlation" if engine == "fireants" else ELASTIX_STAIN_METRIC
