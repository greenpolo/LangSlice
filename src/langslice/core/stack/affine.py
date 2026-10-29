"""A section's in-plane affine, as the stack stores it.

The affine is stored as six normalized numbers ``[a, b, tx, c, d, ty]``
(x as a fraction of the image width, y of its height, so the same numbers
apply at any resolution: :mod:`langslice.core.geometry.affine`), on the
section as the run sees it, after its quarter turn and flip. Its readable
form is a :class:`PhysicalAffine`: the knobs the alignment tools take.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langslice.core.jsonable import JsonObject, empty_object


@dataclass(frozen=True)
class PhysicalAffine:
    """Hold an affine as rotation, two scales, shear and shift about a pivot.

    The five knobs the alignment tools take, plus the shear an affine can
    have and they cannot, about a pivot given as fractions of the canvas.
    Every affine LangSlice makes gives all of them; a host's transform may
    give some, and a knob it leaves out is None.

    Parameters
    ----------
    rotation_deg
        Rotation in degrees, counter-clockwise on the picture.
    scale_x
        Scale along the canvas's x axis, 1 for none.
    scale_y
        Scale along the canvas's y axis, 1 for none.
    translate_x_mm
        Shift along x in millimetres of the section.
    translate_y_mm
        Shift along y in millimetres of the section.
    shear
        Shear factor the knobs cannot set, 0 for none.
    pivot
        The rotation and scale centre as ``(x, y)`` canvas fractions.
    extra
        Other keys a host's knobs carried, kept as they came.
    """

    rotation_deg: float | None = None
    scale_x: float | None = None
    scale_y: float | None = None
    translate_x_mm: float | None = None
    translate_y_mm: float | None = None
    shear: float | None = None
    pivot: tuple[float, float] | None = None
    extra: JsonObject = field(default_factory=empty_object)


#: The identity's knobs about the canvas centre: what a section starts from.
IDENTITY_PHYSICAL = PhysicalAffine(
    rotation_deg=0.0,
    scale_x=1.0,
    scale_y=1.0,
    translate_x_mm=0.0,
    translate_y_mm=0.0,
    shear=0.0,
    pivot=(0.5, 0.5),
)


@dataclass(frozen=True)
class Calibration:
    """Hold the section's scale an affine's millimetres are measured at.

    Parameters
    ----------
    section_um_per_px
        Micrometres per pixel of the section's working frame; None for an
        imported placement of a section without a pixel size.
    source
        Where that scale came from (the supplied pixel size, an estimate).
    """

    section_um_per_px: float | None
    source: str


@dataclass(frozen=True)
class FitRegions:
    """Hold the atlas regions an affine fit was restricted to.

    Parameters
    ----------
    include
        Region entries the fit used alone; empty for all.
    exclude
        Region entries the fit left out.
    """

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoredAffine:
    """Hold a section's in-plane affine, whatever made it.

    Parameters
    ----------
    kind
        What made it: ``elastix``, ``interactive``, ``host``, ``imported``,
        a plugin's or a script's label; None for a host's transform that
        does not say.
    params
        The six normalized numbers ``[a, b, tx, c, d, ty]``.
    physical
        The same affine as knobs; None when it was supplied as numbers only.
    calibration
        The scale the knobs' millimetres are measured at; None without
        knobs.
    mirrored
        Whether the affine's 2 x 2 part has a negative determinant; None
        when the maker did not say.
    note
        The maker's note (an interactive or a plugin's transform); None
        when it wrote none.
    in_plane
        An imported placement's six numbers read on its render (rotation,
        scales, shear); None for every other kind.
    regions
        The regions an Elastix fit was restricted to; None when it used
        all.
    fit_atlas
        The atlas image an Elastix fit read, when not the template.
    orientation
        A host's record of the orientation its affine was made at, which
        must match the section's (``flip``, ``rotation_deg``).
    stale
        A host's mark that the affine no longer holds.
    extra
        Other keys the stored affine carried (a host's, an earlier
        LangSlice's), kept as they came so the state file reads back the
        same.
    """

    kind: str | None
    params: tuple[float, ...]
    physical: PhysicalAffine | None = None
    calibration: Calibration | None = None
    mirrored: bool | None = None
    note: str | None = None
    in_plane: JsonObject | None = None
    regions: FitRegions | None = None
    fit_atlas: str | None = None
    orientation: JsonObject | None = None
    stale: bool | None = None
    extra: JsonObject = field(default_factory=empty_object)
