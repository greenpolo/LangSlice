"""A section's in-plane affine, as the stack stores it.

The affine is stored as six normalized numbers ``[a, b, tx, c, d, ty]``
(x as a fraction of the image width, y of its height, so the same numbers
apply at any resolution: :mod:`langslice.core.geometry.affine`), on the
section as the run sees it, after its quarter turn and flip. Its readable
form is a :class:`PhysicalAffine`: the knobs the alignment tools take.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langslice.core.stack.part import Flag, Number, Object, Part, Point, omitted


class PhysicalAffine(Part):
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

    Other keys a host's knobs carried are kept as they came.
    """

    rotation_deg: Number | None = omitted()
    scale_x: Number | None = omitted()
    scale_y: Number | None = omitted()
    translate_x_mm: Number | None = omitted()
    translate_y_mm: Number | None = omitted()
    shear: Number | None = omitted()
    pivot: Point | None = omitted()


IDENTITY_PHYSICAL = PhysicalAffine(
    rotation_deg=0.0,
    scale_x=1.0,
    scale_y=1.0,
    translate_x_mm=0.0,
    translate_y_mm=0.0,
    shear=0.0,
    pivot=(0.5, 0.5),
)


class Calibration(Part):
    """Hold the section's scale an affine's millimetres are measured at.

    Parameters
    ----------
    section_um_per_px
        Micrometres per pixel of the section's working frame; None for an
        imported placement of a section without a pixel size.
    source
        Where that scale came from (the supplied pixel size, an estimate).
    """

    section_um_per_px: Number | None = None
    source: str


class FitRegions(Part):
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


class StoredAffine(Part):
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
    """

    kind: str | None = omitted()
    params: tuple[Number, ...]
    physical: PhysicalAffine | None = omitted()
    calibration: Calibration | None = omitted()
    mirrored: Flag | None = omitted()
    note: str | None = omitted()
    in_plane: Object | None = omitted()
    regions: FitRegions | None = omitted()
    fit_atlas: str | None = omitted()
    orientation: Object | None = omitted()
    stale: Flag | None = omitted()

    def arranged(self, data: dict[str, Any]) -> dict[str, Any]:
        """Write an imported affine's reading on the render and ``mirrored``
        before its calibration; every other one writes the calibration first."""
        if self.in_plane is None:
            return data
        first = ("kind", "params", "physical", "in_plane", "mirrored", "calibration")
        placed = {key: data[key] for key in first if key in data}
        return {**placed, **{key: value for key, value in data.items() if key not in placed}}


def affine_fields(affine: StoredAffine | None, keys: Sequence[str]) -> dict[str, Any]:
    """Return some keys of a stored affine's JSON form, null where unset.

    What a digest of a placement hashes (its six numbers, its calibration).
    """
    form = {} if affine is None else affine.to_json()
    return {key: form.get(key) for key in keys}
