"""One section's state: order, orientation, position, angles, registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Quarter-turns the render applies before the flip.
ROTATIONS = (0, 90, 180, 270)

#: The five physical knobs of the transform every section starts from, and the
#: B side of an A/B preview when a section carries nothing yet. (The identity's
#: six normalized numbers are ``core.geometry.affine.IDENTITY_PARAMS``.)
IDENTITY_KNOBS: dict[str, float] = {
    "rotation_deg": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "translate_x_mm": 0.0,
    "translate_y_mm": 0.0,
}

#: A plane's cutting angles, ``(pitch_deg, yaw_deg)``.
Angles = tuple[float, float]

#: The key cutting angles are stored under, on a section and (in the
#: serialized state, when every section shares one) on the stack.
ANGLES_KEY = "cutting_angles_deg"


def flat_angles() -> dict[str, float]:
    """The flat plane, ``{"pitch": 0.0, "yaw": 0.0}`` (a new dict)."""
    return {"pitch": 0.0, "yaw": 0.0}


def angles_tuple(angles: dict[str, Any] | None) -> Angles:
    """``(pitch, yaw)`` of a stored angles dict (missing parts are 0)."""
    angles = angles or {}
    return float(angles.get("pitch", 0.0)), float(angles.get("yaw", 0.0))


def serialized_mixed_angles(data: dict[str, Any]) -> bool:
    """Whether a serialized state (:meth:`StackState.to_dict`, a checkpoint's
    fields at format 3) has sections whose cutting angles differ: the stack's
    ``cutting_angles_deg`` is written null exactly then."""
    return ANGLES_KEY in data and data[ANGLES_KEY] is None


class MixedAngles(ValueError):
    """The stack's sections have different cutting angles, so there is no
    one stack-wide angle to read (:attr:`StackState.stack_angles`)."""


@dataclass
class SliceState:
    """Everything the run knows or proposes about one section image.

    ``transform`` is one dict, whatever produced it::

        {"kind": "elastix" | "interactive" | the host's,
         "params": [a, b, tx, c, d, ty],   # normalized 2x3, see
         langslice.core.geometry.affine "physical": {rotation_deg, scale_x,
         scale_y, shear,
                      translate_x_mm, translate_y_mm, pivot},
         "calibration": {section_um_per_px, source},
         "mirrored": bool,                 # det of the 2x2 is negative
         "note": str}                      # interactive only

    For an affine, ``physical`` is the ONE representation it carries, whatever
    made it: the five knobs the alignment tools take (plus the ``shear`` an
    affine can have and they cannot), about a pivot given as canvas
    fractions.

    Coordinates are NORMALIZED — x as a fraction of image width, y as a
    fraction of image height — so the same six numbers apply at any resolution
    (:func:`langslice.core.geometry.affine.normalized_affine`). The transform
    describes the section as the run sees it, i.e. AFTER ``rotation_deg`` and
    ``flip``.
    """

    id: str
    index_original: int
    index_corrected: int
    #: Mirror left-right. Applied AFTER ``rotation_deg``.
    flip: bool = False
    #: Quarter-turn applied before the flip; one of :data:`ROTATIONS`.
    rotation_deg: int = 0
    #: The atlas regions whose tissue this section is missing, or has folded
    #: over other anatomy (``mark_damage``): acronyms or ids, each optionally
    #: one side (``"CTX:left"``, :mod:`langslice.core.atlas.sides`). Every fit
    #: and the image model's trace leave them out
    #: (:func:`langslice.core.sections.damage.exclusions`).
    damaged_regions: list[str] = field(default_factory=list)
    #: What is wrong with the tissue, in words: the agent's ``mark_damage``
    #: note, after the host's ``inputs.damaged`` note (``{filename: note}``)
    #: when it gave one. A note alone does not make the section damaged.
    damage_note: str = ""
    position_mm: float | None = None
    transform: dict[str, Any] | None = None
    #: Raw image-model correction and artifacts; does not replace the transform.
    image_correction: dict[str, Any] | None = None
    #: The applied deformable fit (``fit_deformable``): a reference to its
    #: record on disk (``record``: the ``DeformableRecord`` directory, parent
    #: steps inside it) plus a summary. It lives on top of the linear
    #: placement it was fitted at (``linear_key``); any change to position,
    #: orientation, cutting angles or transform clears it.
    deformation: dict[str, Any] | None = None
    #: Authoritative stable landmark pairs and the next never-reused pair number.
    landmarks: dict[str, Any] = field(default_factory=dict)
    #: The layers tool plugins left on this section (``ctx.add_layer``): one
    #: reference per layer (``id``, ``name``, ``kind``, ``frame``, ``file``
    #: relative to the job folder, ``tool``, ``call``, ``created``). Part of
    #: the checkpoint, so undoing the call that made a layer hides it. Left
    #: out of the serialized state while empty.
    layers: list[dict[str, Any]] = field(default_factory=list)
    #: This section's masks (``make_mask``,
    #: :mod:`langslice.core.sections.masks`), by name: ``{"kind": "em" | "fit",
    #: "section_image", "atlas_image" (resolved), "params", "inputs" (the
    #: digest of the section's registration it was made at), "file" (relative
    #: to the job folder), "left_out"}``. Part of the checkpoint; the files are
    #: never deleted, so undo only hides the reference. Left out of the
    #: serialized state while empty.
    masks: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: This section's made images (``make_image``, a script's ``add_image``,
    #: the opening ``preprocessed``; :mod:`langslice.core.sections.images`), by
    #: name: ``{"source": {"recipe": ...} | {"script": note} | {"opening":
    #: ...}, "file" (relative to the job folder), "digest", "made"}``. Part of
    #: the checkpoint; the files are never deleted, so undo only hides the
    #: reference. Left out of the serialized state while empty.
    images: dict[str, dict[str, Any]] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    #: This section's own cutting angles, ``{"pitch": deg, "yaw": deg}``:
    #: the atlas plane every picture, fit and map of it is drawn at. Equal
    #: on every section of a stack LangSlice angled itself
    #: (``position_sections``' cutting angles set them all); a registration supplied per
    #: section keeps each its own (``inputs.angles``). Serialized on the
    #: stack when all sections share one (:meth:`StackState.to_dict`).
    cutting_angles_deg: dict[str, float] = field(default_factory=flat_angles)

    @property
    def damaged(self) -> bool:
        """Whether the section is marked damaged: it has marked regions.
        Read-only."""
        return bool(self.damaged_regions)

    @property
    def angles(self) -> Angles:
        """``(pitch_deg, yaw_deg)`` of this section's plane."""
        return angles_tuple(self.cutting_angles_deg)

    @property
    def pitch_deg(self) -> float:
        return self.angles[0]

    @property
    def yaw_deg(self) -> float:
        return self.angles[1]

    @property
    def is_oblique(self) -> bool:
        return bool(self.pitch_deg or self.yaw_deg)
