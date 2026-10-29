"""One section's state: order, orientation, position, angles, registration.

A section is one image of the stack. Its order and orientation describe
a view over the discovered file (the file itself is never changed); its
position, cutting angles and in-plane affine place it on the atlas; a
deformation, a border trace, landmark pairs, made images, masks and
plugin layers may refine or annotate that placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langslice.core.stack.affine import StoredAffine
from langslice.core.stack.deformation import DeformationRef
from langslice.core.stack.images import ImageRef, LayerRef, MaskRef
from langslice.core.stack.landmarks import Landmarks
from langslice.core.stack.trace import BorderTrace

#: Quarter-turns the render applies before the flip.
ROTATIONS = (0, 90, 180, 270)

#: A plane's cutting angles, ``(pitch_deg, yaw_deg)``.
Angles = tuple[float, float]

#: The flat plane: no pitch, no yaw.
FLAT: Angles = (0.0, 0.0)


class MixedAngles(ValueError):
    """Signal a stack whose sections have different cutting angles.

    There is then no one stack-wide angle to read
    (:attr:`langslice.core.stack.stack.Stack.stack_angles`).
    """


@dataclass
class Section:
    """Hold everything the run knows or proposes about one section image.

    Mutable: it is the job's live state, written under the job folder's
    lock and refilled in place by undo; its parts are frozen values.

    Parameters
    ----------
    id
        The section's filename.
    index_original
        Its place in the discovered stack.
    index_corrected
        Its place in the stack as ordered.
    flip
        Mirror left-right, applied after ``rotation_deg``.
    rotation_deg
        Quarter turn applied before the flip; one of :data:`ROTATIONS`.
    damaged_regions
        The atlas regions whose tissue this section is missing, or has
        folded over other anatomy: acronyms or ids, each optionally one
        side (``"CTX:left"``). Every fit and the image model's trace leave
        them out.
    damage_note
        What is wrong with the tissue, in words: the host's note first,
        then the agent's. A note alone does not make the section damaged.
    position_mm
        Its atlas position in millimetres; None until placed.
    transform
        Its in-plane affine; None until it has one.
    image_correction
        Its image-model border trace; it does not replace the affine.
    deformation
        The deformation it holds, on top of the linear placement it was
        fitted at; any change to position, orientation, cutting angles
        or affine clears it.
    landmarks
        Its landmark pairs; None until the first pair is made.
    layers
        The layers tool plugins left on it.
    masks
        Its masks, by name.
    images
        Its made images, by name.
    caveats
        Notes on the section the status lists.
    angles
        Its own cutting angles ``(pitch_deg, yaw_deg)``: the atlas plane
        every picture, fit and map of it is drawn at. Equal on every
        section of a stack LangSlice angled itself; a registration
        supplied per section keeps each its own.
    """

    id: str
    index_original: int
    index_corrected: int
    flip: bool = False
    rotation_deg: int = 0
    damaged_regions: list[str] = field(default_factory=list[str])
    damage_note: str = ""
    position_mm: float | None = None
    transform: StoredAffine | None = None
    image_correction: BorderTrace | None = None
    deformation: DeformationRef | None = None
    landmarks: Landmarks | None = None
    layers: list[LayerRef] = field(default_factory=list[LayerRef])
    masks: dict[str, MaskRef] = field(default_factory=dict[str, MaskRef])
    images: dict[str, ImageRef] = field(default_factory=dict[str, ImageRef])
    caveats: list[str] = field(default_factory=list[str])
    angles: Angles = FLAT

    @property
    def damaged(self) -> bool:
        """Return whether the section is marked damaged: it has regions."""
        return bool(self.damaged_regions)

    @property
    def deformation_key(self) -> str | None:
        """Return the key of the deformation the section holds, if any."""
        return None if self.deformation is None else self.deformation.key

    @property
    def pitch_deg(self) -> float:
        """Return the pitch of the section's plane, in degrees."""
        return self.angles[0]

    @property
    def yaw_deg(self) -> float:
        """Return the yaw of the section's plane, in degrees."""
        return self.angles[1]

    @property
    def is_oblique(self) -> bool:
        """Return whether the section's plane is angled."""
        return bool(self.pitch_deg or self.yaw_deg)
