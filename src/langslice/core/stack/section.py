"""One section's state: order, orientation, position, angles, registration.

A section is one image of the stack. Its order and orientation describe
a view over the discovered file (the file itself is never changed); its
position, cutting angles and in-plane affine place it on the atlas; a
deformation, a border trace, landmark pairs, made images, masks and
plugin layers may refine or annotate that placement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    SerializerFunctionWrapHandler,
    WrapSerializer,
)

from langslice.core.jsonable import JsonValue
from langslice.core.stack.affine import StoredAffine
from langslice.core.stack.deformation import DeformationRef
from langslice.core.stack.images import ImageRef, LayerRef, MaskRef
from langslice.core.stack.landmarks import Landmarks
from langslice.core.stack.part import Flag, Number, Shape, Whole
from langslice.core.stack.trace import BorderTrace

ROTATIONS = (0, 90, 180, 270)

Angles = tuple[float, float]

FLAT: Angles = (0.0, 0.0)


def _angles_in(value: object) -> object:
    if value is None:
        return FLAT
    if isinstance(value, Mapping):
        return (value.get("pitch", 0.0), value.get("yaw", 0.0))
    return value


def _angles_out(angles: Angles) -> dict[str, float]:
    return {"pitch": angles[0], "yaw": angles[1]}


def _landmarks_in(value: object) -> object:
    return value or None


def _landmarks_out(value: Landmarks | None, handler: SerializerFunctionWrapHandler) -> Any:
    return {} if value is None else handler(value)


CuttingAngles = Annotated[
    tuple[Number, Number], BeforeValidator(_angles_in), PlainSerializer(_angles_out)
]
SectionLandmarks = Annotated[
    Landmarks | None, BeforeValidator(_landmarks_in), WrapSerializer(_landmarks_out)
]


class MixedAngles(ValueError):
    """Signal a stack whose sections have different cutting angles.

    There is then no one stack-wide angle to read
    (:attr:`langslice.core.stack.stack.Stack.stack_angles`).
    """


PositionSource = Literal["initial", "input", "written"]

INITIAL: PositionSource = "initial"
INPUT: PositionSource = "input"
WRITTEN: PositionSource = "written"
POSITION_SOURCES: tuple[PositionSource, ...] = (INITIAL, INPUT, WRITTEN)


class Section(Shape):
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
        Its atlas position in millimetres. Every section has one from the
        start: the job's initial spacing until the host supplies one or a
        tool writes one.
    position_source
        Who set the position: ``initial`` (the job's start and spacing),
        ``input`` (the host's ``inputs.positions``) or ``written`` (a tool
        or a script).
    transform
        Its in-plane affine; None until it has one.
    border_trace
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

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    index_original: Whole
    index_corrected: Whole
    flip: Flag = False
    rotation_deg: Whole = 0
    damaged_regions: list[str] = Field(default_factory=list[str])
    damage_note: str = ""
    position_mm: Number = 0.0
    position_source: PositionSource = INITIAL
    transform: StoredAffine | None = None
    border_trace: BorderTrace | None = None
    deformation: DeformationRef | None = None
    landmarks: SectionLandmarks = None
    layers: list[LayerRef] = Field(default_factory=list[LayerRef])
    masks: dict[str, MaskRef] = Field(default_factory=dict[str, MaskRef])
    images: dict[str, ImageRef] = Field(default_factory=dict[str, ImageRef])
    caveats: list[str] = Field(default_factory=list[str])
    angles: CuttingAngles = Field(default=FLAT, alias="cutting_angles_deg")

    def fields_json(self, keys: Sequence[str]) -> dict[str, JsonValue]:
        """Return some of the section's row fields, as the row holds them.

        Parameters
        ----------
        keys
            Row keys (``tuple(Section.model_fields)`` names them, the
            cutting angles as ``cutting_angles_deg``), in the order wanted.

        Returns
        -------
        dict
            Each key and its value in the row.
        """
        row = self.to_json()
        return {key: row[key] for key in keys}

    def with_fields(self, fields: Mapping[str, Any]) -> Section:
        """Return a copy of the section with some row fields put back.

        Parameters
        ----------
        fields
            Row keys and their values in a row; the other fields are kept.

        Returns
        -------
        Section
            This section is not changed.
        """
        return Section.from_json({**self.to_json(), **fields})

    def set_position(self, position_mm: float, *, source: PositionSource = WRITTEN) -> None:
        """Set the section's position and who set it.

        Parameters
        ----------
        position_mm
            The atlas position in millimetres.
        source
            Who sets it (:data:`POSITION_SOURCES`).
        """
        self.position_mm = float(position_mm)
        self.position_source = source

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
