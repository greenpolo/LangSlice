"""The stack state: the checkpoint, the result, and what every tool writes.

One object, one JSON shape, three uses. Corrections are DATA — ``flip``,
``rotation_deg`` and ``index_corrected`` describe a view over the discovered
stack, and the user's image files are never modified.

Order and position are separate fields that must agree at submit: reordering
changes only ``index_corrected``, and ``submit`` refuses positions that are not
monotone along the corrected order. Direction is not the agent's concern: the
sections of a stack cut posterior-first are identical to one cut
anterior-first, so at submit :func:`normalize_to_atlas_order` reverses a
descending stack, and the emitted corrected order always runs the atlas way
(positions increasing from the atlas origin end).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from langslice.core.jsonable import JsonObject, empty_object
from langslice.core.stack.section import FLAT, Angles, MixedAngles, Section


def plane_angles(state: Stack, angles: Angles | None) -> Angles:
    """*angles* as floats, or, when None, the stack's one angle
    (:attr:`Stack.stack_angles`, which refuses a stack whose sections
    differ). The rule every atlas-plane picture takes its plane by: a
    section's own angles for a picture of that section, the stack's view
    angles for one without."""
    return (
        state.stack_angles
        if angles is None
        else (float(angles[0]), float(angles[1]))
    )


@dataclass
class Stack:
    """Hold the whole stack: its inputs, its sections and run bookkeeping.

    Mutable: it is the job's live state, written under the job folder's
    lock and refilled in place by undo (:meth:`replace_with`), since the
    tools of a run share one stack object.

    Parameters
    ----------
    image_folder
        The folder the section images were discovered in.
    atlas
        The BrainGlobe atlas name.
    plane
        The cutting plane: ``coronal``, ``sagittal`` or ``horizontal``.
    interval_mm
        The nominal distance between sections; None when not supplied.
    thickness_mm
        The section thickness; None when not supplied.
    spec
        The job's settings as run (the job spec's JSON form).
    interval_breaks
        Corrected indices of the sections after a gap the agent concluded
        is real.
    notes
        The run's notes, in order.
    slices
        The sections, in discovered order.
    submitted
        Whether the registration was submitted.
    debrief
        The agent's post-submit debrief (what it reached for that was not
        there), verbatim: data for the environment's builders.
    """

    image_folder: str = ""
    atlas: str = ""
    plane: str = "coronal"
    interval_mm: float | None = None
    thickness_mm: float | None = None
    spec: JsonObject = field(default_factory=empty_object)
    interval_breaks: list[int] = field(default_factory=list[int])
    notes: list[str] = field(default_factory=list[str])
    slices: list[Section] = field(default_factory=list[Section])
    submitted: bool = False
    debrief: str = ""

    def in_order(self) -> list[Section]:
        """Return the sections sorted by their corrected order.

        Returns
        -------
        list of Section
            The stack as corrected.
        """
        return sorted(self.slices, key=lambda section: section.index_corrected)

    def by_id(self, section_id: str) -> Section | None:
        """Return the section with filename *section_id*.

        Parameters
        ----------
        section_id
            The section's whole filename.

        Returns
        -------
        Section or None
            The section, or None when no section has that filename.
        """
        return next((s for s in self.slices if s.id == section_id), None)

    def resolve(self, ref: object) -> Section | None:
        """Return the section a reference names.

        Sections are named by filename only (:func:`unknown_sections`),
        never by a number.

        Parameters
        ----------
        ref
            A filename, or a filename without its extension when exactly
            one section has that stem.

        Returns
        -------
        Section or None
            The section, or None when *ref* names none.
        """
        text = str(ref).strip()
        hit = self.by_id(text)
        if hit is not None:
            return hit
        stems = [s for s in self.slices if filename_stem(s.id) == text]
        if len(stems) == 1:
            return stems[0]
        return None

    # Each section carries its own plane. Everything LangSlice angles
    # itself is one plane for the whole stack; a registration supplied
    # per section may differ section by section.

    @property
    def mixed_angles(self) -> bool:
        """Return whether the sections' cutting angles differ."""
        return len({section.angles for section in self.slices}) > 1

    @property
    def stack_angles(self) -> Angles:
        """Return the one ``(pitch, yaw)`` every section shares.

        Flat for an empty stack. :class:`MixedAngles` when the sections
        differ, so a reader that should use each section's own angles
        cannot take one silently.
        """
        planes = {section.angles for section in self.slices}
        if len(planes) > 1:
            raise MixedAngles(
                "The sections of this stack have different cutting angles; "
                "read each section's own (Section.angles)."
            )
        return next(iter(planes), FLAT)

    @property
    def view_angles(self) -> Angles:
        """Return the plane an atlas picture without a section is drawn at.

        The stack's one angle, or, when the sections differ, the median
        pitch and the median yaw of its sections (``look``'s atlas, the
        opening's atlas reference).
        """
        if not self.mixed_angles:
            return self.stack_angles
        pitches = [section.pitch_deg for section in self.slices]
        yaws = [section.yaw_deg for section in self.slices]
        return float(statistics.median(pitches)), float(
            statistics.median(yaws)
        )

    def drawn_at_median(self, angles: Angles) -> bool:
        """Return whether a picture at *angles* is at the median plane.

        An atlas picture without a section drawn at the median of
        differing sections' angles (:attr:`view_angles`) must say so in
        its caption.

        Parameters
        ----------
        angles
            The ``(pitch, yaw)`` the picture is drawn at.

        Returns
        -------
        bool
            True when the sections differ and *angles* is their median.
        """
        plane = (float(angles[0]), float(angles[1]))
        return self.mixed_angles and plane == self.view_angles

    def set_angles(self, angles: Angles) -> None:
        """Set every section's cutting angles to *angles*.

        A stack whose sections differed then has one plane
        (``position_sections``).

        Parameters
        ----------
        angles
            The ``(pitch, yaw)`` in degrees.
        """
        pitch, yaw = float(angles[0]), float(angles[1])
        for section in self.slices:
            section.angles = (pitch, yaw)

    @property
    def pitch_deg(self) -> float:
        """Return the stack-wide pitch (:attr:`stack_angles`)."""
        return self.stack_angles[0]

    @property
    def yaw_deg(self) -> float:
        """Return the stack-wide yaw (:attr:`stack_angles`)."""
        return self.stack_angles[1]

    @property
    def is_oblique(self) -> bool:
        """Return whether any section's plane is angled."""
        return any(section.is_oblique for section in self.slices)

    def replace_with(self, new_state: Stack) -> None:
        """Overwrite this stack's contents with another stack's, in place.

        The tools close over one stack object, so undo and redo refill it
        rather than swap it.

        Parameters
        ----------
        new_state
            The stack whose fields this one takes; not copied.
        """
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(new_state, name))


def normalize_to_atlas_order(state: Stack) -> bool:
    """Make the corrected order run the atlas way (positions increasing).

    Called at submit after the monotonicity gate has passed. When the placed
    positions decrease along the corrected order, every ``index_corrected``
    is mirrored (``n - 1 - i``) and each recorded interval break, the index of
    the section after a gap, is mapped to the section after that same gap in
    the new order (``n - i``). Returns True when the stack was reversed.
    """
    ordered = state.in_order()
    placed = [s for s in ordered if s.position_mm is not None]
    if len(placed) < 2:
        return False
    first = float(placed[0].position_mm)  # type: ignore[arg-type]
    last = float(placed[-1].position_mm)  # type: ignore[arg-type]
    if last >= first:
        return False
    n = len(ordered)
    for section in ordered:
        section.index_corrected = n - 1 - section.index_corrected
    state.interval_breaks = sorted(n - i for i in state.interval_breaks if 0 < i < n)
    return True


# --- naming sections ----------------------------------------------------------------
#
# A section is named by its filename everywhere a model, a script or a user
# names one (:meth:`Stack.resolve`); the stack order is internal.

#: How many filenames a refusal lists before it says how many it left out.
NAMES_LISTED = 40


def filename_stem(name: str) -> str:
    """*name* without its extension (``s01.tif`` -> ``s01``)."""
    base, dot, _extension = name.rpartition(".")
    return base if dot and base else name


def break_ids(state: Stack) -> list[str]:
    """The interval breaks as the filenames of the sections after each gap
    (stored as corrected indices, ``Stack.interval_breaks``)."""
    by_index = {
        section.index_corrected: section.id for section in state.slices}
    return [by_index[index] for index in state.interval_breaks if index in by_index]


def unknown_sections(state: Stack, refs: object) -> dict[str, Any]:
    """The facts of a refusal for section names *refs* that name no section:
    ``unknown`` (the refs as text), ``filenames`` (the valid names, the
    first :data:`NAMES_LISTED`) and ``message``, which says that sections
    are named by filename (a number names none; a stem two files share
    needs the whole filename)."""
    unknown = [str(ref) for ref in (refs if isinstance(refs, (list, tuple)) else [refs])]
    names = sorted(section.id for section in state.slices)
    listed = names[:NAMES_LISTED]
    numbers = [ref for ref in unknown if ref.strip().lstrip("-").isdigit()]
    shared = [ref for ref in unknown
              if sum(filename_stem(name) == ref.strip() for name in names) > 1]
    words = [f"No section is named {', '.join(repr(ref) for ref in unknown)}. Sections are "
             "named by filename (or the filename without its extension, when no other "
             "section shares it)."]
    if numbers:
        words.append("Sections have no numbers to name them by.")
    if shared:
        words.append(f"{', '.join(repr(ref) for ref in shared)}: more than one file has "
                     "this name without its extension; give the whole filename.")
    words.append("Filenames: " + ", ".join(listed)
                 + (f" and {len(names) - len(listed)} more (status lists every section)."
                    if len(names) > len(listed) else "."))
    return {"unknown": unknown, "filenames": listed, "message": " ".join(words)}
