"""What a fit was given and what it estimated itself: the two blocks every fit
tool's reply and every deformation record's provenance carry.

``inputs`` names what the engine read: the section image (``{name, kind,
file, digest}``, :meth:`langslice.core.section_images.ImageView.payload`), the
atlas image as asked and as resolved, the section mask (always an object: the
name, ``"none"`` for no mask, its kind and staleness), the atlas-side damage
marks, the regions the fit was restricted to and the engine's own brightness
scaling in words. ``estimated`` names what the engine worked out by itself (its
own sorting of the section and the fraction it left out), or is null. Both name
what was used and nothing about quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The brightness scaling of the native engines (FireANTs, Elastix), in words.
NATIVE_BRIGHTNESS = "1st-99th percentile inside the atlas outline to 0-1"


def mask_block(name: str = "none", kind: str | None = None, stale: bool = False,
               earlier_registration: bool = False) -> dict[str, Any]:
    """The mask entry of ``inputs``, one shape whether a mask was used or not: ``name``
    (``"none"`` when the fit read no section mask), ``kind`` (``em`` / ``fit``, null for
    none), ``stale`` and ``earlier_registration``."""
    return {"name": name, "kind": kind, "stale": bool(stale),
            "earlier_registration": bool(earlier_registration)}


def damage_block(regions: list[str] | tuple[str, ...], margin_mm: float) -> dict[str, Any]:
    """The damage entry of ``inputs``: the atlas-side regions left out of the
    fit and the margin around them, always applied, never on the section side."""
    return {"regions": list(regions), "margin_mm": float(margin_mm), "side": "atlas"}


@dataclass(frozen=True)
class InputsBlock:
    """What one fit read, for one section."""

    #: ``{"name", "kind": "raw" | "made", "file", "digest"}``: the section image read.
    section_image: dict[str, Any]
    #: The atlas image as the call named it (``nissl``) and as it resolved
    #: (``ccfv3augmented_nissl``).
    atlas_image: str
    atlas_image_resolved: str
    #: :func:`mask_block`: ``{"name", "kind", "stale", "earlier_registration"}``, the
    #: name ``"none"`` when no mask was read.
    mask: dict[str, Any]
    #: ``{"regions", "margin_mm", "side": "atlas"}`` (:func:`damage_block`).
    damage: dict[str, Any]
    restrict_to: list[str] = field(default_factory=list)
    #: The engine's own scaling of the images, in words.
    brightness: str = NATIVE_BRIGHTNESS
    #: Said when the default ``nissl`` resolved to the template (``images.nissl_note``).
    atlas_image_note: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "section_image": dict(self.section_image),
            "atlas_image": self.atlas_image, "atlas_image_resolved": self.atlas_image_resolved,
            **({"atlas_image_note": self.atlas_image_note} if self.atlas_image_note else {}),
            "mask": dict(self.mask),
            "damage": dict(self.damage), "restrict_to": list(self.restrict_to),
            "brightness": self.brightness,
        }


@dataclass(frozen=True)
class EstimatedBlock:
    """What an engine worked out itself: where its own sorting is saved, and
    the fraction of the section it left out."""

    #: ``saved with the deformation; make_mask source fit``, or None.
    sorting: str | None = None
    left_out_fraction: float | None = None

    def payload(self) -> dict[str, Any]:
        return {"sorting": self.sorting, "left_out_fraction": self.left_out_fraction}
