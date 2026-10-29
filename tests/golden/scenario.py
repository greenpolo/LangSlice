"""Run the golden scenario: one scripted registration of three real sections.

The scenario drives LangSlice only through the public library
(``langslice.create_job`` and the job handle's tool methods), with fixed
arguments and no model of any kind, so its outputs depend on the code alone.
``tests/test_golden_outputs.py`` runs it in a child interpreter, compares the
job folder it leaves with the goldens and documents the rest.

Run it by hand as ``python tests/golden/scenario.py IMAGES CALLS_JSON``:
IMAGES is a folder holding copies of the golden sections (the job
folder is made inside it), CALLS_JSON receives every call with its reply and
a digest of each picture the reply carried.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import langslice

#: The three sections: real coronal fluorescence sections (DAPI blue, two
#: tracer channels) from the lab's M01 test brain, cut to 768 px wide.
FRONT = "M01_001_005.png"
MIDDLE = "M01_002_005.png"
BACK = "M01_002_009.png"
SECTIONS = (FRONT, MIDDLE, BACK)

#: A coarse real atlas the machine already holds (its Nissl is borrowed from
#: ccfv3augmented_mouse_25um, also cached).
ATLAS = "allen_mouse_50um"
#: Source pixel size given outright, so nothing is inferred: the 4096 px
#: scans are about 2.4 um/px, cut by 5.33.
PIXEL_SIZE_UM = 12.8
#: Positions near the hand registration of these sections, in atlas mm.
POSITIONS_MM = {FRONT: 3.5, MIDDLE: 5.62, BACK: 6.29}
#: A small oblique plane, so the angled-plane code runs.
CUTTING_ANGLES = {"pitch_deg": 1.5, "yaw_deg": -1.0}
#: Hand transforms written before the automatic affine refines them.
HAND_TRANSFORMS = [
    {"id": FRONT, "rotation_deg": 2.0, "scale_x": 1.05, "translate_x_mm": 0.1},
    {"id": MIDDLE, "scale_y": 0.97, "shear": 0.02},
]
#: The zoom box drawn after every write, in overlay-picture pixels.
ZOOM_BOX = [120, 80, 380, 260]
#: Landmark pairs on one overlay picture of FRONT: an atlas point and the
#: section point a few pixels away that it should land on; the third holds
#: its spot (both endpoints one point), as triangulation needs three.
LANDMARK_POINTS = [
    ((200, 150), (208, 155)),
    ((330, 200), (325, 206)),
    ((260, 110), (260, 110)),
]


def picture_digest(image: Image.Image) -> dict[str, object]:
    """Return the size and pixel digest of one picture a reply carried.

    Parameters
    ----------
    image
        A PIL image from ``Reply.images``: the picture before JPEG.

    Returns
    -------
    dict
        ``size`` as ``[width, height]`` and ``sha256`` of its RGB pixels.
    """
    pixels = np.asarray(image.convert("RGB"))
    return {
        "size": [pixels.shape[1], pixels.shape[0]],
        "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }


class Recorder:
    """Call verbs on one job and keep every call with its reply.

    Parameters
    ----------
    job
        The open job handle whose verbs are called.
    """

    def __init__(self, job: object) -> None:
        self.job = job
        self.calls: list[dict[str, object]] = []

    def call(self, verb: str, **arguments: object) -> dict[str, object]:
        """Call one verb and record it.

        Parameters
        ----------
        verb
            The tool's name, as the job handle offers it.
        **arguments
            passed to the verb's method on the job handle.

        Returns
        -------
        dict
            The reply, as a plain dict.
        """
        reply = getattr(self.job, verb)(**arguments)
        self.calls.append(
            {
                "verb": verb,
                "arguments": arguments,
                "reply": dict(reply),
                "pictures": [picture_digest(image) for image in reply.images],
            }
        )
        return reply

    def look_around(self, sections: list[str]) -> None:
        """Draw every look mode, one zoom and the status after a write.

        Parameters
        ----------
        sections
            The sections the write touched; positioning draws them all.
        """
        self.call("look", mode="section", sections=sections)
        self.call("look", mode="atlas", sections=sections)
        overlay = self.call("look", mode="overlay", sections=sections)
        self.call("look", mode="positioning")
        first_overlay = overlay["pictures"][0]["id"]
        self.call("zoom", picture=first_overlay, box=ZOOM_BOX)
        self.call("status")


def landmark_pairs(picture: int) -> list[dict[str, object]]:
    """Return the fixed landmark pairs, both endpoints on one picture.

    Parameters
    ----------
    picture
        The number of an overlay picture of the section.

    Returns
    -------
    list of dict
        Pairs as ``set_landmarks`` takes them.
    """
    pairs = []
    for (atlas_x, atlas_y), (section_x, section_y) in LANDMARK_POINTS:
        pairs.append(
            {
                "atlas": {"picture": picture, "x": atlas_x, "y": atlas_y},
                "section": {
                    "picture": picture,
                    "x": section_x,
                    "y": section_y,
                },
            }
        )
    return pairs


def run(images: Path) -> list[dict[str, object]]:
    """Run the scripted registration in a new job beside *images*.

    Parameters
    ----------
    images
        Folder holding the three section images; the job folder is
        ``images/langslice``.

    Returns
    -------
    list of dict
        Every call: verb, arguments, reply and picture digests.
    """
    every = list(SECTIONS)
    with langslice.create_job(
        str(images),
        atlas=ATLAS,
        tasks=["position", "transform", "nonlinear"],
        pixel_size_um=PIXEL_SIZE_UM,
    ) as job:
        recorder = Recorder(job)
        recorder.call("status")
        recorder.call("look", mode="positioning")
        recorder.call("look", mode="section")

        # linear: place, set by hand, refine with Elastix
        placements = [
            {"id": name, "position_mm": POSITIONS_MM[name]}
            for name in SECTIONS
        ]
        recorder.call(
            "position_sections",
            sections=placements,
            cutting_angles=CUTTING_ANGLES,
        )
        recorder.look_around(every)
        recorder.call("interactive_transform", sections=HAND_TRANSFORMS)
        recorder.look_around([FRONT, MIDDLE])
        recorder.call("elastix_affine", sections=every)
        recorder.look_around(every)

        # inputs of the deformable fit: a mask and a damage mark
        recorder.call("make_mask", name="tissue", sections=[MIDDLE, BACK])
        recorder.look_around([MIDDLE, BACK])
        recorder.call(
            "mark_damage",
            section=BACK,
            regions=["CA1:left"],
            note="left CA1 torn",
        )
        recorder.look_around([BACK])

        # nonlinear: FireANTs on two sections, landmarks on the third
        recorder.call(
            "fireants_syn",
            sections=[MIDDLE, BACK],
            mask="tissue",
            detail="coarse",
        )
        recorder.calls.append(
            {
                "verb": "wait",
                "arguments": {},
                "reply": dict(job.wait()),
                "pictures": [],
            }
        )
        recorder.look_around([MIDDLE, BACK])
        overlay = recorder.call("look", mode="overlay", sections=[FRONT])
        picture = overlay["pictures"][0]["id"]
        recorder.call(
            "set_landmarks", section=FRONT, pairs=landmark_pairs(picture)
        )
        recorder.call("landmark_triangulation", section=FRONT)
        recorder.look_around([FRONT])

        # history and the derived files
        recorder.call("undo")
        recorder.call("redo")
        recorder.call("restore_registration", section=FRONT)
        recorder.call("export_maps")
        recorder.call("status")
    return recorder.calls


def main() -> None:
    """Run the scenario on the first folder; write calls to the second."""
    images, calls_path = Path(sys.argv[1]), Path(sys.argv[2])
    calls = run(images)
    calls_path.write_text(json.dumps(calls, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
