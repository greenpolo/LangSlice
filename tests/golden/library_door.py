"""Run the golden scenario's library door: one scripted registration.

The scenario drives LangSlice only through the public library
(``langslice.create_job`` and the job handle's tool methods), with fixed
arguments and no model of any kind, so its outputs depend on the code alone.
It calls every tool that needs no image model, most of them with more than
one set of options, and some with arguments a tool refuses.
``tests/test_golden_outputs.py`` runs it in a child interpreter, compares the
job folder it leaves with the goldens and documents the rest.

Run it by hand as ``python tests/golden/library_door.py IMAGES CALLS_JSON``:
IMAGES is a folder holding copies of the golden sections (the job
folder is made inside it), CALLS_JSON receives every call with its reply and
a digest of each picture the reply carried.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from stack import (
    ATLAS,
    BACK,
    CUTTING_ANGLES,
    FRONT,
    HAND_TRANSFORMS,
    MIDDLE,
    MISSING,
    PIXEL_SIZE_UM,
    SECTIONS,
    TASKS,
    ZOOM_BOX,
    picture_digest,
    placements,
)

import langslice

#: Landmark pairs on one overlay picture of FRONT: an atlas point and the
#: section point a few pixels away that it should land on; the third holds
#: its spot (both endpoints one point), as triangulation needs three.
LANDMARK_POINTS = [
    ((200, 150), (208, 155)),
    ((330, 200), (325, 206)),
    ((260, 110), (260, 110)),
]
#: Points of an overlay picture whose atlas region ``grep_atlas`` names: one
#: inside the brain, one in the corner outside it.
REGION_POINTS = [(260, 150), (3, 3)]


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


def register(recorder: Recorder) -> int:
    """Place, transform, refine, mask, deform, undo and export the stack.

    Parameters
    ----------
    recorder
        The recorder of the open job.

    Returns
    -------
    int
        The number of the overlay picture of FRONT the landmarks were set
        on, for the calls that name a picture.
    """
    every = list(SECTIONS)
    recorder.call("status")
    recorder.call("look", mode="positioning")
    recorder.call("look", mode="section")

    # linear: place, set by hand, refine with Elastix
    recorder.call(
        "position_sections",
        sections=placements(),
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
            "reply": dict(recorder.job.wait()),
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
    return picture


def look_with_options(recorder: Recorder) -> None:
    """Draw pictures with the options the registration left at defaults.

    Parameters
    ----------
    recorder
        The recorder of the open job.
    """
    recorder.call(
        "look",
        mode="overlay",
        sections=[MIDDLE],
        warp="none",
        atlas_layers=["template", "borders"],
        atlas_opacity=0.3,
        depth=4,
        masks=["tissue"],
    )
    recorder.call("look", mode="atlas", positions_mm=[4.0, 6.0])
    recorder.call(
        "look",
        mode="section",
        sections=[BACK],
        images=["red", "green", "blue"],
        layers=["all"],
    )
    recorder.call(
        "look",
        mode="overlay",
        sections=[FRONT],
        registrations={FRONT: 1},
        resolution=384,
    )
    recorder.call(
        "look", mode="positioning", positions_mm=[3.5, 6.3], sheet=True
    )
    overlay = recorder.call("look", mode="overlay", sections=[MIDDLE])
    recorder.call("zoom", picture=overlay["pictures"][0]["id"], region="TH")


def make_inputs(recorder: Recorder) -> None:
    """Make, replace and delete section images and masks.

    Parameters
    ----------
    recorder
        The recorder of the open job.
    """
    recorder.call(
        "make_image",
        name="dapi",
        sections=[FRONT, MIDDLE],
        weights={"blue": 1},
        clahe_clip=2.0,
    )
    recorder.call(
        "make_image",
        name="tracer",
        sections=[BACK],
        weights={"red": 1, "green": 0.5},
        clahe_clip=0.0,
        contrast=[0.05, 0.95],
        gamma=0.8,
        invert=True,
    )
    recorder.call("make_image", name="tracer", delete=True)
    recorder.call(
        "make_mask",
        name="dapi_tissue",
        sections=[FRONT],
        section_image="dapi",
        atlas_image="template",
        sorting={"match_sd": 0.2},
    )
    recorder.call("make_mask", name="fitted", source="fit", sections=[MIDDLE])
    recorder.call("make_mask", name="dapi_tissue", delete=True, view=False)


def read_atlas_and_files(recorder: Recorder, picture: int) -> None:
    """Look regions up in the atlas and read the job folder.

    Parameters
    ----------
    recorder
        The recorder of the open job.
    picture
        The number of an overlay picture of FRONT.
    """
    recorder.call("grep_atlas", query="CA1")
    recorder.call("grep_atlas", query="HIP", section=MIDDLE)
    points = [{"picture": picture, "x": x, "y": y} for x, y in REGION_POINTS]
    recorder.call("grep_atlas", points=points)
    recorder.call(
        "grep_atlas_view", regions=["CA1", "CTX:left"], positions_mm=[3.5, 6.0]
    )

    # folders and files whose listing holds no size of a file that names
    # the run's own folder (job.json's size changes with its path)
    recorder.call("list_files", path="views")
    recorder.call("list_files", path="sections", glob="*.png")
    recorder.call("search_files", query="CA1", path="state.json")
    recorder.call("read_file", path="job.json", limit=20)
    recorder.call("read_file", path="views/000001_look_positioning/view.jpg")


def refuse_and_submit(recorder: Recorder) -> None:
    """Make calls the tools refuse, note, submit, and write after submitting.

    Parameters
    ----------
    recorder
        The recorder of the open job.
    """
    recorder.call(
        "interactive_transform", sections=[{"id": MISSING, "rotation_deg": 1}]
    )
    recorder.call("look", mode="overlay", sections=[MISSING])
    recorder.call("fireants_syn", sections=[FRONT], mask="no_such_mask")
    recorder.call("note", text="Golden scenario: every tool without a model.")
    recorder.call(
        "submit",
        summary="Placed, refined and deformed three sections by script.",
        notes=["The back section's left CA1 is torn."],
        interval_breaks=[],
    )
    recorder.call("note", text="Written after submitting.")
    recorder.call("status")


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
    with langslice.create_job(
        str(images),
        atlas=ATLAS,
        tasks=list(TASKS),
        pixel_size_um=PIXEL_SIZE_UM,
    ) as job:
        recorder = Recorder(job)
        picture = register(recorder)
        look_with_options(recorder)
        make_inputs(recorder)
        read_atlas_and_files(recorder, picture)
        refuse_and_submit(recorder)
    return recorder.calls


def main() -> None:
    """Run the scenario on the first folder; write calls to the second."""
    images, calls_path = Path(sys.argv[1]), Path(sys.argv[2])
    calls = run(images)
    calls_path.write_text(json.dumps(calls, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
