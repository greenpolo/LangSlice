"""A job's own image-model base prompt (``NonlinearSpec.image_prompt``,
``--image-prompt``), the task notes flags, and the trace replies
``export_maps`` writes to ``exports/traces/<name>.npz``.

Runs on the shared synthetic stack with the library's stub image model
(``tests/test_library.py``): it answers with the placed borders it was sent,
so its exported lines must fall on the atlas borders of the section's maps.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from langslice.core.spec import JobSpec, NonlinearSpec
from langslice.doors.cli import build_parser
from langslice.doors.cli.linear import spec_from_args
from tests.cli_child import install
from tests.synthetic_stack import ID0, ID1, PIXEL_SIZE_UM, atlas_loader, write_sections
from tests.test_library import POSITIONS, TRANSFORMS, Stub, _no_network

JOB_PROMPT = ("Image 1: a {plane} brain section photograph. Image 2: the same photograph "
              "with yellow borders. Return Image 2 with each border moved onto the anatomy.")


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("image_prompt") / "stack"
    write_sections(images)
    return images


@pytest.fixture
def images(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the stack, a private HOME, and no real image model."""
    import langslice.providers.images as transport

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    install(atlas_loader(), monkeypatch.setattr)
    monkeypatch.setattr(transport, "generate_warped_segmentation_image", _no_network)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    return folder


# --- the spec and the flags ------------------------------------------------------------


def test_the_flags_set_the_task_notes_and_the_image_prompt(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(JOB_PROMPT + "\n", encoding="utf-8")
    args = build_parser().parse_args([
        "linear", "run", str(tmp_path), "--tasks", "position,transform,nonlinear",
        "--image-prompt", str(prompt_file), "--position-notes", "Olfactory bulb first",
        "--transform-notes", "Ventral side down", "--nonlinear-notes", "Ignore the tear"])
    spec = spec_from_args(args, str(tmp_path))
    assert spec.nonlinear.image_prompt == JOB_PROMPT  # stripped
    assert (spec.position.notes, spec.transform.notes, spec.nonlinear.notes) == (
        "Olfactory bulb first", "Ventral side down", "Ignore the tear")
    assert JobSpec.from_dict(spec.to_dict()).nonlinear.image_prompt == JOB_PROMPT

    plain = spec_from_args(build_parser().parse_args(["linear", "run", str(tmp_path)]),
                           str(tmp_path))
    assert plain.nonlinear.image_prompt is None
    assert "image_prompt" not in plain.to_dict()["nonlinear"]  # saved specs unchanged

    missing = build_parser().parse_args(
        ["linear", "run", str(tmp_path), "--image-prompt", str(tmp_path / "none.txt")])
    with pytest.raises(ValueError, match="--image-prompt"):
        spec_from_args(missing, str(tmp_path))


@pytest.mark.parametrize("settings, message", [
    ({"image_prompt": "  "}, "non-empty"),
    ({"image_prompt": JOB_PROMPT, "provider": "none"}, "needs an image model"),
])
def test_a_bad_image_prompt_is_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        NonlinearSpec(**settings)


# --- the prompt sent ---------------------------------------------------------------------


def test_the_jobs_prompt_is_the_base_the_model_is_sent_and_the_agent_is_shown(images):
    import langslice
    from langslice.doors.declarations import Variant

    stub = Stub()
    with langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                              pixel_size_um=PIXEL_SIZE_UM, preprocess="none",
                              image_model=stub, nonlinear={"image_prompt": JOB_PROMPT}) as job:
        sent = JOB_PROMPT.replace("{plane}", "coronal")
        assert Variant.of(job.job.spec, auto=False).prompt == sent
        assert job.trace_borders(section=ID1)["status"] == "started"
        job.job.settle_image_corrections()
        job.job.background.wait_all()
        record = job.state.by_id(ID1).image_correction
    assert [request.prompt for request in stub.requests] == [sent]
    attempt = images / "langslice" / record["artifact_dir"]
    assert (attempt / "base_prompt.txt").read_text() == sent
    assert record["untested"] is True and record["prompt_edited"] is False


def test_a_profile_with_its_own_prompt_and_a_job_prompt_is_refused(images):
    import langslice

    profile = langslice.image_model(Stub(), prompt="Another prompt.")
    with pytest.raises(ValueError, match="give only one"):
        langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                             pixel_size_um=PIXEL_SIZE_UM, preprocess="none",
                             image_model=profile, nonlinear={"image_prompt": JOB_PROMPT})


# --- the exported reply ------------------------------------------------------------------


def _borders(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape, bool)
    out[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    out[1:, :] |= labels[1:, :] != labels[:-1, :]
    return out


#: The grey section: no tissue the yellow-line extraction could take for a line.
SECTION = ID0


@pytest.mark.parametrize("orientation", [None, {"flip": True, "rotation_deg": 90}])
def test_export_maps_writes_each_traced_sections_reply_on_its_maps_grid(images, orientation):
    import langslice

    supplied = {SECTION: orientation} if orientation else None
    with langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                              orientation=supplied, pixel_size_um=PIXEL_SIZE_UM,
                              preprocess="none", image_model=Stub()) as job:
        job.trace_borders(section=SECTION)
        job.job.settle_image_corrections()
        job.job.background.wait_all()
        reply = job.export_maps()
        folder = Path(job.folder)
        path = folder / "exports" / "traces" / "s0.npz"
        assert any(item["path"] == str(path) and item["kind"] == "trace_reply"
                   for item in reply["files"])
        assert not (folder / "exports" / "traces" / "s1.npz").exists()  # not traced

        grid = json.loads((folder / "sections" / "s0" / "maps.json").read_text())["grid_size"]
        with np.load(path) as data:
            borders, picture = data["borders"], data["reply"]
            info = json.loads(str(data["info"]))
        assert borders.dtype == bool and borders.shape == (grid[1], grid[0])
        assert picture.dtype == np.uint8 and picture.shape == (grid[1], grid[0], 3)
        assert info["grid_size"] == grid and info["section"] == SECTION
        assert info["placement_current"] is True and info["untested"] is True
        # The stub answers with the placed borders: in the file's frame they
        # lie on the borders of the section's label map at the traced
        # placement, so the deformation fitted to them is undone first.
        job.undo()
        assert job.state.by_id(SECTION).deformation is None
        job.export_maps(slices=[SECTION])
        labels = tifffile.imread(folder / "sections" / "s0" / "labels.tif")
        distance = np.asarray(ndimage.distance_transform_edt(~_borders(labels)))
        assert borders.sum() > 50
        assert np.percentile(distance[borders], 90) <= 1.5

        # A section with no completed trace now loses the reply an earlier export left.
        while job.state.by_id(SECTION).image_correction is not None:  # its fit, then the trace
            job.undo()
        again = job.export_maps(slices=[SECTION])
        assert again.get("status") != "error", again
        assert not path.exists()
