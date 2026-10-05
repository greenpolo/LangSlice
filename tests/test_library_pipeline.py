"""The scripted pipeline through the public library: ``create_job``,
``image_model`` profiles, ``register_section``, ``register_job``, the lean
job folder.

Everything runs on the golden recorder's synthetic stack and atlas
(``tests/golden/record.py``) with a private HOME; the image model is a stub
that answers with the placed borders it was sent, and the real transport
raises if anything reaches it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.cli_child import install
from tests.golden.record import (
    ID0,
    ID1,
    ID2,
    PIXEL_SIZE_UM,
    apply_patches,
    atlas_loader,
    write_sections,
)

#: The supplied placement of the synthetic sections (see test_combinations).
POSITIONS = {ID0: 0.1, ID1: 0.15, ID2: 0.2}
TRANSFORMS = {
    ID0: [1.02, 0.01, -0.01, 0.0, 0.98, 0.01],
    ID1: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ID2: [0.97, -0.02, 0.02, 0.02, 0.97, 0.0],
}
PROFILE_PROMPT = ("Image 1: a {plane} brain section photograph. Image 2: it with yellow "
                  "borders. Return Image 1 with the borders moved onto the anatomy.")


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("pipeline") / "stack"
    write_sections(images)
    return images


def _no_network(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("a real image model was called")


@pytest.fixture
def images(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import langslice.providers.images as transport

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    apply_patches()
    install(atlas_loader(), monkeypatch.setattr)
    monkeypatch.setattr(transport, "generate_warped_segmentation_image", _no_network)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    return folder


def _colour(image: Image.Image) -> float:
    pixels = np.asarray(image.convert("RGB")).astype(np.int16)
    return float(np.abs(pixels[..., 0] - pixels[..., 2]).mean())


class Stub:
    """A model of the caller's own: answers with the placed borders it was
    sent (the more coloured attachment), recording every request."""

    provider = "my-lab"
    model = "lab-edit-1"

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def call(self, request: Any) -> Image.Image:
        self.requests.append(request)
        return max([request.slice_image, *request.reference_images], key=_colour)

    def segmentation(self, request: Any) -> Any:
        """:meth:`call` as an ``ImageCall`` answers (a ``GeneratedSegmentation``)."""
        from langslice.core.nonlinear.types import GeneratedSegmentation

        return GeneratedSegmentation(image=self.call(request), provider=request.provider,
                                     model=str(request.model), route="stub")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The operations the library ran, in order."""
    import langslice.ops.deformable as deformable
    import langslice.ops.submit as submit
    import langslice.ops.traces as traces
    import langslice.ops.transforms as transforms

    seen: list[str] = []
    for module, name in ((transforms, "fit_affine"), (traces, "trace_borders"),
                         (deformable, "fit_deformable"), (submit, "submit")):
        original = getattr(module, name)

        def spy(*args: Any, _original: Any = original, _name: str = name, **kwargs: Any) -> Any:
            seen.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    return seen


def files_in(folder: Path) -> set[str]:
    return {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}


# --- image_model and default_prompt --------------------------------------------------


def test_image_model_builtin_profiles():
    import langslice
    from langslice.core.nonlinear.prompts import border_correction_tool_prompt

    model = langslice.image_model("openai-oauth")
    assert (model.provider, model.model, model.prompt, model.tested, model.profile) == (
        "openai-oauth", "gpt-image-2", None, True, "openai-oauth")
    assert langslice.image_model("chatgpt").provider == "openai-oauth"
    assert langslice.image_model("gemini-api", model="some-image-model").model == \
        "some-image-model"
    assert langslice.image_model(model) is model
    for wrong in ("none", "no-such-provider"):
        with pytest.raises(ValueError):
            langslice.image_model(wrong)
    assert langslice.default_prompt("openai-oauth") == border_correction_tool_prompt(
        "coronal", provider="openai-oauth")
    assert langslice.default_prompt("gemini-api", "sagittal") == border_correction_tool_prompt(
        "sagittal", provider="gemini-api")


def test_image_model_custom_profiles(tmp_path):
    import langslice

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(PROFILE_PROMPT + "\n")
    profile = langslice.image_model("openai-oauth", prompt_file=prompt_file, name="lab-v1")
    assert (profile.prompt, profile.profile, profile.tested) == (PROFILE_PROMPT, "lab-v1", False)
    own = langslice.image_model(Stub(), prompt=PROFILE_PROMPT, photograph_first=True)
    assert (own.provider, own.model, own.tested, own.profile) == (
        "custom", "lab-edit-1", False, "custom")
    function = langslice.image_model(lambda request: request.slice_image)
    assert (function.provider, function.tested, function.prompt) == ("custom", False, None)
    with pytest.raises(ValueError):
        langslice.image_model("openai-oauth", prompt="a", prompt_file=prompt_file)
    with pytest.raises(TypeError):
        langslice.image_model(42)


# --- create_job ---------------------------------------------------------------------------


def test_create_job_from_a_folder(images):
    import langslice

    with langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                              pixel_size_um=PIXEL_SIZE_UM, preprocess="none") as job:
        assert Path(job.folder) == images / "langslice"
        assert job.job.spec.tasks == ["nonlinear"]  # every section has a transform
        assert job.job.spec.nonlinear.provider == "none"
        assert "trace_borders" not in job.verbs and "fit_affine" not in job.verbs
        record = job.state.by_id(ID0)
        assert record.position_mm == POSITIONS[ID0]
        assert record.transform == {"kind": "interactive", "params": TRANSFORMS[ID0],
                                    "mirrored": False}
    elsewhere = images.parent / "jobs" / "arm-a"
    with langslice.create_job(images, positions={ID0: 0.1}, job_dir=elsewhere,
                              image_model=Stub(), preprocess="none") as job:
        assert Path(job.folder) == elsewhere
        assert job.job.spec.tasks == ["transform", "nonlinear"]
        assert job.job.spec.nonlinear.provider == "custom"
        assert {"fit_affine", "trace_borders", "fit_deformable"} <= set(job.verbs)
    record = json.loads((elsewhere / "job.json").read_text())["image_model"]
    assert record["provider"] == "custom" and record["tested"] is False
    with langslice.open_job(elsewhere) as job:  # a model of one's own is never resolved
        assert "trace_borders" not in job.verbs and "fit_deformable" in job.verbs
    with pytest.raises(ValueError, match="Unknown job setting"):
        langslice.create_job(images, no_such_setting=1)


def test_create_job_takes_angles_per_section(images):
    """``angles=`` in either form: the whole stack's, or each section its own
    (a section not named stays flat); a mapping mixing the two is refused."""
    import langslice

    per_section = {ID0: {"pitch": 1.0, "yaw": -0.5}, ID1: {"pitch": -0.75}}
    with langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                              angles=per_section, pixel_size_um=PIXEL_SIZE_UM,
                              preprocess="none") as job:
        assert job.state.mixed_angles
        assert [job.state.by_id(name).angles for name in (ID0, ID1, ID2)] == [
            (1.0, -0.5), (-0.75, 0.0), (0.0, 0.0)]
    with langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                              angles={"pitch": 2, "yaw": 1}, pixel_size_um=PIXEL_SIZE_UM,
                              preprocess="none", fresh=True) as job:
        assert job.state.stack_angles == (2.0, 1.0)
    with pytest.raises(ValueError, match="mixes"):
        langslice.create_job(images, angles={"pitch": 1.0, ID0: {"pitch": 1.0}},
                             fresh=True)


def test_a_job_made_with_an_untested_profile_reopens_without_traces(images):
    import langslice

    langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                         pixel_size_um=PIXEL_SIZE_UM, preprocess="none",
                         image_model=langslice.image_model("openai-oauth",
                                                           prompt=PROFILE_PROMPT)).close()
    with langslice.open_job(images) as job:
        assert "trace_borders" not in job.verbs
    with langslice.open_job(images, image_model=Stub()) as job:
        assert "trace_borders" in job.verbs


def test_open_job_refuses_a_model_for_a_job_without_one(images):
    import langslice

    langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS).close()
    with pytest.raises(ValueError, match="without an image model"):
        langslice.open_job(images, image_model=Stub())


# --- the custom profile's prompt is the one sent and saved ---------------------------------


def test_a_custom_profiles_prompt_is_sent_saved_and_marked_untested(images):
    import langslice

    stub = Stub()
    profile = langslice.image_model(stub, prompt=PROFILE_PROMPT, photograph_first=True,
                                    name="lab-v1")
    with langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                              pixel_size_um=PIXEL_SIZE_UM, preprocess="none",
                              image_model=profile) as job:
        assert job.trace_borders(id=ID0)["status"] in ("running", "ok")
        edited = PROFILE_PROMPT.replace("{plane}", "coronal") + " Keep the lines thin."
        assert job.trace_borders(id=ID1, prompt=edited)["status"] in ("running", "ok")
        job.job.settle_image_corrections()
        first, second = (job.state.by_id(name).image_correction for name in (ID0, ID1))
    sent = PROFILE_PROMPT.replace("{plane}", "coronal")
    assert [request.prompt for request in stub.requests] == [sent, edited]
    # photograph_first: Image 1 is the clean photograph, Image 2 the placed borders.
    request = stub.requests[0]
    assert _colour(request.slice_image) < _colour(request.reference_images[0])
    folder = images / "langslice"
    for record in (first, second):
        assert record["status"] == "ok"
        assert record["untested"] is True and record["profile"] == "lab-v1"
        attempt = folder / record["artifact_dir"]
        assert (attempt / "base_prompt.txt").read_text() == sent
        saved = json.loads((attempt / "result.json").read_text())
        assert saved["untested"] is True and saved["profile"] == "lab-v1"
        assert json.loads((attempt / "request.json").read_text())["untested"] is True
    assert (folder / first["artifact_dir"] / "prompt.txt").read_text() == sent
    assert (folder / second["artifact_dir"] / "prompt.txt").read_text() == edited
    assert "{+Keep the lines thin.+}" in (
        folder / second["artifact_dir"] / "prompt_diff.txt").read_text()
    assert first["prompt_edited"] is False and second["prompt_edited"] is True


def test_a_builtin_profile_is_not_marked(images):
    """A provider's own model and prompt: nothing added to the trace record
    (the goldens hold it byte for byte)."""
    import dataclasses

    import langslice

    stub = Stub()
    model = dataclasses.replace(langslice.image_model("openai-oauth"),
                                call=stub.segmentation)
    with langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                              pixel_size_um=PIXEL_SIZE_UM, preprocess="none",
                              image_model=model) as job:
        job.trace_borders(id=ID0)
        job.job.settle_image_corrections()
        record = job.state.by_id(ID0).image_correction
    assert record["status"] == "ok" and "untested" not in record and "profile" not in record
    assert stub.requests[0].prompt == langslice.default_prompt("openai-oauth")


# --- register_section ----------------------------------------------------------------------


def test_register_section_with_a_transform_traces_fits_and_exports(images, tmp_path, calls):
    import langslice

    stub = Stub()
    result = langslice.register_section(
        images / ID0, position_mm=POSITIONS[ID0], transform=TRANSFORMS[ID0],
        pitch_deg=1.0, yaw_deg=-0.5, pixel_size_um=PIXEL_SIZE_UM,
        image_model=langslice.image_model(stub, prompt=PROFILE_PROMPT),
        folder=tmp_path / "one", arrays=True)
    assert calls == ["trace_borders", "fit_deformable", "submit"]
    assert result.ok and result.submitted
    assert result.job_folder == tmp_path / "one" / "langslice"
    assert (tmp_path / "one" / ID0).is_file()
    section = result.section(ID0)
    assert section.untested is True and section.trace["untested"] is True
    for path in (section.coords, section.labels, section.maps, section.residual,
                 result.registration, result.exports["quicknii"], result.exports["visualign"]):
        assert path is not None and path.is_file(), path
    assert section.arrays is not None
    coords, labels = section.arrays["coords"], section.arrays["labels"]
    assert coords.ndim == 3 and coords.shape[-1] == 3 and coords.dtype == np.float32
    assert labels.shape == coords.shape[:2] and labels.dtype == np.uint32
    assert np.isfinite(coords).any() and (labels > 0).any()
    document = json.loads(result.registration.read_text())
    entry = document["sections"][0]
    assert document["submitted"] is True and entry["mapping"] == "linear + residual"
    assert entry["parameters"]["plane"]["pitch_deg"] == 1.0
    assert entry["parameters"]["affine"]["params"] == TRANSFORMS[ID0]
    assert [request.prompt for request in stub.requests] == [
        PROFILE_PROMPT.replace("{plane}", "coronal")]


def test_register_section_with_a_position_only_aligns_first(images, tmp_path, calls):
    import langslice

    result = langslice.register_section(
        images / ID1, position_mm=POSITIONS[ID1], pixel_size_um=PIXEL_SIZE_UM,
        image_model=Stub(), folder=tmp_path / "one")
    assert calls == ["fit_affine", "trace_borders", "fit_deformable", "submit"]
    assert result.ok
    state = json.loads((result.job_folder / "state.json").read_text())
    assert state["slices"][0]["transform"]["kind"] == "elastix"  # fit_affine's default
    assert result.section(ID1).residual is not None


def test_register_section_takes_an_array_and_no_image_model(images, tmp_path, calls):
    import langslice

    plane = np.asarray(Image.open(images / ID0))[..., 0].astype(np.uint16) * 200
    result = langslice.register_section(
        plane, position_mm=POSITIONS[ID0], transform=TRANSFORMS[ID0],
        pixel_size_um=PIXEL_SIZE_UM, folder=tmp_path / "array", name="M01_s07.tif")
    assert calls == ["fit_deformable", "submit"]  # no image model: the stain fit, no trace
    written = tmp_path / "array" / "M01_s07.tif"
    import tifffile

    assert tifffile.imread(written).dtype == np.uint16
    section = result.section("M01_s07.tif")
    assert section.trace is None and section.untested is False and section.problem is None
    assert section.coords.is_file()


def test_register_section_refuses_a_shared_folder(images, tmp_path):
    import langslice

    before = {path.name: path.read_bytes() for path in images.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="other section images"):
        langslice.register_section(images / ID0, position_mm=0.1, folder=images)
    # A same-named file in the folder is never replaced; nothing is written.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copy2(images / ID1, elsewhere / ID0)
    with pytest.raises(ValueError, match="other section images"):
        langslice.register_section(elsewhere / ID0, position_mm=0.1, folder=images)
    assert {path.name: path.read_bytes() for path in images.iterdir()
            if path.is_file()} == before
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy2(images / ID1, alone / ID0)
    with pytest.raises(ValueError, match="already holds a different"):
        langslice.register_section(images / ID0, position_mm=0.1, folder=alone)
    assert (alone / ID0).read_bytes() == (images / ID1).read_bytes()


def test_register_section_says_why_a_section_failed(images, tmp_path):
    import langslice

    def broken(_request: Any) -> Any:
        raise RuntimeError("model down")

    with pytest.raises(langslice.RegistrationError, match="model down"):
        langslice.register_section(images / ID0, position_mm=POSITIONS[ID0],
                                   transform=TRANSFORMS[ID0], pixel_size_um=PIXEL_SIZE_UM,
                                   image_model=broken, folder=tmp_path / "one")
    assert (tmp_path / "one" / "langslice" / "state.json").is_file()  # kept for a look


# --- lean vs full ------------------------------------------------------------------------


def test_lean_keeps_the_results_only(images, tmp_path):
    import langslice

    def run(output: str) -> set[str]:
        result = langslice.register_section(
            images / ID0, position_mm=POSITIONS[ID0], pixel_size_um=PIXEL_SIZE_UM,
            image_model=Stub(), folder=tmp_path / output, output=output,
            affine_method="silhouette")
        assert result.ok
        return files_in(result.job_folder)

    full, lean = run("full"), run("lean")
    stem = Path(ID0).stem
    results = {"job.json", "state.json", "registration.json", "exports/quicknii.json",
               "exports/visualign.json",
               *(f"sections/{stem}/{name}" for name in (
                   "coords.tif", "labels.tif", "labels_fiji.tif", "labels.csv", "tissue.png",
                   "residual.tif", "maps.json"))}
    assert results <= lean and results <= full
    for kept in ("image_correction/", "deformable/"):
        assert any(f"sections/{stem}/{kept}" in name for name in lean)
    skipped = ("views/", "views.jsonl", "views.seq", "history/", "logs/", "AGENTS.md",
               "CLAUDE.md", f"sections/{stem}/views/")
    assert not [name for name in lean if name.startswith(skipped)]
    assert [name for name in full if name.startswith(skipped)]
    assert {"views.jsonl", "AGENTS.md", "logs/events.jsonl"} <= full
    assert any(name.startswith("history/step-") for name in full)
    assert json.loads((tmp_path / "lean" / "langslice" / "job.json").read_text())[
        "spec"]["output_level"] == "lean"
    assert "output_level" not in json.loads(
        (tmp_path / "full" / "langslice" / "job.json").read_text())["spec"]


def test_a_lean_job_still_undoes_within_its_process(images):
    import langslice

    with langslice.create_job(images, positions=POSITIONS, output="lean",
                              pixel_size_um=PIXEL_SIZE_UM, preprocess="none") as job:
        job.fit_affine(slices=[ID0], method="silhouette")
        assert job.state.by_id(ID0).transform is not None
        assert job.undo()["status"] == "ok"
        assert job.state.by_id(ID0).transform is None
    assert not (images / "langslice" / "history").exists()


# --- the folder-level pipeline -------------------------------------------------------------


def test_register_job_over_a_folder(images, calls):
    import langslice

    stub = Stub()
    job = langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                               pixel_size_um=PIXEL_SIZE_UM, preprocess="none",
                               image_model=stub, output="lean")
    result = langslice.register_job(job)
    assert result.ok and len(stub.requests) == 3
    assert calls == ["trace_borders"] * 3 + ["fit_deformable", "submit"]
    assert result.job_folder == images / "langslice"
    assert [section.id for section in result.sections] == [ID0, ID1, ID2]
    assert all(section.residual is not None and section.untested for section in result.sections)
    document = json.loads(result.registration.read_text())
    assert [entry["mapping"] for entry in document["sections"]] == ["linear + residual"] * 3
