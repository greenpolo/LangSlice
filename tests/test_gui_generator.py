"""The prompt generator (``langslice.hosts.gui.generator``, ``static/generator.js``):
the form's tools come from the tool table, the people's labels never reach a
model, and the message for a few forms, exactly. The message is built in the
page; these tests run its pure functions under node (skipped without node)
and check that the command it holds is one ``langslice-job FOLDER init``
reads into a job spec. The tab's stylesheet uses the window's tokens only."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from langslice.hosts.gui import generator
from langslice.job.settings import labels as tool_labels
from langslice.job.settings import tool_table

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "langslice" / "hosts" / "gui" / "static"
SCRIPT = STATIC / "generator.js"
SNAPSHOTS = ROOT / "tests" / "snapshots" / "agent_text"
NODE = shutil.which("node")

#: A form as the page starts it, the folder filled in.
BASE: dict[str, Any] = {
    "folder": "/data/brain1", "atlas": "allen_mouse_25um", "plane": "", "thickness": "",
    "interval": "", "pixelSize": "", "tasks": ["position", "transform"], "off": [],
    "ordered": False, "damaged": "", "keep": {}, "notes": "",
}

REQUEST = ("Please register my brain sections with LangSlice, following its register-brain "
           "skill: create the job with the init command below, then launch the registration "
           "and check the result.")


def _names() -> set[str]:
    names = set()
    for tool in tool_table.TOOLS:
        names.add(tool)
        names.update(tool_table.option_names(tool))
    return names


# --- the form's data ------------------------------------------------------------


def test_the_tools_are_the_catalogs_without_the_scripting_verbs() -> None:
    rows = generator.tools()
    expected = [entry["name"] for entry in tool_table.catalog()
                if not tool_table.TOOLS[entry["name"]].scripting]
    assert [row["name"] for row in rows] == expected
    assert "export_maps" not in expected
    for row in rows:
        assert row["label"] == tool_labels.TOOL_LABELS[row["name"]]
        assert [o["name"] for o in row["options"]] == list(tool_table.option_names(row["name"]))
        assert all(o["label"] == tool_labels.TOOL_LABELS[o["name"]] for o in row["options"])
    position_only = {row["name"] for row in rows if "position" in row["runs_in"]}
    assert "position_sections" in position_only and "elastix_affine" not in position_only


def test_a_tool_the_table_no_longer_has_is_not_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    fewer = {name: tool for name, tool in tool_table.TOOLS.items() if name != "zoom"}
    monkeypatch.setattr(tool_table, "TOOLS", fewer)
    assert "zoom" not in {row["name"] for row in generator.tools()}


def test_every_tool_and_option_has_a_label_and_no_label_names_another() -> None:
    assert set(tool_labels.TOOL_LABELS) == _names()
    assert set(tool_labels.TASK_LABELS) == set(t["name"] for t in generator.payload()["tasks"])
    assert set(tool_labels.GROUP_LABELS) == set(tool_table.GROUPS)


def test_the_labels_are_never_model_facing() -> None:
    text = "\n".join(path.read_text(encoding="utf-8").lower()
                     for path in sorted(SNAPSHOTS.glob("*.md")))
    assert text
    labels = [*tool_labels.TOOL_LABELS.values(), *tool_labels.TASK_LABELS.values(),
              *tool_labels.TASK_ABOUT.values(),
              *(label for label, _ in tool_labels.GROUP_LABELS.values())]
    assert [label for label in labels if label.lower() in text] == []
    importers = [path.relative_to(ROOT).as_posix()
                 for path in (ROOT / "src" / "langslice").rglob("*.py")
                 if "tool_labels" in path.read_text(encoding="utf-8")
                 and path.name != "tool_labels.py"]
    assert importers == ["src/langslice/hosts/gui/generator.py"]


def test_the_payload_lists_atlases_tasks_and_what_a_kept_property_switches_off() -> None:
    data = generator.payload()
    names = [atlas["name"] for atlas in data["atlases"]]
    assert data["default_atlas"] == "allen_mouse_25um" and data["default_atlas"] in names
    assert {atlas["species"] for atlas in data["atlases"]} <= set(data["species"])
    assert next(a for a in data["atlases"] if a["name"] == "whs_sd_rat_39um")["species"] == "Rat"
    assert [t["name"] for t in data["tasks"] if t["default"]] == ["position", "transform"]
    assert data["frozen_off"]["orientation"] == ["interactive_transform:flip",
                                                 "interactive_transform:rotate"]
    json.dumps(data)  # plain data


def test_the_tab_draws_with_the_window_tokens_and_no_html() -> None:
    """generator.css names only tokens.css's custom properties and writes no
    colour of its own; generator.js puts every text in as text, never HTML."""
    tokens = set(re.findall(r"(--[a-z0-9-]+)\s*:",
                            (STATIC / "tokens.css").read_text(encoding="utf-8")))
    css = (STATIC / "generator.css").read_text(encoding="utf-8")
    assert set(re.findall(r"var\((--[a-z0-9-]+)", css)) <= tokens
    assert re.findall(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?)\(", css) == []
    script = SCRIPT.read_text(encoding="utf-8")
    assert re.findall(r"innerHTML|outerHTML|insertAdjacentHTML|setAttribute\('style'", script) == []


def test_the_server_answers_the_generator_data(tmp_path: Path) -> None:
    from langslice.hosts.gui.server import GuiServer
    from langslice.hosts.gui.watch import JobList

    gui = GuiServer(JobList([], recent=None))
    try:
        host = f"127.0.0.1:{gui.port}"
        answer = gui.answer(f"/{gui.token}/generator", host)
        assert not isinstance(answer, int)
        assert json.loads(answer[0])["default_atlas"] == "allen_mouse_25um"
        for name in ("generator.js", "generator.css"):
            assert not isinstance(gui.answer(f"/{gui.token}/{name}", host), int)
    finally:
        gui.http.server_close()


# --- the message ------------------------------------------------------------------


def _run(forms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node = NODE
    if node is None:
        pytest.skip("node is not installed")
    program = (
        "const G = require(process.argv[1]);"
        "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        "process.stdout.write(JSON.stringify(input.forms.map(f => G.message(input.data, f))));"
    )
    done = subprocess.run([node, "-e", program, str(SCRIPT)], check=True, capture_output=True,
                          text=True, input=json.dumps({"data": generator.payload(),
                                                       "forms": forms}))
    return json.loads(done.stdout)


def _command(text: str) -> list[str]:
    block = text.split("```bash\n", 1)[1].split("\n```", 1)[0]
    return shlex.split(block.replace("\\\n", " "))  # a line continuation, as bash reads it


def _spec(argv: list[str]) -> Any:
    from langslice.doors.agent_cli.calls import init_parser
    from langslice.doors.commands.linear import spec_from_args

    assert argv[0] == "langslice-job" and argv[2] == "init"
    return spec_from_args(init_parser().parse_args(argv[3:]), argv[1])


def test_the_default_form_names_what_it_leaves_to_the_agent() -> None:
    (made,) = _run([BASE])
    assert made["problems"] == [] and made["warnings"] == []
    assert made["text"] == (
        f"{REQUEST} The command leaves out the cutting plane, the section thickness, the "
        "section interval and the pixel size (when the image files do not record it): choose "
        "these yourself, or ask me when the images do not tell you.\n\n"
        "```bash\n"
        "langslice-job /data/brain1 init \\\n"
        "  --tasks position,transform \\\n"
        "  --atlas allen_mouse_25um\n"
        "```\n")
    spec = _spec(_command(made["text"]))
    assert spec.tasks == ["position", "transform"] and spec.atlas == "allen_mouse_25um"


def test_a_filled_form_gives_every_setting_and_init_reads_it() -> None:
    form = {**BASE, "folder": '"C:\\Users\\Lab\\brain 2"', "atlas": "whs_sd_rat_39um",
            "plane": "coronal", "thickness": "40", "interval": "120", "pixelSize": "0.65",
            "tasks": ["position", "transform", "nonlinear"],
            "off": ["zoom", "interactive_transform:flip", "trace_borders", "look:atlas"],
            "ordered": True, "damaged": "s012.tif: left cortex torn\ns013.tif\n",
            "keep": {"orientation": True, "angles": True},
            "notes": "Injection in the right hemisphere; it's bright."}
    (made,) = _run([form])
    assert made["problems"] == []
    assert made["text"] == (
        f"{REQUEST} The command is quoted for bash; if your shell is another, quote it for "
        "yours.\n\n"
        "```bash\n"
        "langslice-job 'C:\\Users\\Lab\\brain 2' init \\\n"
        "  --tasks position,transform,nonlinear \\\n"
        "  --atlas whs_sd_rat_39um \\\n"
        "  --plane coronal \\\n"
        "  --position-thickness-um 40 \\\n"
        "  --position-interval-um 120 \\\n"
        "  --pixel-size-um 0.65 \\\n"
        "  --disabled look:atlas,zoom,trace_borders \\\n"
        "  --frozen '{\"*\":[\"angles\",\"orientation\"]}' \\\n"
        "  --damaged '{\"s012.tif\":\"left cortex torn\",\"s013.tif\":\"damaged\"}' \\\n"
        "  --facts 'The file names sort in the order the sections were cut.' \\\n"
        "  --notes \"Injection in the right hemisphere; it's bright.\"\n"
        "```\n")
    spec = _spec(_command(made["text"]))
    assert spec.image_folder == "C:\\Users\\Lab\\brain 2"
    assert (spec.atlas, spec.plane) == ("whs_sd_rat_39um", "coronal")
    assert (spec.position.thickness_um, spec.position.interval_um) == (40, 120)
    assert spec.inputs["pixel_size_um"] == 0.65
    assert spec.disabled == ["look:atlas", "zoom", "trace_borders"]
    assert spec.frozen == {"*": ["angles", "orientation"]}
    assert spec.inputs["damaged"] == {"s012.tif": "left cortex torn", "s013.tif": "damaged"}
    assert spec.facts == ["The file names sort in the order the sections were cut."]
    assert {"interactive_transform:flip", "interactive_transform:rotate"} <= set(spec.switched_off)


def test_a_switched_off_tool_of_a_task_not_chosen_is_left_out() -> None:
    form = {**BASE, "tasks": ["position"], "off": ["elastix_affine", "fireants_syn", "zoom"],
            "keep": {"orientation": True}}
    (made,) = _run([form])
    argv = _command(made["text"])
    assert argv[argv.index("--disabled") + 1] == "zoom"
    assert "--frozen" not in argv  # orientation is the transform task's, which is off
    assert "elastix_affine" not in made["text"] and "fireants_syn" not in made["text"]


def test_a_choice_of_tools_the_spec_would_refuse_gives_no_message() -> None:
    forms = [
        {**BASE, "folder": "  "},
        {**BASE, "off": ["look:section", "look:atlas", "look:overlay", "look:positioning"]},
        {**BASE, "off": ["position_sections"]},
        {**BASE, "off": ["look:overlay"]},
        {**BASE, "thickness": "40.5"},
    ]
    made = _run(forms)
    assert all(m["text"] == "" for m in made)
    assert made[0]["problems"] == ["Enter the folder that holds the section images."]
    assert made[1]["problems"][0].startswith('"See sections and the atlas" needs at least one')
    assert made[2]["problems"] == [
        '"Place along the atlas" needs one of these tools on: "Place sections along the '
        'atlas, and tilt the cutting plane".']
    assert made[3]["problems"] == [
        '"Move, turn and stretch a section by hand", "Automatic alignment without bending '
        '(Elastix)" and "Return a section to an earlier result" need "See sections and the '
        'atlas: atlas outlines drawn over a section": turn it back on, or turn those off too.']
    assert made[4]["problems"] == ["Section thickness is a whole number of micrometres (µm)."]
    for form in forms[1:4]:  # the spec refuses the same lists
        with pytest.raises(ValueError):
            from langslice.job.settings.spec import JobSpec
            JobSpec(image_folder=".", tasks=form["tasks"], disabled=form["off"])


def test_warping_alone_warns_and_nothing_chosen_leaves_the_tasks_to_the_agent() -> None:
    warp, nothing = _run([{**BASE, "tasks": ["nonlinear"]}, {**BASE, "tasks": []}])
    assert warp["warnings"] and warp["text"]
    assert "--tasks" not in nothing["text"]
    assert "leaves out what to do (place, align, warp), the cutting plane" in nothing["text"]


# --- a job's own settings: the form started from job.json ---------------------------

#: Two snapshots of a job prepared in ABBA (``abba.prepare``): the first
#: registered in ABBA, with a warp of its own.
ABBA_FILES = ("section_0001.tif", "section_0002.tif")


def _prepared_abba_job(root: Path, *, thickness_um: int = 40, interval_um: int = 120) -> Path:
    """A job folder as ``abba.prepare`` writes it (protocol v3): ``job.json``
    with ABBA's inputs, the registered snapshot frozen, ``transform.angles``
    on and the ``host.abba`` block; no checkpoint. Returns the job folder."""
    from langslice.job.layout import JobLayout, job_folder_for, write_job_file
    from langslice.job.settings.spec import JobSpec

    images = root / "abba" / "3f9c2a1b7d0e"
    images.mkdir(parents=True)
    spec = JobSpec.from_dict({
        "image_folder": str(images),
        "inputs": {"positions": {ABBA_FILES[0]: 5.2, ABBA_FILES[1]: 5.45},
                   "order": list(ABBA_FILES), "pixel_size_um": 25.0,
                   "angles": {"pitch": 1.5, "yaw": -0.5}},
        "host_preprocessing": None,
        "frozen": {ABBA_FILES[0]: ["orientation", "affine", "deformation"]},
        "transform": {"angles": True},
        "resume": False,
    })
    layout = JobLayout(job_folder_for(images), image_folder=images)
    layout.folder.mkdir()
    write_job_file(layout, spec=spec.to_dict(), host={
        "format": 3, "kind": "abba", "abba": {
            "project_key": "3f9c2a1b7d0e",
            "atlas": "Adult Mouse Brain - Allen Brain Atlas V3p1",
            "sections": {ABBA_FILES[0]: "M12_s034.czi - Scene 1",
                         ABBA_FILES[1]: "M12_s035.czi - Scene 1"},
            "registered": [ABBA_FILES[0]], "existing_warp": [ABBA_FILES[0]],
            "angles_deg": {"pitch_deg": 1.5, "yaw_deg": -0.5}, "z_offset_mm": 0.0,
            "pixel_size_um": 25.0, "thickness_um": thickness_um, "interval_um": interval_um,
            "prepared_at": "2026-10-30T12:00:00+00:00"}})
    return layout.folder


def _prefill(folder: Path) -> dict[str, Any] | None:
    """The prefill of *folder* as the page receives it (plain data)."""
    found = generator.prefill(folder)
    return asdict(found) if found is not None else None


def test_the_prefill_of_a_job_prepared_in_abba(tmp_path: Path) -> None:
    folder = _prepared_abba_job(tmp_path)
    prefill = _prefill(folder)
    assert prefill is not None
    assert prefill["folder"] == str(folder.parent)
    assert (prefill["atlas"], prefill["plane"], prefill["pixel_size"]) == (
        "allen_mouse_25um", "coronal", "25")
    assert (prefill["thickness"], prefill["interval"]) == ("40", "120")  # ABBA's
    assert prefill["tasks"] == ["position", "transform"] and prefill["off"] == []
    assert prefill["keep"] == {"angles": False, "orientation": False}
    assert prefill["frozen"] == {ABBA_FILES[0]: ["orientation", "affine", "deformation"]}
    assert prefill["abba"] == {
        "atlas": "Adult Mouse Brain - Allen Brain Atlas V3p1",
        "sections": {ABBA_FILES[0]: "M12_s034.czi - Scene 1",
                     ABBA_FILES[1]: "M12_s035.czi - Scene 1"},
        "registered": [ABBA_FILES[0]], "existing_warp": [ABBA_FILES[0]],
        "pixel_size_um": 25.0, "thickness_um": 40, "interval_um": 120}
    json.dumps(prefill)
    unknown = _prepared_abba_job(tmp_path / "other", thickness_um=0, interval_um=0)
    blank = _prefill(unknown)
    assert blank is not None and (blank["thickness"], blank["interval"]) == ("", "")


def test_the_prefill_of_an_opened_job_and_of_a_folder_without_one(tmp_path: Path) -> None:
    from langslice.job.layout import JobLayout, job_folder_for, write_job_file
    from langslice.job.settings.spec import JobSpec

    images = tmp_path / "brain1"
    images.mkdir()
    spec = JobSpec(image_folder=str(images), atlas="whs_sd_rat_39um", plane="sagittal",
                   tasks=["position", "transform", "nonlinear"], disabled=["zoom"],
                   frozen={"*": ["orientation"]},
                   facts=["The file names sort in the order the sections were cut."],
                   inputs={"damaged": {"s1.tif": "torn"}})
    spec.position.thickness_um = 50
    layout = JobLayout(job_folder_for(images), image_folder=images)
    layout.folder.mkdir()
    write_job_file(layout, spec=spec.to_dict(), notes="Right striatum.")
    (layout.folder / "state.json").write_text("{}")
    prefill = _prefill(layout.folder)
    assert prefill is not None and prefill["abba"] is None
    assert (prefill["atlas"], prefill["plane"], prefill["thickness"], prefill["interval"]) == (
        "whs_sd_rat_39um", "sagittal", "50", "")
    assert prefill["off"] == ["zoom"]
    assert prefill["keep"] == {"angles": False, "orientation": True}
    assert prefill["damaged"] == "s1.tif: torn\n" and prefill["notes"] == "Right striatum."
    assert prefill["facts"] == ["The file names sort in the order the sections were cut."]
    assert generator.prefill(tmp_path / "nothing") is None
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "job.json").write_text("not json")
    assert generator.prefill(tmp_path / "broken") is None


def test_the_server_adds_the_prefill_of_a_listed_job(tmp_path: Path) -> None:
    from langslice.hosts.gui.server import GuiServer
    from langslice.hosts.gui.watch import JobList

    folder = _prepared_abba_job(tmp_path)
    empty = tmp_path / "empty" / "langslice"
    empty.mkdir(parents=True)
    gui = GuiServer(JobList([folder, empty], recent=None))
    try:
        host = f"127.0.0.1:{gui.port}"
        plain = gui.answer(f"/{gui.token}/generator", host)
        assert not isinstance(plain, int) and "prefill" not in json.loads(plain[0])
        first = gui.answer(f"/{gui.token}/generator?job=0", host)
        assert not isinstance(first, int)
        assert json.loads(first[0])["prefill"]["abba"]["registered"] == [ABBA_FILES[0]]
        second = gui.answer(f"/{gui.token}/generator?job=1", host)
        assert not isinstance(second, int) and json.loads(second[0])["prefill"] is None
        assert gui.answer(f"/{gui.token}/generator?job=2", host) == 404
        assert gui.answer(f"/{gui.token}/generator?job=x", host) == 400
        listed = gui.jobs.listing()  # the prepared job, without state.json, is listed
        assert [job["folder"] for job in listed] == [str(folder), str(empty)]
    finally:
        gui.http.server_close()


def _run_prefilled(prefill: dict[str, Any],
                   changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The message of the form started from *prefill*, once per change to it."""
    node = NODE
    if node is None:
        pytest.skip("node is not installed")
    program = (
        "const G = require(process.argv[1]);"
        "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        "const start = G.prefilled(input.data, input.prefill);"
        "process.stdout.write(JSON.stringify(input.changes.map("
        "c => G.message(input.data, {...start, ...c}))));"
    )
    done = subprocess.run([node, "-e", program, str(SCRIPT)], check=True, capture_output=True,
                          text=True, input=json.dumps({"data": generator.payload(),
                                                       "prefill": prefill, "changes": changes}))
    return json.loads(done.stdout)


def test_the_message_for_a_job_prepared_in_abba(tmp_path: Path) -> None:
    prefill = _prefill(_prepared_abba_job(tmp_path))
    assert prefill is not None
    images = prefill["folder"]
    kept, freed, kept_straight, freed_upright, odd = _run_prefilled(prefill, [
        {}, {"keepRegistrations": False},
        {"keep": {"angles": True}},
        {"keepRegistrations": False, "keep": {"orientation": True}},
        {"pixelSize": "not a number", "atlas": "", "plane": ""},
    ])
    command = (f"```bash\nlangslice-job {images} init \\\n"
               "  --tasks position,transform \\\n"
               "  --position-thickness-um 40 \\\n"
               "  --position-interval-um 120")
    quoted = " The command is quoted for bash; if your shell is another, quote it for yours."
    # The job holds its atlas, plane and pixel size: no flag, nothing left out.
    assert kept["text"] == f"{REQUEST}\n\n{command}\n```\n"
    assert freed["text"] == f"{REQUEST}{quoted}\n\n{command} \\\n  --frozen '{{}}'\n```\n"
    assert kept_straight["text"] == (
        f"{REQUEST}{quoted}\n\n{command} \\\n  --frozen '{{\"*\":[\"angles\"],"
        f"\"{ABBA_FILES[0]}\":[\"orientation\",\"affine\",\"deformation\"]}}'\n```\n")
    assert freed_upright["text"].endswith("  --frozen '{\"*\":[\"orientation\"]}'\n```\n")
    assert odd["problems"] == [] and odd["text"] == kept["text"]  # fixed fields
    for made in (kept, freed, kept_straight):
        assert "--atlas" not in made["text"] and "--pixel-size-um" not in made["text"]
    spec = _spec(_command(freed["text"]))
    assert spec.frozen == {} and spec.image_folder == images
    spec = _spec(_command(kept_straight["text"]))
    assert spec.frozen == {"*": ["angles"],
                           ABBA_FILES[0]: ["orientation", "affine", "deformation"]}


def test_the_keep_box_names_one_section_in_the_singular() -> None:
    node = NODE
    if node is None:
        pytest.skip("node is not installed")
    program = (
        "const G = require(process.argv[1]);"
        "process.stdout.write(JSON.stringify([1, 3].map(G.keepLabel)));"
    )
    done = subprocess.run([node, "-e", program, str(SCRIPT)], check=True,
                          capture_output=True, text=True)
    one, three = json.loads(done.stdout)
    assert one == ("Keep the 1 section\u2019s existing ABBA registrations "
                   "(its position may still move)")
    assert three == ("Keep the 3 sections\u2019 existing ABBA registrations "
                     "(their positions may still move)")


def test_a_prefill_without_abba_gives_the_message_of_the_same_form() -> None:
    prefill = {**BASE, "keep": {"angles": False, "orientation": False}, "facts": [],
               "frozen": {}, "abba": None}
    (from_prefill,) = _run_prefilled(prefill, [{}])
    (from_form,) = _run([BASE])
    assert from_prefill == from_form


# --- the image model ------------------------------------------------------------------


def test_the_image_models_are_the_providers_a_job_accepts() -> None:
    from langslice.core.connections import CANONICAL_PROVIDERS
    from langslice.providers import connections as registry

    rows = generator.payload()["image_models"]
    assert sorted(row["provider"] for row in rows) == sorted(CANONICAL_PROVIDERS)
    assert rows[0]["provider"] == "openai-oauth" and rows[0]["default"]
    assert rows[-1]["provider"] == "none" and rows[-1]["models"] == []
    by_name = {row["provider"]: row for row in rows}
    assert by_name["openai-oauth"]["models"] == list(registry.OPENAI_OAUTH_IMAGE_MODELS)
    assert by_name["gemini-api"]["models"] == list(registry.GEMINI_IMAGE_MODELS)
    assert by_name["gemini-api"]["default_model"] == registry.GEMINI_DEFAULT_IMAGE_MODEL
    assert [t["name"] for t in generator.tools() if t["image_model"]] == ["trace_borders"]
    for row in rows:  # every choice is one init takes
        for model in [None, *row["models"]]:
            argv = ["--nonlinear-provider", row["provider"]]
            if model is not None:
                argv += ["--nonlinear-image-model", model]
            spec = _spec(["langslice-job", "/data/brain1", "init", *argv])
            assert spec.nonlinear.provider == row["provider"]


def test_the_image_model_flags() -> None:
    warp = {**BASE, "tasks": ["position", "transform", "nonlinear"]}
    chosen, provider_only, none, not_warping, tracing_off, unset = _run([
        {**warp, "imageProvider": "gemini-api", "imageModel": "gemini-3-pro-image"},
        {**warp, "imageProvider": "openai-api"},
        {**warp, "imageProvider": "none", "imageModel": "gemini-3-pro-image"},
        {**BASE, "imageProvider": "gemini-api", "imageModel": "gemini-3-pro-image"},
        {**warp, "imageProvider": "gemini-api", "off": ["trace_borders"]},
        warp,
    ])
    assert chosen["text"].endswith(
        "  --atlas allen_mouse_25um \\\n"
        "  --nonlinear-provider gemini-api \\\n"
        "  --nonlinear-image-model gemini-3-pro-image\n```\n")
    spec = _spec(_command(chosen["text"]))
    assert (spec.nonlinear.provider, spec.nonlinear.image_model) == (
        "gemini-api", "gemini-3-pro-image")
    argv = _command(provider_only["text"])
    assert argv[argv.index("--nonlinear-provider") + 1] == "openai-api"
    assert "--nonlinear-image-model" not in argv
    argv = _command(none["text"])
    assert argv[argv.index("--nonlinear-provider") + 1] == "none"
    assert "--nonlinear-image-model" not in argv
    assert not _spec(argv).nonlinear.uses_image_model
    for made in (not_warping, tracing_off, unset):
        assert "--nonlinear" not in made["text"]
