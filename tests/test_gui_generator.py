"""The prompt generator (``langslice.hosts.gui.generator``, ``static/generator.js``):
the form's tools come from the toolset, the people's labels never reach a
model, and the message for a few forms, exactly. The message is built in the
page; these tests run its pure functions under node (skipped without node)
and check that the command it holds is one ``langslice-job FOLDER init``
reads into a job spec."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from langslice.core import tool_labels, toolset
from langslice.hosts.gui import generator

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "langslice" / "hosts" / "gui" / "static" / "generator.js"
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
    for tool in toolset.TOOLS:
        names.add(tool)
        names.update(toolset.option_names(tool))
    return names


# --- the form's data ------------------------------------------------------------


def test_the_tools_are_the_catalogs_without_the_scripting_verbs() -> None:
    rows = generator.tools()
    expected = [entry["name"] for entry in toolset.catalog()
                if not toolset.TOOLS[entry["name"]].scripting]
    assert [row["name"] for row in rows] == expected
    assert "export_maps" not in expected
    for row in rows:
        assert row["label"] == tool_labels.TOOL_LABELS[row["name"]]
        assert [o["name"] for o in row["options"]] == list(toolset.option_names(row["name"]))
        assert all(o["label"] == tool_labels.TOOL_LABELS[o["name"]] for o in row["options"])
    position_only = {row["name"] for row in rows if "position" in row["runs_in"]}
    assert "position_sections" in position_only and "elastix_affine" not in position_only


def test_a_tool_the_toolset_no_longer_has_is_not_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    fewer = {name: tool for name, tool in toolset.TOOLS.items() if name != "zoom"}
    monkeypatch.setattr(toolset, "TOOLS", fewer)
    assert "zoom" not in {row["name"] for row in generator.tools()}


def test_every_tool_and_option_has_a_label_and_no_label_names_another() -> None:
    assert set(tool_labels.TOOL_LABELS) == _names()
    assert set(tool_labels.TASK_LABELS) == set(t["name"] for t in generator.payload()["tasks"])
    assert set(tool_labels.GROUP_LABELS) == set(toolset.GROUPS)


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
    from langslice.doors.cli.job import init_parser
    from langslice.doors.cli.linear import spec_from_args

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
        "  --thickness 40 \\\n"
        "  --interval 120 \\\n"
        "  --pixel-size-um 0.65 \\\n"
        "  --disable look:atlas,zoom,trace_borders \\\n"
        "  --frozen '{\"*\":[\"angles\",\"orientation\"]}' \\\n"
        "  --damaged '{\"s012.tif\":\"left cortex torn\",\"s013.tif\":\"damaged\"}' \\\n"
        "  --fact 'The file names sort in the order the sections were cut.' \\\n"
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
    assert argv[argv.index("--disable") + 1] == "zoom"
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
            from langslice.core.spec import JobSpec
            JobSpec(image_folder=".", tasks=form["tasks"], disabled=form["off"])


def test_warping_alone_warns_and_nothing_chosen_leaves_the_tasks_to_the_agent() -> None:
    warp, nothing = _run([{**BASE, "tasks": ["nonlinear"]}, {**BASE, "tasks": []}])
    assert warp["warnings"] and warp["text"]
    assert "--tasks" not in nothing["text"]
    assert "leaves out what to do (place, align, warp), the cutting plane" in nothing["text"]
