"""The core library (and the job layer on it) imports without the agent
framework or any model client.

Each core module is imported in a fresh interpreter (a module another test
loaded would hide the leak) and must leave no ``google.adk``,
``google.genai``, ``litellm`` or ``openai`` module behind. Those belong to
the ADK and MCP doors (``langslice.doors.tools``, the MCP server), the agent driver
and the providers; the tool door itself (the toolbox, the ``view`` options)
returns plain pictures and is checked too. The operations
(``langslice.ops``) must also leave no door module behind, and the job
layer's files (``langslice.job``) neither a door nor an operation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

CORE_MODULES = (
    "langslice.core.workspace",
    "langslice.core.display",
    "langslice.core.transform",
    "langslice.core.deformation",
    "langslice.core.appearance",
    "langslice.core.channels",
    "langslice.core.atlas_fetch",
    "langslice.core.opening",
    "langslice.job.job",
    "langslice.core.nonlinear.registration_tool",
    "langslice.core.handoff",
    "langslice.providers.registry",
    "langslice.core.deformable",
    "langslice.core.space",
    "langslice.core.affine",
    "langslice.core.atlas",
    "langslice.core",
    "langslice.core.placement",
    "langslice.core.layers",
    "langslice.core.jpeg",
    "langslice.core.sizes",
    "langslice.core.captions",
    "langslice.core.sections",
    "langslice.core.status",
    "langslice.core.damage",
    "langslice.core.look",
    "langslice.core.positioning",
    "langslice.core.zoom",
    "langslice.core.canvas",
    "langslice.core.maps",
    "langslice.core.import_geometry",
    "langslice.job",
    "langslice.job.layout",
    "langslice.job.history",
    "langslice.job.index",
    "langslice.job.views",
    "langslice.job.formats",
    "langslice.job.imports",
    "langslice.ops",
    "langslice.ops.refusal",
    "langslice.ops.positions",
    "langslice.ops.damage",
    "langslice.ops.appearance",
    "langslice.ops.notes",
    "langslice.ops.transforms",
    "langslice.ops.deformable",
    "langslice.ops.submit",
    "langslice.ops.history",
    "langslice.ops.views",
    "langslice.ops.traces",
    "langslice.ops.atlas",
    "langslice.ops.registry",
    "langslice.ops.exports",
    "langslice.ops.look",
    "langslice.ops.files",
    "langslice.ops.inputs",
)

#: The doors: an operation (``langslice.ops``) must load none of them.
DOORS = (
    "langslice.doors.tools.toolbox",
    "langslice.doors.tools.view_options",
    "langslice.doors.tools",
    "langslice.doors.mcp",
    "langslice.doors",
)

FORBIDDEN = ("google.adk", "google.genai", "litellm", "openai")

#: Doors that return plain pictures and data (the ADK driver and the MCP
#: server package them): they load no agent framework or model client either.
PLAIN_DOORS = (
    "langslice.doors.tools.toolbox",
    "langslice.doors.tools.door",
    "langslice.doors.tools.looking",
    "langslice.doors.tools.changing",
    "langslice.doors.tools.fitting",
    "langslice.doors.tools.view_options",
    "langslice.doors.declarations",
    "langslice.doors.jobs",
    "langslice.doors.library",
    "langslice.providers.profiles",
    "langslice.doors.card",
    "langslice.doors.cli",
    "langslice.doors.cli.job",
)

#: The child interpreters import this tree's sources (not whichever checkout
#: the environment's editable install points at).
_SOURCES = {**os.environ, "PYTHONPATH": os.pathsep.join(
    [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
     os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}

_PROBE = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
forbidden = tuple(sys.argv[2:])
print(json.dumps(sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)))
"""


@pytest.mark.parametrize("module", CORE_MODULES + PLAIN_DOORS)
def test_core_module_loads_no_agent_or_model_client(module: str):
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, module, *FORBIDDEN],
        capture_output=True, text=True, timeout=300, check=False, env=_SOURCES,
    )
    assert done.returncode == 0, done.stderr
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert loaded == [], f"{module} loads {loaded[:5]}"


#: What the job layer's own package may not load: a door or an operation.
OPERATIONS = ("langslice.ops",)


@pytest.mark.parametrize("module", [name for name in CORE_MODULES
                                    if name.startswith(("langslice.ops", "langslice.job"))])
def test_operations_and_job_files_load_no_door(module: str):
    forbidden = DOORS + (OPERATIONS if module.startswith("langslice.job") else ())
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, module, *forbidden],
        capture_output=True, text=True, timeout=300, check=False, env=_SOURCES,
    )
    assert done.returncode == 0, done.stderr
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert loaded == [], f"{module} loads {loaded[:5]}"


_LIBRARY = """
import json, os, sys
from pathlib import Path

import langslice
from tests.golden.record import PIXEL_SIZE_UM, apply_patches, atlas_loader, write_sections
from langslice.doors.jobs import create
from langslice.core.spec import JobSpec

images = Path(sys.argv[1])
write_sections(images)
loader = atlas_loader()
create(JobSpec(image_folder=str(images), preprocess="none",
               tasks=["reorder", "position", "transform"],
               inputs={"pixel_size_um": PIXEL_SIZE_UM}), atlas_loader=loader).close()
job = langslice.open_job(images, atlas_loader=loader)
job.status()
job.position_sections(sections=[{"id": "s0.png", "position_mm": 0.1}])
job.interactive_transform(sections=[{"id": "s0.png", "rotation_deg": 1.0}])
job.look(mode="overlay", sections=["s0.png"])
job.close()
apply_patches()
model = langslice.image_model(lambda request: request.slice_image, prompt="Image 1 ... {plane}")
made = langslice.create_job(images, tasks=["nonlinear"], positions={"s0.png": 0.1},
                            transforms={"s0.png": [1, 0, 0, 0, 1, 0]},
                            pixel_size_um=PIXEL_SIZE_UM, image_model=model,
                            job_dir=images.parent / "one", atlas_loader=loader)
made.status()
made.export_maps(slices=["s0.png"])
made.close()
forbidden = tuple(sys.argv[2:])
print(json.dumps(sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)))
"""


def test_the_library_opens_a_job_without_an_agent_or_model_client(tmp_path):
    """``import langslice``, ``open_job`` and a verb or two, then ``create_job``
    with a model of the script's own and a scripting verb: the script door."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, "-c", _LIBRARY, str(tmp_path / "stack"), *FORBIDDEN],
        capture_output=True, text=True, timeout=300, check=False, cwd=repo,
        env={**os.environ, "HOME": str(tmp_path / "home"),
             "PYTHONPATH": os.pathsep.join([str(repo / "src"), str(repo)])},
    )
    assert done.returncode == 0, done.stderr[-3000:]
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert loaded == [], f"langslice.open_job loads {loaded[:5]}"
