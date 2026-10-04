"""The core library (and the job layer on it) imports without the agent
framework or any model client.

Each core module is imported in a fresh interpreter (a module another test
loaded would hide the leak) and must leave no ``google.adk``,
``google.genai``, ``litellm`` or ``openai`` module behind. Those belong to
the ADK and MCP doors (``langslice.adk``, the MCP server), the agent driver
and the providers; the tool door itself (the toolbox, the ``view`` options)
returns plain pictures and is checked too. The operations
(``langslice.ops``) must also leave no door module behind, and the job
layer's files (``langslice.job``) neither a door nor an operation.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

CORE_MODULES = (
    "langslice.linear.workspace",
    "langslice.linear.render",
    "langslice.linear.display",
    "langslice.linear.transform",
    "langslice.linear.deformation",
    "langslice.linear.appearance",
    "langslice.linear.atlas_fetch",
    "langslice.linear.opening",
    "langslice.linear.job",
    "langslice.registration_handoff",
    "langslice.registration_tool",
    "langslice.core.handoff",
    "langslice.providers.registry",
    "langslice.deformable",
    "langslice.space",
    "langslice.affine",
    "langslice.atlas",
    "langslice.core",
    "langslice.core.pictures",
    "langslice.core.placement",
    "langslice.core.layers",
    "langslice.core.jpeg",
    "langslice.core.sizes",
    "langslice.core.captions",
    "langslice.core.sections",
    "langslice.core.status",
    "langslice.core.sheets",
    "langslice.core.canvas",
    "langslice.job",
    "langslice.job.layout",
    "langslice.job.history",
    "langslice.job.index",
    "langslice.job.migrate",
    "langslice.job.views",
    "langslice.ops",
    "langslice.ops.refusal",
    "langslice.ops.positions",
    "langslice.ops.order",
    "langslice.ops.orientation",
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
)

#: The doors: an operation (``langslice.ops``) must load none of them.
DOORS = (
    "langslice.linear.toolbox",
    "langslice.linear.view_options",
    "langslice.adk",
    "langslice.mcp_server",
)

FORBIDDEN = ("google.adk", "google.genai", "litellm", "openai")

#: Doors that return plain pictures and data (the ADK driver and the MCP
#: server package them): they load no agent framework or model client either.
PLAIN_DOORS = (
    "langslice.linear.toolbox",
    "langslice.linear.view_options",
)

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
        capture_output=True, text=True, timeout=300, check=False,
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
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stderr
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert loaded == [], f"{module} loads {loaded[:5]}"
