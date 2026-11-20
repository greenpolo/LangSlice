"""A script finds every part of the library from ``help()``.

A coding agent learns the library by calling ``help(langslice)``,
``help(job)`` and ``help(job.context)``. These tests hold that every tool of
a job is a method ``help(job)`` lists with its arguments, and that every
public name a script reaches (the package, the job handle, the context and
its six parts, :mod:`langslice.plugin_api`) says what it does in a
docstring.
"""

from __future__ import annotations

import inspect
import pydoc
from pathlib import Path

import langslice
import langslice.plugin_api as plugin_api
from langslice.doors.library.handle import JobHandle
from langslice.doors.library.script_context import ScriptContext
from langslice.plugins.fits import Fits
from langslice.plugins.frames import Frames
from langslice.plugins.pictures import Pictures
from langslice.plugins.proposals import Proposals
from langslice.plugins.reads import Reads
from langslice.plugins.writes import Writes
from tests.synthetic_job import PIXEL_SIZE_UM, POSITIONS

SCRIPT_CLASSES = (JobHandle, ScriptContext, Reads, Frames, Proposals, Fits, Writes, Pictures)


def undocumented_members(owner: type) -> list[str]:
    return [
        f"{owner.__name__}.{name}"
        for name, member in inspect.getmembers(owner)
        if not name.startswith("_")
        and (callable(member) or isinstance(member, property))
        and not (inspect.getdoc(member) or "").strip()
    ]


def test_every_public_name_a_script_reaches_has_a_docstring() -> None:
    missing = [name for owner in SCRIPT_CLASSES for name in undocumented_members(owner)]
    missing += [
        f"langslice.{name}"
        for name in langslice.__all__
        if callable(getattr(langslice, name))
        and not (inspect.getdoc(getattr(langslice, name)) or "").strip()
    ]
    missing += [
        f"plugin_api.{name}"
        for name in plugin_api.__all__
        if (
            inspect.isclass(getattr(plugin_api, name))
            or inspect.isfunction(getattr(plugin_api, name))
        )
        and not (inspect.getdoc(getattr(plugin_api, name)) or "").strip()
    ]
    assert not missing, "Give each a docstring that says what it does:\n" + "\n".join(missing)


def test_the_package_help_names_every_way_in() -> None:
    text = inspect.getdoc(langslice) or ""
    named = ["help(job)", "help(job.context)", "plugin_api", "AGENTS.md", *langslice.__all__]
    absent = [name for name in named if name != "__version__" and name not in text]
    assert not absent, f"help(langslice) does not name {absent}"


def test_help_of_a_job_lists_every_tool_with_its_arguments(images: Path) -> None:
    with langslice.create_job(
        images, positions=POSITIONS, pixel_size_um=PIXEL_SIZE_UM, preprocess="none"
    ) as job:
        text = pydoc.render_doc(job, renderer=pydoc.plaintext)
        verbs = job.verbs
        unlisted = [name for name in verbs if f"- {name}: " not in text]
        unsigned = [
            name for name in verbs if f"{name}{inspect.signature(getattr(job, name))}" not in text
        ]
        undocumented = [name for name in verbs if not inspect.getdoc(getattr(job, name))]
    assert verbs
    assert not unlisted, f"help(job)'s index leaves out {unlisted}"
    assert not unsigned, f"help(job) shows no signature for {unsigned}"
    assert not undocumented, f"these tools have no docstring: {undocumented}"
