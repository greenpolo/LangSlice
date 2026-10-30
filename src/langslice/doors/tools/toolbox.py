"""The toolbox of one run: the tools its spec switches on, built over its job.

:func:`build_tools` gathers each tool's body over one
:class:`~langslice.tools.run.Run` (a function of the tool's module that takes
the run first, :data:`TOOL_BODIES`; the closures of :mod:`.looking` and
:mod:`.fitting`; a plugin's through :mod:`langslice.doors.plugins.runtime`)
and makes each one a tool of the pipeline every call goes through
(:mod:`langslice.tools.pipeline`). It lives at the
doors because the bodies do, and because it resolves the run's image model
through the providers, which the tools do not import.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from langslice.core.fits.trace.request import ImageModel
from langslice.core.fits.trace.trace import profile_prompt
from langslice.core.pictures.sizes import (
    AUTO_RESOLUTION,
    MAX_IMAGES_PER_CALL,
    resolution_level,
)
from langslice.core.space import Plane
from langslice.core.stack.stack import Stack
from langslice.doors.plugins import provenance, runtime
from langslice.doors.tools import fitting, looking
from langslice.doors.tools.view_options import view_edge_limit
from langslice.job.job import Job
from langslice.job.settings.spec import JobSpec
from langslice.providers.connections import resolve_image_model
from langslice.tools import (
    damage,
    history,
    masks,
    notes,
    positions,
    registrations,
    submit,
    transforms,
)
from langslice.tools.catalog import enabled, verbs_for
from langslice.tools.descriptions.description import DescriptionFacts
from langslice.tools.events import EventLog, LiveCallback
from langslice.tools.pipeline import ToolBox, declared_tool
from langslice.tools.run import Run

if TYPE_CHECKING:  # import cycle: the engine builds the toolbox
    from langslice.doors.agent.engine import EngineContext

#: The tools whose body is a function of the tool's module, taking the run.
TOOL_BODIES: dict[str, Callable[..., object]] = {
    "position_sections": positions.position_sections_tool,
    "interactive_transform": transforms.interactive_transform_tool,
    "mark_damage": damage.mark_damage_tool,
    "make_mask": masks.make_mask_tool,
    "note": notes.note_tool,
    "undo": history.undo_tool,
    "redo": history.redo_tool,
    "restore_registration": registrations.restore_registration_tool,
    "submit": submit.submit_tool,
}


def with_job_prompt(spec: JobSpec, image_model: ImageModel) -> ImageModel:
    """*image_model* with the job's own base prompt (``nonlinear.image_prompt``)
    as its profile prompt, an untested ``custom`` profile
    (:func:`langslice.providers.profiles.image_model`); unchanged when the
    job has none. A model whose profile carries another prompt of its own
    is a ``ValueError``: the trace would not know which to send."""
    text = spec.nonlinear.image_prompt
    if text is None or image_model.prompt == text:
        return image_model
    if image_model.prompt is not None:
        raise ValueError("The image model's profile has a prompt of its own, and the job "
                         "sets another (nonlinear.image_prompt); give only one")
    from langslice.providers.profiles import image_model as profile

    return profile(image_model, prompt=text)


def build_tools(
    state: Stack, ctx: EngineContext, spec: JobSpec, *,
    job: Job | None = None,
    on_event: LiveCallback | None = None,
    max_view_edge: int | None = None,
    image_model: ImageModel | None = None,
    level: str | None = None,
    scripting: bool = False,
    image_model_connected: bool = True,
    door: str = "agent",
    event_door: str | None = None,
) -> ToolBox:
    """Build the tools this run's spec switches on, closed over *state*.

    Which tools, in which order, is the registry's
    (:func:`langslice.tools.catalog.enabled`); each tool's name, arguments and
    description are its declaration
    (:func:`langslice.tools.descriptions.render.declare`); each tool's body is
    in its module (:data:`TOOL_BODIES`), :mod:`.looking` or :mod:`.fitting`.
    The tools sit on *job*; without one, a job is made around *state* in the
    context's job folder (an empty undo history, nothing written until the
    first write).
    *max_view_edge* is the largest picture the driver model takes (None: the
    run model's lane,
    :func:`langslice.doors.tools.view_options.view_edge_limit`). *image_model*
    is the image model ``trace_borders`` calls when the run has one (None: this
    door resolves the spec's ``nonlinear.provider`` and ``image_model`` through
    :func:`langslice.providers.connections.resolve_image_model`). *level* is
    the picture-size level the tools work at (None: the run's
    ``image_resolution``; the CLI passes "auto": its caller sizes every
    picture). *scripting* adds the verbs only scripts get (``Verb.scripting``:
    ``export_maps``), for the CLI and the library. *image_model_connected*
    False: the door cannot reach the spec's image model (MCP with none
    connected), so the tools are those of a run without one: no
    ``trace_borders``, ``submit`` not waiting for traces. The spec itself is
    left as it is. *door* is who reads the declarations (``agent``, ``mcp``,
    ``cli``: ``tools.descriptions.description.DOORS``). *event_door*
    is the door the job folder's ``logs/events.jsonl`` names for every call
    (None: *door*; the library's is ``library``).

    Every call of every tool goes through the pipeline's steps
    (:data:`langslice.tools.pipeline.TOOL_STEPS`); its events go to
    *on_event* and to the job folder's ``logs/events.jsonl``.

    A job whose spec names tool plugins (``spec.plugins``) gets their tools
    after the native ones: declared from their own stubs, their bodies run
    by :mod:`langslice.doors.plugins.runtime`, through the same steps; the
    plugins' provenance is recorded in ``job.json`` and kept on the box.
    """
    if job is None:
        job = Job(state, spec, layout=ctx.layout, results_path=ctx.results_path)
    if job.state is not state or job.spec is not spec:
        raise ValueError("build_tools: the job must hold this state and spec")
    if job.workspace is None:
        job.workspace = ctx
    # The agent tools and MCP spend a model's context on pictures: capped. The
    # agent CLI and the library (scripting) leave that to the harness.
    job.picture_cap = None if scripting else MAX_IMAGES_PER_CALL
    level = level or resolution_level(ctx)
    traces_on = spec.nonlinear.uses_image_model and image_model_connected
    if traces_on and image_model is None:
        image_model = resolve_image_model(spec.nonlinear.provider, spec.nonlinear.image_model)
    if traces_on and image_model is not None:
        image_model = with_job_prompt(spec, image_model)
    prompt: str | None = None
    if traces_on and image_model is not None:
        prompt = profile_prompt(image_model, cast(Plane, state.plane))[0]
    variant = DescriptionFacts.from_spec(
        spec, auto=level == AUTO_RESOLUTION, image_model=traces_on,
        door=door, prompt=prompt, uncapped=scripting,
        brain_scale=ctx.brain_scale)
    box = ToolBox(job, variant, max_view_edge=int(max_view_edge or view_edge_limit(ctx)),
                  verbs=verbs_for(spec), door=event_door or door,
                  observers=tuple(observer for observer in (on_event, EventLog(
                      job, event_door or door)) if observer is not None))
    run = Run(job=job, ctx=ctx, spec=spec, box=box, level=level,
              traces_on=traces_on, image_model=image_model)
    box.run = run
    bodies: dict[str, Callable[..., Any]] = {
        **looking.bodies(run), **fitting.bodies(run),
        **{name: functools.partial(body, run)
           for name, body in TOOL_BODIES.items()},
    }
    if spec.plugins:
        bodies.update(runtime.bodies(run, variant))
        box.plugins = provenance.records(variant)
        provenance.record(job, box.plugins)
    box.tools = [declared_tool(name, bodies[name], box) for name in enabled(
        spec, scripting=scripting, image_model=image_model_connected)]
    return box
