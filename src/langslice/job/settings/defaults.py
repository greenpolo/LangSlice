"""The two defaults of a job that depend on what else the job is given.

The same on every door (the commands, MCP, the agent CLI's ``init`` and the
library): a door that does not name the setting gets what these say, and a
door that names it is not asked.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence

from langslice.job.settings.spec import ALL_TASKS

NO_IMAGE_MODEL_SAVED = (
    "trace_borders needs an image model, and this computer has no saved OpenAI "
    "sign-in or key: sign in or save a key in the LangSlice window's Settings "
    "(langslice gui), or name an image provider (--nonlinear-provider, image_model=)"
)


def default_tasks(
    image_folder: str | os.PathLike[str],
    inputs: Mapping[str, object],
    *,
    registration: bool = False,
) -> list[str]:
    """The tasks of a job whose ``tasks`` are not given.

    Every task is on unless the job's supplied answers already settle it:
    ``position`` is off when ``inputs.positions`` names every section of
    *image_folder*, ``transform`` when ``inputs.transforms`` does, and both
    when a linear registration made elsewhere is given (*registration*: it
    supplies every section's position and transform). ``nonlinear`` is never
    settled by what a host supplies.

    Parameters
    ----------
    image_folder
        The folder of section images.
    inputs
        The job's supplied answers (``JobSpec.inputs``).
    registration
        Whether a registration file is given.

    Returns
    -------
    list of str
        The tasks, in ``ALL_TASKS`` order.
    """
    if registration:
        return ["nonlinear"]
    names: set[str] = set()
    if os.path.isdir(image_folder):
        from langslice.core.sections.discovery import discover_slices

        names = {os.path.basename(path) for path in discover_slices(os.fspath(image_folder))}

    def supplied(key: str) -> bool:
        held = inputs.get(key)
        return bool(names) and isinstance(held, Mapping) and names <= set(held)

    settled = {"position": supplied("positions"), "transform": supplied("transforms")}
    return [task for task in ALL_TASKS if not settled.get(task, False)]


def choose_provider(
    given: str | None,
    *,
    enabled: Sequence[str],
    saved: Callable[[], str | None],
) -> str:
    """The image provider of a job whose ``nonlinear.provider`` is not given.

    The provider the caller names (a command's ``--nonlinear-provider``, the
    library's ``image_model=``); else, with ``trace_borders`` enabled, the
    connection saved on this computer; else ``none``: the nonlinear task fits
    deformations to the stain alone.

    Parameters
    ----------
    given
        The provider the caller named, None for none.
    enabled
        The job's ``enabled`` tools.
    saved
        Reads the saved connection
        (:func:`langslice.doors.credentials.saved_image_provider`).

    Returns
    -------
    str
        The provider name.

    Raises
    ------
    ValueError
        When ``trace_borders`` is enabled and no connection is saved.
    """
    if given is not None:
        return given
    if "trace_borders" not in enabled:
        return "none"
    provider = saved()
    if provider is None:
        raise ValueError(NO_IMAGE_MODEL_SAVED)
    return provider
