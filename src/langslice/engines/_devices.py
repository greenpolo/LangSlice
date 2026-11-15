"""Name the torch device an engine fits on in LangSlice's process.

STalign and emlddmm take a ``device`` option: ``auto`` (the GPU when torch
sees one, else the CPU), ``cpu``, ``cuda`` or a numbered GPU such as
``cuda:1``. A fit on a GPU takes the machine's GPU queue
(``langslice.plugin_api.gpu_turn``).
"""

from __future__ import annotations

#: The device option that picks the GPU when there is one.
AUTO = "auto"
#: The start of every GPU device's name, and the option for the first one.
GPU = "cuda"
#: The first GPU, as torch names it.
FIRST_GPU = "cuda:0"
CPU = "cpu"


def resolve_device(option: str) -> str:
    """Resolve a device option to the device a fit runs on.

    Parameters
    ----------
    option
        ``auto``, ``cpu``, ``cuda`` or a numbered GPU such as ``cuda:1``.

    Returns
    -------
    str
        ``cuda:0`` for ``cuda`` (and for ``auto`` when torch sees a GPU),
        a numbered GPU as given, else the option as given (``cpu`` for
        ``auto`` without a GPU).
    """
    import torch  # local: slow to import, needed only when a fit runs

    if option == AUTO:
        option = GPU if torch.cuda.is_available() else CPU
    if option == GPU:
        return FIRST_GPU
    return option


def on_gpu(device: str) -> bool:
    """Return whether a device is a GPU.

    Parameters
    ----------
    device
        A device as :func:`resolve_device` names it.

    Returns
    -------
    bool
        True for ``cuda`` and every numbered GPU.
    """
    return device.startswith(GPU)
