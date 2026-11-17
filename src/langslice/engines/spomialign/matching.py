"""Match two images with RoMa as SPOmiAlign does, in LangSlice's process.

RoMa (Edstedt et al. 2024; its package ``romatch``, MIT) is the dense feature
matcher SPOmiAlign (Wang et al. 2026, Brief Bioinform) builds on; LangSlice
runs the release SPOmiAlign vendors (0.1.2), which comes with the
``engines`` extra and is imported, with torch, only when a call runs
(:func:`is_installed` says whether it can). SPOmiAlign's repository carries
no licence, so none of its code is copied: each step is written from what
its ``align_and_process_images`` does. Both images are resized to its
1152 x 864 grid; RoMa outdoor matches every pixel (coarse resolution 560,
upsampled to the grid; RoMa's plain torch local correlation, its optional
fused CUDA kernel left off); each match's certainty is weighted by the
section's edge map (an FFT band-pass binarised by Otsu's threshold: weight
1 on an edge, 0.1 off it), thinned by a 3 x 3 grid non-maximum
suppression, and the matches are ranked by it.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

if TYPE_CHECKING:
    import torch

H, W = 864, 1152
COARSE_RES = 560
EDGE_SIGMA = 2.0
EDGE_DECAY = 5.0
EDGE_POWER = 1.2
OFF_EDGE_WEIGHT = 0.1
NMS_KERNEL = 3
NMS_FLOOR = 1e-6
GPU_DEVICE = "cuda"
WEIGHTS = {
    "roma_outdoor.pth": "https://github.com/Parskatt/storage/releases/"
    "download/roma/roma_outdoor.pth",
    "dinov2_vitl14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/"
    "dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}
LOCAL_CORR_WARNING = "Local correlation is not supported"


class Matcher(Protocol):
    """RoMa's dense matcher, as this module calls it."""

    def match(
        self,
        atlas: Image.Image,
        section: Image.Image,
        /,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match every pixel of one image pair (RoMa's ``match``).

        Parameters
        ----------
        atlas
            The atlas image, RGB, RoMa's image A.
        section
            The section, RGB, RoMa's image B.
        device
            The torch device the matcher runs on.

        Returns
        -------
        warp : torch.Tensor, shape (1, H, 2 W, 4)
            Both directions' matches in normalized coordinates (-1 to 1):
            the left half A to B, the right half B to A.
        certainty : torch.Tensor, shape (1, H, 2 W)
            Each match's certainty.
        """
        ...


def is_installed() -> bool:
    """Say whether RoMa (``romatch``, the ``engines`` extra) can import.

    Returns
    -------
    bool
        True when the package is installed; nothing is imported.
    """
    return importlib.util.find_spec("romatch") is not None


def weights_missing() -> list[str]:
    """List RoMa's weight files that are not in torch's hub cache.

    Returns
    -------
    list of str
        The names (keys of :data:`WEIGHTS`) of the files missing from
        ``<hub>/checkpoints``, the hub folder torch's own ``get_dir``
        names (``TORCH_HOME``, else ``~/.cache/torch``).
    """
    import torch

    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    return [name for name in WEIGHTS if not (checkpoints / name).exists()]


def resolve_device(device_name: str) -> torch.device:
    """Name the torch device a call runs on, as SPOmiAlign resolves it.

    Parameters
    ----------
    device_name
        A torch device name (``cuda``, ``cuda:1``, ``cpu``, ``mps``), or
        empty for the first GPU when torch sees one, else the CPU.

    Returns
    -------
    torch.device
        The device.

    Raises
    ------
    RuntimeError
        When the device named is not available.
    """
    import torch

    if not device_name:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    device = torch.device(device_name)
    available = True
    if device.type == GPU_DEVICE:
        index = device.index or 0
        available = torch.cuda.is_available()
        available = available and index < torch.cuda.device_count()
    elif device.type == "mps":
        available = torch.backends.mps.is_available()
    if not available:
        raise RuntimeError(
            f"Requested device {device_name!r} is not available"
        )
    return device


def load_matcher(device: torch.device) -> Matcher:
    """Build RoMa outdoor as SPOmiAlign builds it, its weights loaded.

    Parameters
    ----------
    device
        The torch device the matcher runs on.

    Returns
    -------
    Matcher
        RoMa outdoor at coarse resolution 560, upsampled to the 1152 x 864
        grid, with RoMa's plain torch local correlation. The weights come
        from torch's hub cache, downloaded there on first use.
    """
    romatch = importlib.import_module("romatch")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=LOCAL_CORR_WARNING)
        model = romatch.roma_outdoor(  # pyright: ignore[reportAny]
            device=device,
            coarse_res=COARSE_RES,
            upsample_res=(H, W),
            use_custom_corr=False,
        )
    return cast(Matcher, model)


def ranked_matches(
    model: Matcher,
    device: torch.device,
    target: npt.NDArray[np.uint8],
    source: npt.NDArray[np.uint8],
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Rank every match of the pair by its edge-weighted certainty.

    Parameters
    ----------
    model
        RoMa's matcher.
    device
        The torch device it runs on.
    target, source : ndarray, shape (h, w, 3)
        The atlas image and the section, uint8 RGB on one canvas.

    Returns
    -------
    atlas_grid : ndarray, shape (n, 2)
        Each match's atlas end, px of the grid (corner based).
    section_grid : ndarray, shape (n, 2)
        Its section end, px of the grid (corner based).
    certainty : ndarray, shape (n,)
        Its weighted certainty after the suppression, most confident first.
    """
    import torch

    section = Image.fromarray(source).convert("RGB")
    warp, certainty = model.match(
        Image.fromarray(target).convert("RGB"), section, device=device
    )
    edge = edge_weight(section, device)
    if certainty.dim() == 3:
        certainty = certainty.squeeze(0)

    edge_both = torch.cat((torch.zeros_like(edge), edge), dim=1)
    weighted = certainty * (
        edge_both * (1.0 - OFF_EDGE_WEIGHT) + OFF_EDGE_WEIGHT
    )
    weighted[:, :W] = 0
    suppressed = grid_suppression(weighted)

    ranked, order = torch.sort(suppressed.reshape(-1), descending=True)
    ends = warp.reshape(-1, 4)[order]
    atlas_grid = _grid_px(ends[:, :2])
    section_grid = _grid_px(ends[:, 2:])
    return (
        atlas_grid.cpu().numpy().astype(np.float64),
        section_grid.cpu().numpy().astype(np.float64),
        ranked.float().cpu().numpy().astype(np.float64),
    )


def _grid_px(normalized: torch.Tensor) -> torch.Tensor:
    """Take RoMa's normalized coordinates (-1 to 1) to px of the grid."""
    import torch

    x = W / 2 * (normalized[..., 0] + 1)
    y = H / 2 * (normalized[..., 1] + 1)
    return torch.stack((x, y), dim=-1)


def edge_weight(section: Image.Image, device: torch.device) -> torch.Tensor:
    """Map the section's edges on the grid, as SPOmiAlign weights matches.

    Parameters
    ----------
    section
        The section, RGB, at its canvas size.
    device
        The torch device the map is made on.

    Returns
    -------
    torch.Tensor, shape (H, W)
        1.0 on an edge, 0.0 elsewhere: the luminance band-passed in the
        Fourier domain, normalized to 0-1, raised to :data:`EDGE_POWER`
        and binarised by Otsu's threshold.
    """
    import torch

    resized = np.asarray(section.resize((W, H)))
    rgb = (torch.tensor(resized) / 255.0).permute(2, 0, 1).unsqueeze(0)
    rgb = rgb.to(device)
    grey = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    fft = torch.fft
    column = grey.unsqueeze(1)
    transformed = cast(
        "torch.Tensor",
        fft.fft2(column),  # pyright: ignore[reportUnknownMemberType]
    )
    spectrum = cast(
        "torch.Tensor",
        fft.fftshift(transformed),  # pyright: ignore[reportUnknownMemberType]
    )

    y = torch.arange(H, device=device) - H // 2
    x = torch.arange(W, device=device) - W // 2
    rows, cols = torch.meshgrid(y, x, indexing="ij")
    distance_sq = cols**2 + rows**2
    short = min(H, W)
    sigma = EDGE_SIGMA * short / 100.0
    high_pass = 1.0 - torch.exp(-distance_sq / (2 * (sigma**2) + 1e-6))
    low_pass = torch.exp(
        -torch.sqrt(distance_sq) / (EDGE_DECAY * short + 1e-6)
    )
    band = high_pass * low_pass
    unshifted = cast(
        "torch.Tensor",
        fft.ifftshift(spectrum * band),  # pyright: ignore[reportUnknownMemberType]
    )
    inverse = cast(
        "torch.Tensor",
        fft.ifft2(unshifted),  # pyright: ignore[reportUnknownMemberType]
    )
    filtered = inverse.real

    edges = torch.abs(filtered.squeeze())
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    levels = (torch.pow(edges, EDGE_POWER).cpu().numpy() * 255).astype(
        np.uint8
    )
    _, binary = cv2.threshold(
        levels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    edge_map = torch.from_numpy(binary)  # pyright: ignore[reportUnknownMemberType]
    return edge_map.float().to(device) / 255.0


def grid_suppression(certainty: torch.Tensor) -> torch.Tensor:
    """Keep each certainty that is the largest of its 3 x 3 window.

    Parameters
    ----------
    certainty : torch.Tensor, shape (h, w)
        The weighted certainty of every match.

    Returns
    -------
    torch.Tensor, shape (h, w)
        The certainty where it is its window's maximum and above
        :data:`NMS_FLOOR`, zero elsewhere.
    """
    import torch

    batch = certainty.unsqueeze(0).unsqueeze(0)
    window_max = torch.nn.functional.max_pool2d(
        batch, kernel_size=NMS_KERNEL, stride=1, padding=NMS_KERNEL // 2
    )
    peak = (batch == window_max) & (batch > NMS_FLOOR)
    return (batch * peak).squeeze()
