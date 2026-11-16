"""A job's registrations as an ABBA project file (``.abba``).

ABBA 0.25 opens it with File > Load State. The file is a zip of three files,
laid out as ABBA's own ``MultiSlicePositioner.saveState`` writes them:

- ``_bdvdataset_0.xml``: a BigDataViewer dataset opening each section's
  image file with Bio-Formats (one view setup per channel; the view
  transform a bare voxel scale, which ABBA centres on the world origin);
- ``sources.json``: one entry per channel source, naming its view setup;
- ``state.json``: ABBA's version, the slicing rotations, and per slice its
  actions: ``CreateSliceAction`` (its channel sources), ``MoveSliceAction``
  (its location) and the registrations, LangSlice's affine step as ABBA's
  ``AffineRegistration`` and an applied deformation as a
  ``BigWarpSource2DRegistration`` (a thin-plate spline through landmark
  pairs, which ABBA's "Edit last registration" opens in BigWarp).

The geometry is computed by the caller; this module lays it out in ABBA's
format only.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from xml.sax.saxutils import escape

import numpy as np
import tifffile
from PIL import Image

from langslice.core.jsonable import Json, JsonObject

#: The ABBA version the file is written as.
ABBA_VERSION = "0.25.0"
#: The dataset file inside the zip.
DATASET_FILE = "_bdvdataset_0.xml"
#: The display colours of a section's channels, as ABBA stores them.
CHANNEL_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
)
#: A one-channel section's colour.
GRAY = (255, 255, 255)
#: Half the thickness of a slice along the slicing axis, in millimetres:
#: the z extent of a warp step's bounded transform around the location.
WARP_HALF_DEPTH_MM = 0.5
#: How far a warp step's bounded transform reaches in x and y, millimetres:
#: beyond any section placed in the atlas.
WARP_REACH_MM = 50.0
#: The PIL modes Bio-Formats opens as RGB, split into channels.
RGB_MODES = {"RGB": 3, "RGBA": 4, "P": 3, "CMYK": 4, "YCbCr": 3}
#: The identity, as ABBA writes a slice's pre-transform.
IDENTITY_3D = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#: The colour converter and source class ABBA records per source.
CONVERTER = "class net.imglib2.display.RealARGBColorConverterFactory$Imp"
SOURCE_CLASS = "bdv.SpimSource"


@dataclass(frozen=True)
class AbbaWarp:
    """One section's warp step: landmark pairs in world millimetres.

    Attributes
    ----------
    source_mm
        N-by-2 section points as the affine step places them.
    target_mm
        Where the deformation takes them.
    parameters
        Kept with the registration, as text.
    """

    source_mm: np.ndarray
    target_mm: np.ndarray
    parameters: dict[str, str]


@dataclass(frozen=True)
class AbbaSlice:
    """One section as ABBA opens it.

    Attributes
    ----------
    path
        The section's image file.
    size
        Its width and height in pixels.
    channels
        The channel sources Bio-Formats opens it as.
    rgb
        Whether the file is RGB (split into its channels).
    voxel_mm
        Its pixel size in millimetres.
    location_mm
        Its position on ABBA's slicing axis.
    atlas_z_mm
        The atlas position ABBA's slice list gives for that location.
    affine_mm
        3x3 on ``[x, y, 1]``: centred source millimetres to world.
    warp
        The deformation's landmark pairs, or None.
    """

    path: Path
    size: tuple[int, int]
    channels: int
    rgb: bool
    voxel_mm: float
    location_mm: float
    atlas_z_mm: float
    affine_mm: np.ndarray
    warp: AbbaWarp | None = None


def _number(value: float) -> str:
    return repr(float(value))


def _color(item: AbbaSlice, channel: int) -> tuple[int, int, int]:
    """Return a channel's display colour."""
    if item.channels == 1:
        return GRAY
    return CHANNEL_COLORS[channel % len(CHANNEL_COLORS)]


def _opener(item: AbbaSlice) -> JsonObject:
    """Describe how Bio-Formats opens one section's file."""
    opener: JsonObject = {
        "opt": "",
        "unit": "MILLIMETER",
        "nReader": 10,
        "defaultBlockSize": True,
        "blockSize": [512, 512, 1],
        "splitRGB": item.rgb,
        "to16bits": False,
        "autoPyramidize": True,
        "type": "BIOFORMATS",
        "location": str(item.path),
        "id": 0,
        "nChannels": item.channels,
    }
    return opener


def _setup(setup: int, index: int, item: AbbaSlice, channel: int) -> str:
    """Write one channel's view setup."""
    voxel = _number(item.voxel_mm)
    name = escape(f"{item.path.name}-ch_{channel}")
    return f"""      <ViewSetup>
        <id>{setup}</id>
        <name>{name}</name>
        <size>{item.size[0]} {item.size[1]} 1</size>
        <voxelSize>
          <unit>mm</unit>
          <size>{voxel} {voxel} 1.0</size>
        </voxelSize>
        <attributes>
          <filename>{index}</filename>
          <seriesindex>0</seriesindex>
          <illumination>0</illumination>
          <imagename>{index}</imagename>
          <channel>{channel}</channel>
          <tile>{setup}</tile>
          <angle>0</angle>
          <displaysettings>{setup}</displaysettings>
        </attributes>
      </ViewSetup>"""


def _registration(setup: int, item: AbbaSlice) -> str:
    """Write one channel's view transform, a bare voxel scale."""
    voxel = _number(item.voxel_mm)
    matrix = f"{voxel} 0.0 0.0 0.0 0.0 {voxel} 0.0 0.0 0.0 0.0 1.0 0.0"
    return f"""    <ViewRegistration timepoint="0" setup="{setup}">
      <ViewTransform type="affine">
        <affine>{matrix}</affine>
      </ViewTransform>
    </ViewRegistration>"""


def _attributes(name: str, rows: list[str]) -> str:
    """Write one attribute block of the view setups."""
    body = "\n".join(rows)
    return f'      <Attributes name="{name}">\n{body}\n      </Attributes>'


def _named(tag: str, index: int, name: str) -> str:
    """Write one entry of an attribute block that has a name."""
    return (
        f"        <{tag}>\n          <id>{index}</id>\n"
        f"          <name>{escape(name)}</name>\n        </{tag}>"
    )


def _display(setup: int, color: tuple[int, int, int]) -> str:
    """Write one channel's display settings."""
    red, green, blue = color
    return (
        f"        <Displaysettings>\n          <id>{setup}</id>\n"
        "          <isset>true</isset>\n"
        f"          <color>{red} {green} {blue} 255</color>\n"
        "          <min>0.0</min>\n          <max>255.0</max>\n"
        "          <Projection_Mode>Sum</Projection_Mode>\n"
        "          <islabelimage>false</islabelimage>\n"
        "          <lutname />\n        </Displaysettings>"
    )


def dataset_xml(slices: list[AbbaSlice]) -> str:
    """Write the BigDataViewer dataset that opens every section's channels.

    Parameters
    ----------
    slices
        The sections, in the order their sources are numbered.

    Returns
    -------
    str
        The dataset XML.
    """
    openers = json.dumps(
        [_opener(item) for item in slices], separators=(",", ":")
    )
    setups: list[str] = []
    registrations: list[str] = []
    display: list[str] = []
    tiles: list[str] = []
    for index, item in enumerate(slices):
        for channel in range(item.channels):
            setup = len(setups)
            setups.append(_setup(setup, index, item, channel))
            registrations.append(_registration(setup, item))
            display.append(_display(setup, _color(item, channel)))
            tiles.append(_named("Tile", setup, str(setup)))
    names = [item.path.name for item in slices]
    widest = max(item.channels for item in slices)
    blocks = [
        _attributes(
            "filename",
            [_named("filename", i, n) for i, n in enumerate(names)],
        ),
        _attributes(
            "seriesindex",
            [
                "        <seriesindex>\n          <id>0</id>\n"
                "        </seriesindex>"
            ],
        ),
        _attributes("illumination", [_named("Illumination", 0, "0")]),
        _attributes(
            "imagename",
            [_named("imagename", i, n) for i, n in enumerate(names)],
        ),
        _attributes(
            "channel", [_named("Channel", c, f"ch_{c}") for c in range(widest)]
        ),
        _attributes("tile", tiles),
        _attributes("angle", [_named("Angle", 0, "0")]),
        _attributes("displaysettings", display),
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<SpimData version="0.2">\n'
        f'  <BasePath type="relative">{DATASET_FILE}</BasePath>\n'
        "  <SequenceDescription>\n"
        '    <ImageLoader format="spimreconstruction.openersimageloader">\n'
        "      <version>0.23.4</version>\n"
        f"      <openers>{escape(openers)}</openers>\n"
        "    </ImageLoader>\n    <ViewSetups>\n"
        + "\n".join(setups + blocks)
        + "\n    </ViewSetups>\n"
        '    <Timepoints type="range">\n      <first>0</first>\n'
        "      <last>0</last>\n    </Timepoints>\n    <MissingViews />\n"
        "  </SequenceDescription>\n  <ViewRegistrations>\n"
        + "\n".join(registrations)
        + "\n  </ViewRegistrations>\n</SpimData>\n"
    )


def _argb(color: tuple[int, int, int]) -> int:
    """Pack a colour as ABBA stores it: a signed 32-bit ARGB integer."""
    red, green, blue = color
    value = (255 << 24) | (red << 16) | (green << 8) | blue
    return value - (1 << 32)


def sources_json(slices: list[AbbaSlice]) -> list[JsonObject]:
    """List every channel source, as ABBA's ``sources.json``.

    Parameters
    ----------
    slices
        The sections, in the order their sources are numbered.

    Returns
    -------
    list of JsonObject
        One entry per channel source.
    """
    rows: list[JsonObject] = []
    for item in slices:
        for channel in range(item.channels):
            index = len(rows)
            source: JsonObject = {
                "source_name": f"{item.path.name}-ch_{channel}",
                "source_class": SOURCE_CLASS,
                "converter_class": CONVERTER,
                "source_id": index,
                "color": _argb(_color(item, channel)),
                "converter_setup_min": 0.0,
                "converter_setup_max": 255.0,
                "sac": {
                    "spimdata": {"datalocation": DATASET_FILE},
                    "viewsetup": index,
                },
                "string_metadata": {},
            }
            rows.append(source)
    return rows


def _register_action(registration: JsonObject) -> JsonObject:
    """Wrap a registration as ABBA's ``RegisterSliceAction``."""
    action: JsonObject = {
        "type": "RegisterSliceAction",
        "fixed_sources_preprocess": {"type": "SourcesIdentity"},
        "moving_sources_preprocess": {"type": "SourcesIdentity"},
        "registration": registration,
    }
    return action


def _affine_action(affine_mm: np.ndarray, name: str) -> JsonObject:
    """Write the affine step as ABBA's ``AffineRegistration``."""
    rows = cast("list[list[float]]", np.asarray(affine_mm).tolist())
    (a, b, x), (c, d, y) = rows[0], rows[1]
    numbers = [a, b, 0.0, x, c, d, 0.0, y, 0.0, 0.0, 1.0, 0.0]
    transform = json.dumps({"affinetransform3d": numbers}, indent=2)
    compact = json.dumps(numbers, separators=(",", ":"))
    registration: JsonObject = {
        "type": "AffineRegistration",
        "transform": transform,
        "parameters": {"transform": compact, "pz": "0"},
        "name": name,
    }
    return _register_action(registration)


def _warp_action(warp: AbbaWarp, location_mm: float, name: str) -> JsonObject:
    """Write the warp step as ABBA's ``BigWarpSource2DRegistration``.

    ABBA resamples with the pull-back, world to section: ``srcPts`` are
    where the deformation takes the points (the atlas side), ``tgtPts`` the
    points as the affine step placed them.
    """
    atlas_side = cast("Json", np.asarray(warp.target_mm).T.tolist())
    section_side = cast("Json", np.asarray(warp.source_mm).T.tolist())
    spline: JsonObject = {
        "type": "ThinplateSplineTransform",
        "srcPts": atlas_side,
        "tgtPts": section_side,
    }
    reach = WARP_REACH_MM
    transform: JsonObject = {
        "type": "BoundedRealTransform",
        "realTransform": {
            "type": "InvertibleWrapped2DTransformAs3D",
            "wrappedTransform": {
                "type": "WrappedIterativeInvertibleRealTransform",
                "wrappedTransform": spline,
            },
        },
        "interval_min": [-reach, -reach, location_mm - WARP_HALF_DEPTH_MM],
        "interval_max": [reach, reach, location_mm + WARP_HALF_DEPTH_MM],
    }
    registration: JsonObject = {
        "type": "BigWarpSource2DRegistration",
        "transform": json.dumps(transform, indent=2),
        "parameters": dict(warp.parameters),
        "name": name,
    }
    return _register_action(registration)


def _slice_entry(item: AbbaSlice, first: int) -> JsonObject:
    """Write one slice's actions; its sources are numbered from *first*."""
    indexes: Json = list(range(first, first + item.channels))
    create: JsonObject = {
        "type": "CreateSliceAction",
        "original_sources": {"source_indexes": indexes},
        "original_location": item.location_mm,
        "final_thicknessCorrection": 1.0,
        "final_zShiftCorrection": 0.0,
    }
    move: JsonObject = {
        "type": "MoveSliceAction",
        "z_mm": item.atlas_z_mm,
        "location": item.location_mm,
    }
    actions: list[Json] = [
        create,
        move,
        _affine_action(item.affine_mm, f"LangSlice affine [{item.path.name}]"),
    ]
    if item.warp is not None:
        name = f"LangSlice warp [{item.path.name}]"
        actions.append(_warp_action(item.warp, item.location_mm, name))
    entry: JsonObject = {
        "actions": actions,
        "preTransform": {"affinetransform3d": list(IDENTITY_3D)},
    }
    return entry


def state_json(
    slices: list[AbbaSlice], rotation_x: float, rotation_y: float
) -> JsonObject:
    """Write ABBA's ``state.json`` for the slices at these rotations.

    Parameters
    ----------
    slices
        The sections, in the order their sources are numbered.
    rotation_x, rotation_y
        ABBA's slicing rotations, in radians.

    Returns
    -------
    JsonObject
        The state document.
    """
    entries: list[Json] = []
    first = 0
    for item in slices:
        entries.append(_slice_entry(item, first))
        first += item.channels
    state: JsonObject = {
        "version": ABBA_VERSION,
        "rotationX": float(rotation_x),
        "rotationY": float(rotation_y),
        "slices_state_list": entries,
    }
    return state


def project_bytes(
    slices: list[AbbaSlice], rotation_x: float, rotation_y: float
) -> bytes:
    """Write the whole ``.abba`` file.

    Parameters
    ----------
    slices
        The sections, in any order (ABBA orders them by location).
    rotation_x, rotation_y
        ABBA's slicing rotations, in radians.

    Returns
    -------
    bytes
        The zip ABBA's File > Load State opens.

    Raises
    ------
    ValueError
        When there is no section.
    """
    if not slices:
        raise ValueError("no placed section to write")
    state = state_json(slices, rotation_x, rotation_y)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "sources.json", json.dumps(sources_json(slices), indent=2)
        )
        archive.writestr("state.json", json.dumps(state, indent=2))
        archive.writestr(DATASET_FILE, dataset_xml(slices))
    return buffer.getvalue()


def _tiff_channels(path: Path) -> tuple[int, bool]:
    """Count the channels Bio-Formats opens a TIFF as."""
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = dict(zip(series.axes, series.shape, strict=True))
    if "S" in shape:
        return int(shape["S"]) * int(shape.get("C", 1)), True
    if "C" in shape:
        return int(shape["C"]), False
    return int(shape.get("I", 1)), False


def image_channels(path: Path) -> tuple[int, bool]:
    """Count the channels Bio-Formats opens an image file as.

    Parameters
    ----------
    path
        The section's image file.

    Returns
    -------
    tuple
        The number of channel sources, and whether the file is RGB.

    Raises
    ------
    ValueError
        When the file is neither a TIFF nor an image PIL reads.
    """
    if path.suffix.lower() in {".tif", ".tiff"}:
        return _tiff_channels(path)
    try:
        with Image.open(path) as image:
            mode = image.mode
    except OSError as exc:
        message = f"{path.name}: not an image file ABBA can open ({exc})"
        raise ValueError(message) from exc
    if mode in RGB_MODES:
        return RGB_MODES[mode], True
    return 1, False
