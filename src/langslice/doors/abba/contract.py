"""Pydantic models for the LangSlice engine stdio contract.

The Fiji connector speaks protocol 3 (``PROTOCOL_VERSION`` in
``doors/credentials.py``) with ``langslice serve --stdio``: one JSON request
per line, answered by events and then one result or error. Three methods:
``setup.status`` (the installation, offline), ``abba.prepare`` (an ABBA
session's snapshots become a prepared job, ``doors/abba/prepare.py``) and
``abba.follow`` (the job's checkpoints streamed as ABBA rows until stdin
closes, ``doors/abba/follow.py``).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

EngineMethod = Literal["setup.status", "abba.prepare", "abba.follow"]
#: Every method the service answers, in the order the docs list them.
ENGINE_METHODS: tuple[EngineMethod, ...] = (
    "setup.status",
    "abba.prepare",
    "abba.follow",
)


class EngineBaseModel(BaseModel):
    """A wire record: unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")


class EngineError(EngineBaseModel):
    """An error's code, message and facts."""

    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class EngineRequest(EngineBaseModel):
    """One request: its id, method and parameters."""

    id: str
    method: EngineMethod
    params: dict[str, object] = Field(default_factory=dict)


class EngineLogEvent(EngineBaseModel):
    """A line for the host's log."""

    kind: Literal["log"]
    message: str


class EngineDataEvent(EngineBaseModel):
    """Host events: checkpoints, logs and waiting, never credentials."""

    kind: Literal["data"] = "data"
    payload: dict[str, object]


EngineEvent = Annotated[
    EngineLogEvent | EngineDataEvent, Field(discriminator="kind")
]


#: A finite number: NaN and infinity are refused.
Finite = Annotated[float, Field(allow_inf_nan=False)]


class CuttingAngles(EngineBaseModel):
    """ABBA's stack-wide cutting angles in degrees (``ReslicedAtlas``)."""

    pitch_deg: Finite
    yaw_deg: Finite


class AbbaSession(EngineBaseModel):
    """The ABBA session's facts a prepared job keeps (``host.abba``)."""

    #: The project key the connector computes from the session.
    project_key: str = Field(min_length=1, max_length=64)
    #: ABBA's atlas name.
    atlas: str
    #: The ABBA slice name of each of the job's snapshots, by filename.
    sections: dict[str, str]
    #: ABBA's median section thickness in micrometres; 0: not known.
    thickness_um: int = Field(default=0, ge=0)
    #: ABBA's median section interval in micrometres; 0: not known.
    interval_um: int = Field(default=0, ge=0)


class AbbaPrepareRequest(EngineBaseModel):
    """``abba.prepare``: an ABBA session's snapshots and what ABBA knows."""

    #: The folder of calibrated snapshots the connector exported.
    image_folder: str
    #: The snapshots' micrometres per pixel.
    pixel_size_um: Finite = Field(gt=0)
    #: Each snapshot's position, BrainGlobe AP millimetres, by filename.
    positions_mm: dict[str, Finite]
    #: ABBA's slicing-axis offset, kept with the job (the connector converts
    #: positions with it).
    z_offset_mm: Finite | None = None
    angles_deg: CuttingAngles
    #: The snapshots whose ABBA slice carries the user's own warp.
    existing_warp: list[str] = Field(default_factory=list[str])
    #: The snapshots whose ABBA slice the user already registered in-plane.
    registered: list[str] = Field(default_factory=list[str])
    #: A name for each page of a multi-page snapshot.
    channel_names: list[str] | None = None
    abba: AbbaSession


class AbbaFollowRequest(EngineBaseModel):
    """``abba.follow``: the job folder whose checkpoints are streamed."""

    job_folder: str


class EngineEventEnvelope(EngineBaseModel):
    """An event of the request ``id``."""

    id: str
    type: Literal["event"]
    event: EngineEvent


class EngineResultEnvelope(EngineBaseModel):
    """The result that ends the request ``id``."""

    id: str
    type: Literal["result"]
    result: dict[str, object] = Field(default_factory=dict)


class EngineErrorEnvelope(EngineBaseModel):
    """The error that ends the request ``id`` (None: no request was read)."""

    id: str | None
    type: Literal["error"]
    error: EngineError


def export_schema_bundle() -> dict[str, object]:
    """Return the JSON schema of every wire record, by model name.

    Returns
    -------
    dict
        ``schema_version`` and ``schemas``, one JSON schema per model.
    """
    models: dict[str, type[BaseModel]] = {
        "EngineError": EngineError,
        "EngineRequest": EngineRequest,
        "EngineDataEvent": EngineDataEvent,
        "EngineLogEvent": EngineLogEvent,
        "AbbaPrepareRequest": AbbaPrepareRequest,
        "AbbaFollowRequest": AbbaFollowRequest,
        "EngineEventEnvelope": EngineEventEnvelope,
        "EngineResultEnvelope": EngineResultEnvelope,
        "EngineErrorEnvelope": EngineErrorEnvelope,
    }
    schemas = {
        name: model.model_json_schema() for name, model in models.items()
    }
    return {"schema_version": "1", "schemas": schemas}
