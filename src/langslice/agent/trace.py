"""Full-content JSONL traces for linear agent sessions.

Off unless ``LANGSLICE_TRACE_DIR`` is set (``langslice linear run --trace-dir``
sets it for one run): :func:`open_trace` returns ``None`` and the session loop
skips every call. When it is set, each session started by
:func:`~langslice.agent.session.run_agent_session` appends one JSONL file,
``<trace_dir>/<run_label>_<8 hex>.jsonl``, with one record per event — enough
to reconstruct what the agent was shown, said, called, and got back. A session
given its job folder's ``trace/`` (``mirror_dir``) writes every record there
too, line by line, when the trace folder is another (``--trace-dir``,
``LANGSLICE_TRACE_DIR``), so the job folder always holds its run's trace.

This is deliberately not
:class:`~langslice.agent.plugins.RequestCapturePlugin`: that one records shapes
(part kinds, char counts, argument key names) for transport debugging, which
says nothing about *why* a run went wrong. Here the text, the tool arguments
and the tool responses are written verbatim. Images are the one exception:
they are recorded as descriptors (mime type, byte count, pixel size), never
as bytes.

Record kinds, in the order a healthy session writes them:

``session``
    Once, first line: run label, agent name, model name, and the full system
    instruction.
``plugins``
    Only for a job that loaded tool plugins, once, after ``session``: per
    plugin its module, factory, versions, source digests and every tool's
    declaration as the model saw it (``ToolBox.plugins``).
``seed``
    The step's seed message: text parts verbatim, image parts as descriptors
    carrying the label text that preceded them.
``model``
    One model turn: its visible text, any thought summary, and every function
    call with full JSON arguments. Usage-only responses are recorded too.
    Supported providers add sanitized usage diagnostics: numeric attribution,
    request-slot fingerprints and reconciliation residuals, never image bytes,
    raw request payloads or encrypted reasoning. Unknown counts stay unknown.
``tool_result``
    One tool response: the complete payload the model reads, plus descriptors
    for any media riding on the function response.
``nudge``
    A driver-injected user turn (the model answered in prose, or the pass
    continues).
``spend``
    An operator spend warning: the usage-window share or the dollar estimate
    passed another step (:mod:`langslice.agent.spend`). Never shown to the model.
``summary``
    Once, last line: tool-call count, turn count, whether the step's submit
    tool fired, the token tally and the run's spend account.

No secrets: only content parts, the prompt, and a model *name* are read —
never the request config, HTTP options, or a provider wrapper's repr (which
can hold an API key).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from google.genai import types

from langslice.agent.live import _model_name
from langslice.agent.model_resolver import env_value
from langslice.agent.plugins import part_summary
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.trace import TRACE_DIR_ENV as _TRACE_DIR_ENV
from langslice.job.layout import TRACE_DIR

logger = logging.getLogger(__name__)

#: Directory for full-content session traces. Unset means no tracing at all
#: (the one switch every door's trace reads, :mod:`langslice.doors.trace`).
TRACE_DIR_ENV = _TRACE_DIR_ENV


def job_trace_dir(job_folder: str | Path) -> Path:
    """Where a run keeps its full agent trace when nothing else is named:
    ``trace/`` in its job folder (``langslice linear run`` and the ABBA
    worker's ``linear.run``)."""
    return Path(job_folder) / TRACE_DIR


def _describe_parts(parts: Any) -> list[dict[str, Any]]:
    """Text parts verbatim; media parts as descriptors labelled by the text above."""
    described: list[dict[str, Any]] = []
    label: str | None = None
    for part in parts or []:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            described.append({"kind": "text", "text": text})
            label = text
            continue
        summary = part_summary(part)
        summary["label"] = label
        described.append(summary)
    return described


class SessionTrace:
    """Append-only JSONL record of one linear agent session."""

    def __init__(self, trace_dir: str | Path, run_label: str, *, agent: Any,
                 plugins: list[dict[str, Any]] | None = None,
                 mirror_dir: str | Path | None = None) -> None:
        self._steps = 0
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_label)
        # Same uuid trick as RequestCapturePlugin: concurrent sessions can share
        # a run label, and must not share a file.
        self.path = Path(trace_dir) / f"{safe}_{uuid.uuid4().hex[:8]}.jsonl"
        #: Every file the trace is written to: :attr:`path`, then the same
        #: name in *mirror_dir* when that is another folder.
        self.paths = [self.path]
        if mirror_dir is not None and not _same_folder(Path(mirror_dir), Path(trace_dir)):
            self.paths.append(Path(mirror_dir) / self.path.name)
        instruction = getattr(agent, "instruction", "")
        self._write({
            "kind": "session",
            "run_label": run_label,
            "agent": getattr(agent, "name", ""),
            "model": _model_name(getattr(agent, "model", "")),
            "system_instruction": (
                instruction if isinstance(instruction, str) else "<dynamic instruction>"
            ),
        })
        if plugins:
            self._write({"kind": "plugins", "plugins": plugins})

    def _write(self, record: dict[str, Any]) -> None:
        record["at"] = datetime.now().isoformat(timespec="seconds")
        try:
            line = json.dumps(record, default=str) + "\n"
        except Exception:
            logger.warning("Could not encode a trace record", exc_info=True)
            return
        for path in self.paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except Exception:
                # A debug trace must never take down a 40-section run.
                logger.warning("Could not write trace record to %s", path, exc_info=True)

    def seed(self, message: types.Content) -> None:
        self._write({"kind": "seed", "parts": _describe_parts(message.parts)})

    def spend(self, warning: dict[str, Any], *, turn: int) -> None:
        """An operator spend warning (:mod:`langslice.agent.spend`)."""
        self._write({"kind": "spend", "turn": turn, **warning})

    def nudge(self, text: str, *, turn: int) -> None:
        self._write({"kind": "nudge", "turn": turn, "text": text})

    def event(self, event: Any, *, turn: int) -> None:
        """Record one ADK event: a model turn, tool results, or both."""
        if getattr(event, "partial", False):
            return
        content = getattr(event, "content", None)
        text: list[str] = []
        thought: list[str] = []
        calls: list[dict[str, Any]] = []
        for part in getattr(content, "parts", None) or []:
            response = getattr(part, "function_response", None)
            if response is not None:
                self._write(self._tool_record(response, turn))
                continue
            call = getattr(part, "function_call", None)
            if call is not None:
                calls.append({
                    "name": getattr(call, "name", None),
                    "args": dict(getattr(call, "args", None) or {}),
                })
                continue
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                (thought if getattr(part, "thought", False) else text).append(part_text)

        usage = getattr(event, "usage_metadata", None)
        if text or thought or calls or usage is not None:
            # ``turn`` counts driver invocations; ``step`` counts model
            # responses, which is what a reader paging through a trace wants.
            self._steps += 1
            record: dict[str, Any] = {
                "kind": "model",
                "turn": turn,
                "step": self._steps,
                "author": getattr(event, "author", None),
            }
            if text:
                record["text"] = "\n".join(text)
            if thought:
                record["thought"] = "\n".join(thought)
            if calls:
                record["function_calls"] = calls
            if usage is not None:
                record["usage"] = {
                    "input_tokens": getattr(usage, "prompt_token_count", None),
                    "cached_tokens": getattr(usage, "cached_content_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None),
                }
            metadata = getattr(event, "custom_metadata", None) or {}
            diagnostics = metadata.get("usage_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics.get("schema_version") == 1:
                record["usage_diagnostics"] = diagnostics
                writes = diagnostics.get("totals", {}).get("cache_write_tokens")
                if "usage" in record and type(writes) is int and writes >= 0:
                    record["usage"]["cache_write_tokens"] = writes
            self._write(record)

    @staticmethod
    def _tool_record(response: Any, turn: int) -> dict[str, Any]:
        payload = getattr(response, "response", None)
        if isinstance(payload, dict):
            # ADK lifts tool media out of the payload before the model sees it;
            # if anything is left behind it is Part objects, not JSON.
            payload = {k: v for k, v in payload.items() if k != TOOL_MEDIA_PARTS_KEY}
        media = [part_summary(part) for part in getattr(response, "parts", None) or []]
        record: dict[str, Any] = {
            "kind": "tool_result",
            "turn": turn,
            "name": getattr(response, "name", None),
            "response": payload,
        }
        if media:
            record["media"] = media
        return record

    def summary(
        self,
        *,
        tool_calls: int,
        turns: int,
        submitted: bool,
        tokens: dict[str, int] | None = None,
        spend: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "kind": "summary",
            "tool_calls": tool_calls,
            "turns": turns,
            "submitted": submitted,
        }
        if tokens is not None:
            record["tokens"] = tokens
        if spend is not None:
            record["spend"] = spend
        self._write(record)


def _same_folder(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a.absolute() == b.absolute()


def open_trace(run_label: str, *, agent: Any,
               plugins: list[dict[str, Any]] | None = None,
               mirror_dir: str | Path | None = None) -> SessionTrace | None:
    """A trace for this session, or ``None`` when tracing is off; written to
    *mirror_dir* too (the job folder's ``trace/``) when that is another folder."""
    trace_dir = env_value(TRACE_DIR_ENV)
    if trace_dir is None:
        return None
    return SessionTrace(trace_dir, run_label, agent=agent, plugins=plugins,
                        mirror_dir=mirror_dir)
