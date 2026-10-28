"""``langslice session-end``: a coding agent's transcript copied into its job.

The harness plugins (``connectors/claude-code``, ``connectors/antigravity``)
run this when a registration agent's session ends, with the harness's hook
input (JSON) on stdin: Claude Code's ``SubagentStop`` (``agent_transcript_path``,
else ``<session>/subagents/agent-<agent_id>.jsonl`` beside ``transcript_path``)
or Antigravity's ``Stop`` (``transcriptPath``, ``conversationId``). The job
folders the transcript worked on are the ones its ``langslice-job FOLDER ...``
commands and ``open_job(...)`` calls name, and the ones its MCP ``start_job``
calls open (:func:`started_jobs`: an ``image_folder`` or a saved job's
``job_id``, under any harness's name for the tool); each gets the transcript in
``trace/<harness>_<session id>.jsonl`` (rewritten at each end of the session,
so the copy is the latest), beside LangSlice's own agent's traces, with
inline image data replaced by its size (:func:`slim`). When the transcript
submitted a job (:data:`SUBMIT_PATTERN`: the registration agent's, never the
main session's that launched it), that job's own Python environment is then
removed, its package list kept (:func:`langslice.doors.cli.jobenv.remove`).

Never fails a session: anything that goes wrong is said on stderr, and
stdout is ``{}`` (what an Antigravity ``Stop`` hook answers to let the agent
stop).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

#: A job folder named on a command line or in a script; unquoted, a Windows
#: path holds backslashes, which a POSIX shell reads as escapes.
_UNQUOTED = r"[^\s\"';|&]+" if sys.platform == "win32" else r"[^\s\"'\\;|&]+"
FOLDER_PATTERNS = (
    re.compile(r"""langslice-job\s+(?:"([^"]+)"|'([^']+)'|(""" + _UNQUOTED + "))"),
    re.compile(r"""open_job\(\s*(?:r?"([^"]+)"|r?'([^']+)')"""),
)
#: An MCP ``start_job`` call's tool name, as each harness writes it:
#: ``mcp__plugin_langslice_langslice__start_job`` (Claude Code's plugin),
#: ``mcp__langslice__start_job``, or ``start_job`` itself (Antigravity's
#: ``call_mcp_tool`` names it under ``ToolName``).
START_JOB = re.compile(r"(?:^|__)start_job$")
#: The ``start_job`` arguments that name a job.
START_JOB_KEYS = ("image_folder", "job_id")
#: A submit, on the command line or through the library.
SUBMIT_PATTERN = re.compile(r"langslice-job\s+\S+\s+submit\b|\.submit\(")
#: The MCP ``submit`` tool under any harness's name, as :data:`START_JOB`.
MCP_SUBMIT = re.compile(r"(?:^|__)submit$")
#: Words after ``langslice-job`` that are commands, not folders.
NOT_FOLDERS = frozenset({"ops", "schema", "-h", "--help"})
#: A string this long that looks like base64 is inline image data.
INLINE_DATA_CHARS = 4096
_BASE64 = re.compile(r"[A-Za-z0-9+/=\s]+")


def add_parser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "session-end",
        help="Copy a coding agent's transcript into the job folders it worked on "
        "(the hook the harness plugins run; reads the hook's JSON on stdin)")
    p.add_argument("--harness", required=True, choices=("claude-code", "antigravity"),
                   help="The harness whose hook calls this")


def run(args: Any) -> None:
    try:
        hook = json.loads(sys.stdin.read() or "{}")
        for folder in session_end(hook, args.harness):
            print(f"langslice: transcript copied to {folder}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - a hook never fails the session
        print(f"langslice session-end: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("{}")


def session_end(hook: dict[str, Any], harness: str) -> list[Path]:
    """Copy the hook's transcript into each job folder it names; the copies."""
    from langslice.doors.cli import jobenv
    from langslice.job.layout import JobLayout

    transcript = transcript_path(hook)
    if transcript is None or not transcript.is_file():
        return []
    text = transcript.read_text(encoding="utf-8", errors="replace")
    session = str(hook.get("agent_id") or hook.get("conversationId")
                  or hook.get("session_id") or transcript.stem)
    copies = []
    said = _said(text)
    submitted = SUBMIT_PATTERN.search(said) is not None or called(text, MCP_SUBMIT)
    for folder in job_folders(said, Path(str(hook.get("cwd") or ".")),
                              also=started_jobs(text)):
        layout = JobLayout(folder)
        target = layout.trace_dir / f"{harness}_{_safe(session)}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(slim_lines(text), encoding="utf-8")
        copies.append(target)
        if submitted:
            jobenv.remove(layout)
    return copies


def transcript_path(hook: dict[str, Any]) -> Path | None:
    """The transcript of the agent whose session ended."""
    named = hook.get("agent_transcript_path") or hook.get("transcriptPath")
    if named:
        return Path(str(named)).expanduser()
    main, agent = hook.get("transcript_path"), hook.get("agent_id")
    if main and agent:  # Claude Code keeps a subagent's beside its session's
        main_path = Path(str(main)).expanduser()
        return main_path.with_suffix("") / "subagents" / f"agent-{agent}.jsonl"
    return Path(str(main)).expanduser() if main else None


def job_folders(text: str, cwd: Path, also: list[str] | tuple[str, ...] = ()) -> list[Path]:
    """The job folders *text* (a transcript's strings, :func:`_said`) names,
    then those of *also* (folder names found otherwise, :func:`started_jobs`),
    each a job folder or the image folder beside one, in the order first named."""
    from langslice.doors.jobs import NoJob, find

    names = [next(group for group in match.groups() if group is not None)
             for pattern in FOLDER_PATTERNS for match in pattern.finditer(text)]
    found: list[Path] = []
    seen: set[str] = set()
    for name in [*names, *also]:
        if name in NOT_FOLDERS or name in seen or name.startswith("-"):
            continue
        seen.add(name)
        path = Path(name).expanduser()
        try:
            folder = Path(find(path if path.is_absolute() else cwd / path))
        except (NoJob, OSError, ValueError):
            continue
        if folder not in found:
            found.append(folder)
    return found


def started_jobs(text: str) -> list[str]:
    """The folders the MCP ``start_job`` calls of a JSON-lines transcript
    opened: each call's ``image_folder`` (an image folder or its job folder),
    or the job folder a saved job's ``job_id`` names in the index. A call is
    an object one of whose strings is the tool's name (:data:`START_JOB`); its
    arguments are found anywhere in that object, also inside a string that
    holds them as JSON."""
    from langslice.job import index

    found: list[str] = []

    def arguments(value: Any) -> None:
        if isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                value = json.loads(value)
            except ValueError:
                return
        if isinstance(value, dict):
            for key in START_JOB_KEYS:
                named = value.get(key)
                if isinstance(named, str) and named:
                    if key == "job_id":
                        try:
                            entry = index.lookup(index.default_root(), named)
                        except (OSError, ValueError):
                            entry = None
                        named = entry["job_folder"] if entry else ""
                    if named:
                        found.append(named)
            for item in value.values():
                arguments(item)
        elif isinstance(value, list):
            for item in value:
                arguments(item)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if any(isinstance(item, str) and START_JOB.search(item) for item in value.values()):
                arguments(value)
                return
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for line in text.splitlines():
        try:
            walk(json.loads(line))
        except ValueError:
            continue
    return found


def called(text: str, name: re.Pattern[str]) -> bool:
    """Whether a JSON-lines transcript holds an object one of whose strings
    is a tool name *name* matches: an MCP call, under any harness's name."""

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return any(isinstance(item, str) and name.search(item) for item in value.values()) \
                or any(walk(item) for item in value.values())
        if isinstance(value, list):
            return any(walk(item) for item in value)
        return False

    for line in text.splitlines():
        try:
            if walk(json.loads(line)):
                return True
        except ValueError:
            continue
    return False


def slim_lines(text: str) -> str:
    """The transcript, one JSON record per line, inline image data replaced by
    its size (a line that is not JSON is kept as it is)."""
    out = []
    for line in text.splitlines():
        try:
            out.append(json.dumps(slim(json.loads(line)), ensure_ascii=False))
        except ValueError:
            out.append(line)
    return "\n".join(out) + "\n"


def slim(value: Any) -> Any:
    """*value* with every long base64 string replaced by ``<N bytes of inline data>``."""
    if isinstance(value, dict):
        return {key: slim(item) for key, item in value.items()}
    if isinstance(value, list):
        return [slim(item) for item in value]
    if (isinstance(value, str) and len(value) > INLINE_DATA_CHARS
            and _BASE64.fullmatch(value[:INLINE_DATA_CHARS])):
        return f"<{len(value) * 3 // 4} bytes of inline data>"
    return value


def _said(text: str) -> str:
    """Every string of a JSON-lines transcript as it was typed, one per line
    (a line that is not JSON as it is)."""
    out: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for line in text.splitlines():
        try:
            walk(json.loads(line))
        except ValueError:
            out.append(line)
    return "\n".join(out)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120] or "session"
