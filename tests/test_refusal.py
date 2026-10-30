"""The one refusal: its codes, its reply and how it reads."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import langslice
from langslice.core.fits.flags import FitFlag
from langslice.core.refusal import (
    FAILED_STATUSES,
    RefusalCode,
    Refused,
    ReplyStatus,
)
from langslice.plugin_api import RefusalCode as PluginRefusalCode
from langslice.plugin_api import ToolError
from langslice.tools.descriptions.fixes import FIXES


def test_every_code_is_listed_once_by_its_own_name() -> None:
    values = [member.value for member in RefusalCode]
    assert len(values) == len(set(values))
    renamed = {member.name: member.value for member in RefusalCode
               if member.name != member.value}
    assert renamed == {"NO_SUCH_PATH": "NOT_FOUND"}
    assert not {member.value for member in FitFlag} & set(values)
    assert PluginRefusalCode is RefusalCode


def test_every_fix_is_for_a_listed_code() -> None:
    assert set(FIXES) <= {member.value for member in RefusalCode}


def test_a_member_reads_as_its_string() -> None:
    reply = {"error": RefusalCode.BAD_ARGS, "status": ReplyStatus.ERROR}
    assert repr(reply) == repr({"error": "BAD_ARGS", "status": "error"})
    assert json.dumps(reply) == json.dumps({"error": "BAD_ARGS", "status": "error"})
    assert f"{RefusalCode.STALE_INPUT}: x" == "STALE_INPUT: x"
    assert "refused" in FAILED_STATUSES and "ok" not in FAILED_STATUSES


def test_the_message_stands_where_it_is_given() -> None:
    first = Refused(RefusalCode.FROZEN, "words", ids=["a"])
    later = Refused(RefusalCode.FROZEN, ids=["a"], message="words")
    assert list(first.payload()) == ["status", "error", "message", "ids"]
    assert list(later.payload()) == ["status", "error", "ids", "message"]
    assert first.message == later.message == str(first) == "words"
    bare = Refused(RefusalCode.NOTHING_TO_UNDO)
    assert bare.payload() == {"status": "error", "error": "NOTHING_TO_UNDO"}
    assert str(bare) == "NOTHING_TO_UNDO"


def test_a_gate_says_refused() -> None:
    gate = Refused(RefusalCode.OPENING_NOT_READ, status=ReplyStatus.REFUSED, pages=[1])
    assert gate.payload() == {"status": "refused", "error": "OPENING_NOT_READ",
                              "pages": [1]}


def test_a_plugin_tool_error_answers_as_a_refusal() -> None:
    error = ToolError(RefusalCode.BAD_ARGS, "rows is 2 or more.", rows=2)
    assert isinstance(error, Refused) and isinstance(error, ValueError)
    assert error.payload() == {"status": "error", "error": "BAD_ARGS",
                               "message": "rows is 2 or more.", "rows": 2}
    bare = ToolError(RefusalCode.BAD_ARGS, rows=2)  # as a Refused with none
    assert bare.payload() == Refused(RefusalCode.BAD_ARGS, rows=2).payload()


def _gives_a_message(call: ast.Call) -> bool:
    """A refusal built with a message: positional, ``message=``, or spread
    from another refusal's or a helper's facts (``**``)."""
    return len(call.args) > 1 or any(
        keyword.arg in ("message", None) for keyword in call.keywords)


def test_every_refusal_langslice_raises_says_what_to_do() -> None:
    """Every ``Refused`` and ``ToolError`` built in the package carries a
    message (what was attempted, what was found, what to do); a reply whose
    rows each name their own refusal is a reply, not a ``Refused``."""
    root = Path(langslice.__file__).parent
    bare: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name in ("Refused", "ToolError") and not _gives_a_message(node):
                bare.append(f"{path.relative_to(root)}:{node.lineno}")
    assert bare == []
