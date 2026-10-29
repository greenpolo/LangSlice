"""The one refusal: its codes, its reply and how it reads."""

from __future__ import annotations

import json

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


def test_a_plugin_tool_error_is_a_refusal_with_its_message_first() -> None:
    error = ToolError(RefusalCode.BAD_ARGS, "", rows=2)
    assert isinstance(error, Refused) and isinstance(error, ValueError)
    assert error.payload() == {"status": "error", "error": "BAD_ARGS",
                               "message": "BAD_ARGS", "rows": 2}
    assert error.details == {"rows": 2} and error.message == "BAD_ARGS"
