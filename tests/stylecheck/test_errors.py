"""Tests of the error checks."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from tools.stylecheck.core import Config
from tools.stylecheck.testing import flagged_lines

CONFIG = Config.load(Path(__file__).parents[2] / "pyproject.toml")


def test_broad_except_that_drops_the_error_is_flagged() -> None:
    source = """
    def fit():
        try:
            pass
        except Exception:  # an atlas without a hierarchy
            pass

        try:
            pass
        except:
            return None

        try:
            pass
        except (ValueError, BaseException):  # noqa: BLE001 - a bad file
            logger.debug("skipped")
            continue
    """
    assert flagged_lines(source, "broad-except") == [5, 10, 15]


def test_broad_except_needs_a_comment_naming_the_failure() -> None:
    source = """
    def fit():
        try:
            pass
        except Exception:
            logger.warning("failed")

        try:
            pass
        except Exception:  # noqa: BLE001
            return 0

        try:
            pass
        except Exception:  # noqa: BLE001 - an atlas without one
            return 0

        try:
            pass
        except Exception:
            cleanup()
            raise

        try:
            pass
        except ValueError:
            pass
    """
    assert flagged_lines(source, "broad-except") == [5, 10]


def test_print_and_terminal_writes_are_flagged() -> None:
    source = """
    import sys
    from rich import print as rprint

    def report():
        print("a")
        sys.stdout.write("b")
        sys.stderr.write("c")
        rprint("d")
        logger.info("e")
    """
    assert flagged_lines(source, "print-call") == [6, 7, 8, 9]


def test_print_is_allowed_in_command_modules() -> None:
    source = """
    def main():
        print("a")
    """
    config = dataclasses.replace(CONFIG, cli_modules=("langslice.cli",))
    relpath = "src/langslice/cli/run.py"
    assert (
        flagged_lines(source, "print-call", relpath=relpath, config=config)
        == []
    )


def test_error_dicts_are_flagged() -> None:
    source = """
    a = {"status": "error", "error": "BAD", "message": "m"}
    b = dict(status="error", message="m")
    c = {"status": "ok"}
    """
    assert flagged_lines(source, "error-dict") == [2, 3]


def test_error_codes_written_as_strings_are_flagged() -> None:
    source = """
    def fit(reply, flag):
        raise Refused("BAD_ARGS", message="start is linear or current.")
        raise ToolError("UNKNOWN_SLICE_IDS")
        _refuse("NO_ATLAS_YET")
        make(code="NOTHING_WRITTEN")
        if flag.get("code") == "SCALE_CHANGED":
            pass
        if reply["error"] in ("FROZEN_SECTION", "STALE_INPUT"):
            pass

        rows = {"code": "DISPLACEMENT_OUTSIZED"}
        reply["error"] = "RECORD_WRITE_FAILED"
        raise Refused("FROZEN", message="ALL_CAPS_TEXT is a message")
        if name == "TYPE_CHECKING":
            pass
    """

    lines = flagged_lines(source, "error-code-literal")
    assert lines == [3, 4, 5, 6, 7, 9, 9, 12, 13, 14]


def test_error_codes_hidden_from_the_literal_check_are_flagged() -> None:
    """The gaming snippets of the review: a code bound, built or chosen."""
    source = """
    BAD = "BAD_ARGS"

    def bound(kind):
        code = "NO_ATLAS"
        raise Refused(code, message="m")

    def module_constant():
        raise Refused(BAD)

    def built(kind):
        raise Refused("BAD" + "_ARGS")
        raise Refused(f"NO_{kind}")
        raise Refused("NO_" + kind.upper())

    def chosen(code, flag):
        raise Refused(code or "FROZEN")
        raise Refused("STALE" if flag else "FROZEN")

    def compared(error, reply):
        if error.code == "FROZEN":
            pass
        if reply.get("code") in ("STALE", "FROZEN"):
            pass
        if status_code == "UNKNOWN":
            pass

    class Codes:
        rows = {"code": "MISSING"}
    """
    lines = flagged_lines(source, "error-code-literal")
    assert lines == [2, 5, 12, 13, 14, 17, 18, 18, 21, 23, 23, 25, 29]


def test_upper_case_words_that_are_not_codes_are_left_alone() -> None:
    """One word is a code only in a code place; log levels never are."""
    source = """
    def fit(record, image, method, environ, name):
        if record.levelname == "ERROR":
            pass
        if image.mode in ("RGB", "RGBA"):
            pass
        if method == "GET":
            pass
        path = environ["HOME"]
        if name == "TYPE_CHECKING":
            pass
        level = "INFO"
        logger.setLevel(level)
        raise Refused(f"{name}", message="SECTION_NAME is unknown")
        raise ValueError(f"Section {name} has no atlas position yet.")
        code = "NO_ATLAS"
        code = "STALE"
        raise Refused(code)
    """
    assert flagged_lines(source, "error-code-literal") == []


def test_error_codes_are_allowed_in_the_codes_module() -> None:
    source = """
    BAD_ARGS = Code("BAD_ARGS")
    a = {"status": "error"}
    """
    config = dataclasses.replace(CONFIG, codes_module="langslice.ops.codes")
    relpath = "src/langslice/ops/codes.py"
    for rule in ("error-code-literal", "error-dict"):
        assert (
            flagged_lines(source, rule, relpath=relpath, config=config) == []
        )
