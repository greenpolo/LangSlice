from __future__ import annotations

import pytest

import langslice_harness.cli as cli


def test_parser_supports_serve_command() -> None:
    args = cli._build_parser().parse_args(["serve", "--stdio"])
    assert args.command == "serve"
    assert args.stdio is True


def test_serve_requires_stdio() -> None:
    with pytest.raises(SystemExit):
        cli.main(["serve"])
