"""The settings table (``job/settings/schema.py``) and the doors that read it:
the table names exactly the job form's keys and defaults, every spelling
follows the setting's name, the commands' flags are the table's (its old
spellings read and never listed), and ``create_job`` takes each library
setting by the keyword the table records."""

from __future__ import annotations

import argparse
import dataclasses
import inspect

from langslice.doors.agent_cli.calls import init_parser
from langslice.doors.commands.flags import FLAGS
from langslice.doors.commands.parser import build_parser
from langslice.doors.library.handle import create_job
from langslice.hosts.gui import generator
from langslice.job.settings import schema
from langslice.job.settings.spec import INPUT_KEYS, JobSpec

#: Flags of the commands that set no job setting.
NOT_SETTINGS = {"--trace-dir", "--gui", "--job", "-h", "--help"}


def _fields(cls: type) -> dict[str, dataclasses.Field[object]]:
    return {item.name: item for item in dataclasses.fields(cls)}


def _default(item: dataclasses.Field[object]) -> object:
    if item.default_factory is not dataclasses.MISSING:
        return item.default_factory()
    return None if item.default is dataclasses.MISSING else item.default


def test_the_table_names_every_key_of_the_job_form_and_no_other() -> None:
    top, sections = schema.spec_keys()
    fields = _fields(JobSpec)
    assert top == set(fields)
    for name, keys in sections.items():
        assert keys == set(_fields(fields[name].default_factory().__class__)), name
    inputs = {row.name.removeprefix(schema.ANSWER_PREFIX) for row in schema.SETTINGS
              if row.name.startswith(schema.ANSWER_PREFIX)}
    assert inputs == set(INPUT_KEYS)


def test_the_table_holds_the_job_forms_defaults_and_types() -> None:
    fields = _fields(JobSpec)
    for row in schema.SETTINGS:
        if row.json_key is None or not row.json_key.startswith("spec."):
            continue
        section, _dot, key = row.name.partition(".")
        if section == "inputs":
            assert row.default is None
            continue
        held = fields[section]
        if key:
            held = _fields(held.default_factory().__class__)[key]
        assert row.default == _default(held), row.name
        assert row.type == held.type, row.name


def test_every_setting_is_named_once() -> None:
    names = [row.name for row in schema.SETTINGS]
    assert len(names) == len(set(names))
    assert all(schema.setting(name).name == name for name in names)


def _flags(parser: argparse.ArgumentParser) -> list[str]:
    return [action.option_strings[0] for action in parser._actions  # noqa: SLF001
            if action.option_strings and action.option_strings[0] not in NOT_SETTINGS
            and action.help != argparse.SUPPRESS]


#: The settings given by flags the rule does not make: the stack's cutting
#: angles as two numbers, or each section's.
NAMED_OTHERWISE = {"inputs.angles": ("--pitch", "--yaw", "--section-angles")}


def test_every_spelling_follows_the_settings_name() -> None:
    for row in schema.SETTINGS:
        if row.json_key is not None and row.json_key.startswith("spec."):
            assert row.json_key == f"spec.{row.name}", row.name
        if not row.flags:
            continue
        flag = "--" + row.name.removeprefix(schema.ANSWER_PREFIX).replace(".", "-").replace(
            "_", "-")
        if row.default is True:  # a switch turns it off
            flag = "--no-" + flag.removeprefix("--")
        assert row.flags == NAMED_OTHERWISE.get(row.name, (flag,)), row.name
        section, dot, _key = row.name.partition(".")
        wanted = row.name if not dot else "inputs" if section == "inputs" else section
        assert row.keyword in (None, wanted, row.name.removeprefix(schema.ANSWER_PREFIX)), (
            row.name)


def _command(name: str) -> argparse.ArgumentParser:
    parser = build_parser()
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            chosen = action.choices[name]
            if name == "linear":
                return _command_of(chosen, "run")
            return chosen
    raise AssertionError(name)


def _command_of(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return action.choices[name]
    raise AssertionError(name)


def test_the_commands_flags_are_the_tables_in_its_order() -> None:
    spelled = [spelling for row in schema.SETTINGS for spelling in row.flags]
    assert set(spelled) == set(FLAGS)
    aliases = [alias for row in schema.SETTINGS for alias in row.aliases]
    assert not set(aliases) & set(spelled) and len(aliases) == len(set(aliases))
    for door, parser in (("run", _command("linear")), ("mcp", _command("mcp")),
                         ("init", init_parser())):
        flags = _flags(parser)
        wanted = [spelling for row in schema.SETTINGS if door in row.doors
                  for spelling in row.flags]
        assert sorted(flags) == sorted(wanted), door
        assert flags == sorted(flags, key=spelled.index), door


def test_create_job_takes_each_library_setting_by_its_keyword() -> None:
    parameters = inspect.signature(create_job).parameters
    fields = set(_fields(JobSpec))
    for row in schema.SETTINGS:
        if "library" not in row.doors:
            continue
        assert row.keyword is not None, row.name
        assert row.keyword in parameters or row.keyword in fields, row.name
    for name in ("atlas", "plane"):
        assert parameters[name].default == schema.setting(name).default


def test_the_prompt_generator_spells_each_flag_as_the_table_does() -> None:
    flags = generator.payload()["flags"]
    assert flags == schema.flag_spellings()
    assert flags["position.thickness_um"] == "--position-thickness-um"


def test_an_old_spelling_is_read_as_the_new_one_and_listed_nowhere() -> None:
    from langslice.doors.agent_cli.catalog import flags_table

    parser = init_parser()
    listed = {row["flag"] for row in flags_table(parser)} | {
        alias for row in flags_table(parser) for alias in row["aliases"]}
    for row in schema.SETTINGS:
        assert not set(row.aliases) & listed, row.name
    assert "--thickness" not in parser.format_help()
    old = parser.parse_args(["--thickness", "40", "--image-provider", "none", "--angles",
                             "--disable", "zoom", "--fact", "a", "--fresh"])
    new = parser.parse_args(["--position-thickness-um", "40", "--nonlinear-provider", "none",
                             "--transform-angles", "--disabled", "zoom", "--facts", "a",
                             "--no-resume"])
    assert vars(old) == vars(new)
    assert vars(parser.parse_args([])) == vars(init_parser().parse_args([]))
