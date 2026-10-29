"""Tests of the function shape checks: length, parameters, returns, nesting."""

from __future__ import annotations

from tools.stylecheck.testing import check_source


def test_body_lines_flags_a_long_body() -> None:
    statements = "".join(f"    x{index} = {index}\n" for index in range(101))
    violations = check_source(f"def long(a):\n{statements}", "body-lines")
    assert len(violations) == 1
    assert "101 lines" in violations[0].message


def test_body_lines_does_not_count_the_docstring() -> None:
    docstring = '    """Do it.\n\n' + "    More text.\n" * 40 + '    """\n'
    statements = "".join(f"    x{index} = {index}\n" for index in range(90))
    source = f"def documented(a):\n{docstring}{statements}"
    assert check_source(source, "body-lines") == []


def test_parameters_counts_variadics_but_not_self() -> None:
    source = """
    class Fit:
        def run(self, a, b, c, d, e, *args, **kwargs):
            pass

        def small(self, a, b, c, d, e, f):
            pass
    """
    violations = check_source(source, "parameters")
    assert [violation.symbol for violation in violations] == ["Fit.run"]
    assert "7 parameters" in violations[0].message


def test_parameters_counts_keyword_only_options() -> None:
    source = "def wide(a, b, *, c=1, d=2, e=3, f=4, g=5):\n    pass\n"
    assert len(check_source(source, "parameters")) == 1


def test_returned_values_flags_a_four_tuple() -> None:
    source = """
    def box(a):
        return a, a, a, a

    def pair(a):
        return a, a
    """
    violations = check_source(source, "returned-values")
    assert [violation.symbol for violation in violations] == ["box"]


def test_returned_values_flags_lists_names_and_annotations() -> None:
    """The gaming snippets of the review: the same four values, repacked."""
    source = """
    def as_list(a):
        return [a, a, a, a]

    def via_name(a):
        result = (a, a, a, a)
        return result

    def annotated(a) -> tuple[int, int, int, int]:
        return compute(a)

    def short_list(a) -> tuple[int, ...]:
        values = [a, a, a, a]
        values = [a]
        return values

    def three(a) -> tuple[int, int, int]:
        return a, a, a

    def lines(a, rest):
        return [a, a, a, *rest]

    def arguments(a):
        args = [a, a, a, a]
        args.append(a)
        return args
    """
    violations = check_source(source, "returned-values")
    assert [violation.symbol for violation in violations] == [
        "as_list",
        "via_name",
        "annotated",
    ]


def test_returned_values_counts_the_fields_of_a_named_tuple() -> None:
    """A named tuple of four fields is four values, in any of its forms."""
    others = {
        "src/langslice/core/shapes.py": """
        from typing import NamedTuple
        from collections import namedtuple

        class Box(NamedTuple):
            x: int
            y: int
            width: int
            height: int

        class Pair(NamedTuple):
            x: int
            y: int

        Span = namedtuple("Span", "start stop step size")
        """,
    }
    source = """
    from langslice.core.shapes import Box, Pair, Span

    def box(a):
        return Box(a, a, a, a)

    def span(a):
        return Span(a, a, a, a)

    def annotated(a) -> "Box":
        return make(a)

    def pair(a) -> Pair:
        return Pair(a, a)
    """
    violations = check_source(source, "returned-values", others=others)
    assert [violation.symbol for violation in violations] == [
        "box",
        "span",
        "annotated",
    ]
    unimported = source.replace("from langslice.core.shapes", "from shapes")
    assert check_source(unimported, "returned-values", others=others) == []


def test_returned_values_flags_a_dict_of_results() -> None:
    # the game: return a dict, or dict(...), instead of a tuple
    source = """
    def results(a):
        return {"shift": a, "error": a, "phase": a}

    def results_call(a):
        return dict(shift=a, error=a, phase=a)

    def lookup(a):
        return {"shift": a, "error": a}
    """
    violations = check_source(source, "returned-values")
    assert [violation.symbol for violation in violations] == [
        "results",
        "results_call",
    ]


def test_returned_values_allows_dicts_in_json_modules() -> None:
    source = """
    def payload(a):
        return {"shift": a, "error": a, "phase": a}
    """
    violations = check_source(
        source,
        "returned-values",
        relpath="src/langslice/job/public_files.py",
    )
    assert violations == []


def test_nested_definition_flags_a_closure_factory() -> None:
    source = """
    def bodies(job):
        def look(sections):
            return job.look(sections)
        return {"look": look}
    """
    violations = check_source(source, "nested-definition")
    assert [violation.symbol for violation in violations] == ["bodies.look"]


def test_nested_definition_allows_a_short_callback() -> None:
    source = """
    import functools

    def run(items, pool):
        def double(item):
            return item * 2
        pool.map(double, items)
        return functools.partial(sorted, key=double)
    """
    assert check_source(source, "nested-definition") == []


def test_nested_definition_flags_a_long_callback() -> None:
    lines = "".join(f"        x{index} = {index}\n" for index in range(10))
    source = (
        f"def run(items, pool):\n    def step(item):\n{lines}"
        "        return item\n    pool.map(step, items)\n"
    )
    assert len(check_source(source, "nested-definition")) == 1


def test_nested_definition_flags_a_lambda_outside_key() -> None:
    # the game: a lambda in place of a nested def
    source = """
    def order(items, cache):
        ranked = sorted(items, key=lambda item: item.rank)
        best = max(items, key=len, default=None)
        loader = lambda: load(items)
        return ranked, best, cache.get(loader)
    """
    violations = check_source(source, "nested-definition")
    assert len(violations) == 1
    assert "lambda" in violations[0].message


def test_nested_definition_allows_module_level_lambdas_and_classes() -> None:
    source = """
    TABLE = {"double": lambda x: 2 * x}

    class Outer:
        class Inner:
            pass

        def method(self):
            return 1
    """
    assert check_source(source, "nested-definition") == []


def test_nested_definition_flags_a_class_in_a_function() -> None:
    source = """
    def make():
        class Handler:
            def handle(self):
                return 1
        return Handler
    """
    violations = check_source(source, "nested-definition")
    assert [violation.symbol for violation in violations] == ["make.Handler"]


def test_forwarder_flags_a_parameter_pass_through() -> None:
    source = """
    def run_fit(inputs, settings):
        return run_engine(inputs, settings)

    def reordered(inputs, settings):
        return run_engine(settings, inputs)

    async def fetch(url):
        return await client.get(url)
    """
    violations = check_source(source, "forwarder")
    assert [violation.symbol for violation in violations] == [
        "run_fit",
        "reordered",
        "fetch",
    ]


def test_forwarder_flags_an_attribute_rename() -> None:
    source = """
    def _folder(job):
        return job.layout.folder

    class Job:
        def frozen_for(self, section):
            return self.spec.frozen_for(section)
    """
    violations = check_source(source, "forwarder")
    assert [violation.symbol for violation in violations] == [
        "_folder",
        "Job.frozen_for",
    ]


def test_forwarder_sees_through_a_no_op_statement() -> None:
    # the game: pad the forwarder with a statement that does nothing
    source = """
    def run_fit(inputs, settings):
        pass
        return run_engine(inputs, settings)
    """
    assert len(check_source(source, "forwarder")) == 1


def test_forwarder_passes_a_function_that_adds_something() -> None:
    source = """
    def fit(inputs, settings):
        return run_engine(inputs, settings, metric="mi")

    def first(inputs):
        return run_engine(inputs[0])

    def partial(inputs, settings):
        return run_engine(inputs)

    def chained(inputs):
        return make().run(inputs)
    """
    assert check_source(source, "forwarder") == []


def test_forwarder_exemptions() -> None:
    source = '''
    import functools

    class Box:
        def __len__(self):
            return len(self.items)

        @property
        def folder(self):
            return self.layout.folder

        def close(self):
            self._handle.close()

        def name(self):
            return self.name_text

    @functools.cache
    def atlas(name):
        return load_atlas(name)

    def open_job(folder):
        """Open a job folder; the public alias at the library door."""
        return _open(folder)
    '''
    assert check_source(source, "forwarder") == []


def test_forwarder_exempts_an_override() -> None:
    source = """
    import logging

    class Handler(logging.Handler):
        def emit(self, record):
            self.target.emit(record)
    """
    assert check_source(source, "forwarder") == []


def test_forwarder_is_flagged_in_a_class_with_an_external_base() -> None:
    """An external base exempts only the methods it defines."""
    source = """
    from types import SimpleNamespace
    from http.server import BaseHTTPRequestHandler

    class Options(SimpleNamespace):
        def folder(self, job):
            return job.layout.folder

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply()

        def log_message(self, format, *args):
            self.log(format, *args)

        def send(self, body):
            self.write(body)
    """
    violations = check_source(source, "forwarder")
    assert [violation.symbol for violation in violations] == [
        "Options.folder",
        "Handler.send",
    ]
