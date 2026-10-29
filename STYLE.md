# LangSlice code style

This guide says how LangSlice's Python should read. It is written for the
people and agents who change the code, and it is the reference for every
refactor. A reader who knows scientific Python (numpy, scipy, the array
libraries) and the paper should be able to read any module in one pass.

Each rule carries the tool that enforces it:

- **[format]** the formatter (`ruff format`).
- **[ruff]** a ruff rule, in the main lint run (`pyproject.toml`) or in
  the checker's own ruff run (`tools/stylecheck/external.py`).
- **[numpydoc]** the numpydoc validation hook.
- **[types]** basedpyright in strict mode.
- **[linter]** LangSlice's own checker (`tools/stylecheck`), which runs in
  CI with the others. The tag names its checks; `python -m tools.stylecheck
  --explain NAME` says what each one flags.
- **[review]** needs a reader; the linter cannot decide it.

Where a rule gives a number, the number was set where about 95% of the
functions and modules in mature scientific-Python packages already fall, so
it binds on the long tail only. The text follows PEP 8, the numpydoc
docstring standard, and the Cognitive Complexity definition (SonarSource,
2017) where they apply, and says so where LangSlice departs from them.


## 1. Layout and spacing

**1.1 Format with the formatter, at 79 columns.** [format; ruff E501]
The formatter decides where lines break; nobody arranges whitespace by
hand. When a call does not fit on one line it puts one argument per line,
which lines up keyword names and makes a changed setting one changed line
in a diff. A `noqa: E501` is allowed only on a URL or a row of a literal
table [linter `noqa-line-length`].

**1.2 One idea per line.** [linter `nested-conditional`,
`conditional-placement`, `nested-comprehension`, `comprehension-clauses`,
`walrus-in-comprehension`] A line holds one condition, one loop or one call
that does work.

- No conditional expression inside another, inside a call's argument list,
  or inside a comprehension.
- No comprehension inside a comprehension; one `for` and at most one `if`
  per comprehension; no `:=` inside a comprehension.
- A conditional expression is allowed only as the whole right-hand side of
  an assignment or a `return`.

A plain one-level comprehension that reads as a phrase is good code:
`names = [section.name for section in sections if section.placed]`.

```python
# Not this
caption = ("warped" if drawn else "held" if drawn is False else "linear"
           if request.warp == "none" and record.deformation else "none")

# This
if drawn:
    caption = "warped"
elif drawn is False:
    caption = "held"
elif request.warp == "none" and record.deformation:
    caption = "linear"
else:
    caption = "none"
```

**1.3 Write a function in paragraphs.** [linter `paragraph-length`,
warning] Separate the steps
of a function with blank lines. A paragraph longer than 15 lines (comments
count, blank lines end it) is two steps that need a blank line, or a step
that needs its own function. A step that is not obvious opens with a short
comment naming it:

```python
    # whole-pixel shift from the cross-correlation peak
    peak = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    shift_px = np.array(peak, dtype=float)
    shift_px[shift_px > midpoints] -= np.array(shape)[shift_px > midpoints]

    # refine to a fraction of a pixel around that estimate
    ...
```

**1.4 Narrate a pipeline.** [review] A function that runs a sequence of
stages calls one named function per stage and logs each stage as it starts,
in the words a user would recognise in the log. Library functions that
compute one thing do not log.

```python
    logger.info("Fitting the affine for %s", section.name)
    affine = fit_affine(section, plane, settings)

    logger.info("Fitting the deformation for %s", section.name)
    deformation = fit_deformation(section, plane, affine, settings)
```

**1.5 Keep expressions shallow.** [review] A statement whose calls nest more
than two deep, or whose line has to be read twice, is split with a named
intermediate. The name is the comment.


## 2. Modules

**2.1 One topic per module, named by a noun.** [review] `atlas.py`,
`landmarks.py`, `masks.py`. A module whose name needs "and" is two modules.
A module named `utils`, `helpers`, `misc`, `common` or `*_part2` is a
module that has not found its topic [linter `module-name`].

**2.2 Modules are at most 800 lines.** [linter `module-size`] Most are far shorter. A
module over 400 lines is one class or one family of functions on one record;
a module that reaches 800 is split by topic, never by halves.

**2.3 Imports at the top.** [linter `local-import`] A reader learns a
module's dependencies from its first lines. An import inside a function is
allowed for two reasons, and the line says which: a slow optional dependency
(`torch`, `SimpleITK`, `fireants`, `itk`, `cv2`), or a cycle the layer rule
forces, naming the module that would cycle. The linter checks the name
against the import graph.

```python
    import torch  # local: slow to import, needed only for a deformable fit
    from langslice.tools.status import save_view  # local: cycle with tools.status
```

**2.4 `__init__.py` holds no logic.** [linter `init-logic`] It is empty, sets the
version, or re-exports the package's public names. Nothing else.

**2.5 Order within a module.** [review] Docstring, imports, constants and
type aliases, exceptions, then the public functions or the class, with
private helpers next to their first use. A reader meets the public thing
before its parts.

**2.6 Keep infrastructure apart from science.** [review, linter] Code that
exists for speed or process management (the warm fork server, locks, caches)
lives in its own clearly named subpackage with a short README. Code that
reaches into another library's private attributes does so in one module
under that subpackage, with the library version it was written against.
Private access on another package's objects anywhere else is an error
[ruff SLF001; linter `private-getattr` for `getattr(x, "_name")`].

**2.7 No mutable module state.** [linter `global-statement`,
`globals-or-module-setattr`] Module-level names are constants,
types, functions and tables. No `global` statement, no assignment to a
module name from inside a function, no `globals()` lookup. A cache or a
registry is an object in a named module with a documented lifetime.


## 3. Functions

**3.1 Short and flat.** [linter `cognitive-complexity`, `nesting-depth`,
`body-lines`, `parameters`] The limits:

| measure | limit | counted how |
|---|---|---|
| Cognitive complexity | 15 | every branch, loop, `except`, boolean sequence and recursion, charged more at each level of nesting |
| Nesting depth | 4 | blocks (`if`, `for`, `while`, `with`, `try`, `match`) inside the function |
| Body lines | 100 | from the first statement after the docstring to the last; the docstring is not counted |
| Parameters | 6 | positional and keyword, not counting `self` or `cls` |

Cognitive complexity is the measure that follows how hard a function is to
hold in mind; the others are guards against the ways of hiding it. Length is
capped loosely on purpose: a long function that reads as a list of narrated
steps (1.3, 1.4) is easier to follow than the same work cut into pieces that
each need a name.

**3.2 Extract a function for a reason.** [review] Pull a step out into its
own function when it is reused, when it has a name in the domain
(`crop_atlas`, `flip_section`), when it carries a formula worth a docstring,
or when it needs testing alone. Do not split a function to get under a limit;
the parts must each be something a reader would name. A one-statement helper
with a good name and a docstring is fine; a helper whose name says less than
its body goes back inline.

**3.3 No forwarders.** [linter `forwarder`] A function whose body only passes its
parameters to another call, or only renames an attribute, is removed;
callers call the real thing. The one exception is a public alias with its
own docstring at a door.

```python
# Not this
def _folder(job: Job) -> Path:
    return job.layout.folder
```

**3.4 No nested function definitions,** [linter `nested-definition`] with
two exceptions: a
callback of ten lines or fewer handed straight to a framework, to
`functools.partial` or to a `key=`; and the wrapper inside a decorator in the
project's decorators module (10.3). A function that needs its surroundings
takes them as arguments: a tool body is a module-level function that
receives `door`, not a closure inside a factory. Lambdas count as nested
definitions except in `key=` and `default=` positions.

**3.5 Arguments.** [linter unless marked]

- Required inputs first, positionally [linter `keyword-only-options`].
  Options after `*`, with literal defaults, so a call reads `fit_affine(section, plane, metric="mi")` and
  the order of options can change without breaking a caller.
- No mutable defaults [ruff B006]. `None` means "not given" and is resolved
  in the first lines of the body.
- `**kwargs` only when forwarded whole to one named callee, and the
  docstring says so: `**kwargs : passed to fireants.registration.Affine`.
  The name `kwargs` appears nowhere else in the body. `*args` only for a
  homogeneous "one or many" (`add(*layers)`) [linter `kwargs-forwarding`].
- No output parameters [linter `output-parameter`]. A function returns
  what it computes; it does not append to a list or write into a dict its
  caller passed in. A pre-allocated numpy `out=` array is the one accepted
  form.
- [review] More than six parameters means the options belong together:
  group them in a small frozen dataclass (`BorderStyle`, `ZoomWindow`) built
  once where the request comes in and passed down unchanged. An option keeps
  one name from the door to the core; it is not renamed at each layer.

**3.6 Return named results.** [linter `returned-values`, at four] One value, or a tuple of at
most three values each named in the `Returns` section. Four or more values,
or a result whose arity depends on a flag, come back as a frozen dataclass
with named fields. Never a dict of results.

**3.7 Retire a superseded name.** [review] When a parameter is normalised
into another name or copied into `self`, `del` the old one on the next line
so nothing later reads the stale copy.

**3.8 Classes have a size.** [linter `class-size`] At most 30 methods and 600 lines. A
class past that is several roles under one name.


## 4. Names

**4.1 Use the paper's words.** [review] The code uses the same words as the
paper and the docs: section, atlas, atlas plane, position, angles,
orientation, affine, deformation, landmark, annotation, reference. A section
is a `section`; "slice" means a 2D slice of an array and nothing else. Pairs
are named as pairs (`moving`/`reference`, `source`/`target`, `from_frame`/
`to_frame`). A new project noun needs a definition in the docstring of the
module that introduces it, and should be rare.

**4.2 Function names.** [review; linter `function-name` for the
predicates and the suffixes] A function is named for what it does or what
it yields, in full words. A predicate is `is_`, `has_` or `can_` and
returns a `bool`; a reader or writer is `read_`, `write_`, `load_` or
`save_`; a conversion is `<a>_to_<b>` or `<b>_from_<a>`; a constructor on a
class is `from_<source>`. Properties and dataclass fields are nouns. No
function name ends in `_words` or `_of`. The linter checks the predicates'
return type and the two suffixes; the other forms are for review.

**4.3 Variables say what the value is.** [linter `banned-name` for the
banned list; ruff N for PEP 8's case rules] Names
are whole words. Banned as any part of a variable or parameter name, because
each has come to mean many things here: `held`, `shown`, `drawn`, `placed`,
`asked`, `wanted`, `picked`, `found`, `thing`, `stuff`. Short names are
allowed for loop indices (`i`, `j`), coordinates (`x`, `y`, `z`), `key,
value` pairs, and mathematics where a comment or the docstring names the
symbol (`R  # rotation part of the affine`). Accepted abbreviations: `n_`
for counts, `idx`, `img`, `df`, `src`, `dst`, `np`, `pd`, `npt`.

**4.4 One name per type.** [linter `type-name`, from annotations] A parameter of a given
type has one name everywhere: a `Workspace` is `workspace`, never `ws` or
`ctx`; a `Job` is `job`; a `SliceState` is `section`. A qualifier may
precede it (`source_image`, `target_image`). The table of types and names
lives in the linter's configuration.

**4.5 Units and kinds in names.** [review] A quantity carries its unit
(`_mm`, `_um`, `_px`, `_deg`, `_s`); a quantity without one is a bug. Paths
end in `_path` or `_folder`. Counts start with `n_`. Booleans are flag words
(`flip`, `save_pictures`, `preserve_range`), not `is_` names; `is_` is for
predicates.


## 5. Data and types

**5.1 Records have declared shapes.** [linter `loose-record`,
`or-empty-get`; types] Data that is stored,
passed between layers or read in more than one place is a frozen dataclass.
It is not `dict[str, Any]`, `Mapping[str, object]` or an alias for either. A
`TypedDict` is allowed only in the modules that read and write JSON, listed
in the linter's configuration, and there with required keys. A reader learns
a record's fields from its definition, not by finding every writer.

```python
# Not this
held_pivot = ((record.transform or {}).get("physical") or {}).get("pivot")

# This
pivot = record.transform.physical.pivot
```

**5.2 Options and codes.** [linter `shared-option-literal`] A public option may be a `Literal` of
strings, validated once where it comes in (5.6). A value that crosses a
layer boundary, is compared in more than one module, or names a refusal is a
`StrEnum` member. No option value is compared with `==` or `in` against a
bare string literal in more than one module.

**5.3 No `Any` in a signature or a field,** [linter `any-annotation`;
types] and no alias
that hides it. `object` is allowed where the type really is anything.

**5.4 Trust the types.** [types; linter `literal-getattr`] Inside the package a value of a
declared type is used as that type: no `getattr(obj, "field", None)` for a
declared field, no `vars(obj)["field"]`, no `str(x)` on a `str`, no
`assert x is not None` after a lookup that cannot fail, no `isinstance` that
is always true.

**5.5 Type-check strictly.** [types] basedpyright in strict mode, with
`reportAny`, `reportUnnecessaryCast`, `reportUnnecessaryIsInstance` and
`reportUnnecessaryComparison` on. Each `# type: ignore` and each `cast`
names its error code and says why [ruff PGH003].

**5.6 Validate at the door, with the shared checkers.** [review; linter
`isinstance-outside-checkers` warns on `isinstance` outside the checkers
module and the modules that read JSON] Tool arguments, files and host inputs are validated where they come
in, with the functions in `core/checks.py` (`check_option`, `check_type`,
`check_path`, `check_array`), which produce the refusal and its message in
one form; a parsed file is narrowed in the module that reads it. Below the
door, code trusts its inputs.

**5.7 Dispatch with a table.** [review] When an option chooses an
implementation, a module-level dict from option to callable replaces an
`if/elif` chain over that option, and is the single list of what the option
accepts.


## 6. Docstrings

**6.1 numpydoc, validated.** [numpydoc] Every public function, method and
class has a numpydoc docstring that passes the validation hook: a function
with parameters has a `Parameters` section listing every parameter and no
other; one that returns a value has `Returns`; one that raises the project's
refusal has `Raises`. Each parameter has a description of at least three
words that says more than its name [linter `parameter-description`]. The
docstring's layout (blank lines, quotes, section underlines) follows the
numpy convention [ruff D2, D3, D4].

The type after the colon is written when it adds something the signature
cannot say (shape, unit, allowed values); otherwise the name stands alone:

```
    Parameters
    ----------
    plane : array, shape (H, W)
        The atlas plane the section is fitted to, in atlas millimetres.
    metric : {"mi", "ncc"}
        Similarity measure the fit optimises.
    settings
        Engine settings; defaults come from the job.
```

**6.2 The first line says what it does.** [numpydoc SS05; linter
`summary-first-word`] One line,
an imperative verb, ending in a period: "Fit an affine from the section to
the atlas plane." Not a noun phrase ("The affine fit ...") and not a
third-person form ("Fits ..."). The linter rejects a summary that opens with
`The`, `A`, `An`, `Whether`, `What`, `One`, `Every`, `Where`.

**6.3 Say what a caller needs; leave out the rest.** [review] Say units,
axis order, coordinate space, defaults ("None means the job's default"),
assumptions and cost, and cite the source of borrowed code. Put the
mathematics in `Notes`. Give an `Examples` section, as a doctest, for the
public library API where a worked call shows the shapes. Do not narrate the
body step by step, list callers, tell history, or name plans.

**6.4 Keep references to other code rare.** [linter
`docstring-references`, `docstring-layer`] A docstring's own
parameter names are written in double backticks as numpydoc wants; those do
not count. References to other symbols (`:func:`, `:class:`, a dotted name)
are at most four outside `See Also`. Never one from a lower layer to a
higher one: `core/` does not mention `doors/`, by dotted path or by bare
name.

**6.5 Module docstrings.** [numpydoc GL08, review] Every module opens with a
docstring of a few sentences saying what the module is for. A format or
protocol the module owns may be documented at length there, once. Private
helpers may go without docstrings when their name says it all.


## 7. Comments

**7.1 Two kinds of comment.** [review] A step header ("# refine to a
fraction of a pixel") or a reason the code has to be this way ("# plane by
plane: the 3D version needs more memory than a 25 µm atlas allows"). A
comment does not restate the line below it.

**7.2 Present tense, no history.** [linter `history-comment`] No dates, no names, no quotes,
no "used to", "legacy", "no longer" in comments. The reason is stated as it
holds now; git keeps the history.

**7.3 Constants carry their reason.** [review] A module-level constant has a
`#:` comment saying why it has its value.

**7.4 No commented-out code.** [ruff ERA001] Delete it; git keeps it.


## 8. Errors

**8.1 One way to refuse.** [linter `error-dict`, `error-code-literal`] An
error a caller should see is
one exception type, `Refused`, carrying a code from one `StrEnum` of codes
and a message. It is converted to a reply once, at the door. No error dicts
built by hand, no code written as a bare string.

**8.2 Messages are sentences.** [review] Say what was attempted, what was
found and what to do: "Section 12 has no atlas position yet; place it with
position_sections before fitting." Quote the bad value and list the allowed
ones.

**8.3 Never lose data silently.** [ruff BLE001, B904, S110; linter
`broad-except`] No bare
`except`. A broad `except Exception` names in its comment the failure it
expects, and its body raises, returns a value the function's type admits, or
reports the skip through `report_skip`; a body that only passes, continues,
or logs at debug level is an error. Re-raise with `from error`.

**8.4 `assert` is for invariants.** [review] An `assert` states something
valid input cannot break. Input is validated with the checkers (5.6), which
survive `python -O`.

**8.5 Log, don't print.** [linter `print-call`] Library code reports through
`logging`; `print` and `sys.stdout.write` belong only in the command modules
listed in the configuration.


## 9. Model-facing text

**9.1 Write the whole text in one place.** [review; linter
`text-assembly` flags a function with many string literals and many
branches] A tool description, job
statement or caption is a template a reader can read whole: a module of
named constants or a data file, one template per variant, with named slots.
Logic chooses which template and fills its slots; it does not assemble the
text clause by clause inside branches.

**9.2 Structure, not text surgery.** [linter `text-surgery`] Text is never edited by
matching indentation, exact sentences or regular expressions, and `__doc__`
is never assigned or sliced outside the one renderer that builds tool
descriptions. An argument's description is data that is left out when the
argument is disabled.

**9.3 The snapshot stays.** Every wording change shows up in the text
snapshot (`tests/snapshots/agent_text/`) and is reviewed there.


## 10. Idioms

**10.1 Standard library and aliases.** [ruff PTH, ICN001, UP] `pathlib` for paths,
f-strings for formatting (lazy `%s` inside `logger.*` calls). Standard
aliases only: `np`, `pd`, `npt`, `da`, `sitk`.

**10.2 No metaprogramming.** [linter `metaclass`, `getattr-forwarding`,
`runtime-type`, `globals-or-module-setattr`] No metaclasses beyond `abc.ABC`, no
`__getattr__` forwarding, no types built at runtime, no `setattr` on
modules.

**10.3 Decorators.** [linter `decorator-location`, `decorator-count`,
`decorator-composition`] The project's decorators live in one module, each
with a docstring, each doing one thing; `decorator-location` is off while
the linter's configuration names no decorators module. A definition
carries at most two of them. Behaviour shared by every tool (gates,
locking, saving views) is an explicit list of steps, a module constant,
applied in one loop; it is never a wrapper composed by nesting calls
(`_a(_b(_c(f)))`).

**10.4 One helper for one job.** [linter `duplicate-body`] A utility written twice is written
once in `core/` and imported; the linter compares function bodies after
renaming.

**10.5 Delete what a change replaces.** [review] Old formats and migrations
live in one module and go once no job needs them. Nothing is kept "in case".


## 11. What LangSlice already does well

Keep these through every refactor:

- The layer rule (`core < job < tools < doors < hosts`), enforced by
  import-linter with no exceptions.
- One declaration per tool, shared by every way in.
- The text snapshot of everything the model reads.
- Unit suffixes on quantities, and axis conventions in one module
  (`core/space.py`).
- A module docstring on every module.
- Frozen dataclasses for results.
- Named constants with a one-line reason.
- Exceptions chained with `from`.
- Lint exceptions that say why (`# noqa: BLE001 - an atlas without one`).
- Stable, machine-readable refusal codes.


## 12. Enforcement

| tool | what it owns |
|---|---|
| `ruff format` (run by `tools/stylecheck`) | 1.1 |
| `ruff check .`, the main run (`E`, `F`, `I`, `B`, `UP`; at 100 columns, with the package's import order left to the checker's run) | 3.5, 8.3, 10.1 |
| the checker's ruff run, against its baseline (`E501` and `I001` at 79 columns, `D2`, `D3`, `D4` with `convention = "numpy"`, `BLE`, `S110`, `SLF001`, `ERA001`, `ICN001`, `PGH003`, `PTH`, `N`) | 1.1, 2.6, 4.3, 5.5, 6.1, 7.4, 8.3, 10.1 |
| numpydoc validation (run by `tools/stylecheck`) | 6.1, 6.2, 6.5 |
| `basedpyright --strict` | 5.1, 5.3, 5.4, 5.5 |
| `lint-imports` | the layer rule |
| `tools/stylecheck` | every check named in a [linter] tag above; its configuration holds the limits, the type-to-name table, the banned stems, the boundary modules, the CLI allowlist and the decorators module |

The baselines of `tools/stylecheck` and basedpyright only shrink, and the
settings of every tool above only tighten; CI holds both against the
previous commit. A commit that moves code declares each moved definition
in `tools/stylecheck/moves.json`, and its errors and the modules the
settings name follow the move, at most what the old entry lost.

A rule the tools cannot check is marked [review] and is part of code review,
not optional.
