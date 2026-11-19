# LangSlice code style

The style is small on purpose. ruff formats and lints at 100 columns,
basedpyright checks types in standard mode, import-linter holds the layer
order, and `tests/test_no_comments_or_waivers.py` holds the one rule of
our own. None has a baseline: a finding is fixed in the code.

## No comments, no waivers

- No `#` comment in `src/langslice` or `tests`. A name says what a line is;
  a docstring says why.
- No `# noqa`, `# type: ignore`, `# pyright: ignore`, `# pragma` or `# fmt`
  switch. Change the code until the tool has nothing to report.
- A comment is not replaced by a paragraph. Constants that belong together
  go in a frozen dataclass whose docstring explains the algorithm and whose
  field names carry the unit and the meaning:

```python
@dataclass(frozen=True)
class GlareFilter:
    """Drop bright, low-grain scanner glare patches outside the tissue."""

    max_sharpness_ratio: float = 0.55
    min_image_area_fraction: float = 0.0004
    surround_ring_px: int = 6
```

- Code copied from other labs (`src/langslice/_vendor`) stays as published.

## Design before rules

- Fix a bad abstraction at its root. Do not add a lint rule, a wrapper or a
  compatibility path to live with it.
- When a clean result needs a refactor, do the refactor or say so. Do not
  patch around it.
- Delete the old path when a new one replaces it. LangSlice is unpublished;
  nothing depends on its old shapes.
- One definition of each thing: one settings model, one atlas loader, one
  field sampler, one provider table.
- A typed object crosses a function boundary, not a `dict[str, Any]`.
  Untrusted input is checked once, where it enters.
- Compose (`job.undo`, `job.traces`); do not split one object into mixin
  base classes.
- Catch the exceptions a call is known to raise, by name.

## Names and text

- Plain words; units in names (`spacing_um`, `timeout_s`).
- A docstring says what a caller needs and nothing about history.
- Model-facing text is written where its tool is defined, as plain sentences.
