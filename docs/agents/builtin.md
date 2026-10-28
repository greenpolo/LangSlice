# The built-in agent

`langslice linear run FOLDER` runs LangSlice's own agent session on a job,
with a model LangSlice calls itself.

## Run it

```bash
langslice linear run sections/ --tasks position,transform
langslice linear run sections/ --tasks position,transform,nonlinear --image-provider none
langslice linear run sections/ --registration quicknii.json
```

One session handles the whole stack and ends at `submit`. Its system prompt
is one short text, the same for every job (the coding-agent connectors use the
same text); the job statement, the user's notes, the opening pictures and the
status table are its first message.
The default tasks are `position,transform`; `--registration` imports a linear
placement and defaults to `nonlinear`. The agent chooses its sequence of
tools, regions and fit settings. Nonlinear registration can fit to the stain
alone or use optional image-model border traces.

For model access, `langslice login` signs in with ChatGPT; API providers use
`OPENAI_API_KEY` or `GEMINI_API_KEY` / `GOOGLE_API_KEY` (also read from `.env`).
An optional image model has its own provider setting and credentials.

Claude models run in `langslice linear run` through the Anthropic API:
`--model anthropic-api/<model id>` (install the extra with
`pip install -e ".[anthropic]"`). The key is `ANTHROPIC_API_KEY` (a key
saved in the LangSlice window's Settings is loaded as it), else the file
`~/.langslice/anthropic_api_key`. `--reasoning` sets Claude's
effort (`low`, `medium`, `high`, `xhigh`, `max`) with adaptive thinking whose
summary is written to the run trace, and the system prompt and the newest
message are marked for prompt caching.

## Job settings

Every job setting, with its flag, default and when to change it, is in
[Job settings](../reference/settings.md); `langslice linear run --help` and
`langslice-job schema init` are the complete flag reference.
`langslice linear run` takes the same flags as `langslice-job FOLDER init`,
and these of its own:

| Flag | What it does |
|---|---|
| `--model` | The model that drives the run (default `openai-oauth/gpt-6.1-sol`). Without `--model`, the model chosen in the LangSlice window's Settings (`~/.langslice/settings.json`) is used, with its reasoning effort unless `--reasoning` is given; the run prints the model it took from there. |
| `--reasoning` | Reasoning effort (OAuth default `medium`). |
| `--no-debrief` | No closing call asking what tools the agent missed. |
| `--out`, `--fresh` | Where the results JSON goes; start the job over. |
| `--trace-dir` | A second folder for the run's trace (also `langslice mcp`). |
| `--tool-plugin`, `--tool-plugin-config` | [Tool plugins](../developers/plugins.md). |
| `--gui` | Opens the [LangSlice window](../developers/gui.md) on the job. |

`--image-prompt FILE` replaces the base prompt `trace_borders` sends the
image model for this job; start from
`langslice.default_prompt(provider, plane)`. The agent is shown it as the
base it may edit per section, and its traces are recorded as an untested
profile (`custom`).

An interrupted job resumes from its checkpoint. Resuming with different
supplied inputs is refused with `INPUTS_CHANGED`, and `init` with other job
settings than the saved job's with `SETTINGS_CHANGED`; `--fresh` starts the job
over. Every write is checkpointed and undoable. Calls through different
interfaces pick up each other's saved changes. A write waits for the job
folder's lock while another holds it (up to 300 s, then `JOB_BUSY`); a read
(`look`, `zoom`, `status`, the atlas and job-folder tools) never waits: it
reads the job as it was last saved.

The built-in agent prints token usage per model call. LangSlice sets no limit
on model turns, tool calls, tokens or time, and nothing stops a run for what it
has spent. Spend is reported to whoever runs LangSlice, never to the model:

- On the ChatGPT OAuth lane, which reports the account's usage window, a
  `[spend]` warning is logged each time this run's share of the window (its
  rise since the run's first reading) passes another 10 percentage points.
- On the API-key lanes (`anthropic-api`, `openai-api`, `gemini-api`) a running
  dollar estimate from per-token list prices (input, cached input, output;
  thinking tokens as output; Claude input that is not a cache read as a cache
  write) is warned at $5, $10, $20, $40 and so on, doubling. A model without a
  price in `agent/spend.py` is said so once, with no estimate.
- Each warning also reaches the live event stream as a `spend` event (the ABBA
  agent viewer shows it) and the trace as a `spend` record. The account so
  far is written to `exports/spend.json` in the job folder after every model
  call; the final account is written there when the session ends, also when
  it fails part-way, and is the `spend` field of the trace's `summary` and of
  the `complete` event.


## Logs and credentials

The built-in agent always writes a JSONL trace to the job's `trace/` folder.
`--trace-dir`, else `LANGSLICE_TRACE_DIR`, names a second folder: the same
file is written there too, line by line as the run goes. It records model and
tool events; images are described by type, size and byte count rather than
embedded. MCP traces record calls and results, not the host model's private
conversation. Every tool call on every door (the built-in agent, MCP, the
agent CLI, a script) is a `tool_start` and a `tool_end` line in the job's
`logs/events.jsonl` ([format](../reference/file_formats.md#events-logseventsjsonl)). The
agent CLI logs every call in `logs/calls.jsonl` (the verb,
its arguments as given, the outcome, an error's code and message, the
artifacts, the seconds) and keeps each call's whole envelope, as printed, in
`logs/calls/<n>_<verb>.json`, which the call's line names under `reply`, so a
reply the harness cut short or the agent did not keep is still in the job; it
uses `LANGSLICE_TRACE_DIR` for additional tool traces. The Claude Code and
Antigravity plugins copy a registration agent's own transcript into the
job's `trace/` folder when its session ends ([transcripts](cli.md#transcripts)).

The `openai-oauth` agent transport automatically retries connection failures,
timeouts, HTTP 408/409/429 and 5xx errors (including 507), and early
`response.failed` events reporting server overload, server errors or temporary
rate limits. HTTP and event failures share a budget of three retries; event
failures are retried only before any model output arrives. It logs each retry and waits
2, 4 and 8 seconds with jitter, honoring a longer `Retry-After` up to 60
seconds. Longer server delays, explicit quota exhaustion, certificate errors
and other permanent failures stop the run. A 401 can refresh credentials once.
Stream failures are surfaced without replaying partially delivered tool calls;
image edits are not covered by these request retries.

`langslice login` stores OAuth credentials in
`~/.langslice/openai_auth.json`; `LANGSLICE_OPENAI_AUTH` overrides that path.
LangSlice uses its own login. The LangSlice window's Settings and ABBA setup
can also save API keys in `~/.langslice/provider_credentials.json` (the
window also the Claude key, loaded as `ANTHROPIC_API_KEY`); explicit
environment variables take precedence. Credential presence is checked offline when deciding
whether to offer `trace_borders`.
