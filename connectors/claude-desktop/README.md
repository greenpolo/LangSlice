# Claude Desktop / Claude Code connector

LangSlice can serve its linear tools over MCP to a host that brings its own
model. The host runs the conversation and LangSlice supplies the job, the
pictures and the tools. The user's own Claude subscription runs the model in
Anthropic's own app, so LangSlice never handles Claude credentials.

The server is `langslice mcp` (code: `src/langslice/doors/mcp/`). It needs
the MCP SDK: `pip install "langslice[mcp]"`.

## Claude Desktop

Add the server to `claude_desktop_config.json` (Settings > Developer > Edit
Config). Use the absolute path of `langslice` in the LangSlice environment:

```json
{
  "mcpServers": {
    "langslice": {
      "command": "/path/to/langslice-env/bin/langslice",
      "args": ["mcp"]
    }
  }
}
```

Restart Desktop. In ABBA, choose **Claude** in LangSlice Registration, configure
the job, and click **Copy prompt**. Paste it into Claude; the prompt names the
saved job and asks for `start_job(job_id=...)`. Keep the ABBA progress window
open for live section moves. Closing it disconnects ABBA but does not stop
Claude; results remain in the job folder next to the exported snapshots,
`~/.langslice/snapshots/mcp-*/langslice/`. Enable only LangSlice for this
conversation, never a general Fiji scripting connector.

Without a registration host, make the job with the agent CLI, then name its
folder in the chat:

```bash
langslice-job FOLDER init --interval 200 --notes "Section 12 has a large tear."
```

and paste, for example, `Use the LangSlice connector: call
start_job(image_folder="FOLDER") and register the sections.` `init` takes
every `langslice linear run` job flag plus `--notes`, and makes the job in the
job folder next to the sections, `FOLDER/langslice/`. `start_job` opens it as
saved, with its settings and notes; `langslice mcp`'s own job flags never
change it. The job's settings, checkpoint, undo history, the pictures the host
was shown and the results live in that folder (the section images themselves
are never written), and reopening the job (a restarted Desktop, a new chat)
resumes from the checkpoint with its undo history. One image folder holds one
job: `init` on a folder that already holds one continues it, refuses other
supplied inputs (`--positions`, `--transforms`, ...) with `INPUTS_CHANGED`,
and starts over from the new inputs with `--fresh`. `--job-dir PATH` puts the
job folder elsewhere; when the sections' folder cannot be written, the job
folder is `~/.langslice/jobs/<id>/`. `init` prints the job folder, and
`start_job` takes the job folder as well as the image folder.

A folder without a job gets a new one when the host names it: `langslice mcp`
takes every `langslice linear run` flag (`--tasks`, `--interval`, `--atlas`,
`--trace-dir`, ...) for such folders; with `--fresh` they start every folder
the host names over, a saved job included. Put them in `args`.

The Nonlinear task (`--tasks ...,nonlinear`) works through this connector:
its fitting tool (`ants_syn`) is always offered with it. The image-model tool
(`trace_borders`) is offered only when the job's image provider is not
`none` and its key or login is present on this machine (`langslice login`,
or a saved or environment API key); otherwise it is simply not listed, and
the job statement (and ABBA's copy prompt) says that the image-model tool is off
because no image model is connected. The job keeps its provider setting, so
a later session with the login present offers the tool again.

## Claude Code, locked to LangSlice

Use this where Claude Desktop is not available (Linux). It removes every
built-in tool and loads only this server:

```bash
claude --strict-mcp-config --mcp-config connectors/claude-desktop/langslice.mcp.json \
  --tools "" --allowedTools "mcp__langslice"
```

## What the host gets and what it does not

- **Same as the ADK run:** the same tools (an unknown or misplaced
  argument is refused, not dropped),
  docstrings, submit gates, undo stack and checkpoint. Saved ABBA jobs also
  share the same validation, snapshot preprocessing and native update
  translation.
- **Claude briefing:** `start_job` returns the job statement LangSlice's own
  agent gets (the same text, saying where this door's opening pictures are),
  the user's notes, and the status table with the recent run notes, without
  images. Read every `show_stack(page=...)` page before writing.
  Sections arrive as labelled strips in the stack's order (1568 px long, Claude's
  recommended largest image), the atlas at each section's current position
  beneath it; atlas reference strips follow when a section has no position.
  Every page stays below 680,000 serialized bytes (a strip holds fewer
  sections rather than being shrunk), and every write is refused until each
  page was read. Every tool reply stays below the same size: past it, its
  pictures are shrunk together and the reply says so.
- **Different from the ADK run:**
  - There is no turn budget and no nudges.
  - There is no image working set, so the host keeps every picture for the
    whole conversation.
  - The trace (`--trace-dir`) records what was shown and called, but not the
    model's words between calls, which never reach the server.
- **Not yet:** a one-click `.mcpb` Desktop Extension bundle.
