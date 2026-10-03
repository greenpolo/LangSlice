# Claude Desktop / Claude Code connector

LangSlice can serve its linear tools over MCP to a host that brings its own
model. The host runs the conversation and LangSlice supplies the job, the
pictures and the tools. The user's own Claude subscription runs the model in
Anthropic's own app, so LangSlice never handles Claude credentials.

The server is `langslice mcp` (code: `src/langslice/mcp_server/`). It needs
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
Claude; results remain under `~/.langslice/jobs/<job-id>/`. Enable only LangSlice
for this conversation, never a general Fiji scripting connector.

Without a registration host, the command line is the Copy prompt:

```bash
langslice claude prepare FOLDER --interval 200 --notes "Section 12 has a large tear."
```

It takes every `langslice linear run` job flag, saves the job under
`~/.langslice/jobs/<job-id>/` and prints the prompt to paste. The job's
checkpoint and results live in that directory, never in the image folder, and
reopening the job (a restarted Desktop, a new chat) resumes from the checkpoint.

For development, ask Claude to register a folder of sections. Claude calls
`start_job` with the folder path. `langslice mcp` takes every `langslice linear
run` flag (`--tasks`, `--interval`, `--atlas`, `--trace-dir`, ...), and those
flags apply to development folders, not saved jobs. Put them in `args`.
Image-generation tasks are unavailable through this connector.

## Claude Code, locked to LangSlice

Use this for testing on Linux, where Claude Desktop does not run on Fedora. It
removes every built-in tool and loads only this server:

```bash
claude --strict-mcp-config --mcp-config connectors/claude-desktop/langslice.mcp.json \
  --tools "" --allowedTools "mcp__langslice"
```

## What the host gets and what it does not

- **Same as the ADK run:** the same tools (picture options in one `view`
  object; an unknown or misplaced argument is refused, not dropped),
  docstrings, submit gates, undo stack and checkpoint. Saved ABBA jobs also
  share the same validation, snapshot preprocessing and native update
  translation.
- **Claude briefing:** `start_job` returns a compact factual statement and status
  table without images. Read every `show_stack(page=...)` page before writing.
  Sections arrive as labelled strips in the stack's order (1568 px long, Claude's
  recommended largest image), the atlas at each section's current position
  beneath it; atlas reference strips follow when a section has no position.
  Every page stays below 680,000 serialized bytes.
- **Different from the ADK run:**
  - There is no turn budget and no nudges.
  - There is no image working set, so the host keeps every picture for the
    whole conversation.
  - The trace (`--trace-dir`) records what was shown and called, but not the
    model's words between calls, which never reach the server.
- **Not yet:** a one-click `.mcpb` Desktop Extension bundle.
