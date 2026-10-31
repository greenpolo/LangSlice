# Claude Desktop MCP connector

LangSlice serves registration tools over MCP; the host runs the agent
conversation using its own account. Setup, job creation and tool behavior
are described in [The MCP server](../../docs/agents/mcp.md).

[`langslice.mcp.json`](langslice.mcp.json) configures the server with
`langslice mcp`. Use the absolute path to `langslice` in your installed
Python environment when the host does not inherit that environment.

## Claude Code with only MCP tools

For a Claude Code session restricted to this server:

```bash
claude --strict-mcp-config --mcp-config connectors/claude-desktop/langslice.mcp.json \
  --tools "" --allowedTools "mcp__langslice"
```

For a coding agent that also writes scripts and uses a shell, use the
[Claude Code agent CLI setup](../claude-code/README.md).

## Jobs prepared in ABBA

ABBA's **Register › LangSlice…** opens the LangSlice window on the job it
prepared; its prompt generator writes the message to paste into the
conversation, and the job then starts like any other (`start_job` with the
job's folder). While ABBA is open, every write appears in ABBA
([ABBA](../../docs/abba.md)).
