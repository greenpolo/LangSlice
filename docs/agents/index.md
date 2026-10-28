# For agents

These pages are written for an AI agent that installs or drives LangSlice for
a user. They are reference, literal to the code. People start at the front
page instead.

| Page | Read it to |
|---|---|
| [Install LangSlice (for agents)](install.md) | Install LangSlice from source, verify it, and connect it to Codex, Antigravity or Claude Code |
| [The agent CLI: `langslice-job`](cli.md) | Create a job, write its launch prompt, call verbs and read their replies |
| [The MCP server](mcp.md) | Serve the tools to an MCP host and open a job there |
| [The built-in agent](builtin.md) | Run LangSlice's own agent session: model access, job settings, spend, logs |

Every interface works through one job folder (`<images>/langslice/`) and the
same registration verbs:

| Interface | Who drives it | Entry point |
|---|---|---|
| Agent CLI | A coding agent with a shell, such as Claude Code, Antigravity or Codex | `langslice-job FOLDER VERB` |
| MCP | An agent in an MCP host, or a coding agent's `register` role | `langslice mcp` |
| Built-in agent | LangSlice's agent session | `langslice linear run FOLDER` |
| Python | A coding agent writing its own workflow | [Python library](../developers/library.md) |
| ABBA | The agent works on sections selected in ABBA | [ABBA](../abba.md) |

Rules for a main session that sets up a registration (the connectors'
`register-brain` skill holds them in full):

- Never register sections yourself. Create the job, launch one registration
  agent, check its result.
- Put the user's notes in the job (`init --notes`). Pass the launch prompt
  (`langslice-job FOLDER launch --role register|register-scripting`,
  `result.prompt`) to the registration agent unchanged.
- Run `launch` again for every launch; it reflects the job as it stands.
- Check the result by looking at the saved pictures, not only the agent's
  summary.

```{toctree}
:hidden:

install
cli
mcp
builtin
```
