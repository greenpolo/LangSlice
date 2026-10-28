# The MCP server

`langslice mcp` serves the registration tools over MCP (stdio) to a host that
brings its own model.

## Configure a host

Configure an MCP host to launch the `langslice` executable from its installed
environment, with `mcp` as its argument. The coding-agent connectors' `register`
role works through this server ([ready-made setups](cli.md#ready-made-setups)). The server
advertises that its tool list changes, so a host takes the job's tools once
`start_job` has opened it. In Claude Desktop, open Settings >
Developer > Edit Config and add this to `claude_desktop_config.json`, then
restart Desktop:

```json
{
  "mcpServers": {
    "langslice": {
      "command": "/absolute/path/to/langslice-env/bin/langslice",
      "args": ["mcp"]
    }
  }
}
```

Use the environment's `Scripts/langslice.exe` on Windows. The source
environment includes MCP; a minimal pip installation needs the `mcp` extra.

## Start a job

Create a job with `langslice-job FOLDER init`, then launch the host's agent
with `langslice-job FOLDER launch --role register`'s `result.prompt`, which
names the folder and holds the job statement. The agent calls
`start_job(image_folder="/absolute/path/to/FOLDER")`; its reply says only that
the job is open, its section count and the pages of its opening (the server's
instructions are one short text, the same for every job). For a job prepared
in ABBA, paste ABBA's copy prompt, which supplies `start_job(job_id=...)`; that
call answers with the whole job statement, as that path has no launching
message. The agent must read every `show_stack(page=...)` page before writing.
Tool replies include images directly; pages and replies have a size budget,
and a reply reports when its images were reduced to fit.

Opening an existing job keeps its saved settings and checkpoint. For a
folder without a job, the server's job flags set the defaults. Use one
conversation per job. The MCP host controls the conversation, model budget
and image history; LangSlice supplies the tools and submit checks.

