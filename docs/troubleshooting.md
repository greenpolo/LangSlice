# Troubleshooting

Messages from a registration usually reach you through your agent, which may
reword them.

## The terminal does not know `conda`

On Windows, open **Miniforge Prompt** from the Start menu and run the command
there. On macOS and Linux, close the terminal and open a new one. If it still
says `conda: command not found`, run this once, then open a new terminal:

```bash
~/miniforge3/bin/conda init "$(basename "$SHELL")"
```

## The terminal does not know `langslice`

Run `conda activate langslice` in that terminal. Once your AI app has set
LangSlice up, `langslice` works in every new terminal without it.

## The app cannot start LangSlice

The app says LangSlice's server failed to start, or its agent has no
LangSlice tools, asks permission before every tool, or stops a tool after a
minute. In a new conversation, send **Set up LangSlice.**, then quit the app
completely and open it again. Setup is step 3 of each app's page:
[Codex](getting-started/connect-codex.md),
[Antigravity](getting-started/connect-antigravity.md),
[Claude Code](getting-started/connect-claude-code.md).

## The first registration waits a long time before anything happens

LangSlice is downloading the atlas, about 600 MB for the Allen mouse atlas at
25 µm. It does this once per atlas.

## A second registration took over the first one's job

LangSlice's tools hold one job per conversation. Start a new conversation for
each brain.

## “… is locked by another writer”

Another program is writing to the same job, such as a second agent on the
same brain. Wait for it to finish; run one registration per job at a time.

## “The job in this folder was made with other settings”

The folder already holds a job made another way, such as with another atlas.
Continue it as it was made, or ask your agent to start it over, which
discards its work.

## Something else

Open an issue at
[github.com/greenpolo/LangSlice/issues](https://github.com/greenpolo/LangSlice/issues)
with the LangSlice version (`langslice version`), your operating system and
AI app, what you did, and the full message.
