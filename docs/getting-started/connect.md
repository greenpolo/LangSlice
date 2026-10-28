# Connect your AI app

LangSlice has no chat window of its own. You talk to an AI agent in an app
from your AI provider, and the agent uses LangSlice's tools on your computer.
This page helps you choose the app; each app then has its own page of steps.

## Which AI do you use?

Pick the row for the AI account you already have.

| You use | App | Plan | Choose this if |
|---|---|---|---|
| ChatGPT | [Codex, in the ChatGPT desktop app](connect-codex.md) | Free plan works, with low usage limits | You have a ChatGPT account, or no AI account yet (**recommended** then: make a free ChatGPT account) |
| Gemini | [Antigravity](connect-antigravity.md) | Free tier works, with weekly limits | You have a Google account or a Gemini subscription |
| Claude | [Claude Code, in the Claude desktop app](connect-claude-code.md) | Needs a paid Claude plan | You pay for Claude. On the free Claude plan, use Codex with a free ChatGPT account instead |

A paid plan is recommended whichever app you choose. Paid plans give the agent
stronger models and higher usage limits. A registration is a long task for the
agent, so both matter.

## What connecting does

Each app's agent can use tools that run on your computer. LangSlice provides
its tools through **MCP** (Model Context Protocol), a standard way for AI apps
to use programs on your computer. Connecting LangSlice adds three things to
your app:

- LangSlice's tools, as a **local MCP server**: a program the app starts on
  your computer;
- a **skill** named `register-brain`: instructions that tell the app's agent
  how to set up a registration and check it;
- two **registration agents**, `register` and `register-scripting`: the
  helpers the main agent starts to register one brain each.

You talk to the main agent. It asks you about your sections, prepares the
registration, starts a registration agent, and checks the result with you.

## What leaves your computer

The agent looks at pictures LangSlice draws of your sections. Those pictures,
and everything you type, go to the company that runs your app: OpenAI,
Google or Anthropic. Your original image files stay on your computer and are
never changed.

Each registration counts against your plan's usage limits.

```{toctree}
:hidden:

connect-codex
connect-antigravity
connect-claude-code
```
