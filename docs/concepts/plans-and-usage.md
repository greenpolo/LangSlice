# Plans and usage limits

LangSlice is free. The AI model that does the registration is not part of
it. The model runs on your AI provider's plan, and a run uses up part of that
plan's allowance. Subscriptions are highly recommended for both model quality and
usage limits.

## Which plan works with which app

| You have | App | Free plan |
|---|---|---|
| ChatGPT | Codex app | Included on ChatGPT Free, for light use |
| Gemini | Antigravity app | Free tier with weekly limits |
| Claude | Claude Code, in the Code tab of Claude Desktop | Needs a paid Claude plan. With Claude Free, use the Codex app with a free ChatGPT account |

Providers change their plans. Check your provider's plan page for current
limits.

## Why a paid plan helps

- **Model quality.** Registration asks the model to judge anatomy in many
  pictures. Paid plans give access to stronger models.
- **Usage limits.** A run is long, steady work on many pictures. On a free
  plan it is more likely to reach the plan's limit and stop before the stack
  is done.

## What a run uses

Every step the agent takes is a request to the model. Each request carries
the conversation so far, including the pictures, and the model's answer and
reasoning. What counts against your plan grows with:

- **the number of sections**;
- **the stages switched on**: a deformation stage adds work for each section
  the agent deforms;
- **how many pictures the agent looks at, and their size**;
- **the reasoning effort** you choose for the model, where the app lets you
  choose;
- **image-model tracing**, which sends an image-edit request for each traced
  section, on top of the agent's own requests.

LangSlice sets no limit on the number of steps, tool calls, tokens or time,
and stops no run for what it has used. The agent is told so, and told that
quality comes first. It works until it judges the registration finished.

The fits themselves (Elastix, FireANTs, landmarks) run on your computer and
use none of your plan. FireANTs is faster on a computer with a graphics card
(GPU).

## When you reach a limit

When your plan's limit is reached, the provider refuses further requests and
the run stops. Nothing is lost: every change was saved in the job folder as
it was made. When your allowance returns, open the same job again, in your
app or with LangSlice's own agent, and the work continues from where it
stopped.

## Seeing what a run used

- **In your AI app**, use the app's own usage display. LangSlice cannot see
  your app's usage.
- **With LangSlice's own agent**, LangSlice reports usage as the run goes:
  - signed in with ChatGPT, it warns each time the run has used another
    10 percentage points of your plan's usage window;
  - with an API key (a password that bills you per request), it keeps a running dollar estimate from the provider's
    list prices, and warns at $5, $10, $20, $40 and so on.

  The account is saved in the job folder and shown under **Spend** in the
  [LangSlice window](../developers/gui.md). The model never sees it.

With an API key you pay the provider per request instead of a monthly plan.
The estimate is LangSlice's arithmetic, not your bill.
