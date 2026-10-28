# Your data

LangSlice runs on your computer. The registration methods run there too. What
leaves your computer is what an AI model needs to see: pictures of your
sections and text about the job. It goes to the AI provider you chose, under
your account.

## In short

| Leaves your computer | Stays on your computer |
|---|---|
| Pictures of your sections and of the atlas, drawn by LangSlice, sent to your AI provider | Your original image files, never changed |
| Text about the job: filenames, your notes, positions and settings | The job folder: every result, picture, log and transcript |
| Only if image-model tracing is used: two pictures per traced section, sent to the image model's provider | The Elastix, FireANTs and landmark fits |
| | Your sign-in and API keys (passwords that bill you per request) |

## What your AI provider receives

An AI model does the registration, and it runs on the provider's servers.
Everything the model sees is sent there.

**Through your AI app** (Codex, Antigravity, or Claude Code). Your app's agent
calls LangSlice's tools on your computer. The tools answer with pictures and
text. The app sends those to its provider (OpenAI, Google or Anthropic) as
part of the conversation, under your account. LangSlice itself sends
nothing in this route.

**Through LangSlice's own agent** (`langslice linear run`, and the agent built
into the ABBA connector). LangSlice sends the same kind of content directly
to the provider you set up:

| How you connected | Where it goes |
|---|---|
| Signed in with ChatGPT (`langslice login`) | OpenAI, through your ChatGPT plan |
| OpenAI API key | OpenAI |
| Gemini API key | Google |
| Anthropic API key | Anthropic |
| A model server you name yourself (OpenRouter, Ollama, or another server that speaks OpenAI's format) | That server; a server on your own computer keeps everything local |

In both routes the content is:

- **Pictures** of your sections, alone or with atlas borders drawn on them,
  and pictures of the atlas. Most are smaller than your files. A close-up
  (a *zoom*) is redrawn from your original file at its full resolution. It
  can show part of a section at the full detail of your file.
- **Text**: the job statement, your notes for the agent, and the replies of
  the tools. The job statement names the atlas, the number of sections, their
  filenames and any facts you supplied. The replies hold positions,
  transforms, region names and file names in the job folder.

Your app's agent can also read other files on your computer, as far as the
app's own permissions allow. What it reads goes to the provider like
anything else in the conversation. That is how the app works, not something
LangSlice does.

## Image-model tracing

Image-model border tracing is optional (see
[How LangSlice registers a brain](how-registration-works.md#image-model-border-tracing)).
It exists in a job only when the deformation stage is on and an image model
is connected. When the agent traces a section, LangSlice sends the image
model:

- a picture of the section, cropped around the brain;
- the same picture with the atlas borders drawn on it in yellow;
- the instructions for the model, which the agent may edit for that section.

LangSlice sends this itself, to the image model's provider:

| Image model | Where it goes |
|---|---|
| Signed in with ChatGPT (`langslice login`), the default | OpenAI, through your ChatGPT plan |
| OpenAI API key | OpenAI |
| Gemini API key | Google |

This is separate from your AI app. If you use Claude Code and have also
signed in to LangSlice with ChatGPT, traced sections go to OpenAI. Without a
sign-in or key, the tracing tool is not offered at all. To keep it off, set
the job's image model to `none` (see [Job settings](../reference/settings.md)).

## What stays on your computer

- **Your image files.** LangSlice reads them and never changes them.
- **The job folder.** It sits beside your images, in a folder named
  `langslice`. It holds the settings, every result, every picture the agent
  looked at, the undo history, the logs, and the agent's transcript. Nothing
  in it is uploaded by LangSlice.
- **The registration methods.** Elastix, FireANTs and the landmark warp run
  on your computer.
- **The LangSlice window.** It is served to your own computer only, and other
  computers cannot reach it.
- **The ABBA connection.** The ABBA connector talks to LangSlice on your
  computer only.
- **Your sign-in and API keys.** They are kept in a `.langslice` folder in
  your home folder, never in a job folder.

The first time you use an atlas, LangSlice downloads it from BrainGlobe. That
download sends nothing about your data.

## What the provider does with it

How long the provider keeps what it receives, and whether it may use it to
train models, depends on your plan and the provider's terms. LangSlice does
not change either. Check your plan's data settings before you send
unpublished images.
