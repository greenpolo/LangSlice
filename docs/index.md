# LangSlice

LangSlice registers histological brain sections to BrainGlobe atlases.
An agent inspects the stack, sets atlas positions and cutting angles,
aligns sections in-plane, and optionally fits deformations. It chooses
regions and settings, reviews the results, and can use an image model to
trace borders where useful.

Every interface works through a job folder. Source images stay unchanged;
writes are checkpointed and undoable. Results include coordinate and label
maps, a readable registration file, and QuickNII / VisuAlign exports.
Coronal sections of the Allen mouse atlas are the most tested case.

## Start here

| Guide | What it covers |
|---|---|
| [Agents, CLI and MCP](agents.md) | Coding agents, Claude Desktop and LangSlice's built-in agent; setup, settings and replies |
| [ABBA](abba.md) | Install the Fiji connector and register sections inside ABBA |
| [Registration tools](registration.md) | Tasks, calibration, viewing, affine fits, ANTs and optional image-model traces |
| [Python library](library.md) | Write agent workflows using the same job and tools |
| [File formats](file_formats.md) | Job files, coordinate conventions, maps and registration imports/exports |
| [Architecture](architecture.md) | Package layers, shared operations and the Fiji worker protocol |

## Install from source

Download and extract the source, then run in the folder containing
`environment.yml`:

```bash
conda env create -f environment.yml
conda activate langslice
```

The environment includes registration engines, MCP and the Python ABBA
launcher. On Windows, use Miniforge Prompt or PowerShell with conda
initialized. Atlases download on first use. Continue with the setup for
your chosen interface above.
