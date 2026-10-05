# LangSlice in your existing ABBA

The Fiji connector runs LangSlice in a separate Python environment. Open your
usual ABBA installation; its **Register > LangSlice Registration…**
entry opens LangSlice, and Fiji's **Plugins > LangSlice > LangSlice setup…** opens
setup. You may install the Fiji plugin or the Python environment first.
Account setup is done inside ABBA. Command-line login is optional.

**Distribution:** the connector and the Python environment are installed
from source; there is no Fiji update site or conda package (the conda package
needs `itk-elastix`, which conda-forge does not publish; see
[`packaging/README.md`](https://github.com/greenpolo/LangSlice/blob/main/packaging/README.md)).

## Install the Python environment

1. Install [Miniforge](https://github.com/conda-forge/miniforge#download).
2. Download and extract LangSlice's source archive. Open a terminal in the
   extracted folder containing `environment.yml`.
3. Run:

   ```bash
   conda env create -f environment.yml
   ```

This creates an environment named `langslice`, including its Python dependencies.
Leave it installed in that location. You can close the terminal when installation
finishes. You do not need Java, Maven or `abba-python` in this environment.

If an environment named `langslice` already exists, choose another name with
`conda env create -n langslice-fiji -f environment.yml` and select that environment
in Setup. This avoids changing an environment used by another project.

## Install the Fiji connector

Use an existing ABBA **0.24** installation (ABBA 0.24 runs on Java 21). The
connector is built and tested against ABBA 0.24.1; older ABBA versions such as
0.11 are not supported.

For this source preview, build the connector following
[the connector instructions](https://github.com/greenpolo/LangSlice/blob/main/connectors/fiji/README.md),
copy `langslice-fiji-0.2.0.jar` into Fiji's `jars` folder, and restart Fiji before
opening ABBA. Keep only one version of the connector installed.


## Set up LangSlice in ABBA

Open Fiji's **Plugins > LangSlice > LangSlice setup…**, or click **Setup…** in the
LangSlice Registration dialog.

1. Select the environment folder containing LangSlice. The dialog discovers
   common conda environments; **Browse…** lets you select another location.
   Select the environment folder, not `python.exe`, an image folder or the
   LangSlice source folder.
2. Click **Check installation**. The plugin checks that the worker starts and
   speaks a compatible protocol.
3. Choose **Sign in with ChatGPT** and finish in your browser, or use
   **Save API key…** for OpenAI or Gemini. If the browser did not open, use
   **Open sign-in page**. API usage is billed by the provider. Model requests
   send section images to the selected provider. The Registration dialog's agent
   runs on the ChatGPT account; a saved OpenAI or Gemini key can serve as the
   image model for Nonlinear.
4. Save setup. The environment location is remembered for future sessions.

Credentials stay in LangSlice's user settings, not the ABBA project. Setup
reports credential presence; it does not claim that an API key has been validated
online or that the account has access to every model. Environment variables
explicitly supplied to Fiji take precedence over saved API keys.

If the plugin is installed first, Setup still opens. Install the worker, select
its environment folder, and check again. If the environment is moved or removed,
choose its new location. Opening the agent before setup is complete takes you
through setup first.

## Run and save

Import sections into ABBA, select the sections to process (or leave none selected
to use all), and open **Register > LangSlice Registration…** (at the end of ABBA's
Register menu). The dialog has three tabs:

- **Tasks:** turn on any of **Positioning** (order and atlas positions; check the
  section thickness and interval, which are filled in from ABBA), **Linear** (in-plane
  alignment, hemisphere flips with an optional hemisphere cue, and, if you tick
  **Enable slice angle estimation**, the atlas cutting angles) and **Nonlinear** (a
  deformation per slice on top of its linear placement; choose the fitting engine).
  Each has a box for notes to the agent. Nonlinear builds on a linear registration:
  if some slices have none and Linear is off, LangSlice asks whether the agent should
  align them first; answering No leaves those slices out of Nonlinear.
- **Slices:** the slices that will be sent, with their number of ABBA
  registrations. Tick **Damaged** for torn or folded slices. By default, slices that
  already have ABBA registrations keep their in-plane alignment; their positions
  can still move. Tick **Allow the agent to overwrite existing transforms** to let
  the agent realign them too.
- **Preprocessing:** **Auto** usually works. **Custom** lets you weight each image
  channel and set local contrast (CLAHE). **Let the agent drive preprocessing** lets
  the agent adjust what it sees itself (every channel is then sent). The preview shows
  each slice as exported and as the agent will see it. Try to maximize contrast
  between different regions.

Choose the agent model at the top, and the image model for Nonlinear (or **None (fit
to the stain only)**), check the estimated cost at the bottom, and click **Run**.
**Open agent viewer** is available only when ABBA was started with `langslice abba`.
Your choices are remembered for the next run. Tick **Save traces to**
to keep a full record of what the agent saw and did in a folder of your choice;
it is useful when reporting a problem.

The agent's changes appear in ABBA while it works: each step it saves is one ABBA
step, which ABBA's **Undo** reverts. LangSlice adds at most two registrations to a
slice: an affine and, with Nonlinear, a BigWarp warp on top of it; you can open the
warp in BigWarp like any BigWarp registration, and the saved project opens in ABBA
without LangSlice. **Stop run** ends the agent; what it saved before stays in ABBA.
With **Show agent log** on, the run window shows what the agent is doing; with it
off, a small window shows only the status and the final message. Save the project
using ABBA's normal Save command.

The connector currently requires the Allen Mouse V3 or V3p1 atlas in a coronal
session; tilted cutting angles are fine. Avoid changing the listed slices'
registrations while a run is active: LangSlice then leaves such a slice alone and
says so, and offers **Retry failed updates** at the end.

## Optional terminal commands

For troubleshooting, these commands use the same installation and login:

```bash
conda activate langslice
langslice version
langslice login
```

`langslice abba` starts ABBA from Python, with the connector, the agent viewer and
the agent log; it needs Java 21 (fetched on first start) and the connector jar.
The connector does not need it.
